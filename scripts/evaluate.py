#!/usr/bin/env python3
"""Evaluation harness (improvement #6): walk-forward multi-class
scoring of the frozen track records.

Reads data.json (NBA track.nights + NFL nfl.track.weeks) and
optionally data/calibration_log.jsonl, and prints, per market:

  * n scored games, log loss (multi-class, vs uniform ln(10) = 2.303),
  * multi-class Brier score + Murphy decomposition (reliability /
    resolution / uncertainty),
  * ECE (top-pick confidence bins),
  * top-1 hit rate (vs 10% uniform),
  * bootstrap 95% CIs for log loss and hit rate (default 1000
    resamples; seeded, runs in seconds),
  * ROI vs stored odds (closing_odds.jsonl, written by
    scripts/devig.py --odds): flat-stake ROI betting the model's top
    pick at the recorded odds. With no stored odds the ROI section
    is skipped, not invented.

Minimum-N rules (research, Nov 2026): n < 100 -> log-loss and hit
rate are unstable, treat as directional only; n < 500 -> do NOT quote
a yield/ROI claim (a yield needs >= 500 settled bets). The harness
prints those warnings itself so a small-sample number is never
mistaken for a verdict.

This is a READ-ONLY report: it never writes data.json and is NOT
wired into the nightly workflow (run it on demand after a night's
scoring, e.g. `python scripts/evaluate.py --data data.json`).

The walk-forward property comes from the data: every track game was
scored with the probabilities frozen the night BEFORE the game, so
each night's numbers are genuinely out-of-sample for the model as it
stood then. The harness does not refit anything.
"""

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shrinkage import logloss, brier_score, murphy_decomposition, ece  # noqa: E402

UNIFORM_LL = math.log(10)          # 2.303, uniform over 10 starters
UNIFORM_BRIER = 0.90               # 9*0.01 + 0.81 for a uniform 10-way
UNIFORM_HIT = 0.10
MIN_N_STABLE = 100                 # below this, metrics are directional
MIN_N_YIELD = 500                  # below this, no yield/ROI claim


def _norm_probs(pairs):
    """[(name, prob)] -> {name: prob} renormalized to sum 1 (the frozen
    track probs sum to ~0.96 — the bench share)."""
    tot = sum(p for _n, p in pairs)
    if tot <= 0:
        return {}
    return {n: p / tot for n, p in pairs}


def score_games(games):
    """games: [{probs: {name: prob}, scorer: name}]. Returns a dict of
    metrics, or None when no scored games are present. All metrics
    are computed on the renormalized starter distribution."""
    scored = [g for g in games if g.get("scorer") in g.get("probs", {})]
    excluded = len([g for g in games
                    if g.get("scorer")
                    and g.get("scorer") not in g.get("probs", {})])
    n = len(scored)
    if n == 0:
        return None
    lls, briers, hits, top_conf, top_hit = [], [], [], [], []
    murphy_games = []
    for g in scored:
        probs = g["probs"]
        p_scorer = probs[g["scorer"]]
        lls.append(logloss(p_scorer))
        ordered = sorted(probs.items(), key=lambda kv: -kv[1])
        briers.append(brier_score([p for _n, p in ordered],
                                  next(i for i, (n, _p) in enumerate(ordered)
                                       if n == g["scorer"])))
        murphy_games.append(([p for _n, p in ordered],
                             next(i for i, (n, _p) in enumerate(ordered)
                                  if n == g["scorer"])))
        top = ordered[0]
        hits.append(1.0 if top[0] == g["scorer"] else 0.0)
        top_conf.append(top[1])
        top_hit.append(1.0 if top[0] == g["scorer"] else 0.0)
    mur = murphy_decomposition(murphy_games)
    ece_val, _ece_table = ece(top_conf, top_hit)
    return {
        "n": n,
        "excluded": excluded,
        "logloss": sum(lls) / n,
        "brier": sum(briers) / n,
        "hit_rate": sum(hits) / n,
        "ece": ece_val,
        "murphy": {"reliability": mur["reliability"],
                   "resolution": mur["resolution"],
                   "uncertainty": mur["uncertainty"]},
        "_lls": lls, "_hits": hits,
    }


def bootstrap_ci(values, n_boot=1000, seed=7):
    """95% percentile CI of the mean of `values` (seeded)."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(values) for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return lo, hi


def nba_games(data):
    """track.nights -> [{probs, scorer}] for scored games."""
    out = []
    for night in (data.get("track") or {}).get("nights", []):
        for g in night.get("games", []):
            act = g.get("act")
            if not act:
                continue
            probs = _norm_probs([(n, p) for n, _t, p in g.get("p", [])])
            if act.get("fg") in probs:
                out.append({"probs": probs, "scorer": act["fg"],
                            "date": night.get("d"), "game": g.get("g")})
    return out


def nfl_games(data):
    """nfl.track.weeks -> [{probs, scorer}] for scored games."""
    out = []
    for week in (data.get("nfl") or {}).get("track", {}).get("weeks", []):
        for g in week.get("games", []):
            act = g.get("act")
            if not act:
                continue
            probs = _norm_probs([(n, p) for n, _t, p in g.get("p", [])])
            if act in probs:
                out.append({"probs": probs, "scorer": act,
                            "date": g.get("d"), "game": g.get("g")})
    return out


def calibration_log_games(path):
    """calibration_log.jsonl -> [{probs, scorer}] using the stored
    rates + adjusted shares (same normalization the fitter uses:
    team_prob * adj, renormalized by their sum)."""
    if not path or not os.path.exists(path):
        return []
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fit_calibration as F  # noqa: E402
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fg = r.get("first_fg")
            if not fg or r.get("team_first") is None:
                continue
            pa = F.team_first_prob(r["awayTip"], r["awayRate"],
                                   r["homeRate"])
            raw = {}
            for s in r.get("starters", []):
                raw[s["n"]] = ((pa if s["team"] == r["away"] else 1 - pa)
                               * s["adj"])
            probs = _norm_probs(list(raw.items()))
            if fg.get("player") in probs:
                out.append({"probs": probs, "scorer": fg["player"],
                            "date": r.get("date"),
                            "game": f"{r.get('away')}@{r.get('home')}"})
    return out


def roi_vs_odds(games, odds_path):
    """Flat-stake ROI betting the model's top pick at the recorded
    odds. odds_path is a JSONL of {date, game, player, odds}
    (American). Returns None when no odds file or no joined bets."""
    if not odds_path or not os.path.exists(odds_path):
        return None
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from devig import american_to_decimal  # noqa: E402
    odds = {}
    with open(odds_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            odds[(r.get("date"), r.get("game"), r.get("player"))] = \
                r.get("odds")
    staked, returned, bets = 0.0, 0.0, 0
    for g in games:
        if not g.get("probs"):
            continue
        top = max(g["probs"].items(), key=lambda kv: kv[1])[0]
        dec = american_to_decimal(
            odds.get((g.get("date"), g.get("game"), top)))
        if dec is None:
            continue
        bets += 1
        staked += 1.0
        if top == g.get("scorer"):
            returned += dec
    if bets == 0:
        return None
    return {"bets": bets, "staked": staked, "returned": round(returned, 2),
            "roi": round(returned / staked - 1, 4) if staked else None}


def _fmt_ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def report_section(title, metrics, n_boot, seed, odds_path=None,
                   games=None):
    """Print one market's section. metrics is score_games() output."""
    print(f"\n== {title} ==")
    if metrics is None:
        print("  no scored games yet")
        return
    n = metrics["n"]
    ll_lo, ll_hi = bootstrap_ci(metrics["_lls"], n_boot, seed)
    h_lo, h_hi = bootstrap_ci(metrics["_hits"], n_boot, seed)
    print(f"  n = {n} scored games"
          + (f" ({metrics.get('excluded', 0)} scored games excluded — "
             f"scorer outside the ranked 10)"
             if metrics.get("excluded") else ""))
    print(f"  log loss  {metrics['logloss']:.3f} "
          f"(95% CI {_fmt_ci(ll_lo, ll_hi)}; uniform {UNIFORM_LL:.3f})")
    print(f"  Brier     {metrics['brier']:.3f} (uniform {UNIFORM_BRIER:.2f})"
          f"  [reliability {metrics['murphy']['reliability']:.3f}, "
          f"resolution {metrics['murphy']['resolution']:.3f}, "
          f"uncertainty {metrics['murphy']['uncertainty']:.3f}]")
    print(f"  ECE       {metrics['ece']:.3f} (top-pick confidence bins)")
    print(f"  hit rate  {metrics['hit_rate']:.1%} "
          f"(95% CI {_fmt_ci(h_lo, h_hi)}; uniform {UNIFORM_HIT:.0%})")
    if n < MIN_N_STABLE:
        print(f"  WARNING: n={n} < {MIN_N_STABLE} — log-loss / hit rate "
              f"are directional only, not a verdict.")
    roi = roi_vs_odds(games, odds_path) if games is not None else None
    if roi is None:
        print("  ROI       no stored odds — run scripts/devig.py --odds "
              "with FanDuel lines to enable it.")
    else:
        print(f"  ROI       {roi['roi']:+.1%} on {roi['bets']} flat-stake "
              f"top-pick bets (returned {roi['returned']} on "
              f"{roi['staked']:.0f} staked)")
        if roi["bets"] < MIN_N_YIELD:
            print(f"  WARNING: {roi['bets']} bets < {MIN_N_YIELD} — do "
                  f"NOT quote this ROI as a yield.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data.json")
    ap.add_argument("--cal", default=None,
                    help="calibration_log.jsonl (default: data/ "
                         "calibration_log.jsonl next to --data)")
    ap.add_argument("--odds", default=None,
                    help="closing_odds.jsonl (default: data/ "
                         "closing_odds.jsonl next to --data)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default=None,
                    help="also write the metrics as JSON to this path")
    args = ap.parse_args()
    with open(args.data) as f:
        data = json.load(f)
    root = os.path.dirname(os.path.abspath(args.data))
    cal_path = args.cal or os.path.join(root, "calibration_log.jsonl")
    if args.cal is None:
        alt = os.path.join(root, "data", "calibration_log.jsonl")
        cal_path = alt if os.path.exists(alt) else cal_path
    odds_path = args.odds or os.path.join(root, "data",
                                          "closing_odds.jsonl")
    as_of = (data.get("_meta") or {}).get("updated", "?")
    print(f"evaluation harness — data as of {as_of}")
    out = {}
    nba = nba_games(data)
    m_nba = score_games(nba)
    report_section("NBA first basket (frozen track)", m_nba,
                   args.bootstrap, args.seed,
                   odds_path=odds_path, games=nba)
    out["nba_track"] = {k: v for k, v in (m_nba or {}).items()
                        if not k.startswith("_")}
    cal = calibration_log_games(cal_path)
    m_cal = score_games(cal)
    report_section("NBA first basket (calibration log, stored rates)",
                   m_cal, args.bootstrap, args.seed)
    out["nba_calibration_log"] = {
        k: v for k, v in (m_cal or {}).items() if not k.startswith("_")}
    nfl = nfl_games(data)
    m_nfl = score_games(nfl)
    report_section("NFL first TD (frozen track)", m_nfl,
                   args.bootstrap, args.seed,
                   odds_path=odds_path, games=nfl)
    out["nfl_track"] = {k: v for k, v in (m_nfl or {}).items()
                        if not k.startswith("_")}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
