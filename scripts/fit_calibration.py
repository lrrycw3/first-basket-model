#!/usr/bin/env python3
"""On-demand fitter for the First Basket Model calibration log (audit
items #3/#4: fitting K / bench share / the 45% rate anchor).

Reads data/calibration_log.jsonl — one record per game, written nightly by
scripts/update_results.py. Each record carries the model's inputs for that
game (awayTip, awayRate, homeRate, the raw spread/total, and each starter's
model share sh, adjusted share adj, first-basket count y and games-started
g) plus the outcome (team_first + first_fg), filled the next night.

This script does NOT run nightly and does NOT change any parameter. It
reports:
  - how many games have outcomes yet (the fit needs real data),
  - the current model's log-loss against the actual first-FG scorer,
    vs a naive baseline (uniform 1/10 per starter) and vs the page's
    naive 9.6% first-basket baseline,
  - when >= 30 games have outcomes, a grid search over K, bench share
    and rate anchor, reporting the best-fitting values and their log-loss
    vs the current (K=20, bench 96%, anchor 45%).

Do NOT apply a fitted value to the page until the fit is stable across
enough nights and the parent/user has approved it. The page's K, bench
share and anchor stay at 20 / 96% / 45% until then.

Usage: python scripts/fit_calibration.py [--root .] [--min-games 30]
"""
import argparse
import json
import math
import os
import sys

# Current page parameters (the fit compares against these).
CUR_K = 20
CUR_BENCH = 96.0
CUR_ANCHOR = 0.45


def first_poss(a, b):
    return a / (a + b - a * b)


def team_first_prob(away_tip, away_rate, home_rate):
    tip = away_tip / 100.0
    return (tip * first_poss(away_rate / 100.0, home_rate / 100.0)
            + (1 - tip) * (1 - first_poss(home_rate / 100.0,
                                          away_rate / 100.0)))


def rates_from_line(spread_home, total, anchor):
    """Score rates for a given anchor (the page uses 45%). The pipeline's
    score_rates() computes rate = 90 * implied / total at the 45%
    anchor, i.e. 200 * anchor * implied / total in percent units — the
    200 factor is the 2x that turns the 45% possession score rate into
    the ~45 rate the page stores (implied/total is ~0.5). A 100x
    factor here would halve every rate (22 instead of 45) and silently
    corrupt every fit that recomputes rates from the raw line."""
    s = abs(spread_home)
    if spread_home < 0:
        home_imp, away_imp = (total + s) / 2, (total - s) / 2
    else:
        away_imp, home_imp = (total + s) / 2, (total - s) / 2
    return (200 * anchor * away_imp / total,
            200 * anchor * home_imp / total)


def logloss(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -math.log(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".",
                    help="repo root (holds data/calibration_log.jsonl)")
    ap.add_argument("--min-games", type=int, default=30,
                    help="games with outcomes needed before fitting")
    ap.add_argument("--write-params", default=None,
                    help="write candidate parameters (K / Platt) to this "
                         "JSON path when the time-split held-out log-loss "
                         "improves; otherwise writes defaults (K=20, no "
                         "Platt). The nightly run reads this file as "
                         "data/calibration_params.json.")
    args = ap.parse_args()
    path = os.path.join(args.root, "data", "calibration_log.jsonl")
    recs = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    done = [r for r in recs if r.get("team_first") is not None
            and r.get("first_fg")]
    print(f"calibration log: {len(recs)} games, {len(done)} with outcomes")
    if not done:
        print("No completed games yet — nothing to evaluate. "
              "The nightly Action fills outcomes the next morning.")
        return 0
    # Current-model log-loss: P(actual first-FG scorer) = team-first prob
    # x the scorer's adjusted share (adj is a fraction of the team-first
    # probability mass; the 10 starters' probs sum to ~1).
    cur_ll, naive_ll, team_ll = 0.0, 0.0, 0.0
    n_cur = 0
    for r in done:
        pa = team_first_prob(r["awayTip"], r["awayRate"], r["homeRate"])
        fg = r["first_fg"]
        scorer = next((s for s in r["starters"]
                       if s["n"] == fg["player"] and s["team"] == fg["team"]),
                      None)
        if scorer is None:
            continue  # bench scorer: the model only ranks the 10 starters
        # The 10 starters' probabilities are team_prob * adj; they sum
        # to ~0.96 (the bench share). Renormalize by THAT sum (condition
        # on a starter scoring). Dividing by sum(adj) instead would be
        # wrong by ~2x: adj is a per-team share (each side sums to
        # ~0.96), so sum(adj) over 10 starters is ~1.92, not ~0.96.
        probs = {}
        for s in r["starters"]:
            probs[(s["n"], s["team"])] = (
                (pa if s["team"] == r["away"] else 1 - pa) * s["adj"])
        tot = sum(probs.values())
        p_scorer = (probs[(fg["player"], fg["team"])] / tot if tot
                    else 0.0)
        cur_ll += logloss(p_scorer)
        naive_ll += logloss(0.10)
        team_ll += logloss(pa if fg["team"] == r["away"] else 1 - pa)
        n_cur += 1
    n = len(done)
    print(f"current model log-loss (first-FG scorer): {cur_ll / n_cur:.3f} "
          f"over {n_cur} starter-scored games (of {n} completed)")
    print(f"naive uniform 1/10 baseline:              {naive_ll / n_cur:.3f}")
    print(f"team-first log-loss (which team first):    {team_ll / n_cur:.3f}")
    print(f"page naive first-basket baseline is 9.6% "
          f"(log-loss {logloss(0.096):.3f})")
    if n < args.min_games:
        print(f"\nOnly {n} completed games — need {args.min_games} before "
              f"fitting K / bench share / anchor. Keep logging.")
        return 0
    # Grid search. K and bench share change the adjusted shares; the
    # anchor changes the rates (recomputed from the raw spread/total).
    # Games whose first-FG scorer was a bench player (not among the 10
    # starters) are skipped — the model only ranks starters — and the
    # fit is reported over the games actually scored.
    best = None
    n_scored = 0
    for k in (0, 5, 10, 15, 20, 30, 50, 100):
        for bench in (90.0, 94.0, 96.0, 98.0):
            for anchor in (0.40, 0.45, 0.50):
                ll = 0.0
                scored = 0
                for r in done:
                    ar, hr = rates_from_line(r["spread_home"], r["total"],
                                             anchor)
                    pa = team_first_prob(r["awayTip"], ar, hr)
                    fg = r["first_fg"]
                    # Rescale model shares to the candidate bench share.
                    scale = bench / 96.0
                    adj = {}
                    for s in r["starters"]:
                        m = s["sh"] * scale
                        if k + s["g"] == 0:
                            # K=0 and a first-timer (g=0): the blend is
                            # undefined; fall back to the model share.
                            adj[s["n"]] = m
                        else:
                            adj[s["n"]] = (k * m + s["y"]) / (k + s["g"])
                    scorer = next((s for s in r["starters"]
                                   if s["n"] == fg["player"]
                                   and s["team"] == fg["team"]), None)
                    if scorer is None:
                        continue
                    probs = {}
                    for s in r["starters"]:
                        probs[(s["n"], s["team"])] = (
                            (pa if s["team"] == r["away"] else 1 - pa)
                            * adj[s["n"]])
                    tot = sum(probs.values())
                    p = (probs[(fg["player"], fg["team"])] / tot
                         if tot else 0.0)
                    ll += logloss(p)
                    scored += 1
                if scored == 0:
                    continue
                if best is None or ll < best[0]:
                    best = (ll, k, bench, anchor, scored)
    ll, k, bench, anchor, n_scored = best
    print(f"\nbest fit over {n_scored} scored games: K={k}, bench={bench}%, "
          f"anchor={anchor:.2f} — log-loss {ll / n_scored:.3f} "
          f"(current K=20, bench=96%, anchor=0.45: {cur_ll / n_cur:.3f})")
    print("Do NOT apply these values until the fit is stable across more "
          "nights and the parent/user approves.")

    # --- Empirical-Bayes K (MoM) + time-split Platt calibration --------
    # The grid above is in-sample (one parameter set fitted on the same
    # 41 games it is scored on), so a "best" K=100 there is as likely
    # noise as signal. The checks below are the honest ones: a
    # method-of-moments K from the Beta-binomial model (no log-loss
    # chasing), and a TIME-SPLIT evaluation — fit on the first 2/3 of
    # dates, score on the last 1/3 — for both K and the Platt map.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from shrinkage import eb_k_mom, platt_fit, platt_apply  # noqa: E402

    def scorer_ll(records, k, bench=96.0, anchor=0.45):
        """Mean log-loss of the actual first-FG scorer over records
        (same recomputation as the grid above). Returns (ll, n)."""
        total, scored = 0.0, 0
        for r in records:
            ar, hr = rates_from_line(r["spread_home"], r["total"], anchor)
            pa = team_first_prob(r["awayTip"], ar, hr)
            fg = r["first_fg"]
            scale = bench / 96.0
            adj = {}
            for s in r["starters"]:
                m = s["sh"] * scale
                adj[s["n"]] = (m if k + s["g"] == 0 else
                               (k * m + s["y"]) / (k + s["g"]))
            scorer = next((s for s in r["starters"]
                           if s["n"] == fg["player"]
                           and s["team"] == fg["team"]), None)
            if scorer is None:
                continue
            probs = {}
            for s in r["starters"]:
                probs[(s["n"], s["team"])] = (
                    (pa if s["team"] == r["away"] else 1 - pa)
                    * adj[s["n"]])
            tot = sum(probs.values())
            p = (probs[(fg["player"], fg["team"])] / tot
                 if tot else 0.0)
            total += logloss(p)
            scored += 1
        return (total / scored if scored else float("nan"), scored)

    priors, counts, trials = [], [], []
    for r in done:
        for s in r["starters"]:
            priors.append(s["sh"])
            counts.append(s["y"])
            trials.append(s["g"])
    mom_k = eb_k_mom(priors, counts, trials)
    print(f"\nempirical-Bayes K (method of moments, {len(priors)} "
          f"starter-records): "
          f"{mom_k if mom_k is None else round(mom_k, 1)} "
          f"(current K=20; None = data cannot identify K)")

    uniq_dates = sorted({r["date"] for r in done})
    cut = uniq_dates[max(1, (2 * len(uniq_dates)) // 3)]
    fit_recs = [r for r in done if r["date"] < cut]
    eval_recs = [r for r in done if r["date"] >= cut]
    print(f"time split: fit on {len(fit_recs)} games (< {cut}), "
          f"evaluate on {len(eval_recs)} games (>= {cut})")
    cand_k = None
    if fit_recs and eval_recs:
        best_k, best_ll = None, None
        for k in (5, 10, 15, 20, 30, 50, 100):
            ll_fit, _ = scorer_ll(fit_recs, k)
            if best_ll is None or ll_fit < best_ll:
                best_k, best_ll = k, ll_fit
        ll_eval_k20, n_ev = scorer_ll(eval_recs, 20)
        ll_eval_best, _ = scorer_ll(eval_recs, best_k)
        print(f"  K grid on fit split: best K={best_k} "
              f"(fit log-loss {best_ll:.3f})")
        print(f"  held-out log-loss: K=20 -> {ll_eval_k20:.3f}, "
              f"K={best_k} -> {ll_eval_best:.3f} over {n_ev} games")
        if ll_eval_best < ll_eval_k20:
            cand_k = best_k
            print(f"  => held-out improvement: candidate K={best_k}")
        else:
            print("  => no held-out improvement: keep K=20")
        # Platt calibration, binary framing (all 10 starters per game).
        fit_rows = []
        for r in fit_recs:
            pa = team_first_prob(r["awayTip"], r["awayRate"],
                                 r["homeRate"])
            fg = r["first_fg"]
            raw = {}
            for s in r["starters"]:
                raw[s["n"]] = ((pa if s["team"] == r["away"] else 1 - pa)
                               * s["adj"])
            tot = sum(raw.values())
            for s in r["starters"]:
                fit_rows.append((raw[s["n"]] / tot if tot else 0.0,
                                 1 if (s["n"] == fg["player"]
                                       and s["team"] == fg["team"])
                                 else 0))
        platt = platt_fit([p for p, _ in fit_rows],
                          [o for _, o in fit_rows])
        cand_platt = None
        if platt is not None:
            tot_ev, n_ev2 = 0.0, 0
            for r in eval_recs:
                pa = team_first_prob(r["awayTip"], r["awayRate"],
                                     r["homeRate"])
                fg = r["first_fg"]
                raw = {}
                for s in r["starters"]:
                    raw[s["n"]] = ((pa if s["team"] == r["away"] else 1 - pa)
                                   * s["adj"])
                tot = sum(raw.values())
                cal = {n_: platt_apply(p / tot if tot else 0.0, platt)
                       for n_, p in raw.items()}
                ctot = sum(cal.values())
                scorer = next((s for s in r["starters"]
                               if s["n"] == fg["player"]
                               and s["team"] == fg["team"]), None)
                if scorer is None or ctot <= 0:
                    continue
                tot_ev += logloss(cal[fg["player"]] / ctot)
                n_ev2 += 1
            ll_ev_cal = tot_ev / n_ev2 if n_ev2 else float("nan")
            print(f"  Platt fit on fit split: a={platt[0]:.3f}, "
                  f"b={platt[1]:.3f}; held-out log-loss "
                  f"uncalibrated {ll_eval_k20:.3f} -> calibrated "
                  f"{ll_ev_cal:.3f} over {n_ev2} games")
            if ll_ev_cal < ll_eval_k20:
                cand_platt = platt
                print("  => held-out improvement: candidate Platt map")
            else:
                print("  => no held-out improvement: no Platt map")
        else:
            print("  Platt fit not identifiable on the fit split")
    else:
        cand_platt = None
        print("  time split too small to evaluate; no candidates")

    if args.write_params:
        params = {"K": cand_k if cand_k is not None else CUR_K,
                  "K_mom": mom_k,
                  "platt": (list(cand_platt)
                            if cand_platt is not None else None),
                  "fitted_on_games": len(done),
                  "fitted_at": done[-1]["date"],
                  "held_out_note": ("K / Platt applied only when the "
                                    "time-split held-out log-loss "
                                    "improved; otherwise defaults "
                                    "(K=20, no Platt).")}
        with open(args.write_params, "w") as f:
            json.dump(params, f, indent=1)
            f.write("\n")
        print(f"\nwrote candidate parameters to {args.write_params}: "
              f"{params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
