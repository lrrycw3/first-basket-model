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
    """Score rates for a given anchor (the page uses 45%)."""
    s = abs(spread_home)
    if spread_home < 0:
        home_imp, away_imp = (total + s) / 2, (total - s) / 2
    else:
        away_imp, home_imp = (total + s) / 2, (total - s) / 2
    return (100 * anchor * away_imp / total,
            100 * anchor * home_imp / total)


def logloss(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -math.log(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".",
                    help="repo root (holds data/calibration_log.jsonl)")
    ap.add_argument("--min-games", type=int, default=30,
                    help="games with outcomes needed before fitting")
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
        p_scorer = (pa if fg["team"] == r["away"] else 1 - pa) * scorer["adj"]
        # Renormalize the 10 starters' adj shares to sum to 1 for the
        # log-loss (the bench share is ~4% and the scorer is a starter).
        s_adj = sum(s["adj"] for s in r["starters"])
        p_scorer = p_scorer / s_adj if s_adj > 0 else p_scorer
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
                    p = ((pa if fg["team"] == r["away"] else 1 - pa)
                         * adj[fg["player"]])
                    s_adj = sum(adj.values())
                    p = p / s_adj if s_adj > 0 else p
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
