#!/usr/bin/env python3
"""Score a completed NFL week for the Odds Finder First TD model.

Reads the slate that is already in data.json's ``nfl`` block, finds the
actual first OFFENSIVE touchdown scorer of every completed game in that
week from nflverse play-by-play, and appends one entry per scored week
to ``data.json["nfl"]["track"]``:

    {"week": 9, "scored_at": "2026-11-10",
     "games": [{"g": "DAL@IND", "pred": "Jonathan Taylor",
                "pred_p": 0.23, "act": "Jonathan Taylor", "hit": true}]}

The prediction for a game is the slate's top pick: the player with the
highest model probability P = poff * teamFirstTD * sh/100 (poff is
common to every player in the game, so the ranking is team prob x
share). ``pred_p`` is that probability as a fraction, rounded to 4 dp.

Rules (deliberate, matching the basketball track record):
  - A game is scored against its first OFFENSIVE TD scorer, the same
    definition the slate prices (td_team == posteam). A game that
    opened with a return TD is still scored against its first
    offensive TD scorer.
  - A game with no touchdown at all cannot be scored: it is skipped
    with a WARNING, never guessed.
  - A week game with no slate prediction (e.g. Week 9's Thu Nov 5
    JAX@BAL, which was already final before the Week 9 slate was
    built) is skipped with a WARNING -- there was nothing to score.
  - A slate game not yet final in games.csv is skipped with a
    WARNING (a postponed game, or a run before Monday night ends).
  - Scoring is idempotent: re-running for a week that is already in
    ``nfl.track`` re-scores and REPLACES that week's entry (same
    result when the data is unchanged; picks up late-final games or
    stat corrections). Nothing is ever appended twice.
  - ``nfl.track`` is a LIST of week entries. An old single-object
    track (if one ever existed) is converted to a one-element list.

Usage:
    score_week.py [--week N] [--season 2026]
                  [--data-json data.json] [--data-dir nfl/data]
                  [--dry-run]

--week defaults to the week of the slate currently in data.json, so
the Tuesday workflow needs no date arithmetic: it scores whatever
week the site is showing, then builds the next one.
"""
import argparse
import json
import os
import re
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_td_slate import (  # noqa: E402
    first_td_facts,
    load_games,
    load_pbp,
    load_players,
    nk,
    resolve_player,
)


def _strip_q(name):
    return re.sub(r"\s*\(Q\)$", "", str(name)).strip()


def top_pick(game, poff):
    """Highest-model-probability player in a slate game.

    Returns (name, prob_fraction) or (None, None) when the game has
    no players. Names are returned without a " (Q)" suffix.
    """
    away_first = game.get("awayFirstTD")
    if away_first is None:
        return None, None
    best, best_p = None, -1.0
    for p in game.get("players") or []:
        team_p = (away_first / 100.0) if p.get("s") == "away" \
            else (1.0 - away_first / 100.0)
        prob = poff * team_p * (p.get("sh", 0) / 100.0)
        if prob > best_p:
            best, best_p = _strip_q(p.get("n", "")), prob
    if best is None:
        return None, None
    return best, round(best_p, 4)


def score_week(slate, pbp_rows, games_rows, season, players):
    """Score one slate block against PBP + schedule rows.

    ``players`` is the (by_team, by_short) pair from load_players().
    Returns (entry, summary) where entry is the track entry dict (or
    None when nothing could be scored) and summary carries counts and
    warnings for the caller to print.
    """
    week = slate["week"]
    poff = slate.get("poff", 0.0)
    week_games = [g for g in games_rows
                  if int(g["week"]) == week and g["game_type"] == "REG"
                  and int(g["season"]) == season]
    completed = {g["game_id"]: g for g in week_games
                 if g.get("away_score") not in ("", None)}
    completed_ids = set(completed)
    facts = first_td_facts(pbp_rows, completed_ids)
    by_team, by_short = players

    slate_by_game = {(g["away"], g["home"]): g for g in slate.get("games", [])}
    scored, skipped, warnings = [], [], []
    for gid, g in sorted(completed.items()):
        key = (g["away_team"], g["home_team"])
        label = f"{g['away_team']}@{g['home_team']}"
        f = facts.get(gid, {})
        if f.get("first_off_team") is None:
            warnings.append(f"{label}: no touchdown in the game -- skipped")
            skipped.append(label)
            continue
        slate_game = slate_by_game.get(key)
        if slate_game is None:
            warnings.append(
                f"{label}: no slate prediction (game was final before "
                f"the slate was built) -- skipped")
            skipped.append(label)
            continue
        pred, pred_p = top_pick(slate_game, poff)
        if pred is None:
            warnings.append(f"{label}: slate game has no players -- skipped")
            skipped.append(label)
            continue
        short = f.get("first_off_scorer")
        team = f.get("first_off_team")
        hit_name = resolve_player(by_team, by_short, team, short)
        act = hit_name[0] if hit_name else short
        if hit_name is None:
            warnings.append(
                f"{label}: could not resolve scorer {short!r} "
                f"({team}) to a full name -- using short form")
        scored.append({
            "g": label,
            "pred": pred,
            "pred_p": pred_p,
            "act": act,
            "hit": nk(pred) == nk(_strip_q(act)),
        })

    if not scored:
        return None, {"week": week, "scored": 0, "skipped": skipped,
                      "warnings": warnings}
    hits = sum(1 for s in scored if s["hit"])
    entry = {
        "week": week,
        "scored_at": date.today().isoformat(),
        "games": scored,
    }
    return entry, {"week": week, "scored": len(scored), "hits": hits,
                   "skipped": skipped, "warnings": warnings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--data-json", default="data.json")
    ap.add_argument("--data-dir", default=os.path.expanduser(
        "~/workspace/first-basket-model/nfl/data"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.data_json) as f:
        data = json.load(f)
    nfl = data.get("nfl") or {}
    if not nfl.get("games"):
        print("FATAL: no nfl slate in data.json -- nothing to score",
              file=sys.stderr)
        return 2
    week = args.week if args.week is not None else nfl.get("week")
    if week is None:
        print("FATAL: --week not given and data.json has no nfl.week",
              file=sys.stderr)
        return 2
    if week != nfl.get("week"):
        print(f"FATAL: --week {week} does not match the slate in "
              f"data.json (week {nfl.get('week')})", file=sys.stderr)
        return 2

    games = load_games(args.data_dir, args.season)
    pbp = load_pbp(args.data_dir, args.season)
    players = load_players(args.data_dir)
    entry, summary = score_week(nfl, pbp, games, args.season, players)

    for w in summary["warnings"]:
        print("WARNING:", w)
    if entry is None:
        print(f"Week {week}: no games could be scored "
              f"({len(summary['skipped'])} skipped)", file=sys.stderr)
        return 2
    print(f"Week {week}: scored {summary['scored']} games, "
          f"{summary['hits']} hits "
          f"({100.0 * summary['hits'] / summary['scored']:.1f}%)")
    if summary["skipped"]:
        print(f"Skipped: {', '.join(summary['skipped'])}")

    if args.dry_run:
        print(json.dumps(entry, indent=1, ensure_ascii=False))
        return 0

    track = nfl.get("track")
    if track is None:
        track = []
    elif isinstance(track, dict):
        track = [track]
    elif not isinstance(track, list):
        print("FATAL: nfl.track is neither a list nor an object",
              file=sys.stderr)
        return 2
    track = [t for t in track if t.get("week") != week]
    track.append(entry)
    track.sort(key=lambda t: t.get("week", 0))
    data["nfl"]["track"] = track
    with open(args.data_json, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.data_json}: track now has {len(track)} week(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
