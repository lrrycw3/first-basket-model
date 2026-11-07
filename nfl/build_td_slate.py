#!/usr/bin/env python3
"""Build an NFL First Touchdown slate block for the Odds Finder site.

Mirrors the NBA first-basket model: for each game, the chance each team
scores the first touchdown, then each offensive skill player's share of
his team's TDs, blended with first-TD history. NO probability is copied
from any other site -- every number is computed from raw stats below.

Sources (all free, all fetchable from this VM):
  - nflverse schedules (games.csv) ......... schedule + completed games
  - nflverse play-by-play (play_by_play_<season>.csv.gz) ... every TD,
    first-TD history, red-zone touches, player TDs
  - nflverse injuries (injuries_<season>.csv) ......... Out / Questionable
  - nflverse players (players.csv) .................... player positions
  - ESPN scoreboard API (site.api.espn.com) .......... lines (DraftKings)
    Covers NFL odds are the preferred line source, but its odds tables
    are JS-rendered and unreadable from this VM without a browser, so the
    script uses the ESPN API odds and records the provider in the output.

Formulas (same shape as the basketball model):
  - Team first-TD rate:  r = (firstOffTDs + 5) / (gamesPlayed + 10)
    where firstOffTDs = games in which the team scored the game's first
    OFFENSIVE touchdown (scoring team is the team in possession,
    td_team == posteam: includes offensive fumble recoveries in the
    end zone, excludes every return TD).
  - Poff = share of completed games whose very first TD was offensive.
  - Per-game team prob (log5-style, same as basketball tips):
        awayFirstTD = 100*rA*(1-rH) / (rA*(1-rH) + rH*(1-rA))
    This is P(away scores first | the first TD is offensive).
    Then the fitted home-field shift is applied:
        P_home = logistic(logit(P_home_log5) + HOME_LOGIT)
  - Player model share:  modelShare = 0.5*tdShare + 0.5*rzShare
        tdShare = player offensive TDs / team offensive TDs
        rzShare = player red-zone touches / team red-zone touches
        (red-zone touch = rush attempt or pass target with yardline<=20)
  - History blend (same K=20 as basketball):
        adjSh = 100*(20*modelShare/100 + c) / (20 + trials)
        c = games this season the player scored the game's first
            offensive TD; trials = games the player played (any rush
            attempt, target, or pass attempt in the PBP).
  - Page probability (computed on the page, not stored here):
        P(player scores first TD) = Poff * teamFirstTD * adjSh/100

Injuries: players listed Out in the week's injury report are excluded
(the next player by adjSh moves up); Questionable/Doubtful get " (Q)"
appended to the name, the same convention as the basketball page.

Usage:  build_td_slate.py --week 9 --season 2026 [--data-dir DIR]
        [--out FILE] [--sunday-only]  (default: all games of the week
        that have not been played yet at build time)
"""
import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date

import requests

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
ESPN_API = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/"
            "scoreboard")
UA = {"User-Agent": "Mozilla/5.0 (Odds Finder slate builder)"}
K_BLEND = 20          # history-blend constant, same as basketball K
SHRINK_WINS = 5       # team-rate shrinkage wins
SHRINK_GAMES = 10     # team-rate shrinkage games
# Home-field logit shift, fitted Nov 7 (leave-one-out logistic fit on the
# 122 completed games: a = 0.467, SE 0.191; slope b = 0.72, SE 0.40, not
# distinguishable from 1, so the log5 slope is kept and only the
# intercept shifts). Home teams scored the first offensive TD in 73/121
# games (60.3%). Applied to the log5 home probability below.
HOME_LOGIT = 0.467
SKILL_POS = {"QB", "RB", "WR", "TE"}

# nflverse team codes -> ESPN abbreviations (only the ones that differ)
ESPN_ABBR = {"LA": "LAR", "WAS": "WSH"}


def nk(name):
    """Name key: strip accents, punctuation, suffixes; lowercase."""
    s = unicodedata.normalize("NFD", str(name))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower().replace(".", "").replace("'", "")
    s = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", s)
    return s.strip()


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def load_games(data_dir, season):
    path = download(f"{NFLVERSE}/schedules/games.csv",
                    os.path.join(data_dir, "games.csv"))
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["season"]) == season:
                rows.append(r)
    return rows


def load_pbp(data_dir, season):
    path = download(f"{NFLVERSE}/pbp/play_by_play_{season}.csv.gz",
                    os.path.join(data_dir, f"play_by_play_{season}.csv.gz"))
    rows = []
    with gzip.open(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_injuries(data_dir, season, week):
    path = download(f"{NFLVERSE}/injuries/injuries_{season}.csv",
                    os.path.join(data_dir, f"injuries_{season}.csv"))
    out, q = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["season"]) != season or int(r["week"]) != week:
                continue
            st = (r.get("report_status") or "").strip()
            if st == "Out":
                out[nk(r["full_name"])] = r["full_name"]
            elif st in ("Questionable", "Doubtful"):
                q[nk(r["full_name"])] = r["full_name"]
    return out, q


def load_players(data_dir):
    """PBP names are short forms ('J.Allen'); resolve them via players.csv.

    Returns {(short_key, team): (display_name, position)} preferring ACT
    rows, plus {short_key: [(display_name, position, team)]} fallback.
    Short names collide often (712 ambiguous), so (short, team) is the
    primary key.
    """
    path = download(f"{NFLVERSE}/players/players.csv",
                    os.path.join(data_dir, "players.csv"))
    by_team = {}
    by_short = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            short = (r.get("short_name") or "").strip()
            pos = (r.get("position") or "").strip()
            team = (r.get("latest_team") or "").strip()
            name = (r.get("display_name") or "").strip()
            first = (r.get("first_name") or "").strip()
            if not short or not pos or not name:
                continue
            key = (nk(short), team)
            if key not in by_team or r.get("status") == "ACT":
                by_team[key] = (name, pos, first)
            by_short[nk(short)].append(
                (name, pos, team, r.get("status"), first))
    return by_team, by_short


def resolve_player(by_team, by_short, team, short):
    """(team, PBP short name) -> (full display name, position) or None.

    PBP short names are not always players.csv short names: the PBP
    lengthens the initial to disambiguate same-team same-last-name
    players ('Bi.Robinson' vs 'Br.Robinson' for Bijan/Brian Robinson)
    and uses legal first names where players.csv uses a nickname
    ('Z.Knight' for Bam Knight). Fallbacks, in order:
      1. exact (short, team)
      2. collapse to first initial ('Bi.Robinson' -> 'B.Robinson'),
         same team; if several, first-name-prefix match
      3. last name + team unique among skill players
      4. short name unique among ACT skill players
    """
    hit = by_team.get((nk(short), team))
    if hit:
        return (hit[0], hit[1])
    m = re.match(r"^([A-Za-z]+)\.(.+)$", short)
    if not m:
        return None
    first_part, last = m.group(1), m.group(2)
    collapsed = nk(f"{first_part[0]}.{last}")
    cands = [c for c in by_short.get(collapsed, [])
             if c[2] == team and c[1] in SKILL_POS]
    if len({c[0] for c in cands}) == 1:
        return (cands[0][0], cands[0][1])
    pref = [c for c in cands
            if c[4].lower().startswith(first_part.lower())]
    if len({c[0] for c in pref}) == 1:
        return (pref[0][0], pref[0][1])
    # last name + team unique among skill players (nickname cases)
    last_cands = [c for lst in by_short.values() for c in lst
                  if c[2] == team and c[1] in SKILL_POS
                  and c[0].split()[-1].lower().rstrip(".") == last.lower()]
    if len({c[0] for c in last_cands}) == 1:
        return (last_cands[0][0], last_cands[0][1])
    act = [c for c in by_short.get(collapsed, [])
           if c[1] in SKILL_POS and c[3] == "ACT"]
    if len({c[0] for c in act}) == 1:
        return (act[0][0], act[0][1])
    return None


def fetch_lines(dates):
    """ESPN scoreboard API (DraftKings) -> {(away, home): line str}.

    ESPN's `spread` is quoted from the home team's perspective
    (negative = home favorite); `details` names the favorite, e.g.
    "CIN -3". Line is rendered like the basketball page:
    "CIN -3, total 52.5".
    """
    lines = {}
    for d in sorted(dates):
        url = f"{ESPN_API}?dates={d.strftime('%Y%m%d')}"
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        for ev in r.json().get("events", []):
            comp = ev["competitions"][0]
            teams = {c["homeAway"]: c["team"]["abbreviation"]
                     for c in comp["competitors"]}
            away, home = teams.get("away"), teams.get("home")
            odds = (comp.get("odds") or [{}])[0]
            spread, total = odds.get("spread"), odds.get("overUnder")
            details = (odds.get("details") or "").strip()
            # details uses ESPN abbreviations (LAR/WSH); translate to
            # nflverse codes (LA/WAS) so the line matches the slate
            for nfl_code, espn_code in ESPN_ABBR.items():
                details = re.sub(rf"\b{espn_code}\b", nfl_code, details)
            if details and total is not None:
                lines[(away, home)] = f"{details}, total {total:g}"
            elif spread is not None and total is not None:
                lines[(away, home)] = f"spread {spread:g}, total {total:g}"
    # translate ESPN abbreviations back to nflverse codes
    back = {v: k for k, v in ESPN_ABBR.items()}
    return {(back.get(a, a), back.get(h, h)): v
            for (a, h), v in lines.items()}


def first_td_facts(pbp_rows, completed_ids):
    """Per completed game: first TD overall and first OFFENSIVE TD.

    Returns {game_id: (first_td_offensive: bool,
                       first_off_team, first_off_scorer)}.
    A TD is offensive iff the scoring team is the team in possession
    (td_team == posteam). That includes rush/pass TDs AND an offensive
    fumble recovery in the end zone (Week 8 ARI@DAL: J.Mitchell), and
    excludes every return TD (defensive/ST). Verified: this gives
    poff = 118/121 = 0.9752 with team first-TD counts identical to the
    old rush/pass definition. File order is chronological within a game.
    """
    facts = {}
    for r in pbp_rows:
        gid = r["game_id"]
        if gid not in completed_ids:
            continue
        if r.get("touchdown") not in ("1", "1.0"):
            continue
        is_off = (r.get("td_team") and r.get("td_team") == r.get("posteam"))
        cur = facts.setdefault(gid, {"first_any_off": None,
                                     "first_off_team": None,
                                     "first_off_scorer": None})
        if cur["first_any_off"] is None:
            cur["first_any_off"] = is_off
        if is_off and cur["first_off_team"] is None:
            cur["first_off_team"] = r.get("td_team")
            cur["first_off_scorer"] = r.get("td_player_name")
    return facts


def build(season, week, data_dir, today=None):
    today = today or date.today()
    games = load_games(data_dir, season)
    week_games = [g for g in games
                  if int(g["week"]) == week and g["game_type"] == "REG"]
    completed = [g for g in games if g.get("away_score") not in ("", None)]
    completed_ids = {g["game_id"] for g in completed}

    # --- team games played / first-offensive-TD counts -------------------
    games_played = Counter()
    for g in completed:
        games_played[g["away_team"]] += 1
        games_played[g["home_team"]] += 1

    pbp = load_pbp(data_dir, season)
    facts = first_td_facts(pbp, completed_ids)
    first_off = Counter()
    n_first_td_games = 0
    n_first_off = 0
    for gid, f in facts.items():
        if f["first_any_off"] is None:
            continue  # game with no TD at all
        n_first_td_games += 1
        if f["first_any_off"]:
            n_first_off += 1
        if f["first_off_team"]:
            first_off[f["first_off_team"]] += 1
    poff = round(n_first_off / n_first_td_games, 4) if n_first_td_games else 0.0

    def team_rate(team):
        return ((first_off[team] + SHRINK_WINS)
                / (games_played[team] + SHRINK_GAMES))

    # --- player stats from PBP (completed games only) --------------------
    # NOTE: PBP player names are SHORT forms ('J.Allen'); resolved to full
    # names + positions later via resolve_player(). All counters below are
    # keyed by (team, short name) so same-short-name collisions on
    # different teams never mix.
    off_tds = Counter()       # (team, short) -> offensive TDs
    rz_touches = Counter()    # (team, short) -> red-zone touches
    team_off_tds = Counter()
    team_rz = Counter()
    played = defaultdict(set)  # (team, short) -> game_ids played
    first_td_count = Counter()  # (team, short) -> games scoring first off TD
    for r in pbp:
        gid = r["game_id"]
        if gid not in completed_ids:
            continue
        team = r.get("posteam")
        if not team:
            continue
        pt = r.get("play_type")
        # participation (trials): any rush, target, or pass attempt.
        # A kneel-down is not a touch: it does not count as playing.
        rusher = r.get("rusher_player_name") or ""
        receiver = r.get("receiver_player_name") or ""
        passer = r.get("passer_player_name") or ""
        if pt == "run" and rusher and r.get("qb_kneel") not in ("1", "1.0"):
            played[(team, rusher)].add(gid)
        if pt == "pass" and receiver:
            played[(team, receiver)].add(gid)
        if pt == "pass" and passer:
            played[(team, passer)].add(gid)
        # offensive TDs
        if r.get("rush_touchdown") in ("1", "1.0") and rusher:
            off_tds[(team, rusher)] += 1
            team_off_tds[team] += 1
        elif r.get("pass_touchdown") in ("1", "1.0") and receiver:
            off_tds[(team, receiver)] += 1
            team_off_tds[team] += 1
        # red-zone touches (yardline_100 <= 20). Kneel-downs and 2-pt
        # conversion attempts are not TD-scoring touches: excluded.
        try:
            yl = float(r.get("yardline_100")) if r.get("yardline_100") else None
        except ValueError:
            yl = None
        if (yl is not None and yl <= 20
                and r.get("qb_kneel") not in ("1", "1.0")
                and r.get("two_point_attempt") not in ("1", "1.0")):
            if pt == "run" and rusher:
                rz_touches[(team, rusher)] += 1
                team_rz[team] += 1
            elif pt == "pass" and receiver:
                rz_touches[(team, receiver)] += 1
                team_rz[team] += 1
    for gid, f in facts.items():
        if f["first_off_scorer"] and f["first_off_team"]:
            first_td_count[(f["first_off_team"],
                            f["first_off_scorer"])] += 1

    by_team, by_short = load_players(data_dir)
    out_status, q_status = load_injuries(data_dir, season, week)
    unresolved = []  # (team, short) that could not be resolved to a player

    # --- build the slate --------------------------------------------------
    targets = [g for g in week_games if not g.get("away_score")]
    if not targets:
        targets = week_games
    dates = {date.fromisoformat(g["gameday"]) for g in targets}
    lines = fetch_lines(dates)

    out_games = []
    warnings = []
    for g in sorted(targets, key=lambda x: (x["gameday"], x["gametime"] or "")):
        away, home = g["away_team"], g["home_team"]
        rA, rH = team_rate(away), team_rate(home)
        away_first = round(
            100 * rA * (1 - rH) / (rA * (1 - rH) + rH * (1 - rA)), 1)
        # Fitted home-field shift (see HOME_LOGIT): move the log5 home
        # probability up the logit scale by HOME_LOGIT, then convert back.
        home_p = (100 - away_first) / 100
        home_p = 1 / (1 + math.exp(
            -(math.log(home_p / (1 - home_p)) + HOME_LOGIT)))
        away_first = round(100 * (1 - home_p), 1)
        players = []
        for team, side in ((away, "away"), (home, "home")):
            cands = []
            keys = {k for k in off_tds if k[0] == team} | \
                   {k for k in rz_touches if k[0] == team}
            for (t, short) in keys:
                hit = resolve_player(by_team, by_short, team, short)
                if hit is None:
                    if (team, short) not in unresolved:
                        unresolved.append((team, short))
                    continue
                name, pos = hit
                if pos not in SKILL_POS:
                    continue
                tds = off_tds.get((team, short), 0)
                rz = rz_touches.get((team, short), 0)
                td_share = tds / team_off_tds[team] if team_off_tds[team] else 0
                rz_share = rz / team_rz[team] if team_rz[team] else 0
                model_share = 0.5 * td_share + 0.5 * rz_share
                trials = len(played.get((team, short), set()))
                c = first_td_count.get((team, short), 0)
                adj = 100 * (K_BLEND * model_share + c) / (K_BLEND + trials) \
                    if (K_BLEND + trials) else 0
                cands.append({
                    "n": name, "s": side, "pos": pos,
                    "sh": round(adj, 1), "td": int(tds),
                    "g": int(trials), "c": int(c),
                    "trials": int(trials),
                })
            cands.sort(key=lambda c: -c["sh"])
            kept = 0
            for c in cands:
                if kept >= 5:
                    break
                if nk(c["n"]) in out_status:
                    warnings.append(
                        f"{g['gameday']} {team} {c['n']} excluded (Out)")
                    continue
                if nk(c["n"]) in q_status:
                    c["n"] = c["n"] + " (Q)"
                players.append(c)
                kept += 1
        line = lines.get((away, home))
        if line is None:
            warnings.append(f"no line for {away}@{home}")
        out_games.append({
            "away": away, "home": home,
            "awayFirstTD": away_first,
            "line": line,
            "players": players,
        })

    block = {
        "week": week,
        "season": season,
        "as_of": today.isoformat(),
        "poff": poff,
        "note": (f"Week {week} First TD — team first-TD rates from 2026 "
                 f"nflverse PBP (shrinkage r=(c+5)/(n+10)), log5 team prob "
                 f"plus fitted home-field shift (+{HOME_LOGIT} logit), "
                 f"player shares 0.5*TD share + 0.5*red-zone-touch share, "
                 f"history blend K=20. Lines: ESPN/DraftKings. "
                 f"Poff (first TD offensive) = {poff}."),
        "games": out_games,
    }
    return block, {
        "games_played": dict(games_played),
        "first_off_td": dict(first_off),
        "poff_games": (n_first_off, n_first_td_games),
        "warnings": warnings,
        "unresolved": unresolved,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--data-dir", default=os.path.expanduser(
        "~/workspace/first-basket-model/nfl/data"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    block, meta = build(args.season, args.week, args.data_dir)
    out = args.out or os.path.join(
        os.path.dirname(args.data_dir.rstrip("/")),
        f"week{args.week}_slate.json")
    with open(out, "w") as f:
        json.dump(block, f, indent=1, ensure_ascii=False)
    print(f"wrote {out}: {len(block['games'])} games, poff={block['poff']}")
    print(f"first-off-TD games: {meta['poff_games'][0]}/"
          f"{meta['poff_games'][1]}")
    for w in meta["warnings"]:
        print("WARNING:", w)
    for u in meta["unresolved"]:
        print("UNRESOLVED (skipped, no full name/position):", u)


if __name__ == "__main__":
    main()
