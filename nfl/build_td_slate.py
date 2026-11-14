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
  - Anytime TD (parallel block ``nfl.anytime``, same games/players):
        lam_team = (team_off_TDs + 5*lam_league) / (games + 5)
        lam_league = league-average offensive TDs per team per game
        P(player scores >= 1 TD) = 1 - exp(-lam_team * adjSh/100)
        (Poisson; team_off_TDs counts rush+pass TDs, the same counter
        as the shares. The daily refresh never touches this block.)

Injuries: players listed Out in the week's injury report are excluded
(the next player by adjSh moves up); Questionable/Doubtful get " (Q)"
appended to the name, the same convention as the basketball page.

Usage:  build_td_slate.py --week 9 --season 2026 [--data-dir DIR]
        [--out FILE] [--sunday-only]  (default: all games of the week
        that have not been played yet at build time)
        build_td_slate.py --week 10 --merge-into data.json
        (Tuesday workflow: build next week's slate straight into
        data.json, basketball keys + nfl.track preserved)
        build_td_slate.py --refresh [--data-json data.json]
        (daily ~11am check: fresh injuries + lines for games that
        have not started; drops newly-Out players, marks new
        Questionables, never rebuilds shares, never touches
        basketball keys or nfl.track)
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


def fmt_gametime(gt):
    """nflverse gametime ("20:15", a 24-hour ET clock time) -> "8:15 PM".

    Returns None for a missing/blank value. A value that already
    carries AM/PM passes through whitespace-normalized. Never guesses:
    an unparseable value returns None, and the page then shows the
    matchup and date only.
    """
    if gt is None:
        return None
    s = str(gt).strip()
    if not s:
        return None
    if re.search(r"[APap][Mm]\s*$", s):
        return " ".join(s.split())
    m = re.match(r"^(\d{1,2}):(\d{2})", s)
    if not m:
        return None
    h, minute = int(m.group(1)), m.group(2)
    if h == 0:
        return f"12:{minute} AM"
    if h < 12:
        return f"{h}:{minute} AM"
    if h == 12:
        return f"12:{minute} PM"
    return f"{h - 12}:{minute} PM"


def team_td_rates(stats, games_played):
    """League-average and per-team expected offensive TDs per game.

    lam_league = total rush+pass TDs / total team-games (completed
    games only). lam_team = (team_off_TDs + 5*lam_league) / (games+5):
    a 5-game prior toward league average. Uses the same rush+pass TD
    counter as the player shares, so lam stays consistent with sh.
    """
    team_off = stats["team_off_tds"]
    total_tds = sum(team_off.values())
    total_games = sum(games_played.values())
    lam_league = total_tds / total_games if total_games else 0.0
    lam = {team: (team_off.get(team, 0) + 5 * lam_league)
           / (games_played[team] + 5)
           for team in games_played}
    return lam_league, lam


def anytime_prob(lam_team, sh):
    """P(player scores >= 1 TD) = 1 - exp(-lam_team * sh/100).

    lam_team = the player's team expected offensive TDs this game;
    sh = the player's history-blended share of team TDs (percent).
    Rounded to 4 decimals, like every other stored probability.
    """
    if not lam_team or not sh:
        return 0.0
    return round(1.0 - math.exp(-lam_team * sh / 100.0), 4)


def nk(name):
    """Name key: strip accents, punctuation, suffixes; lowercase."""
    s = unicodedata.normalize("NFD", str(name))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower().replace(".", "").replace("'", "")
    s = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", s)
    return s.strip()


def download(url, path, force=False):
    if not force and os.path.exists(path) and os.path.getsize(path) > 1000:
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


def load_injuries(data_dir, season, week, force=False):
    path = download(f"{NFLVERSE}/injuries/injuries_{season}.csv",
                    os.path.join(data_dir, f"injuries_{season}.csv"),
                    force=force)
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
            odds_list = comp.get("odds") or [{}]
            # Prefer the DraftKings entry explicitly: ESPN returns
            # several providers and the [0] slot is not guaranteed
            # stable, so a provider reorder must never flip the line.
            odds = next(
                (o for o in odds_list
                 if "draft" in str((o.get("provider") or {}).get("name", "")).lower()),
                odds_list[0])
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


def strip_q(name):
    """Remove a trailing " (Q)" injury marker from a stored name."""
    return re.sub(r"\s*\(Q\)$", "", str(name)).strip()


def compute_player_stats(pbp, completed_ids, facts):
    """Per-player counters from PBP (completed games only).

    Returns a dict with the counters build() and --refresh share:
    off_tds, rz_touches, team_off_tds, team_rz, played (game_ids per
    player), first_td_count. PBP names are SHORT forms ('J.Allen');
    all counters are keyed by (team, short name) so same-short-name
    collisions on different teams never mix.

    Trials proxy (kept deliberately, Nov 8): trials = games with a
    rush attempt, target, or pass attempt. The design doc suggested
    swapping this for weekly player-stats games-played, but a
    player-stats row only exists when a player records a stat, so it
    misses blocking-only appearances exactly like the proxy does --
    and it would add a second download plus a new failure mode for no
    measured gain. The proxy already counts pass attempts (passer),
    so QBs are covered. Revisit only with a GP source that counts
    snaps, not stat lines.
    """
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
    return {
        "off_tds": off_tds, "rz_touches": rz_touches,
        "team_off_tds": team_off_tds, "team_rz": team_rz,
        "played": played, "first_td_count": first_td_count,
    }


def rank_team_players(team, side, stats, by_team, by_short,
                      out_status, q_status, gameday=""):
    """Rank one team's skill players by blended first-TD share.

    Same formula as the slate build: modelShare = 0.5*TD share +
    0.5*red-zone-touch share, history blend K=20. Returns
    (players, warnings, unresolved): the top 5 after dropping Out
    players, with " (Q)" appended for Questionable/Doubtful. This is
    the single implementation used by both build() and --refresh, so
    a refresh can promote the next man up with exactly the ranking
    the build would have produced.
    """
    off_tds = stats["off_tds"]
    rz_touches = stats["rz_touches"]
    team_off_tds = stats["team_off_tds"]
    team_rz = stats["team_rz"]
    played = stats["played"]
    first_td_count = stats["first_td_count"]
    cands, warnings, unresolved = [], [], []
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
    players, kept = [], 0
    for c in cands:
        if kept >= 5:
            break
        if nk(c["n"]) in out_status:
            warnings.append(
                f"{gameday} {team} {c['n']} excluded (Out)".strip())
            continue
        if nk(c["n"]) in q_status:
            c["n"] = c["n"] + " (Q)"
        players.append(c)
        kept += 1
    return players, warnings, unresolved


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

    stats = compute_player_stats(pbp, completed_ids, facts)

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
            side_players, w, u = rank_team_players(
                team, side, stats, by_team, by_short,
                out_status, q_status, gameday=g["gameday"])
            warnings.extend(w)
            for item in u:
                if item not in unresolved:
                    unresolved.append(item)
            players.extend(side_players)
        line = lines.get((away, home))
        if line is None:
            warnings.append(f"no line for {away}@{home}")
        out_games.append({
            "away": away, "home": home,
            "date": g["gameday"],
            "time": fmt_gametime(g.get("gametime")),
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

    # --- anytime TD block (parallel to the First TD slate) ---------------
    # Same games, same top-5 players, same history-blended shares. The
    # only new number is p: Poisson P(>= 1 TD) from the team's expected
    # offensive TDs for the game times the player's share of them.
    lam_league, lam_by_team = team_td_rates(stats, games_played)
    at_games = []
    for g in out_games:
        at_players = []
        for p in g["players"]:
            team = g["away"] if p["s"] == "away" else g["home"]
            at_players.append({
                "n": p["n"], "s": p["s"], "pos": p["pos"],
                "sh": p["sh"],
                "p": anytime_prob(lam_by_team.get(team, lam_league),
                                  p["sh"]),
            })
        at_games.append({
            "away": g["away"], "home": g["home"],
            "date": g["date"], "time": g["time"],
            "players": at_players,
        })
    block["anytime"] = {
        "week": week,
        "season": season,
        "as_of": today.isoformat(),
        "note": (f"Week {week} Anytime TD -- P = 1-exp(-lam), lam = team "
                 f"expected offensive TDs x player share of team TDs. "
                 f"Team rate shrunk 5 games toward league average "
                 f"(lam_league = {lam_league:.3f} off TDs/team-game). "
                 f"Same top-5 players and shares as First TD."),
        "games": at_games,
    }
    return block, {
        "games_played": dict(games_played),
        "first_off_td": dict(first_off),
        "poff_games": (n_first_off, n_first_td_games),
        "lam_league": round(lam_league, 4),
        "warnings": warnings,
        "unresolved": unresolved,
    }


def merge_into_data_json(data_path, block):
    """Merge a freshly built slate block into data.json's ``nfl`` block.

    Only the slate fields (week, season, as_of, poff, note, games,
    anytime) are replaced. Everything else in the file -- the
    basketball keys and, crucially, ``nfl.track`` and ``nfl.atrack``
    -- passes through untouched. A block that carries no ``anytime``
    (e.g. one built before anytime existed) leaves the stored
    anytime block in place, exactly like ``track``. Refuses a block
    for an EARLIER week than the one already stored (rebuilding the
    same week is allowed).
    """
    with open(data_path) as f:
        data = json.load(f)
    old = data.get("nfl") or {}
    if old.get("week") is not None and block["week"] < old["week"]:
        raise ValueError(
            f"refusing to merge week {block['week']} over stored "
            f"week {old['week']}")
    merged = dict(old)
    merged.update(block)
    if "track" not in block:
        merged["track"] = old.get("track")
    if "anytime" not in block:
        merged["anytime"] = old.get("anytime")
    data["nfl"] = merged
    with open(data_path, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    return merged


def refresh_slate(data, season, data_dir, today=None):
    """Daily ~11am injury/line refresh of the current week's slate.

    Updates the ``nfl`` block inside ``data`` IN PLACE and returns
    (changed, summary). Rules:
      - Only games that have NOT started yet are touched. A game is
        skipped entirely when games.csv shows a score OR the PBP
        already has plays for it (covers in-progress London games).
      - For a touched game, each side's top 5 is recomputed with the
        same rank_team_players() the build uses, against FRESH
        injuries (force-downloaded) and fresh ESPN lines. Out
        players drop out and the next man moves up; newly
        Questionable/Doubtful players get " (Q)"; cleared players
        lose it.
      - Shares are NEVER re-priced by a refresh: for a team whose
        Week N game has not started, every input to its shares is
        its own completed games, which have not changed since the
        build -- so a recompute is identical by construction. If a
        kept player's share nevertheless moved by more than 0.05
        (a stat correction, or a stale games.csv), that is printed
        as a loud WARNING; the recomputed list still stands, since
        it is the current ranking from the same formula.
      - A missing fresh line never blanks a line: the old line is
        kept and a warning is printed.
      - Date/kickoff (Nov 14): backfilled from the schedule when a
        game lacks them, and never dropped afterwards. The schedule
        is the source of truth, so a refresh sets them to the
        scheduled values (and records a change if they moved).
      - Never touches the basketball keys, ``nfl.track``,
        ``nfl.anytime``, poff, or the team probabilities.
        ``nfl.as_of`` moves to today only when something actually
        changed.
    """
    today = today or date.today()
    nfl = data.get("nfl") or {}
    if not nfl.get("games"):
        raise ValueError("no nfl slate in data.json -- nothing to refresh")
    week = nfl["week"]

    games = load_games(data_dir, season)
    sched = {(g["away_team"], g["home_team"]): g for g in games
             if int(g["week"]) == week and g["game_type"] == "REG"
             and int(g["season"]) == season}
    pbp = load_pbp(data_dir, season)
    pbp_ids = {r["game_id"] for r in pbp}
    completed_ids = {g["game_id"] for g in games
                     if g.get("away_score") not in ("", None)}
    facts = first_td_facts(pbp, completed_ids)
    stats = compute_player_stats(pbp, completed_ids, facts)
    by_team, by_short = load_players(data_dir)
    out_status, q_status = load_injuries(data_dir, season, week, force=True)

    dates = set()
    for game in nfl["games"]:
        row = sched.get((game["away"], game["home"]))
        if row is not None and not row.get("away_score") \
                and row["game_id"] not in pbp_ids:
            dates.add(date.fromisoformat(row["gameday"]))
    try:
        lines = fetch_lines(dates)
    except Exception as exc:  # lines are best-effort; injuries still apply
        lines = {}
        fetch_warning = f"ESPN lines fetch failed ({exc}) -- keeping old lines"
    else:
        fetch_warning = None

    changes, warnings = [], []
    if fetch_warning:
        warnings.append(fetch_warning)
    for game in nfl["games"]:
        away, home = game["away"], game["home"]
        label = f"{away}@{home}"
        row = sched.get((away, home))
        if row is None:
            warnings.append(f"{label}: not in the Week {week} schedule "
                            f"-- left untouched")
            continue
        # Date/kickoff from the schedule (Nov 14): the schedule is the
        # source of truth. A slate built before date/time existed gets
        # them backfilled here; a refresh never drops them, and the
        # schedule never moves them. This runs before the started-game
        # skip so even final games get their date/time.
        sched_date = row["gameday"]
        sched_time = fmt_gametime(row.get("gametime"))
        if game.get("date") != sched_date or game.get("time") != sched_time:
            changes.append(
                f"{label}: date/time {game.get('date')} "
                f"{game.get('time')} -> {sched_date} {sched_time}")
            game["date"], game["time"] = sched_date, sched_time
        if row.get("away_score") or row["game_id"] in pbp_ids:
            continue  # final or in progress -- leave exactly as built
        for team, side in ((away, "away"), (home, "home")):
            stored = [p for p in game.get("players", [])
                      if p.get("s") == side]
            fresh, w, u = rank_team_players(
                team, side, stats, by_team, by_short,
                out_status, q_status, gameday=row["gameday"])
            warnings.extend(w)
            for item in u:
                warnings.append(f"UNRESOLVED (skipped): {item}")
            stored_by_name = {strip_q(p["n"]): p for p in stored}
            for p in fresh:
                old = stored_by_name.get(strip_q(p["n"]))
                if old is not None and abs(p["sh"] - old["sh"]) > 0.05:
                    warnings.append(
                        f"{label} {team} {p['n']}: share moved "
                        f"{old['sh']} -> {p['sh']} in refresh "
                        f"(stat correction or stale schedule?)")
            if [strip_q(p["n"]) for p in fresh] != \
                    [strip_q(p["n"]) for p in stored] or \
                    [p["n"] for p in fresh] != [p["n"] for p in stored]:
                dropped = [strip_q(p["n"]) for p in stored
                           if strip_q(p["n"]) not in
                           {strip_q(q["n"]) for q in fresh}]
                added = [strip_q(p["n"]) for p in fresh
                         if strip_q(p["n"]) not in stored_by_name]
                changes.append(
                    f"{label} {team}: "
                    + (f"dropped {', '.join(dropped)}; " if dropped else "")
                    + (f"promoted {', '.join(added)}; " if added else "")
                    + "Q marks updated")
                game["players"] = [p for p in game.get("players", [])
                                   if p.get("s") != side] + fresh
                # keep away players before home players, as built
                game["players"].sort(key=lambda p: 0 if p["s"] == "away" else 1)
        new_line = lines.get((away, home))
        if new_line and new_line != game.get("line"):
            changes.append(f"{label}: line {game.get('line')} -> {new_line}")
            game["line"] = new_line
        elif not new_line and not game.get("line"):
            warnings.append(f"{label}: still no line available")
    if changes:
        nfl["as_of"] = today.isoformat()
    return bool(changes), {"week": week, "changes": changes,
                           "warnings": warnings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=None,
                    help="required to build; in --refresh mode defaults "
                         "to the week in data.json")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--data-dir", default=os.path.expanduser(
        "~/workspace/first-basket-model/nfl/data"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-json", default="data.json",
                    help="data.json to refresh or merge into")
    ap.add_argument("--refresh", action="store_true",
                    help="daily injury/line refresh of the current slate "
                         "in --data-json; never rebuilds shares")
    ap.add_argument("--merge-into", default=None, metavar="DATA_JSON",
                    help="merge the built slate into this data.json "
                         "(basketball keys + nfl.track preserved)")
    args = ap.parse_args()

    if args.refresh:
        with open(args.data_json) as f:
            data = json.load(f)
        nfl = data.get("nfl") or {}
        if not nfl.get("games"):
            print("FATAL: no nfl slate in data.json -- nothing to refresh",
                  file=sys.stderr)
            return 2
        if args.week is not None and args.week != nfl.get("week"):
            print(f"FATAL: --week {args.week} does not match the slate "
                  f"in data.json (week {nfl.get('week')})", file=sys.stderr)
            return 2
        try:
            changed, summary = refresh_slate(data, args.season, args.data_dir)
        except ValueError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
        for w in summary["warnings"]:
            print("WARNING:", w)
        if not changed:
            print(f"Week {summary['week']}: no changes -- "
                  f"{args.data_json} untouched")
            return 0
        with open(args.data_json, "w") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        print(f"Week {summary['week']}: refreshed {args.data_json}")
        for c in summary["changes"]:
            print("  -", c)
        return 0

    if args.week is None:
        print("FATAL: --week is required to build a slate",
              file=sys.stderr)
        return 2
    block, meta = build(args.season, args.week, args.data_dir)
    out = args.out
    if args.merge_into and not out:
        import tempfile
        out = os.path.join(tempfile.gettempdir(),
                           f"week{args.week}_slate.json")
    if not out:
        out = os.path.join(
            os.path.dirname(args.data_dir.rstrip("/")),
            f"week{args.week}_slate.json")
    with open(out, "w") as f:
        json.dump(block, f, indent=1, ensure_ascii=False)
    print(f"wrote {out}: {len(block['games'])} games, "
          f"poff={block['poff']}, lam_league={meta['lam_league']} "
          f"(anytime block: {len(block['anytime']['games'])} games)")
    print(f"first-off-TD games: {meta['poff_games'][0]}/"
          f"{meta['poff_games'][1]}")
    for w in meta["warnings"]:
        print("WARNING:", w)
    for u in meta["unresolved"]:
        print("UNRESOLVED (skipped, no full name/position):", u)
    if args.merge_into:
        try:
            merge_into_data_json(args.merge_into, block)
        except ValueError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
        print(f"merged week {args.week} slate into {args.merge_into} "
              f"(basketball keys + nfl.track preserved)")
    return 0


if __name__ == "__main__":
    main()
