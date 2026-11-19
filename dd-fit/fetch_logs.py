#!/usr/bin/env python3
"""Fetch NBA player game logs from Basketball-Reference (free, no key).

Resolves each player's BRef page URL from the 2026-27 per-game table,
then downloads the player game log for 2025-26 and 2026-27, saving them
to dd-fit/raw/<slug>_gamelog_<season>.json.

Players: the 4 on the First Basket Model Nov 5 slate DD list, plus 6 added
high reb/ast candidates (Jokic, Jalen Johnson, Cade Cunningham, Wembanyama,
Towns, Sengun) so the pooled correlation is not fit on 4 players alone.

Why BRef and not stats.nba.com: the stats.nba.com version of this script
died on BOTH the VM and the user's Mac — commonallplayers read-timed-out
at 30 s with 0 bytes, 3 tries (the host tarpits/blocks both egress IPs;
cdn.nba.com answers 403). BRef answers fine from this VM (HTTP 200 in
~1 s), so the fetch now runs here and the user runs nothing.

Approach reused from the pipeline (pipeline/scripts/update_results.py):
same per-game table for player-URL resolution (_parse_bref_html), same
game-log URL pattern (gamelog_url), same game-log table
(player_game_log_reg) and row parsing (_parse_gamelog_html). The raw
files are written in the stats.nba.com resultSets shape (headers +
rowSet with PTS/REB/AST columns) so dd-fit/fit_dd.py reads them
unchanged. A season with no games (e.g. a player who was in college
that year) is written as an empty rowSet with a WARNING, never guessed.

Robots.txt tension, stated plainly: BRef's robots.txt asks clients to
stay off game logs, and the pipeline already fetches them for the
matchup-adjusted DD feature. This script keeps the same politeness as
that fetch — one request at a time, 1.5 s between requests (the matchup
fetch uses 1.0 s), browser User-Agent, retries with backoff, no
parallelism. It fetches 21 pages total (1 per-game table + 20 game
logs) and is run by hand, not on a schedule.

Requires: requests, beautifulsoup4 (both already installed for the
pipeline; run with ~/workspace/.sttest-venv/bin/python on this VM).
"""
import json
import os
import re
import sys
import time
import unicodedata

import requests
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
BASE = "https://www.basketball-reference.com"
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
# (display season, BRef season code) — BRef uses 2027 for 2026-27.
SEASONS = [("2025-26", 2026), ("2026-27", 2027)]
SLEEP = 1.5  # seconds between requests; at least as polite as the pipeline

# (display name, why included)
PLAYERS = [
    ("Wendell Carter Jr.", "DD list, Nov 5 slate"),
    ("Giannis Antetokounmpo", "DD list, Nov 5 slate"),
    ("Bam Adebayo", "DD list, Nov 5 slate"),
    ("Yaxel Lendeborg", "DD list, Nov 5 slate"),
    ("Nikola Jokic", "added: 13.7 reb / 10.8 ast post-break"),
    ("Jalen Johnson", "added: 9.5 reb / 7.1 ast post-break"),
    ("Cade Cunningham", "added: 10.7 ast post-break"),
    ("Victor Wembanyama", "added: 12.2 reb post-break"),
    ("Karl-Anthony Towns", "added: 11.8 reb post-break"),
    ("Alperen Sengun", "added: 8.0 reb / 6.0 ast post-break"),
]


def nk(x):
    """Name key, mirroring the page's nk() exactly (copied from the
    pipeline so player matching behaves identically)."""
    s = unicodedata.normalize("NFD", str(x))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = re.sub(r"\s*\(q\)", "", s)
    s = s.replace(".", "").replace("'", "")
    s = re.sub(r" (jr|iii|ii)$", "", s)
    return s.strip()


def norm_name(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = re.sub(r"\s*\(Q\)\s*$", "", n).strip().lower()
    n = re.sub(r"[^a-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def slug(n):
    return re.sub(r"[^a-z0-9]+", "-", norm_name(n)).strip("-")


class FetchError(Exception):
    pass


def http_get(url, tries=3):
    """GET with a browser UA. Raises FetchError after retries."""
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content.decode("utf-8", errors="replace")
            last = FetchError(f"{url} -> HTTP {r.status_code} "
                              f"({len(r.content)} bytes)")
        except requests.RequestException as e:
            last = FetchError(f"{url} -> {e}")
        if i < tries - 1:
            time.sleep(5)
    raise last


def resolve_player_urls():
    """{nk(name): (player page URL, team code)} from the BRef 2026-27
    per-game table — the same source the pipeline uses for rec["url"].
    Raises FetchError naming any PLAYERS entry it cannot resolve."""
    url = f"{BASE}/leagues/NBA_2027_per_game.html"
    print("Resolving player URLs (BRef 2026-27 per-game table)...")
    html = http_get(url)
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", id="per_game_stats")
    if tbl is None:
        raise FetchError("BRef per-game table not found — markup may have "
                         "changed; refusing to guess URLs")
    by_name = {}
    for tr in tbl.find("tbody").find_all("tr"):
        name = None
        href = None
        team = None
        for c in tr.find_all(["th", "td"]):
            stat = c.get("data-stat")
            if stat == "name_display":
                name = c.get_text(strip=True)
                a = c.find("a")
                if a and a.get("href"):
                    href = a["href"]
            elif stat == "team_name_abbr":
                team = c.get_text(strip=True)
        if name and href:
            by_name.setdefault(nk(name), (href, team))
    resolved = {}
    for name, _why in PLAYERS:
        hit = by_name.get(nk(name))
        if hit is None:
            raise FetchError(f"cannot resolve BRef player URL for {name!r} "
                             f"— refusing to guess the slug")
        resolved[name] = hit
        print(f"  {name}: {hit[0]} ({hit[1]})")
    return resolved


def gamelog_url(player_url, season):
    """Player page URL (/players/x/xxxxxx01.html) -> game-log URL for a
    BRef season (2027 = 2026-27, 2026 = 2025-26). Same as the pipeline."""
    base = (player_url[:-len(".html")] if player_url.endswith(".html")
            else player_url)
    return f"{BASE}{base}/gamelog/{season}/"


def _parse_mp(text):
    """BRef game-log minutes ("32:15") -> float minutes; None when
    missing or unparseable (DNP rows carry no minutes). Same as pipeline."""
    if not text:
        return None
    try:
        parts = text.strip().split(":")
        if len(parts) == 2:
            return float(parts[0]) + float(parts[1]) / 60.0
        return float(parts[0])
    except (ValueError, TypeError):
        return None


def _parse_gamelog_html(html):
    """Parse a BRef player game-log page into [{date, opp, loc, trb, ast,
    pts, mp}] in chronological order. Same table and row rules as the
    pipeline's _parse_gamelog_html: rows without a parseable date or
    trb/ast (DNP, inactive, header/footer) are skipped."""
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", id="player_game_log_reg")
    if tbl is None:
        return None  # no table: player did not play that season
    games = []
    for tr in tbl.find_all("tr"):
        cells = {}
        for c in tr.find_all(["th", "td"]):
            cells[c.get("data-stat")] = c.get_text(strip=True)
        d = cells.get("date") or ""
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            continue
        try:
            trb = float(cells.get("trb"))
            ast = float(cells.get("ast"))
        except (ValueError, TypeError):
            continue

        def _f(key):
            try:
                return float(cells.get(key))
            except (ValueError, TypeError):
                return 0.0

        games.append({"date": d, "opp": cells.get("opp_name_abbr") or "",
                      "loc": cells.get("game_location") or "",
                      "trb": trb, "ast": ast, "pts": _f("pts"),
                      "mp": _parse_mp(cells.get("mp"))})
    return games


def fetch_gamelog(player_url, season):
    """One season of a player's game log, or None when the season has no
    game-log table (player did not play that season). Raises FetchError
    on HTTP failure."""
    url = gamelog_url(player_url, season)
    html = http_get(url)
    return _parse_gamelog_html(html)


def main():
    os.makedirs(RAW, exist_ok=True)
    urls = resolve_player_urls()
    total_games = 0
    for name, _why in PLAYERS:
        player_url, team = urls[name]
        for season, code in SEASONS:
            out = os.path.join(RAW, f"{slug(name)}_gamelog_{season}.json")
            if os.path.exists(out):
                print(f"  skip {name} {season} (already saved)")
                with open(out) as f:
                    total_games += len(json.load(f)["resultSets"][0]["rowSet"])
                continue
            print(f"  fetching {name} {season}...")
            games = fetch_gamelog(player_url, code)
            if games is None:
                print(f"    WARNING: no game-log table for {name} {season} "
                      f"(did not play that season?) — writing empty log",
                      file=sys.stderr)
                games = []
            rows = []
            for g in games:
                matchup = (f"{team} vs {g['opp']}" if g["loc"] != "@"
                           else f"{team} at {g['opp']}")
                rows.append([g["date"], matchup, g["mp"], g["pts"],
                             g["trb"], g["ast"]])
            # Same resultSets shape the stats.nba.com version wrote, so
            # fit_dd.py reads these unchanged (PTS/REB/AST by header).
            data = {"source": "basketball-reference.com",
                    "player": name, "season": season,
                    "resultSets": [{"headers": ["GAME_DATE", "MATCHUP",
                                                "MIN", "PTS", "REB",
                                                "AST"],
                                    "rowSet": rows}]}
            with open(out, "w") as f:
                json.dump(data, f)
            total_games += len(rows)
            print(f"    saved {len(rows)} games -> {out}")
            time.sleep(SLEEP)
    if total_games == 0:
        print("ERROR: fetched 0 games total — BRef markup may have changed "
              "or every player failed to resolve; refusing to fit on "
              "nothing", file=sys.stderr)
        return 1
    print(f"Done. {total_games} games across {len(PLAYERS)} players.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
