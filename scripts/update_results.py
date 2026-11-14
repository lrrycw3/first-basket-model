#!/usr/bin/env python3
"""Nightly updater for the First Basket Model data.json.

Reads data.json + data/tip_priors.json from the repo, fetches tonight's inputs
from the user's actual sources, rebuilds the slate + track predictions, appends
yesterday's first-basket results to the history log, and writes data.json back.

Sources (the user's, verbatim):
  - Basketball-Reference per-game stats (this season + full 2025-26 for padding)
  - JediBets NBA First Baskets (per-date results + team Tip %)
  - Covers NBA odds board (spread + total; spread = Open, total = median book)
  - Basketball Monster lineups (starters + injury tags; also totals fallback)
  - StatPick first basket (per-center "won X of Y tips")
  - Fox Sports NBA schedule (cross-check only; Covers board is the game list)

Rules:
  - Never invent numbers. If a required input cannot be read, fail loudly
    (non-zero exit + debug artifacts), do not write partial data.
  - History blend uses PER-PLAYER games started (g_i), K=20:
        m_hat_i = (K*m_i + y_i) / (K + g_i)
    NO renormalization (dropped Nov 5 — it would fight the bench-share
    fit and change numbers without basis). g_i is the player's starts in
    the tracked window: a fblog.g entry if the pipeline has tracked his
    starts; 0 for a genuine first-time starter (no first-basket history
    and no post-All-Star / BRef record); otherwise the team-games
    fallback n_T (the established-player default). The initial data.json
    ships with NO g entries at all — do NOT seed g from the seed game
    count (tested: it inflated Bam 17.1% -> 20.0%, because the seed n is
    stale once the log grows the team total). The pipeline creates g
    entries from real Basketball Monster lineups: an established starter
    is initialized at the current team total n_T (the fallback
    approximation, not the stale seed count) and a first-timer at 0, then
    +1 per start. g entries are keyed by plain display name (no injury
    tag).
  - Double-double: pts via the normal approx; reb/ast via the page's
    exact Poisson poisGE (page edit Nov 5). Log-loss floor 0.02 -> 0.01
    and the 9.6% naive baseline are page track constants, not pipeline
    inputs. DD correlation (audit #6) is excluded — no game logs exist.
    The log5 tip formula was already correct (audit #1 was a misread).
  - Calibration logging (audit #3/#4): every night the run appends one
    record per game to data/calibration_log.jsonl (date, teams, awayTip,
    awayRate, homeRate, each starter's model share sh and adjusted share,
    team-first outcome + first-FG scorer, filled the next night). K (20),
    the bench share (4%) and the 45% rate anchor are NOT changed until a
    fit is computed from accumulated log data; the fitter is
    scripts/fit_calibration.py (on demand, not nightly).
  - Tip precedence: StatPick player record > user prior > team 2026-27
    tip wins/games (JediBets stats).
  - Tip formula: r = (wins+5)/(attempts+10);
    awayTip = 100*rA*(1-rH) / (rA*(1-rH) + rH*(1-rA)).
  - All model math mirrors the page's JS exactly (see SCHEMA.md).
"""
import argparse
import difflib
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

NY = ZoneInfo("America/New_York")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
K_BLEND = 20
BENCH_SHARE = 96.0  # model shares of the 5 starters sum to ~96%; rest is bench
DD_TOP = 25  # double-double tab shows the top 25 starters by ddProb

# ---------------------------------------------------------------------------
# Team-code normalization. The page uses one set of abbreviations; every source
# has its own quirks. Everything is normalized to the page's codes on ingest.
# ---------------------------------------------------------------------------
PAGE_TEAMS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
}
TEAM_ALIASES = {
    # Basketball-Reference
    "BRK": "BKN", "CHO": "CHA", "PHO": "PHX",
    # Basketball Monster
    "NOR": "NOP",
    # Covers
    "BK": "BKN", "GS": "GSW",
    # JediBets
    "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS",
}
FULL_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU",
    "Indiana Pacers": "IND", "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def norm_team(code):
    """Normalize any source's team abbreviation to the page's code."""
    code = (code or "").strip().upper()
    if code in PAGE_TEAMS:
        return code
    if code in TEAM_ALIASES:
        return TEAM_ALIASES[code]
    if code in FULL_NAME_TO_ABBR:
        return FULL_NAME_TO_ABBR[code]
    raise ValueError(f"unknown team code: {code!r}")


def nk(x):
    """Name key, mirroring the page's nk() exactly."""
    s = unicodedata.normalize("NFD", str(x))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = re.sub(r"\s*\(q\)", "", s)
    s = s.replace(".", "").replace("'", "")
    s = re.sub(r" (jr|iii|ii)$", "", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Player-name aliases. Basketball Monster shortens some first names
# ("Herb Jones"); postasb and Basketball-Reference use the full form
# ("Herbert Jones"). BM starter names are normalized through this map at
# ingest, so stats lookups, the slate display, and the g map all agree.
# Add entries here as new short forms surface — the run fails loudly on
# an unmapped name, which is how the next one gets found.
# ---------------------------------------------------------------------------
PLAYER_ALIASES = {
    "Herb Jones": "Herbert Jones",
    "Alexandre Sarr": "Alex Sarr",
    "Cam Johnson": "Cameron Johnson",
}


# ---------------------------------------------------------------------------
# Fuzzy player-name resolution. Basketball Monster shortens first names
# ("Cam Johnson", "Herb Jones") or uses a fuller form than the stats
# sources ("Alexandre Sarr" vs "Alex Sarr"); an exact miss used to kill
# the run in blend_stats ("no stats anywhere"). PLAYER_ALIASES above wins
# over fuzzy matching. The resolver below is the safety net for the NEXT
# short form: exact nk match first, then a fuzzy match that requires the
# SAME last name and either a first-name prefix (BM's truncation pattern)
# or a difflib ratio >= 0.85 on the full normalized name. Ambiguous
# matches (two candidates tied at the top score) do NOT resolve — the run
# fails loudly on the name instead of blending the wrong player's stats.
#
# Calibration (Nov 7): a full-name difflib ratio of 0.85 alone does NOT
# catch the known cases — "cam johnson" vs "cameron johnson" is 0.846 and
# "alexandre sarr" vs "alex sarr" is 0.783 — which is why the first-name
# prefix rule exists. The prefix rule needs a first name of >= 3 letters
# so a bare initial ("J. Smith") can never match.
# ---------------------------------------------------------------------------
def resolve_player_name(name, universe):
    """Canonical display name for a starter name. universe: {nk: display}.
    Returns (resolved_name, kind) where kind is 'alias', 'exact',
    'fuzzy', or 'none'. Fuzzy matches print a WARNING with both names."""
    if name in PLAYER_ALIASES:
        return PLAYER_ALIASES[name], "alias"
    key = nk(name)
    if key in universe:
        return universe[key], "exact"
    parts = key.split()
    if len(parts) < 2:
        return name, "none"
    first, last = parts[0], parts[-1]
    best, best_score, tied = None, 0.0, False
    for cand_key, disp in universe.items():
        cparts = cand_key.split()
        if len(cparts) < 2 or cparts[-1] != last:
            continue  # never fuzzy-match across different last names
        cfirst = cparts[0]
        if len(first) >= 3 and (cfirst.startswith(first)
                                or first.startswith(cfirst)):
            score = 0.9
        else:
            score = difflib.SequenceMatcher(None, key, cand_key).ratio()
        if score > best_score:
            best, best_score, tied = disp, score, False
        elif score == best_score and score > 0:
            tied = True
    if best is not None and best_score >= 0.85 and not tied:
        print(f"WARNING: fuzzy player-name match: {name!r} -> {best!r} "
              f"(score {best_score:.2f})", file=sys.stderr)
        return best, "fuzzy"
    return name, "none"


def build_player_universe(postasb_players, bref_cur, bref_full, fblog):
    """{nk: display} over every known player name: post-All-Star export,
    both BRef tables, fblog seed + log, and alias targets."""
    universe = {}
    for name in postasb_players:
        universe.setdefault(nk(name), name)
    for tbl in (bref_cur, bref_full):
        for rec in tbl.values():
            universe.setdefault(nk(rec["name"]), rec["name"])
    for v in (fblog.get("seed") or {}).values():
        for name in v.get("p", {}):
            universe.setdefault(nk(name), name)
    for e in fblog.get("log") or []:
        universe.setdefault(nk(e["fg"]), e["fg"])
    for target in PLAYER_ALIASES.values():
        universe.setdefault(nk(target), target)
    return universe


def canonicalize_lineups(bm, universe):
    """Rewrite BM starter names to canonical display names (alias, exact
    nk, or fuzzy). Returns the number of fuzzy resolutions (each already
    logged as a WARNING by resolve_player_name)."""
    fuzzy_n = 0
    for sides in bm.values():
        for side in ("away", "home"):
            for s in sides[side]:
                resolved, kind = resolve_player_name(s["name"], universe)
                if kind == "fuzzy":
                    fuzzy_n += 1
                s["name"] = resolved
    return fuzzy_n


# ---------------------------------------------------------------------------
# HTTP helpers (plain requests first; Playwright only where JS is required)
# ---------------------------------------------------------------------------
class FetchError(Exception):
    pass


def debug_dump(root, name, content, ext="html"):
    path = os.path.join(root, f"debug-{name}.{ext}")
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as f:
        f.write(content)
    return path


def _decode_html(content):
    """Decode a response body as UTF-8. requests guesses ISO-8859-1 when
    the server omits a charset (BRef sends bare 'Content-Type: text/html'),
    which mojibakes non-ASCII names ('Jokić' -> 'JokiÄ\\x87') and silently
    breaks every nk() lookup for diacritic players (Jokić, Dončić,
    Porziņģis, ...). All of this pipeline's sources serve UTF-8."""
    return content.decode("utf-8", errors="replace")


def http_get(url, root, debug_name, timeout=30, retries=2):
    """GET with a browser UA. Raises FetchError (after saving a debug copy
    when a response body exists)."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 1000:
                return _decode_html(r.content)
            last_exc = FetchError(
                f"{url} -> HTTP {r.status_code} ({len(r.content)} bytes)")
            if r.content:
                debug_dump(root, debug_name, r.content)
        except requests.RequestException as e:
            last_exc = FetchError(f"{url} -> {e}")
        if attempt < retries:
            time.sleep(5)
    raise last_exc


def playwright_fetch(url, root, debug_name, wait_for=None, timeout_ms=60000):
    """Fetch a JS-rendered page with Playwright. Returns (html, screenshot_path).
    Raises FetchError on failure (saves screenshot + HTML first).

    Waits for domcontentloaded, NOT networkidle: ad-heavy pages like Covers
    never go idle (trackers keep the network busy) and goto times out at
    60s — that killed nightly run #2 before Basketball Monster was even
    reached. The wait_for selector gates on the rendered content, with a
    60s budget of its own. It waits for the selector to be ATTACHED, not
    visible: nightly run #3 showed Covers' odds tables present in the DOM
    (5, then 9 matches) but never visible within 60s, so a visibility wait
    times out on a page whose content is already there. A second wait then
    requires the matched elements to actually carry text, so slow JS fill
    is caught before page.content() is read."""
    from playwright.sync_api import sync_playwright
    shot = os.path.join(root, f"debug-{debug_name}.png")
    html_path = os.path.join(root, f"debug-{debug_name}.html")
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(user_agent=UA)
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            if wait_for:
                page.wait_for_selector(wait_for, timeout=60000,
                                       state="attached")
                page.wait_for_function(
                    "() => [...document.querySelectorAll('"
                    + wait_for.replace("'", "\\'")
                    + "')].some(el => (el.textContent || '').trim().length > 100)",
                    timeout=60000)
            page.wait_for_timeout(1500)
            html = page.content()
            page.screenshot(path=shot)
            browser.close()
            return html
    except Exception as e:
        # Best-effort debug capture even on failure.
        try:
            with open(html_path, "w") as f:
                f.write(f"<!-- playwright fetch failed: {e} -->\n")
        except OSError:
            pass
        raise FetchError(f"playwright {url} -> {e} (see {shot} / {html_path})")


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------
def fetch_jedibets_day(root, day):
    """JediBets tracker for one date. Returns (per_team_rows, games).
    per_team_rows: [{matchup, team, scorer, clock_sec}]
    games: {matchup: [rows]}. Plain HTTP: date is a GET param."""
    url = f"https://jedibets.com/nba-first-baskets?date={day.isoformat()}"
    html = http_get(url, root, f"jedibets-{day.isoformat()}")
    return _parse_jedibets_html(html, day, root=root)


def _parse_jedibets_html(html, day, root=None):
    """Parse a JediBets tracker page (already fetched)."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if len(tables) < 2:
        if root:
            debug_dump(root, f"jedibets-{day.isoformat()}", html)
        raise FetchError(f"JediBets {day}: expected 2 tables, found {len(tables)}")
    per_team = tables[1]
    hdrs = [th.get_text(strip=True) for th in per_team.find_all("th")]
    if "First FG" not in hdrs:
        if root:
            debug_dump(root, f"jedibets-{day.isoformat()}", html)
        raise FetchError(f"JediBets {day}: Per-Team table headers unexpected: {hdrs}")
    rows, games, cur_game = [], {}, None
    for tr in per_team.find_all("tr")[1:]:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) == 5:
            _, matchup, team, _fp, fg = cells
            cur_game = matchup
        elif len(cells) == 3:
            team, _fp, fg = cells
            matchup = cur_game
        else:
            continue
        m = re.match(
            r"(.+?)\s+(?:3pt Shot|Layup|Dunk|Jumper|Hook|Floater|FT[^@]*|Shot)\s*@\s*(\d+):(\d+)",
            fg)
        if not m:
            raise FetchError(f"JediBets {day}: cannot parse first-FG cell: {fg!r}")
        scorer = m.group(1).strip()
        clock = int(m.group(2)) * 60 + int(m.group(3))
        row = {"matchup": matchup, "team": norm_team(team),
               "scorer": scorer, "clock": clock}
        rows.append(row)
        games.setdefault(matchup, []).append(row)
    if not rows:
        raise FetchError(f"JediBets {day}: no per-team rows parsed")
    return rows, games


def fetch_jedibets_team_tips(root):
    """JediBets stats page: team tip wins/games (fallback tip source)."""
    url = "https://jedibets.com/nba-first-basket-stats"
    html = http_get(url, root, "jedibets-stats")
    out = {}
    for m in re.finditer(r'\{[^{}]*"tip_wins"[^{}]*\}', html):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        out[norm_team(d["team_abbr"])] = (int(d["tip_wins"]),
                                           int(d["total_games"]))
    if len(out) < 25:
        debug_dump(root, "jedibets-stats", html)
        raise FetchError(f"JediBets stats: only {len(out)} team tip records")
    return out


def fetch_statpick_tips(root):
    """StatPick first-basket board: per-center 'Won X of Y tips'.
    Returns {nk(name): (wins, attempts)}."""
    url = "https://www.statpick.ai/nba/first-basket"
    html = http_get(url, root, "statpick")
    return _parse_statpick_html(html, root=root)


def _parse_statpick_html(html, root=None):
    """Parse a StatPick board page (already fetched). The tip records read
    like '<TEAM> Tip-off <Name> Won W of N tips' — the team code and the
    'Tip-off' label must be stripped from the captured name, or the nk()
    lookup misses (e.g. 'ORL Tip-off Wendell Carter Jr.' is not Carter)."""
    text = BeautifulSoup(html, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text)
    out, conflicts = {}, []
    for m in re.finditer(
            r"([A-Z][A-Za-z'\.\-]+(?: [A-Z][A-Za-z'\.\-]+){0,3})\s+Won (\d+) of (\d+) tips?",
            text):
        name = m.group(1).strip()
        # Strip a leading team code and/or the 'Tip-off' label.
        name = re.sub(r"^(?:[A-Z]{2,4}\s+)?Tip-off\s+", "", name)
        name = re.sub(r"^[A-Z]{2,4}\s+", "", name)
        w, n = int(m.group(2)), int(m.group(3))
        key = nk(name)
        if key in out and out[key] != (w, n):
            conflicts.append((name, out[key], (w, n)))
        out.setdefault(key, (w, n))
    if conflicts:
        raise FetchError(f"StatPick: conflicting tip records: {conflicts[:3]}")
    if not out:
        if root:
            debug_dump(root, "statpick", html)
        raise FetchError("StatPick: no tip records parsed")
    return out


def fetch_covers_board(root, target_day):
    """Covers NBA odds board. Returns {game_key: {away, home, spread_open,
    total_open, totals(list), spreads(list)}} for games on target_day.
    spread_open: the Open-column spread from the HOME team's perspective
    (negative = home favored). total_open: Open total. totals: per-book totals.
    Plain HTTP first; Playwright fallback if the tables are missing."""
    url = "https://www.covers.com/sport/basketball/nba/odds"
    try:
        html = http_get(url, root, "covers")
        games = _parse_covers_html(html, target_day)
        if games:
            return games
        print("Covers plain HTTP returned no games for today; "
              "trying Playwright", file=sys.stderr)
        debug_dump(root, "covers", html)
    except FetchError as e:
        print(f"Covers plain HTTP failed ({e}); trying Playwright",
              file=sys.stderr)
    html = playwright_fetch(url, root, "covers", wait_for="table")
    games = _parse_covers_html(html, target_day)
    if not games:
        debug_dump(root, "covers", html)
        raise FetchError("Covers: no games parsed from board (plain + Playwright)")
    return games


def _parse_covers_html(html, target_day):
    """Content-based parse of the Covers odds tables. The board renders one
    table per odds type; rows carry the game in the first cell and per-book
    prices in the rest. We merge across tables by (away, home)."""
    soup = BeautifulSoup(html, "html.parser")
    games = {}
    skipped_codes = set()  # Covers codes norm_team() doesn't know — a game
    # dropped here never reaches the slate, so surface it loudly instead
    # of silently skipping the row (BKN@ATL vanished this way: Covers
    # abbreviates Brooklyn as BK, which was missing from TEAM_ALIASES).
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True)
                     for c in tr.find_all(["td", "th"])]
            if not cells or not cells[0]:
                continue
            first = cells[0]
            m = re.match(
                r"(Today|Nov \d+|Dec \d+|Jan \d+|Feb \d+|Mar \d+|Apr \d+|May \d+|"
                r"Jun \d+|Jul \d+|Aug \d+|Sep \d+|Oct \d+)?,?\s*"
                r"(\d{1,2}:\d{2}(?:\s*[AP]M)?)\s+([A-Z]{2,4})\s+([A-Z]{2,4})\b",
                first)
            if not m:
                continue
            datelabel, _time, away_raw, home_raw = m.groups()
            if datelabel == "Today":
                gday = target_day
            elif datelabel:
                mon = [k for k, v in month_names.items()
                       if v == datelabel[:3]][0]
                gday = date(target_day.year, mon, int(datelabel[4:]))
                if gday < target_day - timedelta(days=1):
                    gday = date(target_day.year + 1, mon, int(datelabel[4:]))
            else:
                continue
            if gday != target_day:
                continue
            try:
                away, home = norm_team(away_raw), norm_team(home_raw)
            except ValueError:
                skipped_codes.add(away_raw if away_raw not in PAGE_TEAMS
                                  and away_raw not in TEAM_ALIASES
                                  else home_raw)
                continue
            key = f"{away}@{home}"
            g = games.setdefault(key, {"away": away, "home": home,
                                       "spread_open": None, "total_open": None,
                                       "totals": [], "spreads": []})
            # Open total: "o 224.0" in the first cell.
            mo = re.search(r"\bo\s+(\d{2,3}\.\d)\b", first)
            if mo and g["total_open"] is None:
                g["total_open"] = float(mo.group(1))
            # Open spread: bare signed numbers in the first cell (e.g. the
            # Spread table's first cell reads 'ORL PHO +2.5 +2.5 +2.5
            # -2.5 -2.5 -2.5'). The lines are listed in team order — the
            # AWAY team's spread(s) first, then the HOME team's — so the
            # home spread is the second distinct value (or the only value
            # when the cell lists just the home line, as the 9:00 PM table
            # does). Taking the first match would give the away spread
            # (+2.5) with the wrong sign.
            rest = re.sub(r"[ou]\s*\d{2,3}\.\d", " ", first)
            ms = [float(s) for s in re.findall(
                r"(?<!\d)([+-]\d+(?:\.\d)?)(?!\d)", rest)
                if -30 < float(s) < 30]
            if ms and g["spread_open"] is None:
                distinct = []
                for v in ms:
                    if v not in distinct:
                        distinct.append(v)
                if len(distinct) == 2:
                    g["spread_open"] = distinct[1]  # away, then home
                elif len(distinct) == 1:
                    g["spread_open"] = distinct[0]
                else:
                    g["spread_open"] = ms[len(ms) // 2]
            # Per-book cells: totals ("o 222.5") and spreads (bare signed).
            for cell in cells[1:]:
                for mt in re.finditer(r"\bo\s+(\d{2,3}\.\d)\b", cell):
                    g["totals"].append(float(mt.group(1)))
                cleaned = re.sub(r"[ou]\s*\d{2,3}\.\d", " ", cell)
                for s in re.findall(r"(?<!\d)([+-]\d+(?:\.\d)?)(?!\d)", cleaned):
                    try:
                        v = float(s)
                    except ValueError:
                        continue
                    if -30 < v < 30:
                        g["spreads"].append(v)
    if skipped_codes:
        print("WARNING: Covers rows skipped — unknown team code(s) "
              f"{sorted(skipped_codes)}; add them to TEAM_ALIASES or "
              "those games are dropped from the slate",
              file=sys.stderr)
    # Keep only games with an Open spread and at least one total.
    return {k: v for k, v in games.items()
            if v["spread_open"] is not None and (v["totals"] or v["total_open"] is not None)}


def fetch_bm_lineups(root):
    """Basketball Monster lineups (JS-rendered; Playwright).
    Returns {game_key: {'away': [starter dicts], 'home': [starter dicts]}}.
    Each starter: {name, pos, tag} where tag is an injury tag like 'Q', 'GTD',
    'O', or ''.

    Real markup (from the Nov 5, 2026 runner debug dump, debug-bm-lineups.html):
    one <table class="datatable"> per game; row 0 is the header 'AAA @ BBB
    9:00 PM ET ...'; row 1 is the team cells; then one row per position
    [PG, away starter, home starter]. A name cell may carry a status note
    after the name ('S' confirmed starter, 'IN', 'Off Inj'). BM uses PHO
    for Phoenix; keys are normalized to the page's codes (PHO -> PHX)."""
    url = "https://basketballmonster.com/nbalineups.aspx"
    html = playwright_fetch(url, root, "bm-lineups", wait_for="table")
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    # Strategy 1: datatables with the layout above.
    games = {}
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True)
                     for c in tr.find_all(["td", "th"])]
            rows.append(cells)
        _bm_absorb_table(rows, games)
    if games:
        return games
    # Strategy 2: whole-page text scan for lineup blocks.
    _bm_absorb_text(text, games)
    if games:
        return games
    debug_dump(root, "bm-lineups", html)
    raise FetchError("Basketball Monster: no lineups parsed (see debug-bm-lineups.*)")


def _bm_split_name(cell):
    """Split a BM name cell into (name, injury tag). The cell may carry a
    status note after the name: 'S' (confirmed starter), 'IN', 'Off Inj',
    'NOTE' (a Basketball Monster note badge — Nov 13: 'Jeremiah Fears
    NOTE' and 'Domantas Sabonis NOTE' killed the nightly run because the
    suffixed name matched no stats table) — none of those is an injury
    tag, so they are dropped. Only a Q/GTD/O/D/P note becomes a tag."""
    text = " ".join((cell or "").split())  # collapse all whitespace/newlines
    if not text:
        return "", ""
    m = re.match(r"^(.*?)\s*\((Q|GTD|O|D|P)\)\s*$", text)
    if m:
        name = m.group(1).strip()
        if name.endswith(" NOTE"):
            name = name[:-len(" NOTE")].strip()
        return name, m.group(2)
    for suffix, tag in ((" Off Inj", ""), (" IN", ""), (" S", ""),
                        (" NOTE", ""),
                        (" Out", "O"), (" Q", "Q"), (" GTD", "GTD"),
                        (" D", "D"), (" P", "P")):
        if text.endswith(suffix):
            return text[:-len(suffix)].strip(), tag
    return text, ""


def _bm_absorb_table(rows, games):
    """Read one BM datatable as a game: header row 'AAA @ BBB', then one row
    per position with the away starter in cell 1 and the home starter in
    cell 2. Only accepts a table that yields exactly 5 starters per side."""
    teams = None
    for cells in rows:
        for c in cells:
            m = re.search(r"\b([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\b", c or "")
            if m:
                try:
                    a, h = norm_team(m.group(1)), norm_team(m.group(2))
                except ValueError:
                    print(f"BM: unknown team code in header: {m.group(0)!r}",
                          file=sys.stderr)
                    continue
                if a in PAGE_TEAMS and h in PAGE_TEAMS:
                    teams = (a, h)
                    break
        if teams:
            break
    if not teams:
        return games
    away, home = [], []
    for cells in rows:
        if len(cells) < 3:
            continue
        pos = (cells[0] or "").strip()
        if pos not in ("PG", "SG", "SF", "PF", "C"):
            continue
        for side_list, cell in ((away, cells[1]), (home, cells[2])):
            name, tag = _bm_split_name(cell)
            name = PLAYER_ALIASES.get(name, name)
            if name:
                side_list.append({"pos": pos, "name": name, "tag": tag})
    if len(away) == 5 and len(home) == 5:
        games[f"{teams[0]}@{teams[1]}"] = {"away": away, "home": home}
    return games


def _bm_absorb_text(text, games):
    """Fallback: scan the page text for 'TEAM at TEAM' headers + starter lines."""
    return games


def fetch_bref_per_game(root, season, label):
    """Basketball-Reference per-game table for a season (2027 = 2026-27).
    Returns {nk(name): {name, team, g, fgm, pts, reb, ast}}.
    Prefers the TOT row for multi-team players. Plain HTTP first,
    Playwright fallback, then the NBA stats API."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"
    html = None
    try:
        html = http_get(url, root, f"bref-{season}")
        out = _parse_bref_html(html)
        if out:
            return out
    except FetchError:
        pass
    if html:
        debug_dump(root, f"bref-{season}", html)
    try:
        html = playwright_fetch(url, root, f"bref-{season}", wait_for="table")
        out = _parse_bref_html(html)
        if out:
            return out
    except FetchError:
        pass
    # Last resort: NBA stats API.
    return fetch_nba_stats_per_game(root, season, label)


def _parse_bref_html(html):
    """Parse a BRef per-game table. Current BRef markup uses data-stat names
    name_display / team_name_abbr / games / fg_per_g / pts_per_g /
    fg3_per_g / trb_per_g / ast_per_g (the classic player/fg/pts names
    are gone). fg3 is the per-game 3-pointers-made mean (props fallback
    when a player's game log cannot be fetched)."""
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", id="per_game_stats")
    if tbl is None:
        return {}
    # Collect every row per player (a traded player has a TOT row plus one
    # row per team). Prefer the TOT row's stats with the most recent team's
    # abbreviation; otherwise use the single/last row.
    per_player = {}
    for tr in tbl.find("tbody").find_all("tr"):
        cells = {}
        for c in tr.find_all(["th", "td"]):
            cells[c.get("data-stat")] = c.get_text(strip=True)
            if c.get("data-stat") == "name_display":
                a = c.find("a")
                if a and a.get("href"):
                    cells["_url"] = a["href"]
        name = cells.get("name_display")
        if not name:
            continue
        team_raw = cells.get("team_name_abbr", "")
        try:
            rec = {"name": name,
                   "team": norm_team(team_raw) if team_raw != "TOT" else None,
                   "g": int(cells.get("games") or 0),
                   "fgm": float(cells.get("fg_per_g") or 0),
                   "pts": float(cells.get("pts_per_g") or 0),
                   "fg3": float(cells.get("fg3_per_g") or 0),
                   "reb": float(cells.get("trb_per_g") or 0),
                   "ast": float(cells.get("ast_per_g") or 0),
                   "url": cells.get("_url"),
                   "_tot": team_raw == "TOT"}
        except (TypeError, ValueError):
            continue
        per_player.setdefault(nk(name), []).append(rec)
    rows = {}
    for key, recs in per_player.items():
        tot = [r for r in recs if r["_tot"]]
        non_tot = [r for r in recs if not r["_tot"]]
        if tot and non_tot:
            best = dict(tot[0])
            best["team"] = non_tot[-1]["team"]  # most recent team
            best.pop("_tot", None)
            rows[key] = best
        else:
            best = dict(recs[-1])
            best.pop("_tot", None)
            rows[key] = best
    return rows


def fetch_nba_stats_per_game(root, season, label):
    """Fallback: NBA stats API leaguedashplayerstats (PerGame)."""
    nba_season = f"{season - 1}-{str(season)[-2:]}"  # e.g. 2026-27
    url = ("https://stats.nba.com/stats/leaguedashplayerstats"
           f"?Season={nba_season}&SeasonType=Regular%20Season&PerMode=PerGame"
           "&MeasureType=Base&PlusMinus=N&PaceAdjust=N&Rank=N&Outcome=&"
           "Location=&Month=0&SeasonSegment=&DateFrom=&DateTo=&OpponentTeamID=0"
           "&VsConference=&VsDivision=&GameSegment=&Period=0&ShotClockRange=&"
           "LastNGames=0&GameScope=&PlayerExperience=&PlayerPosition=&StarterBench=")
    headers = {"User-Agent": UA, "Referer": "https://www.nba.com/",
               "Accept": "application/json, text/plain, */*",
               "Accept-Language": "en-US,en;q=0.9",
               "Host": "stats.nba.com", "Connection": "keep-alive"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        data = r.json()
    except Exception as e:
        raise FetchError(f"NBA stats API {nba_season} -> {e}")
    try:
        headers_row = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]
    except (KeyError, IndexError):
        raise FetchError(f"NBA stats API {nba_season}: unexpected payload")
    idx = {h: i for i, h in enumerate(headers_row)}
    out = {}
    for row in rows:
        name = row[idx["PLAYER_NAME"]]
        team = norm_team(row[idx["TEAM_ABBREVIATION"]])
        out[nk(name)] = {"name": name, "team": team,
                         "g": int(row[idx["GP"]]),
                         "fgm": float(row[idx["FGM"]]),
                         "pts": float(row[idx["PTS"]]),
                         "reb": float(row[idx["REB"]]),
                         "ast": float(row[idx["AST"]])}
    if len(out) < 100:
        raise FetchError(f"NBA stats API {nba_season}: only {len(out)} players")
    return out


# ---------------------------------------------------------------------------
# Putback check (forward-only, Nov 7). A first basket that was a putback
# off an offensive rebound was not a designed play for the scorer, so it
# counts 0.5 in the history count c instead of 1. JediBets records WHO
# scored first, not HOW — the how comes from Basketball-Reference
# play-by-play pages, which return HTTP 200 from this VM and the runner
# (stats.nba.com is blocked from both). The check is best-effort: an
# unreachable or unparsable PBP page is a WARNING and the entry is logged
# at full weight — a missed putback check never kills the run. Existing
# fblog seed/log history is NOT backfilled (that would need a PBP fetch
# for every past game); only results scored after this lands carry "w".
# ---------------------------------------------------------------------------
BREF_TEAM_CODES = {"BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}


def bref_team_code(page_code):
    """Page team code -> Basketball-Reference code (BRef uses BRK/CHO/PHO)."""
    return BREF_TEAM_CODES.get(page_code, page_code)


def bref_pbp_url(day, home_team):
    """BRef play-by-play URL: boxscores/pbp/<YYYYMMDD>0<HOME>.html."""
    return ("https://www.basketball-reference.com/boxscores/pbp/"
            f"{day.strftime('%Y%m%d')}0{bref_team_code(home_team)}.html")


_MADE_FG = re.compile(r"\bmakes\s+(?:\d+-pt|dunk|layup|hook|floater)")


def _pbp_events(html):
    """Parse a BRef PBP page into [(side, text)] in game order. side is
    'away'/'home' (which column carried the play). Quarter headers, the
    column header row and the jump-ball row have no play text and are
    skipped. Real markup (verified Nov 7, POR@LAL): 6 columns
    [time, away play, '', score, '', home play]; play text sits in
    column 1 (away) or column 5 (home)."""
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", id="pbp")
    if tbl is None:
        raise FetchError("BRef PBP: no pbp table")
    events = []
    for tr in tbl.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue
        away = tds[1].get_text(" ", strip=True)
        home = tds[5].get_text(" ", strip=True)
        if away:
            events.append(("away", away))
        elif home:
            events.append(("home", home))
    return events


def _pbp_first_fg_putbacks(events):
    """First field goal per side -> True if it was a putback. A putback is
    a made basket by the team that missed a shot and grabbed the
    offensive rebound in the same possession: no made basket, defensive
    rebound, turnover or made free throw in between resets the state.
    Free throws never count as a first FG (a missed FT can still set up
    a putback). Returns {side: bool}; a side is absent if no first FG
    was found for it."""
    first = {}
    miss_team = oreb_team = None
    for side, text in events:
        if "free throw" in text:
            if "makes" in text:
                miss_team = oreb_team = None
            elif "misses" in text:
                miss_team, oreb_team = side, None
            continue
        if _MADE_FG.search(text):
            if side not in first:
                first[side] = (miss_team == side and oreb_team == side)
            miss_team = oreb_team = None
            if len(first) == 2:
                break
            continue
        if "misses" in text:
            miss_team, oreb_team = side, None
            continue
        if text.startswith("Offensive rebound"):
            oreb_team = side
            continue
        if (text.startswith("Defensive rebound") or "Turnover" in text
                or text.startswith("Offensive foul")
                or "violation" in text.lower()):
            miss_team = oreb_team = None
    return first


def fetch_bref_pbp_putbacks(root, day, away, home):
    """BRef PBP for one game -> {side: bool} for each side's first FG.
    Best-effort: any failure returns {} with a WARNING (the caller logs
    the entry at full weight)."""
    url = bref_pbp_url(day, home)
    try:
        html = http_get(url, root, f"bref-pbp-{day.isoformat()}-{away}-{home}",
                        timeout=20, retries=0)
    except FetchError as e:
        print(f"WARNING: putback check skipped for {away}@{home} {day}: {e}",
              file=sys.stderr)
        return {}
    try:
        return _pbp_first_fg_putbacks(_pbp_events(html))
    except Exception as e:
        print(f"WARNING: putback check unparsable for {away}@{home} {day}: {e}",
              file=sys.stderr)
        return {}


def bref_boxscore_url(day, home_team):
    """BRef box-score URL: boxscores/<YYYYMMDD>0<HOME>.html."""
    return ("https://www.basketball-reference.com/boxscores/"
            f"{day.strftime('%Y%m%d')}0{bref_team_code(home_team)}.html")


def _parse_boxscore_html(html, want_fg3=False):
    """Parse a BRef box-score page into {nk_name: (pts, reb, ast)} — or,
    with want_fg3, (pts, reb, ast, fg3) for the props track fill.

    Reads both teams' "Basic Box Score" tables (id box-<TEAM>-game-basic).
    DNP rows have no stat cells and Team Totals rows are not players;
    both are skipped. A player who did not record a stat still gets a
    row with 0s, so a missing name means "not in the box score at all".
    """
    soup = BeautifulSoup(html, "html.parser")
    out = {}
    for table in soup.find_all("table", id=re.compile(r"^box-.*-game-basic$")):
        for row in table.find_all("tr"):
            name_cell = row.find("th", {"data-stat": "player"})
            if name_cell is None:
                continue
            name = name_cell.get_text(strip=True)
            if not name or name in ("Team Totals", "Player"):
                continue
            cells = {td.get("data-stat"): td.get_text(strip=True)
                     for td in row.find_all("td")}
            if "pts" not in cells:
                continue  # DNP / inactive row

            def _num(key):
                try:
                    return int(float(cells.get(key) or 0))
                except ValueError:
                    return 0

            if want_fg3:
                out[nk(name)] = (_num("pts"), _num("trb"), _num("ast"),
                                 _num("fg3"))
            else:
                out[nk(name)] = (_num("pts"), _num("trb"), _num("ast"))
    return out


def fetch_bref_boxscore(root, day, away, home, want_fg3=False):
    """BRef box score for one game -> {nk_name: (pts, reb, ast)} — or
    (pts, reb, ast, fg3) with want_fg3=True.

    Best-effort: any failure returns {} with a WARNING (the caller
    leaves those DD entries unscored for that night).
    """
    url = bref_boxscore_url(day, home)
    try:
        html = http_get(url, root, f"bref-box-{day.isoformat()}-{away}-{home}",
                        timeout=20, retries=0)
    except FetchError as e:
        print(f"WARNING: box-score fetch skipped for {away}@{home} {day}: {e}",
              file=sys.stderr)
        return {}
    try:
        return _parse_boxscore_html(html, want_fg3=want_fg3)
    except Exception as e:
        print(f"WARNING: box score unparsable for {away}@{home} {day}: {e}",
              file=sys.stderr)
        return {}


def is_double_double(pts, reb, ast):
    """True when at least two of pts/reb/ast reach 10."""
    return sum(1 for v in (pts, reb, ast) if v >= 10) >= 2


def build_track_dd(dd_ranked):
    """Frozen DD predictions for tonight's track night.

    Takes dd_ranked (top DD_TOP, with _p probabilities) and returns the
    list stored on the night: name without injury tag, team, model
    probability (4 dp), the means the probability was computed from,
    and act=None until tomorrow night's run fills it from box scores.
    """
    return [{"n": re.sub(r"\s*\([A-Z]+\)$", "", d["n"]), "t": d["t"],
             "p": round(d["_p"], 4),
             "pts": d["pts"], "reb": d["reb"], "ast": d["ast"],
             "act": None} for d in dd_ranked[:DD_TOP]]


def fill_dd_actuals(night, yesterday, matchups, fetch_box):
    """Fill 'act' on a track night's DD entries from box scores.

    ``matchups`` maps "Full Away at Full Home" -> (away_code, home_code)
    (the same map the putback check builds). ``fetch_box(away, home)``
    returns {nk_name: (pts, reb, ast)} for that game ({} on failure).
    Returns (filled, warnings). Never raises for a missing page or a
    missing player line: those entries simply stay unscored.
    """
    filled, warnings = 0, []
    entries = night.get("dd") or []
    if not entries:
        return 0, []
    for m, (aw, hm) in sorted(matchups.items()):
        pending = [e for e in entries
                   if e.get("act") is None and e.get("t") in (aw, hm)]
        if not pending:
            continue
        box = fetch_box(aw, hm)
        if not box:
            continue
        for e in pending:
            key = nk(re.sub(r"\s*\([A-Z]+\)$", "", e["n"]))
            got = box.get(key)
            if got is None:
                warnings.append(f"no box-score line for {e['n']} "
                                f"({e['t']}) on {yesterday}")
                continue
            pts, reb, ast = got[:3]  # 4-tuples (with fg3) also accepted
            e["act"] = {"pts": pts, "reb": reb, "ast": ast,
                        "hit": is_double_double(pts, reb, ast)}
            filled += 1
    return filled, warnings


# ---------------------------------------------------------------------------
# Matchup-adjusted double-double means (Nov 7). A DD-list player's
# season-average rebounds/assists are shrunk toward how he has played
# against TONIGHT'S opponent:
#     adjusted = (n*matchup_mean + 5*season_mean) / (n + 5)
# with n = matchup games found — this season first, then last season if
# fewer than 3, most recent meetings, up to 8 total. 5 meetings = 50/50,
# 1 meeting barely moves it, 0 meetings = season means unchanged. The
# exact-Poisson DD calc then runs on the ADJUSTED means; nothing else in
# the DD math changes. Source: BRef player game logs
# (/players/<initial>/<slug>/gamelog/<season>/, table
# player_game_log_reg). Only the top-25 DD players' logs are fetched,
# ~1s apart; any failure is a WARNING + season-means fallback, never
# fatal. The player page URL comes from the BRef per-game table's name
# link (captured in _parse_bref_html as rec["url"]).
# ---------------------------------------------------------------------------
MATCHUP_K = 5
MATCHUP_MAX_GAMES = 8
MATCHUP_MIN_CURRENT = 3


def gamelog_url(player_url, season):
    """Player page URL (/players/x/xxxxxx01.html) -> game-log URL for a
    BRef season (2027 = 2026-27, 2026 = 2025-26)."""
    base = (player_url[:-len(".html")] if player_url.endswith(".html")
            else player_url)
    return f"https://www.basketball-reference.com{base}/gamelog/{season}/"


def _parse_gamelog_html(html):
    """Parse a BRef player game-log page into
    [{date, opp, trb, ast, pts, fg3}] in chronological order. Rows
    without a parseable date or trb/ast (DNP, inactive, header/footer)
    are skipped. opp is normalized to the page's team codes (BRef uses
    BRK/CHO/PHO). pts/fg3 default to 0.0 when a game row lacks them, so
    a real game is never dropped for a missing shooting column."""
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", id="player_game_log_reg")
    if tbl is None:
        raise FetchError("BRef game log: no player_game_log_reg table")
    games = []
    for tr in tbl.find_all("tr"):
        cells = {}
        for c in tr.find_all(["th", "td"]):
            cells[c.get("data-stat")] = c.get_text(strip=True)
        d = cells.get("date") or ""
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            continue
        try:
            opp = norm_team(cells.get("opp_name_abbr") or "")
            trb = float(cells.get("trb"))
            ast = float(cells.get("ast"))
        except (ValueError, TypeError):
            continue

        def _f(key):
            try:
                return float(cells.get(key))
            except (ValueError, TypeError):
                return 0.0

        games.append({"date": d, "opp": opp, "trb": trb, "ast": ast,
                      "pts": _f("pts"), "fg3": _f("fg3")})
    return games


def fetch_bref_gamelog(root, player_url, season):
    """One season of a player's game log. Raises FetchError on failure."""
    url = gamelog_url(player_url, season)
    html = http_get(url, root, f"bref-gamelog-{season}", timeout=20,
                    retries=0)
    return _parse_gamelog_html(html)


def _matchup_means(games, opponent, max_n=MATCHUP_MAX_GAMES):
    """Most recent games vs opponent (up to max_n): (n, mean_trb,
    mean_ast). games are chronological; the most recent are at the end."""
    vs = [g for g in games if g["opp"] == opponent][-max_n:]
    if not vs:
        return 0, None, None
    return (len(vs),
            sum(g["trb"] for g in vs) / len(vs),
            sum(g["ast"] for g in vs) / len(vs))


def blend_matchup(season_mean, matchup_mean, n, k=MATCHUP_K):
    """Shrinkage blend: (n*matchup + k*season) / (n + k). n=0 -> season."""
    return (n * matchup_mean + k * season_mean) / (n + k) if n else season_mean


def adjust_dd_for_matchup(name, opponent, season_reb, season_ast,
                          url_map, root, fetch=fetch_bref_gamelog):
    """Adjusted (reb, ast, n, matchup_reb, matchup_ast) for one DD-list
    player. url_map: {nk: player page URL} from the BRef tables. Any
    resolution/fetch failure is a WARNING + season means (n=0)."""
    key = nk(re.sub(r"\s*\([A-Z]+\)$", "", name))
    url = url_map.get(key)
    if not url:
        print(f"WARNING: matchup adjustment skipped for {name}: "
              f"no BRef player page resolved", file=sys.stderr)
        return season_reb, season_ast, 0, None, None
    try:
        cur = fetch(root, url, 2027)
    except Exception as e:
        print(f"WARNING: matchup adjustment skipped for {name}: "
              f"2026-27 game log unreadable: {e}", file=sys.stderr)
        return season_reb, season_ast, 0, None, None
    n, m_reb, m_ast = _matchup_means(cur, opponent)
    if n < MATCHUP_MIN_CURRENT:
        time.sleep(1.0)  # polite delay between BRef game-log pages
        try:
            prev = fetch(root, url, 2026)
        except Exception as e:
            print(f"WARNING: matchup adjustment for {name}: last-season "
                  f"game log unreadable ({e}); using {n} current-season "
                  f"meeting(s)", file=sys.stderr)
            prev = []
        pn, pm_reb, pm_ast = _matchup_means(
            prev, opponent, MATCHUP_MAX_GAMES - n)
        if pn:
            if n:
                m_reb = (n * m_reb + pn * pm_reb) / (n + pn)
                m_ast = (n * m_ast + pn * pm_ast) / (n + pn)
            else:
                m_reb, m_ast = pm_reb, pm_ast
            n += pn
    if not n:
        return season_reb, season_ast, 0, None, None
    return (blend_matchup(season_reb, m_reb, n),
            blend_matchup(season_ast, m_ast, n),
            n, m_reb, m_ast)


# ---------------------------------------------------------------------------
# Model math (mirrors the page's JS exactly; see SCHEMA.md)
# ---------------------------------------------------------------------------
def first_poss(a, b):
    return a / (a + b - a * b)


def game_calc(away_tip, away_rate, home_rate):
    t, a, h = away_tip / 100.0, away_rate / 100.0, home_rate / 100.0
    away_first = t * first_poss(a, h) + (1 - t) * (1 - first_poss(h, a))
    return away_first, 1 - away_first


def fair_odds(p):
    if p <= 0:
        return "—"
    if p >= 1:
        return "-inf"
    if p >= 0.5:
        return "-" + str(round(100 * p / (1 - p)))
    return "+" + str(round(100 * (1 - p) / p))


def _norm_ge(mean, sd, k):
    # P(X >= k) for a normal, with the page's continuity correction (k - 0.5).
    z = (k - 0.5 - mean) / sd
    return 0.5 * (1 - math.erf(z / math.sqrt(2)))


def _pois_ge(mean, k):
    # P(X >= k) for Poisson(mean) — the page's poisGE, used for rebounds
    # and assists since the Nov 5 page edit (pts still uses the normal).
    c, t = 0.0, math.exp(-mean)
    for i in range(k):
        c += t
        t *= mean / (i + 1)
    return 1 - c


def dd_prob(pts, reb, ast):
    pp = _norm_ge(pts, max(3.0, 0.28 * pts + 2), 10)
    pr = _pois_ge(reb or 0, 10)
    pa = _pois_ge(ast or 0, 10)
    return pp * pr + pp * pa + pr * pa - 2 * pp * pr * pa


# ---------------------------------------------------------------------------
# Player props: points + threes (Nov 13). Manual-entry lines on the page:
# the pipeline supplies each player's scoring distribution (points mean +
# SD from season game logs; threes mean from the same logs) and a
# reference line (nearest .5 to the mean — the model's median line, NOT a
# book line). The page computes P(over) live from the distribution when
# the user types a line, using the same normGE/poisGE math as the DD tab.
# The track record freezes the reference-line call each night and scores
# it against the box score the next night, like the DD track.
# ---------------------------------------------------------------------------
PROPS_TOP = 40  # props table covers the top 40 starters by points mean


def ref_line(mean):
    """Nearest .5 line to a mean. The .5 lines sit exactly at the
    midpoint of each unit interval, so the nearest one is always
    floor(mean) + 0.5 — and it is always a half-point line (no pushes).
    (A round-to-nearest-half would be wrong here: for mean 24.8 it
    rounds to 25.0 and bumps to 25.5, but 24.5 is closer.)"""
    return math.floor(mean) + 0.5


def _mean_sd(values):
    """(mean, sample SD, n) for a list of numbers; SD needs n >= 2."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var), n


def props_stats_from_log(games):
    """(pts_mean, pts_sd, fg3_mean, n) from parsed game-log games."""
    pm, psd, n = _mean_sd([g["pts"] for g in games])
    fm, _, _ = _mean_sd([g["fg3"] for g in games])
    return pm, psd, fm, n


def assumed_pts_sd(pts_mean):
    """Fallback SD when no game log could be fetched — the same assumed
    spread the DD points model uses (0.28*mean + 2, min 3.0)."""
    return max(3.0, 0.28 * pts_mean + 2)


def build_props(all_starters, url_map, bref_cur, root,
                fetch=fetch_bref_gamelog):
    """Tonight's props players: top PROPS_TOP starters by blended points
    mean, each with a season game log (best-effort, 1s polite delay).
    Returns (players, warnings). A failed log fetch falls back to the
    blended season mean + assumed SD for points, and the BRef per-game
    3PM mean for threes when available; a player with no threes source
    gets fg3=None (points-only row on the page)."""
    ranked = sorted(all_starters, key=lambda s: -s[2][1])[:PROPS_TOP]
    players, warnings = [], []
    first = True
    for name, team, vals in ranked:
        _fgm, pts_mean, _reb, _ast = vals
        pname = re.sub(r"\s*\([A-Z]+\)$", "", name)
        key = nk(pname)
        pts, sd, fg3, g, fg3g = pts_mean, assumed_pts_sd(pts_mean), None, 0, 0
        url = url_map.get(key)
        log = None
        if url:
            if not first:
                time.sleep(1.0)  # polite delay between BRef game-log pages
            first = False
            try:
                log = fetch(root, url, 2027)
            except Exception as e:
                warnings.append(f"props log unreadable for {name}: {e}")
        if log:
            pm, psd, fm, n = props_stats_from_log(log)
            if n:
                pts, fg3, g, fg3g = pm, fm, n, n
                sd = psd if psd > 0 else assumed_pts_sd(pm)
        if fg3 is None:
            rec = bref_cur.get(key) or {}
            if rec.get("fg3") is not None:
                fg3 = round(rec["fg3"], 2)
                fg3g = rec.get("g") or 0
        line_pts = ref_line(pts)
        entry = {"n": pname, "t": team, "pts": round(pts, 1),
                 "sd": round(sd, 1), "g": g,
                 "fg3": (round(fg3, 2) if fg3 is not None else None),
                 "fg3g": fg3g,
                 "line_pts": line_pts,
                 "p_pts": round(_norm_ge(pts, sd, int(line_pts + 0.5)), 4),
                 "line_fg3": None, "p_fg3": None}
        if fg3 is not None:
            line_fg3 = ref_line(fg3)
            entry["line_fg3"] = line_fg3
            entry["p_fg3"] = round(_pois_ge(fg3, int(line_fg3 + 0.5)), 4)
        players.append(entry)
    return players, warnings


def build_track_props(players):
    """Frozen props calls for tonight's track night: one entry per player
    per market (points + threes where available) at the reference line,
    with the model P(over). act=None until tomorrow night's run fills it
    from box scores."""
    out = []
    for p in players:
        out.append({"n": p["n"], "t": p["t"], "mkt": "pts",
                    "line": p["line_pts"], "p": p["p_pts"], "act": None})
        if p.get("fg3") is not None:
            out.append({"n": p["n"], "t": p["t"], "mkt": "fg3",
                        "line": p["line_fg3"], "p": p["p_fg3"],
                        "act": None})
    return out


def fill_props_actuals(night, yesterday, matchups, fetch_box):
    """Fill 'act' on a track night's props entries from box scores.
    fetch_box(away, home) returns {nk_name: (pts, reb, ast, fg3)}.
    Returns (filled, warnings). Never raises: missing pages/players stay
    unscored, mirroring fill_dd_actuals."""
    filled, warnings = 0, []
    entries = night.get("props") or []
    if not entries:
        return 0, []
    for m, (aw, hm) in sorted(matchups.items()):
        pending = [e for e in entries
                   if e.get("act") is None and e.get("t") in (aw, hm)]
        if not pending:
            continue
        box = fetch_box(aw, hm)
        if not box:
            continue
        for e in pending:
            key = nk(re.sub(r"\s*\([A-Z]+\)$", "", e["n"]))
            got = box.get(key)
            if got is None:
                warnings.append(f"no box-score line for {e['n']} "
                                f"({e['t']}) on {yesterday}")
                continue
            pts, _reb, _ast, fg3 = got
            val = pts if e["mkt"] == "pts" else fg3
            e["act"] = {"val": val, "hit": val > e["line"]}
            filled += 1
    return filled, warnings


def tip_pct(wins_a, att_a, wins_h, att_h):
    """Log5 tip probability from shrunk rates. Returns away win %."""
    ra = (wins_a + 5) / (att_a + 10)
    rh = (wins_h + 5) / (att_h + 10)
    return 100 * ra * (1 - rh) / (ra * (1 - rh) + rh * (1 - ra))


def build_history(fblog):
    """Fold seed + log into {team: {'n': games, 'c': {nk_name: firsts}}}."""
    hist = {}
    for team, v in (fblog.get("seed") or {}).items():
        hist[team] = {"n": int(v["n"]), "c": {nk(p): int(c)
                                              for p, c in v["p"].items()}}
    for e in fblog.get("log") or []:
        h = hist.setdefault(e["t"], {"n": 0, "c": {}})
        h["n"] += 1
        key = nk(e["fg"])
        # Log entries carry a weight ("w", default 1): a first basket that
        # was a putback off an offensive rebound counts 0.5 (see the
        # putback check). The page's blend accepts fractional c, but the
        # page's own log accumulation must sum (e.w || 1) for the display
        # to match — see PROBE_NOTES.md (page one-liner, Nov 7).
        h["c"][key] = h["c"].get(key, 0) + float(e.get("w", 1))
    return hist


def blend_stats(name, postasb_nk, bref_cur, bref_full):
    """Blend one player's per-game stats: post-All-Star 2025-26 (padded with
    full-season 2025-26 when < 15 post-break games) + current season.
    Weight w = g/(g+15), or g/(g+8) on a new team.
    postasb_nk is keyed by nk(name); bref tables are keyed by nk(name).
    Returns (fgm, pts, reb, ast) or raises FetchError when no stats exist."""
    post = postasb_nk.get(nk(name))
    cur = bref_cur.get(nk(name))
    full = bref_full.get(nk(name))
    cur_team = cur["team"] if cur else None
    cur_g = cur["g"] if cur else 0
    # Base (last season).
    base, base_team = None, None
    if post:
        base_team = post[0]
        post_n = (float(post[2]), float(post[3]), float(post[4]), float(post[5]))
        if post[1] >= 15:
            base = post_n
        elif full:
            full_n = (full["fgm"], full["pts"], full["reb"], full["ast"])
            gp = post[1]
            base = tuple((gp * post_n[i] + (15 - gp) * full_n[i]) / 15
                         for i in range(4))
        else:
            base = post_n
    elif full:
        base = (full["fgm"], full["pts"], full["reb"], full["ast"])
        base_team = full["team"]
    cur_n = None
    if cur:
        cur_n = (cur["fgm"], cur["pts"], cur["reb"], cur["ast"])
    if base is None and cur_n is None:
        raise FetchError(f"no stats anywhere for {name}")
    if cur_n is None or cur_g == 0:
        return base
    if base is None:
        return cur_n
    new_team = (base_team is None) or (cur_team != base_team)
    w = cur_g / (cur_g + 8) if new_team else cur_g / (cur_g + 15)
    return tuple((1 - w) * base[i] + w * cur_n[i] for i in range(4))


def compute_shares(blended_fgm):
    """shares = 96 * FGM^0.75 / sum(FGM^0.75) over the 5 starters.
    blended_fgm: {name: fgm}. Returns {name: share rounded to 1 decimal}."""
    raw = {n: (max(f, 0.01) ** 0.75) for n, f in blended_fgm.items()}
    total = sum(raw.values())
    return {n: round(BENCH_SHARE * v / total, 1) for n, v in raw.items()}


def score_rates(spread_home, total):
    """Score rates from the line. spread_home: spread from the HOME team's
    perspective (negative = home favored). Returns (away_rate, home_rate)
    rounded to 1 decimal. rate = 90 * implied team total / game total."""
    s = abs(spread_home)
    if spread_home < 0:  # home favored
        home_imp, away_imp = (total + s) / 2, (total - s) / 2
    else:  # away favored (or pick'em)
        away_imp, home_imp = (total + s) / 2, (total - s) / 2
    return (round(90 * away_imp / total, 1),
            round(90 * home_imp / total, 1))


def resolve_tip_record(name, team, statpick, priors, team_tips):
    """Tip record for one jumper. Precedence: StatPick > user prior > team.
    Returns ((wins, attempts), source)."""
    key = nk(name)
    if key in statpick:
        return statpick[key], "statpick"
    for pname, rec in priors.items():
        if nk(pname) == key:
            return (rec["wins"], rec["attempts"]), "prior"
    if team in team_tips:
        return team_tips[team], "team"
    raise FetchError(f"no tip record for {name} ({team}): "
                     "no StatPick, no prior, no team tip record")


def _g_for_blend(key, y_i, team_g_nk, n_t, established_nk):
    """Games-started denominator for the blend. gmap entry if the pipeline
    has tracked the player's starts; 0 for a genuine first-time starter
    (no first-basket history and no record in the post-All-Star / BRef
    tables — a debut or a return from obscurity); otherwise the team-games
    fallback n_T, the established-player default for a player whose starts
    are not yet tracked. The fallback is what keeps an established player
    from being shrunk as if he went 0-for-n_T — and what keeps a tracked
    player's small early g from inflating him (see the g-increment step,
    which initializes an established player's entry at n_T, not at 0)."""
    if key in team_g_nk:
        return team_g_nk[key]
    if y_i == 0 and key not in established_nk:
        return 0
    return n_t


def adjusted_shares(game, hist, gmap, established_nk):
    """Per-player adjusted shares with the corrected blend:
        m_hat_i = (K*m_i + y_i) / (K + g_i)
    NO renormalization (dropped per the Nov 5 correction — renormalizing
    starter shares to the model-share sum would fight the bench-share fit
    and change numbers without basis). With g_i = n_T for every starter
    the frozen Nov 5 slate reproduces exactly (Bam 0.1707, Booker 0.1638);
    a synthetic g=0 player keeps his full model share.
    Returns ({name: adj_share_fraction}, {name: {"y": y_i, "g": g_i}}) —
    the y/g detail is logged to the calibration record so the on-demand
    fitter can recompute the blend under a different K."""
    out, detail = {}, {}
    for side in ("away", "home"):
        team = game[side]
        starters = [p for p in game["players"] if p["s"] == side]
        h = hist.get(team, {"n": 0, "c": {}})
        n_t = h["n"]
        team_g_nk = {nk(k): v for k, v in (gmap.get(team) or {}).items()}
        for p in starters:
            key = nk(p["n"])
            y_i = h["c"].get(key, 0)
            g_i = _g_for_blend(key, y_i, team_g_nk, n_t, established_nk)
            out[p["n"]] = (K_BLEND * (p["sh"] / 100.0) + y_i) / (K_BLEND + g_i)
            detail[p["n"]] = {"y": y_i, "g": g_i}
    return out, detail


# ---------------------------------------------------------------------------
# Calibration log (audit items #3/#4: fitting K / bench share / rate anchor)
# ---------------------------------------------------------------------------
def load_calibration(root):
    """Read data/calibration_log.jsonl (one record per game). Returns
    (records, path). Missing file -> empty list (first run)."""
    path = os.path.join(root, "data", "calibration_log.jsonl")
    recs = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs, path


def save_calibration(path, recs):
    """Atomic rewrite. The log is append-only in spirit: records are never
    removed; yesterday's records just get their outcome fields filled in."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def fill_calibration_yesterday(recs, day, games_by_matchup):
    """Fill team_first / first_fg for `day`'s records from the JediBets
    per-date results (winner = larger 1st-quarter clock). Returns the
    number of records filled. Records whose game is not in the results
    (postponed?) stay null and are reported by the caller."""
    filled = 0
    for r in recs:
        if r["date"] != day.isoformat() or r.get("team_first") is not None:
            continue
        matchup = None
        for m in games_by_matchup:
            parts = re.match(r"(.+?) at (.+)", m)
            if not parts:
                continue
            if (FULL_NAME_TO_ABBR.get(parts.group(1)) == r["away"]
                    and FULL_NAME_TO_ABBR.get(parts.group(2)) == r["home"]):
                matchup = m
                break
        if matchup is None:
            continue
        first = max(games_by_matchup[matchup], key=lambda x: x["clock"])
        r["team_first"] = first["team"]
        r["first_fg"] = {"team": first["team"], "player": first["scorer"]}
        filled += 1
    return filled


# ---------------------------------------------------------------------------
# Validation gate (pre-commit). validate_slate() returns a list of problem
# strings (empty = pass). main() calls it after the slate/track are built
# and before ANY write, so a failed gate leaves the previous data.json
# (and calibration log) untouched on disk. The shares band is 94-98 per
# side: compute_shares() targets BENCH_SHARE=96.0 with each share rounded
# to 0.1, so a good slate sums to ~96.0, never ~100 — a 96-104 band would
# be wrong for this code.
# ---------------------------------------------------------------------------
def validate_slate(games_out, track_games, dd_out, all_starters, line_for_cal,
                   props_out=None):
    """Pre-commit checks on the built slate. Returns [problems] (empty =
    pass). Checks: >= 1 game; exactly 5 starters per side per game;
    shares sum 94-98 per side; 0 < awayTip < 100; rates 25-60; every game
    has a line and a total in range; dd list non-empty and the expected
    length; track prob sums 0.85-1.05 with all probs in (0,1). When
    props_out is given: props list non-empty and the expected length
    (top PROPS_TOP starters), every points mean/SD and reference-line
    probability in range, threes means in range where present."""
    problems = []
    if len(games_out) < 1:
        problems.append("slate has no games")
    for g in games_out:
        key = f"{g['away']}@{g['home']}"
        for side in ("away", "home"):
            n = sum(1 for p in g["players"] if p["s"] == side)
            if n != 5:
                problems.append(f"{key} {side} has {n} starters (need 5)")
            sh = sum(p["sh"] for p in g["players"] if p["s"] == side)
            if not (94.0 <= sh <= 98.0):
                problems.append(f"{key} {side} shares sum {sh}")
        if not (0 < g["awayTip"] < 100):
            problems.append(f"{key} awayTip {g['awayTip']}")
        if not (25 <= g["awayRate"] <= 60 and 25 <= g["homeRate"] <= 60):
            problems.append(f"{key} rates {g['awayRate']}/{g['homeRate']}")
        line = g.get("line") or ""
        if not line or "total" not in line:
            problems.append(f"{key} missing line/total")
        cal = line_for_cal.get(key)
        if cal is None:
            problems.append(f"{key} missing calibration line")
        elif not (150 <= float(cal[1]) <= 300):
            problems.append(f"{key} total {cal[1]} out of range")
    for tg in track_games:
        s = sum(p[2] for p in tg["p"])
        if not (0.85 <= s <= 1.05):
            problems.append(f"track {tg['g']} prob sum {s:.4f}")
        if not all(0 < p[2] < 1 for p in tg["p"]):
            problems.append(f"track {tg['g']} has out-of-range prob")
    if len(dd_out) < 1:
        problems.append("dd list is empty")
    if len(dd_out) != min(DD_TOP, len(all_starters)):
        problems.append(f"dd list has {len(dd_out)} entries "
                        f"(need {min(DD_TOP, len(all_starters))})")
    if props_out is not None:
        if len(props_out) < 1:
            problems.append("props list is empty")
        if len(props_out) != min(PROPS_TOP, len(all_starters)):
            problems.append(f"props list has {len(props_out)} entries "
                            f"(need {min(PROPS_TOP, len(all_starters))})")
        for p in props_out:
            if not (0 <= p["pts"] < 60 and 0 < p["sd"] <= 20):
                problems.append(f"props {p['n']} pts/sd out of range")
            if not (0 < p["p_pts"] < 1):
                problems.append(f"props {p['n']} p_pts out of range")
            if p.get("fg3") is not None and not (0 <= p["fg3"] <= 10):
                problems.append(f"props {p['n']} fg3 out of range")
            if p.get("p_fg3") is not None and not (0 < p["p_fg3"] < 1):
                problems.append(f"props {p['n']} p_fg3 out of range")
    return problems


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".",
                    help="repo root (holds data.json, data/tip_priors.json)")
    ap.add_argument("--date", default=None,
                    help="run date YYYY-MM-DD (default: today, America/New_York)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute everything but do not write data.json")
    args = ap.parse_args()
    root = args.root
    data_path = os.path.join(root, "data.json")
    priors_path = os.path.join(root, "data", "tip_priors.json")
    with open(data_path) as f:
        data = json.load(f)
    with open(priors_path) as f:
        priors = json.load(f)["priors"]
    today = (date.fromisoformat(args.date) if args.date
             else datetime.now(NY).date())
    yesterday = today - timedelta(days=1)
    notes, warnings = [], []
    fblog = data["fblog"]
    track = data["track"]
    postasb_players = data["postasb"]["players"]
    cal_recs, cal_path = load_calibration(root)

    # ---- 1. Yesterday's results: JediBets -> fblog log + track act -------
    try:
        rows, games_by_matchup = fetch_jedibets_day(root, yesterday)
    except FetchError as e:
        print(f"FATAL: yesterday's results unreadable: {e}", file=sys.stderr)
        return 2
    filled_cal = fill_calibration_yesterday(cal_recs, yesterday,
                                            games_by_matchup)
    if filled_cal:
        notes.append(f"calibration: filled outcome for {filled_cal} "
                     f"games on {yesterday}")
    # Putback check (forward-only): one BRef PBP page per game; a first
    # basket that was a putback counts 0.5 in the history count. Best
    # effort — a skipped page only costs the discount for that game.
    putbacks = {}  # (team, scorer) -> True
    matchups = {}
    for m in games_by_matchup:
        parts = re.match(r"(.+?) at (.+)", m)
        if not parts:
            continue
        aw = FULL_NAME_TO_ABBR.get(parts.group(1))
        hm = FULL_NAME_TO_ABBR.get(parts.group(2))
        if aw and hm:
            matchups[m] = (aw, hm)
    first_page = True
    for m, (aw, hm) in sorted(matchups.items()):
        if not first_page:
            time.sleep(1.0)  # polite delay between BRef PBP pages
        first_page = False
        flags = fetch_bref_pbp_putbacks(root, yesterday, aw, hm)
        if not flags:
            continue  # fetch/parse already warned; log at full weight
        for r in games_by_matchup[m]:
            side = "away" if r["team"] == aw else "home"
            if side not in flags:
                print(f"WARNING: putback check found no first FG for "
                      f"{r['team']} in {aw}@{hm} on {yesterday}",
                      file=sys.stderr)
            elif flags[side]:
                putbacks[(r["team"], r["scorer"])] = True
    if putbacks:
        notes.append(f"putback: {len(putbacks)} of yesterday's first "
                     f"baskets were putbacks (counted 0.5)")
    existing_log = {(e["d"], e["t"], e["fg"]) for e in fblog.get("log") or []}
    added_log = 0
    for r in rows:
        key = (yesterday.isoformat(), r["team"], r["scorer"])
        if key not in existing_log:
            entry = {"d": yesterday.isoformat(), "t": r["team"],
                     "fg": r["scorer"]}
            if putbacks.get((r["team"], r["scorer"])):
                entry["w"] = 0.5  # putback: half weight in the history count
            fblog.setdefault("log", []).append(entry)
            existing_log.add(key)
            added_log += 1
    notes.append(f"fblog: +{added_log} log entries for {yesterday}")
    # Fill track act for yesterday's night (winner = larger 1st-Q clock).
    night = next((n for n in track["nights"] if n["d"] == yesterday.isoformat()),
                 None)
    filled_act = 0
    if night is None:
        warnings.append(f"no track night for {yesterday} (nothing to score)")
    else:
        for g in night["games"]:
            if g.get("act"):
                continue
            away, home = g["g"].split("@")
            matchup = None
            for m in games_by_matchup:
                parts = re.match(r"(.+?) at (.+)", m)
                if not parts:
                    continue
                if (FULL_NAME_TO_ABBR.get(parts.group(1)) == away
                        and FULL_NAME_TO_ABBR.get(parts.group(2)) == home):
                    matchup = m
                    break
            if matchup is None:
                warnings.append(f"no JediBets game for {g['g']} on {yesterday}")
                continue
            rs = games_by_matchup[matchup]
            first = max(rs, key=lambda r: r["clock"])
            g["act"] = {"t": first["team"], "fg": first["scorer"]}
            filled_act += 1
        notes.append(f"track: filled act for {filled_act} games on {yesterday}")

    # Fill DD actuals for yesterday's night from BRef box scores.
    # Best-effort like the putback check: a missed page only leaves
    # those entries unscored for that night (they are not retried on
    # later nights, mirroring the track-act fill). The box fetch is
    # shared with the props fill (same pages, cached per game) and
    # returns fg3 as well, so one fetch serves both fills.
    _box_cache = {}
    _box_first = [True]

    def _fetch_box(aw, hm, _first=_box_first):
        if (aw, hm) in _box_cache:
            return _box_cache[(aw, hm)]
        if not _first[0]:
            time.sleep(1.0)  # polite delay between BRef box pages
        _first[0] = False
        box = fetch_bref_boxscore(root, yesterday, aw, hm, want_fg3=True)
        _box_cache[(aw, hm)] = box
        return box

    if night is not None and night.get("dd"):
        dd_filled, dd_warnings = fill_dd_actuals(
            night, yesterday, matchups, _fetch_box)
        warnings.extend(dd_warnings)
        notes.append(f"track: filled DD actuals for {dd_filled} players "
                     f"on {yesterday}")

    # Fill props actuals for yesterday's night (same box scores).
    if night is not None and night.get("props"):
        props_filled, props_warnings = fill_props_actuals(
            night, yesterday, matchups, _fetch_box)
        warnings.extend(props_warnings)
        notes.append(f"track: filled props actuals for {props_filled} "
                     f"entries on {yesterday}")

    # ---- 2. Tonight's inputs ---------------------------------------------
    covers = fetch_covers_board(root, today)
    if not covers:
        print(f"FATAL: no Covers games for {today}", file=sys.stderr)
        return 2
    notes.append(f"covers: {len(covers)} games with lines for {today}")
    try:
        bm = fetch_bm_lineups(root)
    except FetchError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    notes.append(f"basketball-monster: lineups for {len(bm)} games")
    # A BM game with no Covers entry is skipped from the slate by the
    # loop below — never silently. If this fires, check the game on a
    # second source (FanDuel / BetMGM / Action Network) before trusting
    # the drop: it may be a missing TEAM_ALIASES entry (BKN@ATL, Nov 7:
    # Covers calls Brooklyn "BK"), not a genuinely unlined game.
    bm_only = sorted(set(bm) - set(covers))
    if bm_only:
        warnings.append(
            f"BM games with NO Covers lines (skipped from slate): "
            f"{', '.join(bm_only)} — verify on a second source before "
            f"trusting the drop")
    try:
        bref_cur = fetch_bref_per_game(root, 2027, "2026-27")
        bref_full = fetch_bref_per_game(root, 2026, "2025-26")
    except FetchError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    notes.append(f"bref: {len(bref_cur)} current / {len(bref_full)} full-25-26 players")
    try:
        statpick = fetch_statpick_tips(root)
        team_tips = fetch_jedibets_team_tips(root)
    except FetchError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    notes.append(f"statpick: {len(statpick)} tip records; "
                 f"jedibets team tips: {len(team_tips)} teams")

    # Canonicalize BM starter names against the known player universe
    # (alias > exact > fuzzy). A new BM short form resolves here instead
    # of killing the run in blend_stats.
    universe = build_player_universe(postasb_players, bref_cur, bref_full,
                                     fblog)
    fuzzy_n = canonicalize_lineups(bm, universe)
    if fuzzy_n:
        notes.append(f"names: {fuzzy_n} fuzzy BM name resolution(s) "
                     f"(see WARNINGs)")

    hist = build_history(fblog)
    gmap = fblog.get("g") or {}
    postasb_nk = {nk(k): v for k, v in postasb_players.items()}
    games_out, all_starters = [], []
    line_for_cal = {}  # game key -> (spread_home, total), for calibration
    for key in sorted(covers):
        cov = covers[key]
        away, home = cov["away"], cov["home"]
        if key not in bm:
            print(f"FATAL: no Basketball Monster lineup for {key} "
                  f"({today}); cannot build its slate", file=sys.stderr)
            return 2
        lineups = bm[key]
        starters = {"away": lineups["away"], "home": lineups["home"]}
        for side in ("away", "home"):
            if len(starters[side]) != 5:
                print(f"FATAL: {key} {side} has "
                      f"{len(starters[side])} starters (need 5)",
                      file=sys.stderr)
                return 2
        # Blended stats + shares per team.
        players_out = []
        blended = {}
        for side in ("away", "home"):
            fgm_map = {}
            for s in starters[side]:
                try:
                    vals = blend_stats(s["name"], postasb_nk,
                                       bref_cur, bref_full)
                except Exception as e:
                    # Was an uncaught traceback (exit 1) in run #8; now a
                    # FATAL naming the game/side/player (exit 2).
                    print(f"FATAL: {key} {side} {s['name']}: "
                          f"blend failed: {e}", file=sys.stderr)
                    return 2
                blended[s["name"]] = vals
                fgm_map[s["name"]] = vals[0]
            shares = compute_shares(fgm_map)
            for s in starters[side]:
                tag = f" ({s['tag']})" if s.get("tag") else ""
                players_out.append({"n": s["name"] + tag, "s": side,
                                    "sh": shares[s["name"]]})
                # dd entries carry the display name (with tag), like the page.
                all_starters.append((s["name"] + tag,
                                     away if side == "away" else home,
                                     blended[s["name"]]))
        # Lines: spread = Open (home perspective), total = median book total.
        spread_home = cov["spread_open"]
        total = (statistics.median(cov["totals"]) if cov["totals"]
                 else cov["total_open"])
        total = round(float(total), 1)
        line_for_cal[key] = (spread_home, total)
        away_rate, home_rate = score_rates(spread_home, total)
        fav = home if spread_home < 0 else away
        line = f"{fav} -{abs(spread_home):.1f}, total {total:.1f}"
        # Tip: jumpers are the starting Cs.
        jumpers = {}
        for side in ("away", "home"):
            cands = [s for s in starters[side] if s.get("pos") == "C"]
            if not cands:
                cands = starters[side]
            # Prefer the candidate with a StatPick/prior record.
            cands.sort(key=lambda s: (nk(s["name"]) in statpick,
                                      any(nk(p) == nk(s["name"])
                                          for p in priors)), reverse=True)
            jumpers[side] = cands[0]["name"]
        tip_src = {}
        recs = {}
        for side in ("away", "home"):
            team = away if side == "away" else home
            try:
                recs[side], tip_src[side] = resolve_tip_record(
                    jumpers[side], team, statpick, priors, team_tips)
            except Exception as e:
                print(f"FATAL: {key} {side} jumper {jumpers[side]} "
                      f"({team}): tip record failed: {e}", file=sys.stderr)
                return 2
        away_tip = round(tip_pct(recs["away"][0], recs["away"][1],
                                 recs["home"][0], recs["home"][1]), 1)
        notes.append(
            f"{key}: line {line} (open spread {spread_home}, "
            f"median total {total} from {len(cov['totals'])} books); "
            f"tip {jumpers['away']} {recs['away']} ({tip_src['away']}) vs "
            f"{jumpers['home']} {recs['home']} ({tip_src['home']}) -> "
            f"awayTip {away_tip}")
        games_out.append({
            "away": away, "home": home, "awayTip": away_tip,
            "awayRate": away_rate, "homeRate": home_rate,
            "tipNote": f"{jumpers['away']} vs {jumpers['home']}",
            "line": line, "players": players_out})
    # Double-double list: top DD_TOP by ddProb among tonight's starters.
    # (On a light night there may be fewer starters than DD_TOP; the list
    # is then every starter, and validation expects min(DD_TOP, starters).)
    dd_ranked = []
    for name, team, vals in all_starters:
        _fgm, pts, reb, ast = vals
        dd_ranked.append({"n": name, "t": team,
                          "pts": round(pts, 1), "reb": round(reb, 1),
                          "ast": round(ast, 1),
                          "_reb": reb, "_ast": ast,
                          "matchup_n": 0, "matchup_reb": None,
                          "matchup_ast": None,
                          "_p": dd_prob(pts, reb, ast)})
    dd_ranked.sort(key=lambda d: -d["_p"])
    # Matchup adjustment (Nov 7): for the top-25 DD players only, shrink
    # reb/ast means toward their games vs tonight's opponent. The blend
    # runs on full-precision season means; the stored (and page-displayed)
    # values are the rounded adjusted means, and _p is recomputed from
    # those so the notes match what the page will show.
    team_to_opp = {}
    for g in games_out:
        team_to_opp[g["away"]] = g["home"]
        team_to_opp[g["home"]] = g["away"]
    url_map = {}
    for tbl in (bref_cur, bref_full):
        for k, rec in tbl.items():
            if rec.get("url"):
                url_map.setdefault(k, rec["url"])
    # Memoized game-log fetch, shared by the DD matchup adjustment and
    # the props build below so a player in both lists is fetched once.
    _log_cache = {}

    def _fetch_log(root_, url_, season_):
        key = (url_, season_)
        if key not in _log_cache:
            _log_cache[key] = fetch_bref_gamelog(root_, url_, season_)
        return _log_cache[key]

    first_log = True
    for d in dd_ranked[:DD_TOP]:
        opp = team_to_opp.get(d["t"])
        if opp is None:
            continue
        if not first_log:
            time.sleep(1.0)  # polite delay between BRef game-log pages
        first_log = False
        adj_reb, adj_ast, n, m_reb, m_ast = adjust_dd_for_matchup(
            d["n"], opp, d["_reb"], d["_ast"], url_map, root,
            fetch=_fetch_log)
        d["reb"], d["ast"] = round(adj_reb, 1), round(adj_ast, 1)
        d["matchup_n"] = n
        d["matchup_reb"] = round(m_reb, 1) if n else None
        d["matchup_ast"] = round(m_ast, 1) if n else None
        d["_p"] = dd_prob(d["pts"], d["reb"], d["ast"])
    dd_out = [{k: d[k] for k in ("n", "t", "pts", "reb", "ast",
                                 "matchup_n", "matchup_reb", "matchup_ast")}
              for d in dd_ranked[:DD_TOP]]
    notes.append("dd top25: " + ", ".join(
        f"{d['n']} {d['_p']*100:.1f}%" for d in dd_ranked[:DD_TOP]))

    # Player props (Nov 13): top PROPS_TOP starters by points mean, with
    # season game-log distributions. Best-effort: a failed log falls back
    # to season means (see build_props); props never fail the run.
    props_out, props_warnings = build_props(
        all_starters, url_map, bref_cur, root, fetch=_fetch_log)
    warnings.extend(props_warnings)
    notes.append(f"props: {len(props_out)} players "
                 f"({sum(1 for p in props_out if p.get('fg3') is not None)} "
                 f"with threes)")

    # ---- 3. Tonight's slate + frozen track predictions --------------------
    slate = {"date": today.isoformat(),
             "note": (f"{len(games_out)} games · post-All-Star '25-26 + "
                      "'26-27 blend, first-basket history, Covers lines, "
                      "tip records"),
             "games": games_out, "dd": dd_out}
    # Track predictions use the UPDATED history (incl. yesterday) and the
    # corrected blend (per-player g, no renormalization).
    hist = build_history(fblog)
    # Established players (for the first-timer g=0 rule): anyone with a
    # post-All-Star record, a current-season BRef record, or a full-season
    # 2025-26 BRef record. A starter in none of those tables and with no
    # first-basket history is a genuine first-time starter.
    established_nk = (set(postasb_nk)
                      | set(bref_cur) | set(bref_full))
    track_games = []
    cal_tonight = []
    for g in games_out:
        g_key = f"{g['away']}@{g['home']}"
        af, hf = game_calc(g["awayTip"], g["awayRate"], g["homeRate"])
        adj, detail = adjusted_shares(g, hist, gmap, established_nk)
        plist = []
        starters_cal = []
        for p in g["players"]:
            team_first = af if p["s"] == "away" else hf
            prob = team_first * adj[p["n"]]
            team = g["away"] if p["s"] == "away" else g["home"]
            # Track entries use the name WITHOUT the injury tag,
            # matching the frozen track format.
            pname = re.sub(r"\s*\([A-Z]+\)$", "", p["n"])
            plist.append([pname, team, round(prob, 4)])
            starters_cal.append({"n": pname, "team": team,
                                 "sh": round(p["sh"] / 100.0, 4),
                                 "adj": round(adj[p["n"]], 4),
                                 "y": detail[p["n"]]["y"],
                                 "g": detail[p["n"]]["g"]})
        plist.sort(key=lambda x: -x[2])
        track_games.append({"g": f"{g['away']}@{g['home']}",
                            "pa": round(af, 4), "p": plist, "act": None})
        cal_tonight.append({"date": today.isoformat(),
                            "away": g["away"], "home": g["home"],
                            "awayTip": g["awayTip"],
                            "awayRate": g["awayRate"],
                            "homeRate": g["homeRate"],
                            # Raw line, so the fitter can recompute the
                            # rates under a different anchor.
                            "spread_home": line_for_cal[g_key][0],
                            "total": line_for_cal[g_key][1],
                            "starters": starters_cal,
                            "team_first": None, "first_fg": None})
    track_dd = build_track_dd(dd_ranked)
    track_props = build_track_props(props_out)
    # Replace tonight's night on re-runs (idempotent); else append.
    track["nights"] = [n for n in track["nights"]
                       if n["d"] != today.isoformat()]
    track["nights"].append({"d": today.isoformat(), "games": track_games,
                            "dd": track_dd, "props": track_props})

    # ---- 4. Increment g for tonight's starters (once per date) ------------
    # g entries are created here, from real Basketball Monster lineups —
    # never seeded from the seed game count. An established starter with no
    # entry yet is initialized at the CURRENT team total n_T (the fallback
    # approximation, which keeps his next-night blend at the right
    # magnitude); a genuine first-timer is initialized at 0. Then +1 for
    # tonight's start. Keys are plain display names (injury tags stripped)
    # so the blend's nk() lookup matches.
    meta = data.setdefault("_meta", {})
    if meta.get("g_increment_date") != today.isoformat():
        for g in games_out:
            for p in g["players"]:
                team = g["away"] if p["s"] == "away" else g["home"]
                n_t = hist.get(team, {"n": 0})["n"]
                pname = re.sub(r"\s*\([A-Z]+\)$", "", p["n"])
                key = nk(pname)
                team_g = gmap.setdefault(team, {})
                existing = next((k for k in team_g if nk(k) == key), None)
                if existing is None:
                    y_i = hist.get(team, {"c": {}})["c"].get(key, 0)
                    init = (0 if (y_i == 0 and key not in established_nk)
                            else n_t)
                    team_g[pname] = init + 1
                else:
                    team_g[existing] += 1
        meta["g_increment_date"] = today.isoformat()
        notes.append(f"g: incremented games-started for "
                     f"{sum(len(g['players']) for g in games_out)} starters")
    else:
        notes.append("g: already incremented today (re-run); skipped")
    fblog["g"] = gmap

    # Calibration: append tonight's records (outcomes filled tomorrow).
    cal_recs = [r for r in cal_recs if r["date"] != today.isoformat()]
    cal_recs.extend(cal_tonight)
    notes.append(f"calibration: +{len(cal_tonight)} records for {today}")

    # ---- 5. Validation ------------------------------------------------------
    # Pre-commit gate (validate_slate): a failed gate returns 2 BEFORE
    # any write, so the previous data.json stays live on disk.
    problems = validate_slate(games_out, track_games, dd_out, all_starters,
                              line_for_cal, props_out)
    if problems:
        print("FATAL: validation failed:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2

    # ---- 6. Write -----------------------------------------------------------
    data["slate"] = slate
    data["props"] = {
        "as_of": today.isoformat(),
        "note": ("Points: Normal(mean, sd) from season game logs. Threes: "
                 "Poisson(mean 3PM). Reference line = nearest .5 to the "
                 "mean (the model's median line, NOT a book line). Book "
                 "lines are entered by hand on the page; the pipeline "
                 "never copies a book line."),
        "players": props_out}
    meta["updated"] = (datetime.now(NY).isoformat(timespec="seconds"))
    meta["as_of"] = today.isoformat()
    meta["version"] = int(meta.get("version") or 1)
    if args.dry_run:
        print("[dry-run] would write data.json")
    else:
        tmp = data_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
            f.write("\n")
        os.replace(tmp, data_path)
        print(f"wrote {data_path}")
        save_calibration(cal_path, cal_recs)
        print(f"wrote {cal_path} ({len(cal_recs)} records)")
    print(f"run {today.isoformat()}: {len(games_out)} games, "
          f"{len(track_games)} track games")
    for n in notes:
        print(f"  - {n}")
    for w in warnings:
        print(f"  ! {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
