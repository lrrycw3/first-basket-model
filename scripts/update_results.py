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
import json
import math
import os
import re
import statistics
import sys
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
    "GS": "GSW",
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


def http_get(url, root, debug_name, timeout=30, retries=2):
    """GET with a browser UA. Raises FetchError (after saving a debug copy
    when a response body exists)."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 1000:
                return r.text
            last_exc = FetchError(
                f"{url} -> HTTP {r.status_code} ({len(r.content)} bytes)")
            if r.content:
                debug_dump(root, debug_name, r.content)
        except requests.RequestException as e:
            last_exc = FetchError(f"{url} -> {e}")
        if attempt < retries:
            import time
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
    status note after the name: 'S' (confirmed starter), 'IN', 'Off Inj' —
    none of those is an injury tag, so they are dropped. Only a Q/GTD/O/D/P
    note becomes a tag."""
    text = " ".join((cell or "").split())  # collapse all whitespace/newlines
    if not text:
        return "", ""
    m = re.match(r"^(.*?)\s*\((Q|GTD|O|D|P)\)\s*$", text)
    if m:
        return m.group(1).strip(), m.group(2)
    for suffix, tag in ((" Off Inj", ""), (" IN", ""), (" S", ""),
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
    trb_per_g / ast_per_g (the classic player/fg/pts names are gone)."""
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
                   "reb": float(cells.get("trb_per_g") or 0),
                   "ast": float(cells.get("ast_per_g") or 0),
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
        h["c"][key] = h["c"].get(key, 0) + 1
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
    existing_log = {(e["d"], e["t"], e["fg"]) for e in fblog.get("log") or []}
    added_log = 0
    for r in rows:
        key = (yesterday.isoformat(), r["team"], r["scorer"])
        if key not in existing_log:
            fblog.setdefault("log", []).append(
                {"d": yesterday.isoformat(), "t": r["team"], "fg": r["scorer"]})
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
                vals = blend_stats(s["name"], postasb_nk,
                                   bref_cur, bref_full)
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
            recs[side], tip_src[side] = resolve_tip_record(
                jumpers[side], team, statpick, priors, team_tips)
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
    # Double-double list: top 4 by ddProb among tonight's starters.
    dd_ranked = []
    for name, team, vals in all_starters:
        _fgm, pts, reb, ast = vals
        dd_ranked.append({"n": name, "t": team,
                          "pts": round(pts, 1), "reb": round(reb, 1),
                          "ast": round(ast, 1),
                          "_p": dd_prob(pts, reb, ast)})
    dd_ranked.sort(key=lambda d: -d["_p"])
    dd_out = [{k: d[k] for k in ("n", "t", "pts", "reb", "ast")}
              for d in dd_ranked[:4]]
    notes.append("dd top4: " + ", ".join(
        f"{d['n']} {d['_p']*100:.1f}%" for d in dd_ranked[:4]))

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
    # Replace tonight's night on re-runs (idempotent); else append.
    track["nights"] = [n for n in track["nights"]
                       if n["d"] != today.isoformat()]
    track["nights"].append({"d": today.isoformat(), "games": track_games})

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
    problems = []
    for g in games_out:
        for side in ("away", "home"):
            sh = sum(p["sh"] for p in g["players"] if p["s"] == side)
            if not (94.0 <= sh <= 98.0):
                problems.append(f"{g['away']}@{g['home']} {side} shares sum {sh}")
        if not (0 < g["awayTip"] < 100):
            problems.append(f"{g['away']}@{g['home']} awayTip {g['awayTip']}")
        if not (25 <= g["awayRate"] <= 60 and 25 <= g["homeRate"] <= 60):
            problems.append(f"{g['away']}@{g['home']} rates "
                            f"{g['awayRate']}/{g['homeRate']}")
    for tg in track_games:
        s = sum(p[2] for p in tg["p"])
        if not (0.85 <= s <= 1.05):
            problems.append(f"track {tg['g']} prob sum {s:.4f}")
        if not all(0 < p[2] < 1 for p in tg["p"]):
            problems.append(f"track {tg['g']} has out-of-range prob")
    if len(dd_out) != 4:
        problems.append(f"dd list has {len(dd_out)} entries (need 4)")
    if problems:
        print("FATAL: validation failed:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2

    # ---- 6. Write -----------------------------------------------------------
    data["slate"] = slate
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
