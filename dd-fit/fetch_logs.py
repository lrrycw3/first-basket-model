#!/usr/bin/env python3
"""Fetch NBA player game logs from stats.nba.com (free, no key).

Resolves player IDs via commonallplayers, then downloads playergamelog for
2025-26 (full season) and 2026-27 (to date) for each player, saving the raw
API responses untouched to dd-fit/raw/<slug>_gamelog_<season>.json.

Players: the 4 on the First Basket Model Nov 5 slate DD list, plus 6 added
high reb/ast candidates (Jokic, Jalen Johnson, Cade Cunningham, Wembanyama,
Towns, Sengun) so the pooled correlation is not fit on 4 players alone.

Run from any network where stats.nba.com answers. The VM's egress IP is
blocked by stats.nba.com (connection opens, 0 bytes, read timeout at 30s;
cdn.nba.com returns HTTP 403) — run this from the GitHub Actions runner or
a local machine instead.
"""
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

BASE = "https://stats.nba.com/stats/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
SEASONS = ["2025-26", "2026-27"]

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


def norm_name(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = re.sub(r"\s*\(Q\)\s*$", "", n).strip().lower()
    n = re.sub(r"[^a-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def slug(n):
    return re.sub(r"[^a-z0-9]+", "-", norm_name(n)).strip("-")


def get(url, tries=3):
    last = None
    for i in range(tries):
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:200]
            print(f"  HTTP {e.code} {url[:100]}: {body}", file=sys.stderr)
            last = e
            if e.code in (429, 500, 502, 503) and i < tries - 1:
                time.sleep(5 * (i + 1))
                continue
            raise
        except Exception as e:
            print(f"  ERR {type(e).__name__}: {e} {url[:100]}", file=sys.stderr)
            last = e
            if i < tries - 1:
                time.sleep(5 * (i + 1))
                continue
            raise
    raise last


def resolve_ids():
    data = get(BASE + "commonallplayers?LeagueID=00&Season=2026-27&IsOnlyCurrentSeason=0")
    with open(os.path.join(RAW, "commonallplayers_2026-27.json"), "w") as f:
        json.dump(data, f)
    rs = data["resultSets"][0]
    h = rs["headers"]
    i_id = h.index("PERSON_ID")
    i_first = h.index("DISPLAY_FIRST_NAME")
    i_last = h.index("DISPLAY_LAST_NAME")
    by_name = {}
    for r in rs["rowSet"]:
        by_name[norm_name(r[i_first] + " " + r[i_last])] = r[i_id]
    ids = {}
    for name, _why in PLAYERS:
        pid = by_name.get(norm_name(name))
        if pid is None:
            print(f"  !! no ID found for {name}")
        else:
            ids[name] = pid
            print(f"  {name}: {pid}")
    return ids


def main():
    os.makedirs(RAW, exist_ok=True)
    print("Resolving player IDs (commonallplayers 2026-27)...")
    ids = resolve_ids()
    for name, why in PLAYERS:
        pid = ids.get(name)
        if pid is None:
            continue
        for season in SEASONS:
            out = os.path.join(RAW, f"{slug(name)}_gamelog_{season}.json")
            if os.path.exists(out):
                print(f"  skip {name} {season} (already saved)")
                continue
            url = (BASE + "playergamelog?LeagueID=00&PerMode=PerGame&PlayerID="
                   f"{pid}&Season={season}&SeasonType=Regular+Season")
            print(f"  fetching {name} {season}...")
            data = get(url)
            with open(out, "w") as f:
                json.dump(data, f)
            n = len(data["resultSets"][0]["rowSet"])
            print(f"    saved {n} games -> {out}")
            time.sleep(1.5)
    print("Done.")


if __name__ == "__main__":
    main()
