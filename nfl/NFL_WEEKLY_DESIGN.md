# NFL Weekly Automation — Design (written Nov 7, 2026)

User's cadence decisions (Nov 7): data refreshes **early Tuesday** after the
week's games are done; a **daily ~11am ET check** picks up injury updates.
Thursday + Monday night games are in the slate. Offensive players only.

## Tuesday early update (weekly) — `nfl-weekly.yml`
Schedule: **11:00 UTC Tuesdays** (6:00am EST / 7:00am EDT — "early Tuesday").
Runs on the GitHub Actions runner (plain HTTP fetches only: nflverse + ESPN;
no Playwright needed, no Covers — lines come from the ESPN scoreboard API).

Steps:
1. Download fresh `play_by_play_2026.csv.gz`, `games.csv`,
   `injuries_2026.csv` from nflverse-data (cache in `nfl/data/`, refresh weekly).
2. **Score the prior week** (new script `score_week.py`, to be written):
   for each completed game, take the actual first OFFENSIVE TD scorer from
   the PBP; compare against the slate's top pick per game (highest model
   probability) and record hit/miss. Append to `data.json`'s `nfl.track`:
   `{"week": 9, "games": [{"g": "DAL@IND", "pred": "Jonathan Taylor",
   "pred_p": 0.24, "act": "Jonathan Taylor", "hit": true}], ...}`.
   The page's First TD Track record view already reads `nfl.track`.
3. Refresh team first-TD rates + player shares with the new week's data
   (same formulas as `build_td_slate.py`: shrinkage `r=(c+5)/(n+10)`,
   log5 team prob, share = 0.5*TD share + 0.5*red-zone-touch share,
   history blend K=20).
4. Build next week's slate: `build_td_slate.py --week N --season 2026`
   (already reusable; Week 10 = Nov 12–16).
5. Merge the new `nfl` block into `data.json` (preserving the basketball
   keys) and commit via GITHUB_TOKEN — same pattern as the basketball
   nightly job. The basketball pipeline round-trips unknown top-level
   keys (verified in code Nov 7), so the `nfl` block survives the
   nightly basketball runs.

## Daily injury/line check — `nfl-daily.yml`
Schedule: **16:00 UTC daily** (11:00am EST / 12:00pm EDT).
Steps:
1. Re-fetch `injuries_2026.csv` + ESPN scoreboard lines.
2. Update the CURRENT week's slate in place (new `--refresh` mode on
   `build_td_slate.py`, to be written): drop newly-Out players (next man
   moves up), mark newly-Questionable with " (Q)", refresh spreads/totals.
   Never re-scores, never rebuilds shares, never touches the basketball keys.
3. Commit `data.json` only if something changed.

## Still to build
- `score_week.py` (Tuesday scoring + `nfl.track` append).
- `--refresh` mode on `build_td_slate.py` (daily injury/line update).
- The two workflow files (`.github/workflows/nfl-weekly.yml`,
  `.github/workflows/nfl-daily.yml`) — via web UI like the basketball one,
  since the Muse GitHub App can't push.
- Double-double result logging is a SEPARATE basketball-pipeline addition
  (see todo); it is not part of this NFL design.

## Model fits — home field (Nov 7) + pre-registered fit grid
**Home field, fitted Nov 7 from the 122 completed games (leave-one-out
rates — the first fit used in-sample rates and leaked, giving a bogus
slope of 3.0; the LOO fit is the honest one):** logistic regression
logit(P(home scores first offensive TD)) = a + b*(logit rH − logit rA):
a = 0.450 (SE 0.190), b = 0.720 (SE 0.398). b is not distinguishable
from 1, so the recommended application is the one-parameter shift on top
of log5: P_home = logistic(logit(P_home_log5) + 0.467), where 0.467 is
the b=1 fit (SE 0.191, home edge 61.5% at even rates; raw count 73/121
= 60.3%). **APPLIED Nov 7 at his yes** — `HOME_LOGIT = 0.467` in
`build_td_slate.py`, landed with the Week 9 slate the same evening.
Effect on Week 9 headliners (as landed, from the rounded team probs):
Taylor +261→+208, Walker +297→+237, Gibbs +507→+713, Irving
+448→+612, Hubbard +372→+299.

**Pre-registered fit grid — run once the Tuesday track record has enough
scored weeks; fit ONLY on scored outcomes, never on the slate being
priced; pick by out-of-sample log-loss on first-TD outcomes:**
- Blend weight w in {0, 0.1, …, 1} for
  modelShare = w*tdShare + (1−w)*rzShare (current: 0.5, unfitted).
- History-blend K in {5, 10, 20, 30, 50} (current: 20, unfitted).
- Shrinkage (SHRINK_WINS, SHRINK_GAMES) in {(2,4), (5,10), (8,16),
  (10,20)} (current: (5,10); a crude empirical-Bayes moment fit on the
  32 team rates implied ~17 pseudo-games, so (5,10) is roughly right).
- Refit the home logit a each time the grid runs.
- Also fix the trials proxy when building the automation: use weekly
  player-stats games-played instead of touch-games (the proxy undercounts
  GP for non-ball-carriers and inflates fringe players, e.g. Quitoriano
  trials=1, c=1, sh 11.2).

## Proven so far (Nov 7)
- Re-running the ORIGINAL `build_td_slate.py` reproduced the Week 9
  slate byte-identically (apart from live line movement) except poff
  (script computed 0.9669, hand-fixed to 0.9752 in the live file).
- **Safe fixes applied Nov 7 (in the workspace script; landing pending
  his call on the home term):** offensive-TD definition aligned to
  `td_team == posteam` — the script now computes poff = 0.9752 itself
  (verified: team first-TD counts identical under both definitions;
  only player-level change is Flournoy c 1→0 / sh 10.0→6.2 as
  J.Mitchell takes the ARI@DAL first-TD credit); kneel-downs excluded
  from trials, kneel-downs + 2-pt tries excluded from red-zone touches.
  Net effect on re-run: 91/140 shares move ±0.1–0.7, two fringe top-5
  swaps (Waddle out / R.J. Harvey in for DEN; Cody White out /
  Jack Bech in for LV), no team-probability changes.
- All 14 Week 9 `awayFirstTD` values were independently recomputed from
  raw PBP with 0 diffs; lines verified against nflverse `games.csv`
  (13/14 spreads, 14/14 totals; DAL@IND moved -4.5 → -5.5).
- `poff` = 0.9752 (118/121 first TDs offensive).
