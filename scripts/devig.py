#!/usr/bin/env python3
"""Market blending for the first-basket / first-TD model (improvement #4).

The pipeline's probabilities come from raw stats — NEVER from another
site's model. Market blending uses only the market's ODDS (de-vigged
to implied probabilities), never a site's model probabilities:

    model prob (ours) + market odds (FanDuel, user-supplied)
        -> Shin de-vig -> 5% model / 95% market blend -> edge flag

There is no free first-basket / first-TD odds feed and no free
closing-odds archive, so the odds come from a JSON file the user
supplies (paste FanDuel first-basket odds) — see the CLI below. The
CLI appends the supplied odds to data/closing_odds.jsonl so the
evaluation harness (#6) can score CLV later.

De-vig methods (Wunderlich & Memmel; Shin 1993):
  * proportional: p_i = q_i / sum(q) — the baseline.
  * Shin: p_i(z) = (sqrt(z^2 + 4(1-z) q_i^2) - z) / (2(1-z)), with z
    solved by bisection so sum(p) = 1. Shin removes more margin from
    longshots than from favorites — the literature standard for
    multi-outcome markets, and the default here.
  * power: q_i = p_i^alpha / sum(p_j^alpha), inverted as
    p_i ~ q_i^(1/alpha). alpha has no closed-form identification from
    odds alone — it must be FIT from settled odds + outcomes (a job
    for the harness, #6). It is therefore an explicit parameter with
    no silent default; alpha=1 reproduces proportional.

Blend weight: 5% model / 95% market is the research default (NBA
moneyline optimal blend, Nov 2026 research). The model is young
(41 scored games, log-loss ~2.31 vs uniform 2.30) — it earns a small
weight until the harness shows it beating the market out of sample.

Edge: expected value per unit staked at the quoted decimal odds,
edge = blended_p * decimal_odds - 1. A play is flagged only when
edge >= EDGE_MIN (0.02 — a starting default, not a fitted constant;
raise it if the harness shows the model over-flagging).
"""

import argparse
import json
import math
import os
import sys

W_MODEL = 0.05   # model weight in the blend (research default)
EDGE_MIN = 0.02  # minimum flagged edge (2% EV per unit staked)


def american_to_prob(odds):
    """American odds ("+500", "-200", 500, -200) -> implied probability.
    Returns None for a missing/placeholder line ("—", "", None)."""
    if odds is None:
        return None
    if isinstance(odds, (int, float)):
        a = float(odds)
    else:
        s = str(odds).strip()
        if not s or s in ("—", "-", "n/a"):
            return None
        a = float(s.replace("+", ""))
    if a == 0:
        return None
    if a > 0:
        return 100.0 / (a + 100.0)
    return -a / (-a + 100.0)


def american_to_decimal(odds):
    """American odds -> decimal odds (stake included). None if missing."""
    p = american_to_prob(odds)
    if p is None or p <= 0:
        return None
    return 1.0 / p


def devig_proportional(probs):
    """p_i = q_i / sum(q). The baseline every other method is judged
    against. probs: {name: implied_prob}; returns {name: prob}."""
    tot = sum(probs.values())
    if tot <= 0:
        return dict(probs)
    return {k: v / tot for k, v in probs.items()}


def _shin_p(q, z):
    if z >= 1.0:
        return q * q
    return (math.sqrt(z * z + 4 * (1 - z) * q * q) - z) / (2 * (1 - z))


def devig_shin(probs):
    """Shin de-vig (z solved by bisection so the probabilities sum to 1).
    Fair odds (no overround) return the input unchanged (z = 0).
    probs: {name: implied_prob}; returns {name: prob}."""
    qs = list(probs.values())
    if not qs or sum(qs) <= 0:
        return dict(probs)
    if sum(qs) <= 1.0:
        # No margin to remove (fair odds or an underround/arbitrage
        # book): Shin's z would be 0, so proportional is the answer.
        return devig_proportional(probs)
    lo, hi = 0.0, 1.0 - 1e-9
    for _ in range(100):
        z = (lo + hi) / 2
        s = sum(_shin_p(q, z) for q in qs)
        if s > 1.0:
            lo = z
        else:
            hi = z
        if hi - lo < 1e-12:
            break
    z = (lo + hi) / 2
    tot = sum(_shin_p(q, z) for q in qs)
    return {k: _shin_p(v, z) / tot for k, v in probs.items()}


def devig_power(probs, alpha):
    """Power de-vig with an explicit alpha (fit from settled odds +
    outcomes; alpha=1 reproduces proportional). p_i ~ q_i^(1/alpha),
    renormalized. alpha must be > 0."""
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if alpha == 1.0:
        return devig_proportional(probs)
    inv = 1.0 / alpha
    raw = {k: v ** inv for k, v in probs.items() if v > 0}
    tot = sum(raw.values())
    if tot <= 0:
        return dict(probs)
    out = {k: raw.get(k, 0.0) / tot for k in probs}
    return out


def devig(probs, method="shin", alpha=None):
    """Dispatch: 'shin' (default), 'power' (needs alpha), 'prop'."""
    if method == "shin":
        return devig_shin(probs)
    if method == "prop":
        return devig_proportional(probs)
    if method == "power":
        if alpha is None:
            raise ValueError("power de-vig needs an explicit alpha")
        return devig_power(probs, alpha)
    raise ValueError(f"unknown de-vig method: {method}")


def blend(model_p, market_p, w_model=W_MODEL):
    """w_model * model + (1 - w_model) * market. Both are probabilities
    for the same outcome."""
    return w_model * model_p + (1 - w_model) * market_p


def edge_report(model_probs, market_odds, method="shin", w_model=W_MODEL,
                edge_min=EDGE_MIN, alpha=None):
    """One game: model_probs {player: prob} (frozen track values),
    market_odds {player: american_odds}. Returns a list of dicts, one
    per player with a quoted line, sorted by edge descending:
    {player, model, market, blended, decimal, edge, flag}.
    'market' is the de-vigged market probability; 'edge' is the EV per
    unit staked at the quoted odds using the BLENDED probability.
    Players without a quoted line are skipped; model players missing
    from the market get market=None and no edge (they cannot be bet)."""
    implied = {}
    for name, odds in market_odds.items():
        p = american_to_prob(odds)
        if p is not None:
            implied[name] = p
    if not implied:
        return []
    market = devig(implied, method=method, alpha=alpha)
    rows = []
    for name, mp in model_probs.items():
        if name not in market:
            continue
        dec = american_to_decimal(market_odds[name])
        bl = blend(mp, market[name], w_model=w_model)
        edge = bl * dec - 1 if dec else None
        rows.append({"player": name, "model": round(mp, 4),
                     "market": round(market[name], 4),
                     "blended": round(bl, 4),
                     "decimal": round(dec, 2) if dec else None,
                     "edge": round(edge, 4) if edge is not None else None,
                     "flag": bool(edge is not None and edge >= edge_min)})
    rows.sort(key=lambda r: (r["edge"] is None, -(r["edge"] or 0)))
    return rows


def load_track_probs(data):
    """data.json -> {game_key: {player: prob}} from the frozen track."""
    out = {}
    for g in (data.get("track") or {}).get("games", []):
        out[g["g"]] = {n: p for n, _t, p in g["p"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data.json",
                    help="data.json with the frozen track (default data.json)")
    ap.add_argument("--odds", required=True,
                    help="JSON file of market odds: "
                         '{"GAMEKEY": {"Player": "+500", ...}} '
                         "(GAMEKEY like 'MIA@BOS')")
    ap.add_argument("--method", default="shin",
                    choices=["shin", "prop", "power"])
    ap.add_argument("--alpha", type=float, default=None,
                    help="power-method exponent (required with --method power)")
    ap.add_argument("--w-model", type=float, default=W_MODEL)
    ap.add_argument("--edge-min", type=float, default=EDGE_MIN)
    ap.add_argument("--record", default=None,
                    help="append the supplied odds to this JSONL file "
                         "(default: data/closing_odds.jsonl next to --data)")
    args = ap.parse_args()
    with open(args.data) as f:
        data = json.load(f)
    with open(args.odds) as f:
        odds = json.load(f)
    track = load_track_probs(data)
    as_of = (data.get("_meta") or {}).get("updated", "?")
    any_flag = False
    for game, lines in sorted(odds.items()):
        model = track.get(game)
        if model is None:
            print(f"{game}: no track game in {args.data}; skipped")
            continue
        rows = edge_report(model, lines, method=args.method,
                           w_model=args.w_model, edge_min=args.edge_min,
                           alpha=args.alpha)
        print(f"\n{game} (model as of {as_of}; {args.method} de-vig, "
              f"w_model={args.w_model})")
        for r in rows:
            mark = "  <-- EDGE" if r["flag"] else ""
            print(f"  {r['player']:<24} model {r['model']:.3f}  "
                  f"market {r['market']:.3f}  blended {r['blended']:.3f}  "
                  f"edge {r['edge']:+.3f}{mark}")
            any_flag = any_flag or r["flag"]
    if not any_flag:
        print("\nno flagged edges at this threshold.")
    rec_path = args.record
    if rec_path is None:
        rec_path = os.path.join(os.path.dirname(
            os.path.abspath(args.data)), "closing_odds.jsonl")
    os.makedirs(os.path.dirname(rec_path) or ".", exist_ok=True)
    with open(rec_path, "a") as f:
        for game, lines in sorted(odds.items()):
            for player, line in sorted(lines.items()):
                f.write(json.dumps({"date": as_of[:10], "game": game,
                                    "player": player, "odds": line,
                                    "method": args.method}) + "\n")
    print(f"\nrecorded {sum(len(v) for v in odds.values())} odds lines to "
          f"{rec_path} (closing-price store for CLV, harness #6)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
