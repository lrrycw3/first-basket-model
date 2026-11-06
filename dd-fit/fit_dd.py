#!/usr/bin/env python3
"""Fit double-double correlation + overdispersion from NBA game logs.

Reads stats.nba.com playergamelog responses from a raw directory
(default: dd-fit/raw, files <slug>_gamelog_<season>.json) and:
  * per player: n, mean/sd of PTS/REB/AST, pairwise Pearson correlations,
    overdispersion phi = var/mean for REB and AST (Poisson is phi = 1);
  * pooled correlations via Fisher-z average (weighted by n-3) and pooled phi
    (n-weighted mean), with per-player spread;
  * recomputes P(double-double) for the page's current DD list under
      (a) the current independent formula (page ddProb: normal pts,
          exact Poisson reb/ast, independence), and
      (b) the recommended Gaussian-copula formula: Normal pts marginal,
          negative-binomial reb/ast marginals with fitted phi, pooled
          correlation matrix, Monte Carlo (200k draws, seed 42).

Refuses to run if no real log files are present. Never invents logs.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

PAGE = os.path.expanduser("~/workspace/first-basket-model/index.html")
N_MC = 200_000
SEED = 42


def norm_name(n):
    n = n.replace("(Q)", "").strip()
    n = n.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-")


def page_dd_list(slate_json="data.json"):
    if os.path.exists(slate_json):
        with open(slate_json) as f:
            data = json.load(f)
        return data["slate"]["dd"]
    html = open(PAGE).read()
    m = re.search(r'<script id="slate"[^>]*>(.*?)</script>', html, re.S)
    slate = json.loads(m.group(1))
    return slate["dd"]


def load_logs(raw_dir):
    """Return {norm_name: (pts, reb, ast) float arrays} from raw log files."""
    players = {}
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".json") or "_gamelog_" not in fn:
            continue
        slug = fn.split("_gamelog_")[0]
        with open(os.path.join(raw_dir, fn)) as f:
            data = json.load(f)
        rs = data["resultSets"][0]
        h = rs["headers"]
        need = ["PTS", "REB", "AST"]
        if not all(c in h for c in need):
            raise ValueError(f"{fn}: missing PTS/REB/AST columns")
        ip, ir, ia = h.index("PTS"), h.index("REB"), h.index("AST")
        arr = np.array([[r[ip], r[ir], r[ia]] for r in rs["rowSet"]], dtype=float)
        if slug in players:
            players[slug] = np.vstack([players[slug], arr])
        else:
            players[slug] = arr
    return players


def corr_matrix(x):
    return np.corrcoef(x.T)


def fisher_pool(rs, ns):
    """Fisher-z pooled correlation from per-player r values and sample sizes."""
    zs = [0.5 * math.log((1 + r) / (1 - r)) for r in rs]
    w = [n - 3 for n in ns]
    z = sum(a * b for a, b in zip(zs, w)) / sum(w)
    return math.tanh(z)


def norm_ge(m, sd, k):
    return 0.5 * math.erfc((k - 0.5 - m) / sd / math.sqrt(2))


def pois_ge2(m, k):
    c = 0.0
    t = math.exp(-m)
    for i in range(k):
        c += t
        t *= m / (i + 1)
    return 1 - c


def dd_before(d):
    pp = norm_ge(d["pts"], max(3, 0.28 * d["pts"] + 2), 10)
    pr = pois_ge2(d["reb"], 10)
    pa = pois_ge2(d["ast"], 10)
    return pp * pr + pp * pa + pr * pa - 2 * pp * pr * pa


# --- Acklam's inverse normal CDF (accurate to ~1e-9) ---
def ndtri(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    q = p - 0.5
    m = (p > 0) & (p < plow)
    if m.any():
        qq = np.sqrt(-2 * np.log(p[m]))
        out[m] = (((((c[0] * qq + c[1]) * qq + c[2]) * qq + c[3]) * qq + c[4]) * qq + c[5]) / \
                 ((((d[0] * qq + d[1]) * qq + d[2]) * qq + d[3]) * qq + 1)
    m = (p >= plow) & (p <= phigh)
    if m.any():
        qq = q[m]
        r = qq * qq
        out[m] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * qq / \
                 (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    m = (p > phigh) & (p < 1)
    if m.any():
        qq = np.sqrt(-2 * np.log(1 - p[m]))
        out[m] = -(((((c[0] * qq + c[1]) * qq + c[2]) * qq + c[3]) * qq + c[4]) * qq + c[5]) / \
                  ((((d[0] * qq + d[1]) * qq + d[2]) * qq + d[3]) * qq + 1)
    return out


def nb_quantile_table(mean, phi, tol=1e-12):
    """Inverse-CDF table for NB(mean, var=phi*mean); phi<=1 -> Poisson."""
    if phi <= 1.0:
        # Poisson table
        pmf = [math.exp(-mean)]
        while pmf[-1] > 0:
            pmf.append(pmf[-1] * mean / len(pmf))
        cdf = np.cumsum(pmf)
        return cdf
    n = mean / (phi - 1.0)
    p = n / (n + mean)
    pmf = [p ** n]
    i = 1
    while True:
        pmf.append(pmf[-1] * (n + i - 1) / i * (1 - p))
        if pmf[-1] < tol and sum(pmf) > 1 - tol:
            break
        i += 1
        if i > 500:
            break
    return np.cumsum(pmf)


def nb_ppf(u, cdf):
    return np.searchsorted(cdf, u, side="right")


def dd_copula(d, R, phi_reb, phi_ast, n_mc=N_MC, seed=SEED):
    """Gaussian-copula P(at least two of pts/reb/ast reach 10).

    Marginals: pts ~ Normal(proj, max(3, .28*proj+2)) [page's formula];
    reb/ast ~ NegBin(mean=proj, var=phi*mean), phi<=1 falls back to Poisson.
    """
    rng = np.random.default_rng(seed)
    evals, evecs = np.linalg.eigh(np.asarray(R, dtype=float))
    evals = np.clip(evals, 1e-8, None)
    L = evecs @ np.diag(np.sqrt(evals))
    z = rng.standard_normal((n_mc, 3)) @ L.T
    u = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    pts = d["pts"] + max(3, 0.28 * d["pts"] + 2) * ndtri(np.clip(u[:, 0], 1e-12, 1 - 1e-12))
    reb = nb_ppf(np.clip(u[:, 1], 0, 1 - 1e-12), nb_quantile_table(d["reb"], phi_reb))
    ast = nb_ppf(np.clip(u[:, 2], 0, 1 - 1e-12), nb_quantile_table(d["ast"], phi_ast))
    hits = ((pts >= 10).astype(int) + (reb >= 10).astype(int) + (ast >= 10).astype(int)) >= 2
    p = float(hits.mean())
    se = math.sqrt(p * (1 - p) / n_mc)
    return p, se


def fair(p):
    if p >= 0.5:
        return "-" + str(round(100 * p / (1 - p)))
    return "+" + str(round(100 * (1 - p) / p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.expanduser("~/workspace/first-basket-model/dd-fit/raw"))
    ap.add_argument("--slate-json", default="data.json",
                    help="repo data.json with the slate DD list (falls back to the page HTML)")
    ap.add_argument("--mc", type=int, default=N_MC)
    args = ap.parse_args()

    if not os.path.isdir(args.raw_dir):
        print(f"No raw directory {args.raw_dir} — fetch logs first (see fetch_logs.py).")
        return 1
    players = load_logs(args.raw_dir)
    if not players:
        print(f"No game-log files in {args.raw_dir} — fetch logs first (see fetch_logs.py).")
        return 1

    print(f"Players with logs: {len(players)}")
    per = {}
    for slug, x in sorted(players.items()):
        n = len(x)
        mu = x.mean(axis=0)
        sd = x.std(axis=0, ddof=1)
        var = x.var(axis=0, ddof=1)
        C = corr_matrix(x)
        phi_r = var[1] / mu[1] if mu[1] > 0 else float("nan")
        phi_a = var[2] / mu[2] if mu[2] > 0 else float("nan")
        per[slug] = dict(n=n, mu=mu, sd=sd, C=C, phi_reb=phi_r, phi_ast=phi_a)
        print(f"\n{slug}: n={n}  mean P/R/A = {mu[0]:.2f}/{mu[1]:.2f}/{mu[2]:.2f}  "
              f"sd = {sd[0]:.2f}/{sd[1]:.2f}/{sd[2]:.2f}")
        print(f"  corr(P,R)={C[0,1]:+.3f}  corr(P,A)={C[0,2]:+.3f}  corr(R,A)={C[1,2]:+.3f}  "
              f"phi_reb={phi_r:.3f}  phi_ast={phi_a:.3f}")

    pairs = [(0, 1, "corr(P,R)"), (0, 2, "corr(P,A)"), (1, 2, "corr(R,A)")]
    pooled = {}
    print("\nPooled (Fisher-z, weighted by n-3):")
    for i, j, name in pairs:
        rs = [per[s]["C"][i, j] for s in per]
        ns = [per[s]["n"] for s in per]
        r = fisher_pool(rs, ns)
        pooled[(i, j)] = r
        print(f"  {name}: pooled {r:+.3f}  per-player range [{min(rs):+.3f}, {max(rs):+.3f}]")
    phi_reb_pool = float(np.average([per[s]["phi_reb"] for s in per],
                                   weights=[per[s]["n"] for s in per]))
    phi_ast_pool = float(np.average([per[s]["phi_ast"] for s in per],
                                   weights=[per[s]["n"] for s in per]))
    print(f"  phi_reb pooled: {phi_reb_pool:.3f}  range "
          f"[{min(per[s]['phi_reb'] for s in per):.3f}, {max(per[s]['phi_reb'] for s in per):.3f}]")
    print(f"  phi_ast pooled: {phi_ast_pool:.3f}  range "
          f"[{min(per[s]['phi_ast'] for s in per):.3f}, {max(per[s]['phi_ast'] for s in per):.3f}]")

    R = np.array([[1.0, pooled[(0, 1)], pooled[(0, 2)]],
                  [pooled[(0, 1)], 1.0, pooled[(1, 2)]],
                  [pooled[(0, 2)], pooled[(1, 2)], 1.0]])
    print("\nPooled correlation matrix R:")
    print(R)

    print("\nDouble-double before/after (current DD list, projections from page slate):")
    print(f"{'Player':28s} {'proj P/R/A':>16s} {'before':>8s} {'after':>8s} {'delta pp':>9s} {'fair before':>11s} {'fair after':>10s}")
    for d in page_dd_list(args.slate_json):
        b = dd_before(d)
        a, se = dd_copula(d, R, phi_reb_pool, phi_ast_pool, n_mc=args.mc)
        proj = f"{d['pts']}/{d['reb']}/{d['ast']}"
        print(f"{d['n']:28s} {proj:>16s} {b*100:7.1f}% {a*100:7.1f}% "
              f"{(a-b)*100:+8.1f} {fair(b):>11s} {fair(a):>10s}  (MC se {se*100:.2f}pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
