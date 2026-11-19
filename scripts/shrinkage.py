#!/usr/bin/env python3
"""Empirical-Bayes shrinkage + calibration for the First Basket Model
(improvement #2, Nov 2026).

The fixed K=20 history blend is a coincidence, not a tuned value: the
right shrinkage constant varies wildly by quantity (Darko: K~16 for tip
rates vs ~260 for 3P%). This module estimates it from the data
(method of moments on the Beta-binomial model) instead of guessing,
and fits a Platt (logit) calibration map on held-out data so
over-confident tails are pulled toward the base rate.

All functions are pure (no I/O, no network) so they are unit-testable
and importable from fit_calibration.py, evaluate.py and
update_results.py alike.

Beta-binomial setup for the history blend: player i has prior mean
m_i (his model share), g_i games started, y_i first baskets (weights
allowed, e.g. a putback counts 0.5). theta_i ~ Beta(mean m_i,
concentration K); y_i | theta_i ~ Binomial(g_i, theta_i). Then
    Var(y_i / g_i) = m_i (1 - m_i) / (K + 1) * (1 + K / g_i).
Matching the observed squared deviations to that expectation gives
    K = (A - S) / (S - B),  A = sum m_i (1 - m_i),
    B = sum m_i (1 - m_i) / g_i,  S = sum (y_i / g_i - m_i)^2.
No overdispersion (S -> B) sends K -> infinity (clamped); more
dispersion than the model allows (S >= A) returns None, meaning the
data cannot identify K and the caller should keep its default.
"""
import math

K_MIN = 1
K_MAX = 500


def eb_k_mom(priors, counts, trials):
    """Method-of-moments shrinkage constant K for Beta-binomial data.

    priors/counts/trials are parallel sequences (m_i, y_i, g_i).
    Rows with g_i == 0 carry no information and are skipped. Returns
    a float in [K_MIN, K_MAX], or None when the data cannot identify
    K (S >= A: more dispersion than any K allows; or too few rows).
    """
    a = b = s = 0.0
    n = 0
    for m, y, g in zip(priors, counts, trials):
        if g is None or g <= 0 or m is None:
            continue
        m = min(max(float(m), 1e-6), 1 - 1e-6)
        v = m * (1 - m)
        a += v
        b += v / g
        s += (y / g - m) ** 2
        n += 1
    if n < 10 or a <= 0:
        return None
    if s <= b:
        return float(K_MAX)  # no overdispersion: shrink hard to the prior
    if s >= a:
        return None  # more dispersion than the model allows
    k = (a - s) / (s - b)
    return min(max(k, float(K_MIN)), float(K_MAX))


def eb_shrink(y, n, prior_mean, k):
    """Beta-binomial posterior mean: (y + K * prior) / (n + K)."""
    if k is None:
        k = 20.0
    return (y + k * prior_mean) / (n + k)


def _logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def platt_fit(probs, outcomes, iters=50):
    """Fit a Platt (logit) calibration map: calibrated = sigmoid(a +
    b * logit(p)). IRLS on the intercept + slope, with a tiny ridge
    for stability. probs/outcomes are parallel sequences (p in (0,1),
    outcome 0/1). Returns (a, b) or None when the fit is not
    identifiable (< 30 rows, or all outcomes the same)."""
    rows = [(min(max(float(p), 1e-6), 1 - 1e-6), int(o))
            for p, o in zip(probs, outcomes)]
    rows = [r for r in rows if r[0] > 0 and r[0] < 1]
    if len(rows) < 30 or len({o for _, o in rows}) < 2:
        return None
    a, b = 0.0, 1.0
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for p, o in rows:
            x = _logit(p)
            q = _sigmoid(a + b * x)
            d = q - o
            w = max(q * (1 - q), 1e-9)
            g0 += d
            g1 += d * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        # Ridge for stability on small samples.
        h00 += 1e-6
        h11 += 1e-6
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        a -= da
        b -= db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    return (a, b)


def platt_apply(p, params):
    """Apply a fitted (a, b) Platt map to a probability. params None
    -> identity (no calibration)."""
    if params is None:
        return p
    a, b = params
    return _sigmoid(a + b * _logit(p))


def logloss(p):
    """Multi-class log loss for one assigned probability (floored)."""
    return -math.log(min(max(p, 1e-9), 1 - 1e-9))


def brier_score(probs, outcome_idx):
    """Multi-class Brier score: sum (p_i - o_i)^2 over the outcome set."""
    return sum((p - (1.0 if i == outcome_idx else 0.0)) ** 2
               for i, p in enumerate(probs))


def murphy_decomposition(games):
    """Murphy decomposition of the multi-class Brier score over a list
    of games. Each game is (probs, outcome_idx). Returns dict with
    brier, reliability, resolution, uncertainty. Reliability is the
    calibration term (lower better); resolution is discrimination
    (higher better); uncertainty is the irreducible base-rate term.
    Computed per outcome slot vs the empirical base rate of that slot
    (slots are sorted by predicted probability desc, so slot 0 is the
    top pick)."""
    n = len(games)
    if n == 0:
        return {"brier": 0.0, "reliability": 0.0, "resolution": 0.0,
                "uncertainty": 0.0, "n": 0}
    k = len(games[0][0])
    base = [sum(1.0 for probs, idx in games if idx == j) / n
            for j in range(k)]
    brier = sum(brier_score(p, idx) for p, idx in games) / n
    # Reliability/resolution via 10 equal-width forecast bins per slot.
    reliability = 0.0
    resolution = 0.0
    for j in range(k):
        bins = {}
        for probs, idx in games:
            b = min(int(probs[j] * 10), 9)
            bins.setdefault(b, [0, 0.0, 0])
            bins[b][0] += 1
            bins[b][1] += probs[j]
            bins[b][2] += 1 if idx == j else 0
        for cnt, psum, hits in bins.values():
            fbar = psum / cnt
            obar = hits / cnt
            reliability += cnt * (fbar - obar) ** 2
            resolution += cnt * (obar - base[j]) ** 2
    reliability /= n
    resolution /= n
    uncertainty = sum(b * (1 - b) for b in base)
    return {"brier": brier, "reliability": reliability,
            "resolution": resolution, "uncertainty": uncertainty, "n": n}


def ece(probs, outcomes, buckets=((0.0, 0.05), (0.05, 0.10),
                                  (0.10, 0.15), (0.15, 0.25),
                                  (0.25, 1.01))):
    """Expected Calibration Error over low-probability buckets (most
    first-scorer forecasts live in 3-25%). probs/outcomes are parallel
    flat sequences (one row per starter per game). Returns (ece, table)
    where table is [(bucket, n, mean_forecast, observed_freq)]."""
    table = []
    total = 0.0
    n_all = 0
    for lo, hi in buckets:
        sel = [(p, o) for p, o in zip(probs, outcomes)
               if lo <= p < hi]
        if not sel:
            table.append(((lo, hi), 0, None, None))
            continue
        mf = sum(p for p, _ in sel) / len(sel)
        of = sum(o for _, o in sel) / len(sel)
        table.append(((lo, hi), len(sel), mf, of))
        total += len(sel) * abs(mf - of)
        n_all += len(sel)
    return (total / n_all if n_all else 0.0), table
