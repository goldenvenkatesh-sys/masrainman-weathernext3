# ============================================================
# EXTREME FORECAST INDEX (EFI) + SHIFT OF TAILS (SOT) — the maths
# ============================================================
#
# Same definitions as ECMWF (Lalaurette 2003, Zsoter 2006/2009):
#
#   EFI = 2/pi * integral_0^1  (p - F_f(q_c(p))) / sqrt(p (1 - p))  dp
#
#     F_f    forecast CDF (from ensemble members, or from forecast percentiles)
#     q_c(p) climate quantile for probability p
#     EFI is +1 when every member is above the climate maximum,
#     -1 when every member is below the climate minimum, 0 when the
#     forecast distribution equals the climate.
#
#   SOT (upper) = (Q_f(90) - Q_c(99)) / (Q_c(99) - Q_c(90))
#   SOT (lower) = (Q_c(1)  - Q_f(10)) / (Q_c(10) - Q_c(1))
#
#     SOT > 0 means at least 10 % of the forecast is beyond the climate
#     99th (1st) percentile; it says how far into the tail the event goes,
#     which the EFI (bounded at 1) cannot.
#
# Only numpy; every function works on whole grids at once.
# Arrays: forecast knots (K, ...) and climate quantiles (101, ...) where
# "..." is any grid shape (ny, nx).
# ============================================================

import numpy as np

CLIM_PROBS = np.linspace(0.0, 1.0, 101)          # climate stored at 0,1,...,100 %

# Mid-points of the 100 climate bins and the EFI weights 1/sqrt(p(1-p)).
_P_MID = (CLIM_PROBS[:-1] + CLIM_PROBS[1:]) / 2.0
_W = np.diff(CLIM_PROBS) / np.sqrt(_P_MID * (1.0 - _P_MID))
# Normalise so that F_f = 0 everywhere gives exactly +1 with this discretisation
# (the continuous integral of p / sqrt(p(1-p)) is pi/2).
_NORM = float(np.sum(_P_MID * _W))


def knots_from_percentiles(q, probs, lower_bound=None):
    """Turn forecast percentiles into a full CDF outline.

    q      (P, ...) forecast values at probabilities `probs` (e.g. 0.1 .. 0.9)
    probs  (P,) increasing probabilities, strictly inside (0, 1)
    Adds a 0 % and a 100 % knot by continuing the outermost slopes
    (uniform tails), clips at `lower_bound` (0 for rain) and makes the
    knots non-decreasing. Returns (knots (P+2, ...), knot_probs (P+2,)).
    """
    q = np.asarray(q, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float64)
    lo_slope = (q[1] - q[0]) / (probs[1] - probs[0])
    hi_slope = (q[-1] - q[-2]) / (probs[-1] - probs[-2])
    q0 = q[0] - lo_slope * probs[0]
    q1 = q[-1] + hi_slope * (1.0 - probs[-1])
    knots = np.concatenate([q0[None], q, q1[None]], axis=0)
    if lower_bound is not None:
        knots = np.maximum(knots, lower_bound)
    knots = np.maximum.accumulate(knots, axis=0)
    return knots, np.concatenate([[0.0], probs, [1.0]])


def cdf_from_knots(x, knots, knot_probs):
    """Piecewise-linear forecast CDF at values x.

    x      (M, ...) values (e.g. climate quantiles)
    knots  (K, ...) non-decreasing forecast values at knot_probs (0 .. 1)
    Uses the mid-point rule for ties, (P(X < x) + P(X <= x)) / 2, so a dry
    forecast in a dry climate (both all zero) gives F = 0.5 and EFI = 0.
    """
    x = np.asarray(x, dtype=np.float32)
    knots = np.asarray(knots, dtype=np.float32)
    f_le = np.zeros(x.shape, dtype=np.float32)
    f_lt = np.zeros(x.shape, dtype=np.float32)
    for k in range(len(knot_probs) - 1):
        a, b = knots[k][None], knots[k + 1][None]
        pa, pb = float(knot_probs[k]), float(knot_probs[k + 1])
        width = b - a
        frac = np.where(width > 0, (x - a) / np.where(width > 0, width, 1), 0)
        inside = pa + (pb - pa) * np.clip(frac, 0, 1)
        f_le = np.maximum(f_le, np.where(x >= b, pb, np.where(x >= a, inside, 0)))
        f_lt = np.maximum(f_lt, np.where(x > b, pb, np.where(x > a, inside, 0)))
    return 0.5 * (f_le + f_lt)


def cdf_from_members(x, members):
    """Empirical forecast CDF from ensemble members (N, ...) at values x (M, ...)."""
    x = np.asarray(x, dtype=np.float32)[:, None]
    m = np.asarray(members, dtype=np.float32)[None]
    return 0.5 * ((m <= x).mean(axis=1) + (m < x).mean(axis=1))


def efi_from_cdf(clim_q, cdf):
    """EFI from climate quantiles (101, ...) and a function x -> F_f(x)."""
    clim_q = np.asarray(clim_q, dtype=np.float32)
    q_mid = 0.5 * (clim_q[:-1] + clim_q[1:])                      # (100, ...)
    f = cdf(q_mid)
    shape = (-1,) + (1,) * (q_mid.ndim - 1)
    p = _P_MID.reshape(shape)
    w = _W.reshape(shape)
    efi = np.sum((p - f) * w, axis=0) / _NORM
    return np.clip(efi, -1.0, 1.0).astype(np.float32)


def efi_from_percentiles(clim_q, fc_q, fc_probs, lower_bound=None):
    """EFI when the forecast is known only through a few percentiles (WeatherNext 3)."""
    knots, kp = knots_from_percentiles(fc_q, fc_probs, lower_bound)
    return efi_from_cdf(clim_q, lambda x: cdf_from_knots(x, knots, kp))


def efi_from_members(clim_q, members):
    """EFI from ensemble members (N, ...) — e.g. ECMWF ENS, 51 members."""
    return efi_from_cdf(clim_q, lambda x: cdf_from_members(x, members))


def _ratio(num, den, min_den):
    den = np.asarray(den, dtype=np.float32)
    ok = den > min_den
    out = np.where(ok, num / np.where(ok, den, 1), np.nan)
    return out.astype(np.float32)


def sot_upper(clim_q, fc_q90, min_spread=1e-3):
    """Shift of Tails for the upper tail (heavy rain, heat). NaN where Q_c99 = Q_c90."""
    c90, c99 = clim_q[90], clim_q[99]
    return _ratio(np.asarray(fc_q90) - c99, c99 - c90, min_spread)


def sot_lower(clim_q, fc_q10, min_spread=1e-3):
    """Shift of Tails for the lower tail (cold nights)."""
    c1, c10 = clim_q[1], clim_q[10]
    return _ratio(c1 - np.asarray(fc_q10), c10 - c1, min_spread)


def quantile_at(knots, knot_probs, p):
    """Forecast value at probability p from CDF knots (inverse of cdf_from_knots)."""
    knots = np.asarray(knots)
    i = int(np.searchsorted(knot_probs, p, side="right") - 1)
    i = min(max(i, 0), len(knot_probs) - 2)
    pa, pb = knot_probs[i], knot_probs[i + 1]
    t = (p - pa) / (pb - pa)
    return knots[i] + t * (knots[i + 1] - knots[i])
