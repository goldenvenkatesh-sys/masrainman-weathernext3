# ============================================================
# WEIGHTED MULTI-MODEL BLEND
# ============================================================

import logging

import numpy as np

from . import config as C
from .fetch import daily_totals

LOG = logging.getLogger("mme")


def load_group(group, init):
    """Fetch every model of a group. Returns {model: (days, run)} for models that worked."""
    out = {}
    for model in C.GROUPS[group]["models"]:
        try:
            days, run = daily_totals(model, init)
        except Exception as exc:  # one broken source must not stop the others
            LOG.exception("%s failed: %s", model, exc)
            continue
        if days is not None:
            out[model] = (days, run)
    return out


def blend(group, members):
    """Weighted mean per day, weights renormalised over models that reach that day.

    Returns (days[n_days, ny, nx] with NaN for days without enough models,
             used[n_days] = list of (model, weight) actually used).
    """
    weights = C.GROUPS[group]["models"]
    shape = next(iter(members.values()))[0].shape
    out = np.full(shape, np.nan, np.float32)
    used = []
    for k in range(shape[0]):
        avail = [m for m, (d, _) in members.items() if np.isfinite(d[k]).mean() > 0.95]
        need = C.MIN_MODELS_LATE if k + 1 >= C.LATE_DAY else C.MIN_MODELS_PER_DAY
        if len(avail) < need:
            used.append([])
            continue
        wsum = sum(weights[m] for m in avail)
        acc = np.zeros(shape[1:], np.float32)
        for m in avail:
            acc += np.nan_to_num(members[m][0][k]) * (weights[m] / wsum)
        out[k] = acc
        used.append([(m, round(weights[m] / wsum, 3)) for m in avail])
    return out, used


def totals(days, n):
    """Sum of the first n days, or None if any of them is missing."""
    part = days[:n]
    if not np.isfinite(part[:, 0, 0]).all():
        return None
    return part.sum(0)
