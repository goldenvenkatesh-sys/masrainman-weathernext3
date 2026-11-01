# ============================================================
# WeatherNext 3 (statistics mean) — rain + MSLP for the blends
# ============================================================
#
# Uses render.open_run_zarr from the masrainman-weathernext3 repo (one folder up),
# so the same Google login as the WeatherNext 3 maps is used.
#   total_precipitation_1hr_mean : m, index i = rain in the hour ending T+(i+1)
#   mean_sea_level_pressure_mean : Pa, index i = T+(i+1)

import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .grib import RAIN, TRACK
from . import config as C

RAIN_VAR = os.environ.get("WN3_RAIN_VAR", "total_precipitation_1hr_mean")
MSL_VAR = "mean_sea_level_pressure_mean"


def _root(init):
    repo = os.path.dirname(C.HERE)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import render  # noqa: E402  (masrainman-weathernext3/render.py)
    return render.open_run_zarr(f"{init:%Y%m%d}_{init:%H}hr_01_preds")


def _window(root, grid):
    lat = np.asarray(root["lat_0p1"][:])
    lon = (np.asarray(root["lon_0p1"][:]) + 180) % 360 - 180
    li = np.where((lat >= grid.lat[0] - 0.5) & (lat <= grid.lat[-1] + 0.5))[0]
    lo = np.where((lon >= grid.lon[0] - 0.5) & (lon <= grid.lon[-1] + 0.5))[0]
    sl = (slice(int(li[0]), int(li[-1]) + 1), slice(int(lo[0]), int(lo[-1]) + 1))
    return sl, lat[sl[0]], lon[sl[1]]


def _to_grid(v, lat, lon, grid):
    if lat[0] > lat[-1]:
        lat, v = lat[::-1], v[::-1]
    f = RegularGridInterpolator((lat, lon), v, bounds_error=False, fill_value=np.nan)
    return f((grid.LAT2, grid.LON2)).astype(np.float32)


def wn3_accum(init, leads):
    """{lead: accumulated rain mm since init} on the rain grid."""
    from .fetch import NotAvailable
    root = _root(init)
    tp = root[RAIN_VAR]
    n = int(tp.shape[0])
    sl, lat, lon = _window(root, RAIN)
    out, total, done = {}, None, 0
    for L in sorted(leads):
        if L > n:
            break
        chunk = np.asarray(tp[done:L, sl[0], sl[1]], dtype=np.float32) * 1000.0
        if np.mean(np.isfinite(chunk)) < 0.99:  # run not complete that far
            break
        part = np.maximum(np.nan_to_num(chunk), 0).sum(axis=0)
        total = part if total is None else total + part
        done = L
        out[L] = _to_grid(total, lat, lon, RAIN)
    if len(out) < 2:
        raise NotAvailable("WeatherNext 3 rain: run not complete")
    return out


def wn3_track_fields(init):
    """{0: {"msl": [n_steps, ny, nx] hPa}} on the tracking grid (ensemble mean = one member)."""
    root = _root(init)
    msl = root[MSL_VAR]
    n = int(msl.shape[0])
    sl, lat, lon = _window(root, TRACK)
    arr = np.full((len(C.TRACK_STEPS),) + TRACK.LAT2.shape, np.nan, np.float32)
    for k, L in enumerate(C.TRACK_STEPS):
        i = max(L - 1, 0)
        if i >= n:
            break
        v = np.asarray(msl[i, sl[0], sl[1]], dtype=np.float32) / 100.0
        if np.isfinite(v).mean() > 0.99:
            arr[k] = _to_grid(v, lat, lon, TRACK)
    return {0: {"msl": arr}}
