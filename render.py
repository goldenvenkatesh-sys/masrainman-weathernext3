# ============================================================
# MASRAINMAN — V3 SURFACE SYSTEM TRACKING — AUTO RENDERER
#   WeatherNext 3  +  ECMWF IFS (HRES)  +  ECMWF AIFS Single
# ============================================================
#
# GitHub Actions version of the MasRainman Colab script.
#
# Every time it runs, for each model it:
#   1. Lists the 00/06/12/18 UTC runs
#   2. Picks the newest runs that are not yet on the website
#   3. Checks the run is complete
#        WeatherNext 3 : all 360 h written
#        ECMWF IFS/AIFS: last step on data.ecmwf.int
#                        (00/12Z -> T+360, 06/18Z -> T+144)
#   4. Loads regional MSLP / U10 / V10 (55–100E, 2–28N)
#   5. Runs V3 surface-system detection + persistent tracking
#      (same logic for all three models)
#   6. Draws T+24, T+48 … maps, with and without 10-m wind barbs
#      (same design as the Colab map)
#   7. Builds an animated GIF for each layer
#   8. Writes CSVs + systems JSON + a manifest per model
#   9. Keeps only the latest KEEP_RUNS runs per model
#
# Website layout (WeatherNext 3 paths are unchanged):
#   site/runs/<YYYYMMDD_HHhr_01_preds>/   site/data/manifest.json
#   site/ecmwf/ifs/<YYYYMMDD_HH>/         site/data/manifest_ifs.json
#   site/ecmwf/aifs/<YYYYMMDD_HH>/        site/data/manifest_aifs.json
#   site/data/models.json                 list of models for the page
#
# Usage:
#   python render.py                              # all models, latest runs
#   python render.py --models ifs,aifs            # only some models
#   python render.py --model wn3 --run 20261019_00hr_01_preds   # force one run
#   python render.py --model ifs --run 20261019_12               # force one run
#   python render.py --demo                       # synthetic test, all models
#
# ============================================================

import os
import re
import sys
import json
import shutil
import argparse
import tempfile
import warnings

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import minimum_filter

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from shapely.geometry import shape

from PIL import Image

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# SETTINGS
# ============================================================

STATS_BUCKET = "weathernext3_statistics_spatial"
STATS_PREFIX = "weathernext_3_0_0_statistics/zarr/"
RUN_ROOT_PREFIX = "weathernext_3_0_0_statistics/zarr/2026_to_present/"

RUN_PATTERN = r"^(\d{8})_(00|06|12|18)hr_01_preds$"
ECMWF_RUN_PATTERN = r"^(\d{8})_(00|06|12|18)$"

# Website output folder
SITE_DIR = os.environ.get("SITE_DIR", os.path.join(HERE, "site"))

# How many runs stay selectable on the website (per model)
KEEP_RUNS = int(os.environ.get("KEEP_RUNS", "4"))

# Max new runs rendered in one job, per model (protects job time)
MAX_NEW_RUNS_PER_JOB = int(os.environ.get("MAX_NEW_RUNS", "2"))

# Frames: every 24 h until the end of the forecast
FRAME_STEP_H = 24
FRAME_END_H = 360

# GIF settings
GIF_WIDTH_PX = 1400
GIF_FRAME_MS = 900

# Map domain
LAT_MIN, LAT_MAX = 2.0, 28.0
LON_MIN, LON_MAX = 55.0, 100.0

# Rain-chart domain (wider, like the usual South Asia rain charts)
RAIN_LAT_MIN, RAIN_LAT_MAX = 0.0, 42.0
RAIN_LON_MIN, RAIN_LON_MAX = 50.0, 110.0

# WeatherNext 3 rain field: hourly total precipitation, ensemble mean (m).
# Others in the store: _p10.._p90, imerg_tp_1hr_*, experimental_tp_1hr_*
WN3_RAIN_VAR = os.environ.get("WN3_RAIN_VAR", "total_precipitation_1hr_mean")

# Rain colour scale (mm) — grey / green / blue / yellow-red / brown / purple
RAIN_LEVELS = [0.2, 0.6, 1.2, 1.8, 2.5, 3.75, 5, 6.25, 7.5, 10, 12.5, 15, 17.5, 20, 22.5,
               26, 30, 35, 40, 45, 50, 62.5, 75, 87.5, 100, 125, 150, 175, 200, 225, 250,
               275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
RAIN_TICKS = [0.2, 1.2, 2.5, 5, 7.5, 12.5, 17.5, 22.5, 30, 40, 50, 75, 100, 150, 200,
              250, 300, 350, 400, 450, 500]
RAIN_COLOURS = [  # one colour per band above (40 bands); above 500 = RAIN_OVER
    "#d0d0d0", "#bdbdbd", "#a3a3a3", "#8c8c8c", "#6f6f6f",                       # 0.2 – 3.75 grey
    "#c4f7b8", "#a2ef8f", "#7fe36a", "#5ad146", "#33b52b", "#12951c",            # 3.75 – 15 green
    "#0b3fb8", "#2266dd", "#4a95ee", "#8fc8f7",                                  # 15 – 26 blue
    "#fff3a3", "#ffde5c", "#ffc430", "#ffa51f", "#ff8a16", "#ff6a10",            # 26 – 62.5 yellow-orange
    "#f2450c", "#df1a0a", "#c40606", "#a10000", "#7c0b05",                       # 62.5 – 150 red
    "#5e2a1e", "#7d4a3c", "#a07264", "#c49d91", "#e6cfc8",                       # 150 – 275 brown
    "#ddd0ee", "#c4b3e6", "#a991d8", "#8e6fc8", "#7650b5", "#6630a0",            # 275 – 425 lavender-purple
    "#8020a8", "#a01ab8", "#c626cf",                                             # 425 – 500 magenta
]
RAIN_OVER = "#e83ce8"

# India national boundary as per Government of India
# (land borders only, from Natural Earth "India point-of-view" countries)
BOUNDARY_FILE = os.path.join(HERE, "boundaries", "borders_india_view.geojson")

# ECMWF open data
ECMWF_ROOT = os.environ.get("ECMWF_ROOT", "https://data.ecmwf.int/forecasts")
ECMWF_STEP_H = 6            # 6-hourly fields for tracking (both models have these)
# Grid resolutions to try, best first. ECMWF added free 0.1° (~9 km) IFS data
# in Oct 2026 (folder .../ifs/0p1/oper/, files *-oper-fc-sfc.grib2). The 0.1°
# files appear about 2 h after the 0.25° ones; a run first drawn at 0.25° is
# drawn again at 0.1° once that is fully published.
ECMWF_RESOL_DEG = {"0p1": 0.1, "0p25": 0.25}
# File-name endings: new style (separate sfc / pl files) and the old single file
ECMWF_FILE_KINDS = ["oper-fc-sfc", "oper-fc"]
# Which ECMWF runs to draw. "00,12" = only the 15-day runs (saves GitHub minutes)
ECMWF_CYCLES = [int(c) for c in os.environ.get("ECMWF_CYCLES", "00,06,12,18").split(",") if c.strip()]
HTTP_TIMEOUT = (10, 120)    # connect, read seconds


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "wn3": {
        "label": "WeatherNext 3",
        "file_prefix": "weathernext3",
        "runs_dir": "runs",                       # unchanged
        "manifest": "manifest.json",              # unchanged
        "pattern": RUN_PATTERN,
        "title": "WeatherNext 3 — Mean MSLP with V3 Low Centres",
        "info_head": "WeatherNext 3 Statistics Mean",
        "rain_label": "WeatherNext 3 Mean",
        "barbs_label": "Mean 10-m wind barbs: ON",
        "credits": [
            "WeatherNext 3 © 2026 DeepMind Technologies Limited | Experimental model output",
            "Historical WeatherNext 3 data: CC BY 4.0 | "
            "Future/real-time data: applicable GDM experimental-data terms | "
            "V3 analysis & visualisation: MasRainman",
            "WeatherNext 3 predictions are experimental and are not official weather warnings or forecasts.",
        ],
    },
    "ifs": {
        "label": "ECMWF IFS",
        "file_prefix": "ecmwf_ifs",
        "runs_dir": os.path.join("ecmwf", "ifs"),
        "manifest": "manifest_ifs.json",
        "pattern": ECMWF_RUN_PATTERN,
        "ecmwf_model": "ifs",
        "resols": [r for r in os.environ.get("IFS_RESOLS", "0p1,0p25").split(",") if r.strip()],
        "title": "ECMWF IFS HRES — MSLP with V3 Low Centres",
        "info_head": "ECMWF IFS (operational deterministic, {grid})",
        "barbs_label": "10-m wind barbs: ON",
        "credits": [
            "Source: ECMWF open data © European Centre for Medium-Range Weather Forecasts | CC BY 4.0",
            "V3 analysis & visualisation: MasRainman",
            "This is not an official forecast. Please follow IMD for official forecasts and warnings.",
        ],
    },
    "aifs": {
        "label": "ECMWF AIFS",
        "file_prefix": "ecmwf_aifs",
        "runs_dir": os.path.join("ecmwf", "aifs"),
        "manifest": "manifest_aifs.json",
        "pattern": ECMWF_RUN_PATTERN,
        "ecmwf_model": "aifs-single",
        "resols": ["0p25"],
        "title": "ECMWF AIFS — MSLP with V3 Low Centres",
        "info_head": "ECMWF AIFS Single (AI model, {grid})",
        "barbs_label": "10-m wind barbs: ON",
        "credits": [
            "Source: ECMWF open data © European Centre for Medium-Range Weather Forecasts | CC BY 4.0",
            "V3 analysis & visualisation: MasRainman",
            "This is not an official forecast. Please follow IMD for official forecasts and warnings.",
        ],
    },
}
MODEL_ORDER = ["wn3", "ifs", "aifs"]


# ============================================================
# V3 DETECTION PARAMETERS (unchanged from the Colab script)
# ============================================================

MINIMUM_FILTER_SIZE = 21          # grid points on the 0.1° WeatherNext grid (~2.1°)
BOUNDARY_MARGIN_DEG = 2.0
PRESSURE_RING_INNER_DEG = 2.0
PRESSURE_RING_OUTER_DEG = 4.0
CIRC_RING_INNER_DEG = 0.8
CIRC_RING_OUTER_DEG = 2.0
MAX_MOVEMENT_KM_PER_HOUR = 125.0
MIN_PRESSURE_DEPRESSION_HPA = 0.8
CIRCULATION_FLOOR = 0.55
MIN_TRACK_HOURS = 6
MAX_CANDIDATES_PER_HOUR = 20

# For 3/6-hourly models the per-step movement limit is 125 km/h × step,
# capped so a 6-hourly step cannot jump more than this:
MAX_MOVEMENT_KM_PER_STEP_CAP = 500.0


# ============================================================
# GOOGLE CLOUD ACCESS (WeatherNext 3)
# ============================================================

def get_session():
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    print(f"Google project: {project}")
    return AuthorizedSession(credentials)


def list_weathernext_runs(session):
    print("\nSearching WeatherNext 3 runs...")

    runs = []
    url = f"https://storage.googleapis.com/storage/v1/b/{STATS_BUCKET}/o"
    params = {"prefix": RUN_ROOT_PREFIX, "delimiter": "/", "maxResults": 1000}

    while True:
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        for prefix in data.get("prefixes", []):
            run_name = prefix.rstrip("/").split("/")[-1]
            if re.match(RUN_PATTERN, run_name):
                runs.append(run_name)

        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token

    return sorted(set(runs), reverse=True)


def open_run_zarr(run_name):
    import zarr
    import obstore

    prefix = f"{STATS_PREFIX}2026_to_present/{run_name}/predictions.zarr"

    store = None
    try:
        # Uses the same Google credentials as google.auth.default()
        from obstore.auth.google import GoogleCredentialProvider
        store = obstore.store.GCSStore(
            bucket=STATS_BUCKET,
            prefix=prefix,
            credential_provider=GoogleCredentialProvider(),
        )
    except Exception as exc:
        print(f"(credential provider not used: {exc})")

    if store is None:
        key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        config = {"google_service_account_path": key_path} if key_path else {}
        store = obstore.store.GCSStore(bucket=STATS_BUCKET, prefix=prefix, config=config)

    zstore = zarr.storage.ObjectStore(store, read_only=True)
    return zarr.open_group(zstore, mode="r", zarr_format=3)


def decode_init_time(run_name):
    m = re.match(RUN_PATTERN, run_name) or re.match(ECMWF_RUN_PATTERN, run_name)
    if not m:
        raise RuntimeError(f"Unable to decode run time: {run_name}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)


# ============================================================
# DATA LOADING — WeatherNext 3 (unchanged)
# ============================================================

def load_run(run_name):
    """Return dict with lat/lon/mslp/u10/v10 for the region, or None if incomplete."""

    print(f"\nOpening WeatherNext 3 Zarr for {run_name} ...")
    root = open_run_zarr(run_name)

    mslp_arr = root["mean_sea_level_pressure_mean"]
    u10_arr = root["u_component_of_wind_10m_mean"]
    v10_arr = root["v_component_of_wind_10m_mean"]

    lat_full = np.asarray(root["lat_0p1"][:])
    lon_full = np.asarray(root["lon_0p1"][:])

    lat_ids = np.where((lat_full >= LAT_MIN) & (lat_full <= LAT_MAX))[0]
    lon_ids = np.where((lon_full >= LON_MIN) & (lon_full <= LON_MAX))[0]
    if len(lat_ids) == 0 or len(lon_ids) == 0:
        raise RuntimeError("Requested map domain is outside available coordinates.")

    lat_start, lat_end = int(lat_ids[0]), int(lat_ids[-1]) + 1
    lon_start, lon_end = int(lon_ids[0]), int(lon_ids[-1]) + 1

    n_hours = min(int(mslp_arr.shape[0]), int(u10_arr.shape[0]), int(v10_arr.shape[0]))

    # ---- Completeness check: last forecast hour must hold real values
    if n_hours < FRAME_END_H:
        print(f"Run has only {n_hours} hours — skipping for now.")
        return None

    lat_mid = (lat_start + lat_end) // 2
    lon_mid = (lon_start + lon_end) // 2
    probe = float(np.asarray(mslp_arr[n_hours - 1, lat_mid, lon_mid]))
    if not np.isfinite(probe) or probe < 80000 or probe > 110000:
        print(f"Run not complete yet (last-hour probe = {probe}) — skipping for now.")
        return None

    print("Loading MSLP / U10 / V10 for the region ...")
    sl = (slice(0, n_hours), slice(lat_start, lat_end), slice(lon_start, lon_end))

    mslp = np.asarray(mslp_arr[sl], dtype=np.float32) / 100.0  # Pa -> hPa
    u10 = np.asarray(u10_arr[sl], dtype=np.float32)
    v10 = np.asarray(v10_arr[sl], dtype=np.float32)

    print(f"Grid {mslp.shape}, MSLP {np.nanmin(mslp):.1f}–{np.nanmax(mslp):.1f} hPa")

    data = {
        "lat": lat_full[lat_start:lat_end],
        "lon": lon_full[lon_start:lon_end],
        "mslp": mslp,
        "u10": u10,
        "v10": v10,
        "n_hours": n_hours,
        "hours": list(range(1, n_hours + 1)),   # array index i = T+(i+1) h
        "grid_deg": 0.1,
    }

    # Rain: hourly ensemble-mean precipitation (m) -> running total since T+0 (mm)
    try:
        tp_arr = root[WN3_RAIN_VAR]
        rlat = np.where((lat_full >= RAIN_LAT_MIN) & (lat_full <= RAIN_LAT_MAX))[0]
        rlon = np.where((lon_full >= RAIN_LON_MIN) & (lon_full <= RAIN_LON_MAX))[0]
        rsl = (slice(0, n_hours), slice(int(rlat[0]), int(rlat[-1]) + 1),
               slice(int(rlon[0]), int(rlon[-1]) + 1))
        print(f"Loading rain ({WN3_RAIN_VAR}) for the rain-chart area ...")
        hourly = np.nan_to_num(np.asarray(tp_arr[rsl], dtype=np.float32)) * 1000.0  # m -> mm
        data["rain"] = np.cumsum(np.maximum(hourly, 0), axis=0, dtype=np.float32)
        data["rlat"] = lat_full[rsl[1]]
        data["rlon"] = lon_full[rsl[2]]
        del hourly
        print(f"Rain grid {data['rain'].shape}, max total {np.nanmax(data['rain'][-1]):.0f} mm")
    except Exception as exc:   # never lose the MSLP maps because of rain
        print(f"  (rain not loaded: {type(exc).__name__}: {exc} - rain charts skipped)")

    return data


# ============================================================
# DATA LOADING — ECMWF IFS / AIFS (open data)
# ============================================================

def _http():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "MasRainman-V3-renderer"
    return s


def ecmwf_run_url(key, run_time, resol="0p25"):
    cfg = MODELS[key]
    ymd = run_time.strftime("%Y%m%d")
    hh = run_time.hour
    # All four IFS/AIFS runs are published under stream "oper"
    return f"{ECMWF_ROOT}/{ymd}/{hh:02d}z/{cfg['ecmwf_model']}/{resol}/oper", f"{ymd}{hh:02d}0000"


def grid_label(deg):
    """0.1 -> '0.1° ~9 km', 0.25 -> '0.25° ~28 km' (ECMWF's own figures)."""
    km = {0.1: 9, 0.25: 28}.get(round(deg, 3), round(deg * 111))
    return f"{deg:g}° ~{km} km"


def ecmwf_best_source(key, run_time, http):
    """Best fully published version of a run, or None.

    Returns {"resol": "0p1", "step": 360, "kind": "oper-fc-sfc"}.
    A run counts as complete once its last-step .index file exists.
    00/12Z need T+360; 06/18Z accept T+144 (or T+360).
    The longest forecast wins; for equal length the finer grid wins.
    """
    wanted = [360] if run_time.hour in (0, 12) else [360, 144]
    for step in wanted:                       # longest first
        for resol in MODELS[key]["resols"]:   # finest first
            base, stamp = ecmwf_run_url(key, run_time, resol)
            for kind in ECMWF_FILE_KINDS:
                try:
                    r = http.head(f"{base}/{stamp}-{step}h-{kind}.index",
                                  timeout=(5, 15), allow_redirects=True)
                except Exception as exc:
                    print(f"  ({key} {stamp} {resol} check failed: {type(exc).__name__})")
                    continue
                if r.status_code == 200:
                    return {"resol": resol, "step": step, "kind": kind}
    return None


def ecmwf_last_step(key, run_time, http):
    """Kept for older callers: the final step of the best source, else None."""
    src = ecmwf_best_source(key, run_time, http)
    return src["step"] if src else None


def source_rank(key, step, resol):
    """Bigger is better: longer forecast first, then finer grid."""
    order = MODELS[key]["resols"]
    fineness = len(order) - order.index(resol) if resol in order else 0
    return (int(step), fineness)


def rendered_rank(model, run_name):
    """Rank of a run already on the site (old runs without 'resol' were 0.25°)."""
    path = os.path.join(runs_root(model), run_name, "run.json")
    try:
        with open(path) as f:
            info = json.load(f)
    except Exception:
        return None
    return source_rank(model, info.get("last_hour", 0), info.get("resol", "0p25"))


def list_ecmwf_runs(key):
    """Newest complete runs first: [(run_name, source), ...]."""
    print(f"\nSearching {MODELS[key]['label']} runs...")
    http = _http()
    t = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 6)
    found = []
    for _ in range(16):
        if t.hour not in ECMWF_CYCLES:
            t -= timedelta(hours=6)
            continue
        src = ecmwf_best_source(key, t, http)
        if src:
            found.append((t.strftime("%Y%m%d_%H"), src))
            if len(found) == KEEP_RUNS:
                break
        t -= timedelta(hours=6)
    return found


def _download_step(http, base, stamp, step, kind, params, out_path, optional=()):
    """Fetch only the wanted GRIB messages of one step, using the .index byte ranges."""
    idx_url = f"{base}/{stamp}-{step}h-{kind}.index"
    grib_url = f"{base}/{stamp}-{step}h-{kind}.grib2"
    r = http.get(idx_url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    parts = []
    for line in r.text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("param") in set(params) | set(optional) and rec.get("levtype", "sfc") == "sfc":
            parts.append((int(rec["_offset"]), int(rec["_length"]), rec["param"]))
    got = {p for _, _, p in parts}
    missing = set(params) - got
    if missing:
        raise RuntimeError(f"step {step}: {', '.join(sorted(missing))} not in index")
    with open(out_path, "wb") as f:
        for off, length, _ in sorted(parts):
            for attempt in range(3):
                try:
                    rr = http.get(grib_url, timeout=HTTP_TIMEOUT,
                                  headers={"Range": f"bytes={off}-{off + length - 1}"})
                    rr.raise_for_status()
                    if len(rr.content) != length:
                        raise RuntimeError(f"short read {len(rr.content)}/{length}")
                    break
                except Exception:
                    if attempt == 2:
                        raise
            f.write(rr.content)
    return out_path


def _cut(da, lat0, lat1, lon0, lon1):
    lon = da["longitude"].values
    if lon.max() > 180 and lon0 < 0:
        da = da.assign_coords(longitude=((lon + 180) % 360) - 180).sortby("longitude")
    da = da.sortby("latitude")                       # ascending like WeatherNext
    return da.sel(latitude=slice(lat0, lat1), longitude=slice(lon0, lon1))


def _read_step_domain(path):
    """Read msl/10u/10v (+tp if present) of one step file, cut to the map domains
    right away (a global 0.1° field is 26 MB, the domain cut is under 1 MB)."""
    import xarray as xr

    out = {}
    for short_name in ("msl", "10u", "10v", "tp"):
        try:
            ds = xr.open_dataset(path, engine="cfgrib",
                                 backend_kwargs={"filter_by_keys": {"shortName": short_name},
                                                 "indexpath": "",
                                                 "read_keys": ["startStep", "endStep"]})
        except Exception:
            if short_name == "tp":
                continue
            raise
        if not ds.data_vars:
            ds.close()
            if short_name == "tp":
                continue
            raise RuntimeError(f"{short_name} missing in {os.path.basename(path)}")
        da = ds[list(ds.data_vars)[0]]
        if short_name == "tp":
            da = _cut(da, RAIN_LAT_MIN, RAIN_LAT_MAX, RAIN_LON_MIN, RAIN_LON_MAX)
            out["tp"] = da.values.astype(np.float32) * 1000.0       # m -> mm
            out["tp_start"] = int(da.attrs.get("GRIB_startStep", 0))   # 0 = since T+0
            out["rlat"] = da["latitude"].values
            out["rlon"] = da["longitude"].values
        else:
            da = _cut(da, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            out[short_name] = da.values.astype(np.float32)
            out["lat"] = da["latitude"].values
            out["lon"] = da["longitude"].values
        ds.close()
    return out


def rain_since_start(hours, fields, starts):
    """Accumulated rain since T+0 for each hour.

    ECMWF says in each GRIB message what period tp covers (startStep-endStep).
    startStep 0 everywhere -> already a running total since T+0 (normal case).
    Otherwise each field covers its own period and the periods are added up.
    (Do not guess from the values: packing noise makes totals dip slightly.)"""
    acc = np.stack(fields).astype(np.float32)
    since_start = all(st == 0 for h, st in zip(hours, starts) if h > 0)
    if not since_start:
        print(f"  tp covers separate periods (start steps {sorted(set(starts))}) -> adding up")
        acc = np.cumsum(acc, axis=0)
    # running totals can only grow; packing noise can dip them a little
    acc = np.maximum.accumulate(np.maximum(acc, 0), axis=0)
    return acc


def load_ecmwf_run(key, run_name, source):
    cfg = MODELS[key]
    if isinstance(source, int):                       # old style: just a step
        source = {"resol": "0p25", "step": source, "kind": "oper-fc"}
    resol, last_step, kind = source["resol"], source["step"], source["kind"]
    run_time = decode_init_time(run_name)
    base, stamp = ecmwf_run_url(key, run_time, resol)
    steps = list(range(0, last_step + 1, ECMWF_STEP_H))
    params = ["msl", "10u", "10v"]
    grid = ECMWF_RESOL_DEG.get(resol, 0.25)

    print(f"\nDownloading {cfg['label']} {run_name} at {grid_label(grid)}: "
          f"{len(steps)} steps (T+0 … T+{last_step}) ...")
    tmp = tempfile.mkdtemp(prefix=f"{key}_{run_name}_")
    try:
        http = _http()

        def fetch_and_cut(s):
            p = _download_step(http, base, stamp, s, kind, params,
                               os.path.join(tmp, f"s{s:03d}.grib2"), optional=("tp",))
            try:
                return s, _read_step_domain(p)
            finally:
                os.remove(p)                          # keep disk use small

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = dict(pool.map(fetch_and_cut, steps))

        hours = sorted(results)
        first = results[hours[0]]
        for h in hours:
            r = results[h]
            if r["msl"].shape != first["msl"].shape:
                raise RuntimeError(f"step {h}: grid changed {r['msl'].shape} vs {first['msl'].shape}")

        mslp = np.stack([results[h]["msl"] for h in hours]) / 100.0   # Pa -> hPa
        data = {
            "lat": first["lat"],
            "lon": first["lon"],
            "mslp": mslp.astype(np.float32),
            "u10": np.stack([results[h]["10u"] for h in hours]),
            "v10": np.stack([results[h]["10v"] for h in hours]),
            "n_hours": int(hours[-1]),
            "hours": [int(h) for h in hours],
            "grid_deg": grid,
            "resol": resol,
        }
        # Rain: needs tp at every step after T+0 (T+0 is zero by definition)
        rain_hours = [h for h in hours if h > 0]
        if rain_hours and all("tp" in results[h] for h in rain_hours):
            ref = results[rain_hours[0]]
            zero = np.zeros_like(ref["tp"])
            fields = [results[h]["tp"] if "tp" in results[h] else zero for h in hours]
            starts = [results[h].get("tp_start", 0) for h in hours]
            if all(f.shape == zero.shape for f in fields):
                data["rain"] = rain_since_start(hours, fields, starts)
                data["rlat"], data["rlon"] = ref["rlat"], ref["rlon"]
                print(f"Rain grid {data['rain'].shape}, max total {np.nanmax(data['rain'][-1]):.0f} mm")
        else:
            print("  (no total precipitation in this run - rain charts skipped)")
        print(f"Grid {mslp.shape}, MSLP {np.nanmin(mslp):.1f}–{np.nanmax(mslp):.1f} hPa")
        return data
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# DEMO DATA
# ============================================================

def make_demo_data(n_hours=360, step_h=1, grid=0.1):
    """Synthetic fields for testing the pipeline without any downloads."""
    lat = np.round(np.arange(LAT_MIN, LAT_MAX + 0.001, grid), 2)
    lon = np.round(np.arange(LON_MIN, LON_MAX + 0.001, grid), 2)
    LON2, LAT2 = np.meshgrid(lon, lat)
    hours = list(range(step_h if step_h == 1 else 0, n_hours + 1, step_h))
    mslp = np.empty((len(hours), len(lat), len(lon)), np.float32)
    u10 = np.empty_like(mslp)
    v10 = np.empty_like(mslp)
    for i, t in enumerate(hours):
        clat = 11.0 + 0.02 * t
        clon = 90.0 - 0.045 * t
        r2 = ((LAT2 - clat) ** 2 + ((LON2 - clon) * np.cos(np.radians(clat))) ** 2)
        depth = 4 + 10 * np.sin(np.pi * t / n_hours)
        p = 1008 + 0.25 * (LAT2 - 15) - depth * np.exp(-r2 / 6.0)
        mslp[i] = p
        dy = LAT2 - clat
        dx = (LON2 - clon) * np.cos(np.radians(clat))
        speed = 2.2 * depth * np.sqrt(r2) * np.exp(-r2 / 6.0)
        rr = np.sqrt(r2) + 1e-6
        u10[i] = -speed * dy / rr - 0.3 * dx / rr * speed - 4
        v10[i] = speed * dx / rr - 0.3 * dy / rr * speed
    return {"lat": lat, "lon": lon, "mslp": mslp, "u10": u10, "v10": v10,
            "n_hours": n_hours, "hours": hours, "grid_deg": grid}


# ============================================================
# V3 DETECTION (logic unchanged from the Colab script)
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def calculate_circulation_metrics(lat0, lon0, u, v, LAT2D, LON2D):
    dist = haversine_km(lat0, lon0, LAT2D, LON2D)
    ring = (dist >= CIRC_RING_INNER_DEG * 111.0) & (dist <= CIRC_RING_OUTER_DEG * 111.0)

    if np.count_nonzero(ring) < 20:
        return {"circulation": 0.0, "max_wind_kt": 0.0, "mean_tangential": 0.0, "convergence": 0.0}

    dx = np.radians(LON2D - lon0) * np.cos(np.radians(lat0))
    dy = np.radians(LAT2D - lat0)
    radius = np.sqrt(dx * dx + dy * dy)
    radius[radius == 0] = np.nan

    tx = -dy / radius
    ty = dx / radius
    tangential_ring = (u * tx + v * ty)[ring]

    circulation = max(np.mean(tangential_ring > 0), 0.0)
    mean_tangential = np.nanmean(tangential_ring)

    wind_speed = np.sqrt(u * u + v * v)
    max_wind_kt = np.nanpercentile(wind_speed[ring], 95) * 1.94384

    radial_ring = (u * (dx / radius) + v * (dy / radius))[ring]
    convergence = max(0.0, -np.nanmean(radial_ring))

    return {
        "circulation": float(circulation),
        "max_wind_kt": float(max_wind_kt),
        "mean_tangential": float(mean_tangential),
        "convergence": float(convergence),
    }


def filter_size_for(grid_deg):
    """Same ~2.1° search window on any grid (21 points at 0.1° -> 9 points at 0.25°)."""
    n = int(round(MINIMUM_FILTER_SIZE * 0.1 / grid_deg))
    return max(3, n | 1)   # odd


def detect_v3_candidates(hour, mslp, u, v, LAT2D, LON2D, filter_size=MINIMUM_FILTER_SIZE):
    candidates = []

    local_min = mslp == minimum_filter(mslp, size=filter_size, mode="nearest")

    for iy, ix in np.argwhere(local_min):
        lat0 = float(LAT2D[iy, ix])
        lon0 = float(LON2D[iy, ix])

        if (lat0 <= LAT_MIN + BOUNDARY_MARGIN_DEG or lat0 >= LAT_MAX - BOUNDARY_MARGIN_DEG
                or lon0 <= LON_MIN + BOUNDARY_MARGIN_DEG or lon0 >= LON_MAX - BOUNDARY_MARGIN_DEG):
            continue

        p0 = float(mslp[iy, ix])

        dist = haversine_km(lat0, lon0, LAT2D, LON2D)
        pressure_ring = (dist >= PRESSURE_RING_INNER_DEG * 111.0) & (dist <= PRESSURE_RING_OUTER_DEG * 111.0)
        if np.count_nonzero(pressure_ring) < 30:
            continue

        depression = np.nanmedian(mslp[pressure_ring]) - p0
        if depression < MIN_PRESSURE_DEPRESSION_HPA:
            continue

        circ = calculate_circulation_metrics(lat0, lon0, u, v, LAT2D, LON2D)
        circulation = circ["circulation"]
        if circulation < CIRCULATION_FLOOR:
            continue

        max_wind_kt = circ["max_wind_kt"]
        convergence = circ["convergence"]

        pressure_score = np.clip(depression / 4.0, 0.0, 1.0)
        circulation_score = np.clip(circulation, 0.0, 1.0)
        wind_score = np.clip(max_wind_kt / 25.0, 0.0, 1.0)
        convergence_score = np.clip(convergence / 5.0, 0.0, 1.0)
        tropical_bonus = 0.08 if (8.0 <= lat0 <= 25.0 and 65.0 <= lon0 <= 100.0) else 0.0

        score = (0.35 * pressure_score + 0.30 * circulation_score + 0.20 * wind_score
                 + 0.15 * convergence_score + tropical_bonus)

        candidates.append({
            "hour": hour, "lat": lat0, "lon": lon0, "mslp_hpa": p0,
            "pressure_depression_hpa": float(depression),
            "circulation": float(circulation),
            "max_wind_kt": float(max_wind_kt),
            "convergence": float(convergence),
            "score": float(score),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    for candidate in candidates:
        too_close = any(
            haversine_km(candidate["lat"], candidate["lon"], e["lat"], e["lon"]) < 150.0
            for e in selected
        )
        if not too_close:
            selected.append(candidate)
        if len(selected) >= MAX_CANDIDATES_PER_HOUR:
            break
    return selected


def run_tracking(data):
    """V3 detection + tracking.

    "hour" in the output is the array index (as before); "fhour" is the
    real forecast hour. For hourly WeatherNext 3 data every limit is exactly
    as in the Colab script. For 6-hourly ECMWF data the movement limit is
    scaled by the step (125 km/h × step, capped) and the persistence test is
    the same 6 hours (= 2 points at 6-hourly steps).
    """
    lat_sub, lon_sub = data["lat"], data["lon"]
    LON2D, LAT2D = np.meshgrid(lon_sub, lat_sub)
    hours = data["hours"]
    n_steps = len(hours)
    fsize = filter_size_for(data.get("grid_deg", 0.1))

    typical_dt = int(np.median(np.diff(hours))) if n_steps > 1 else 1
    min_points = max(2, int(np.ceil(MIN_TRACK_HOURS / typical_dt))) if typical_dt > 1 else MIN_TRACK_HOURS

    print(f"\nRUNNING V3 SURFACE SYSTEM DETECTION ({n_steps} steps, filter {fsize}, "
          f"min track {min_points} points)")
    by_hour = []
    for i in range(n_steps):
        if i % max(1, 48 // typical_dt) == 0:
            print(f"  forecast hour {hours[i]:03d}/{hours[-1]:03d}")
        cands = detect_v3_candidates(i, data["mslp"][i], data["u10"][i], data["v10"][i],
                                     LAT2D, LON2D, fsize)
        for c in cands:
            c["fhour"] = int(hours[i])
        by_hour.append(cands)

    all_candidates = [c for h in by_hour for c in h]
    print("V3 candidate records:", len(all_candidates))

    active_tracks, finished_tracks, next_track_id = {}, [], 1

    for i in range(n_steps):
        hour_candidates = by_hour[i]

        if i == 0:
            for c in hour_candidates:
                c = dict(c, track_id=next_track_id)
                active_tracks[next_track_id] = [c]
                next_track_id += 1
            continue

        dt = hours[i] - hours[i - 1]
        max_move = min(MAX_MOVEMENT_KM_PER_HOUR * dt, max(MAX_MOVEMENT_KM_PER_HOUR,
                                                          MAX_MOVEMENT_KM_PER_STEP_CAP))

        previous_tracks = dict(active_tracks)
        active_tracks = {}
        assigned_ci = set()
        possible = []

        for tid, pts in previous_tracks.items():
            last = pts[-1]
            for ci, c in enumerate(hour_candidates):
                d = haversine_km(last["lat"], last["lon"], c["lat"], c["lon"])
                if d <= max_move:
                    possible.append((d + 20.0 * abs(c["mslp_hpa"] - last["mslp_hpa"]), d, tid, ci))

        possible.sort(key=lambda x: x[0])
        assigned_tracks = set()
        for _, _, tid, ci in possible:
            if tid in assigned_tracks or ci in assigned_ci:
                continue
            active_tracks[tid] = previous_tracks[tid] + [dict(hour_candidates[ci], track_id=tid)]
            assigned_tracks.add(tid)
            assigned_ci.add(ci)

        for tid, pts in previous_tracks.items():
            if tid not in assigned_tracks:
                finished_tracks.append(pts)

        for ci, c in enumerate(hour_candidates):
            if ci in assigned_ci:
                continue
            active_tracks[next_track_id] = [dict(c, track_id=next_track_id)]
            next_track_id += 1

    finished_tracks.extend(active_tracks.values())

    persistent = [t for t in finished_tracks if len(t) >= min_points]
    print("Raw tracks:", len(finished_tracks), "| Persistent tracks:", len(persistent))

    candidates_df = pd.DataFrame(all_candidates)
    tracks_df = pd.DataFrame([p for t in persistent for p in t])
    if tracks_df.empty:
        tracks_df = pd.DataFrame(columns=[
            "hour", "lat", "lon", "mslp_hpa", "pressure_depression_hpa",
            "circulation", "max_wind_kt", "convergence", "score", "fhour", "track_id"])
    return candidates_df, tracks_df, len(persistent)


# ============================================================
# MAP (same design as the Colab script)
# ============================================================

_BORDER_FEATURE = None


def india_view_borders():
    """Land borders (like cfeature.BORDERS) with India's border as per the Government of India."""
    global _BORDER_FEATURE
    if _BORDER_FEATURE is None:
        with open(BOUNDARY_FILE) as f:
            gj = json.load(f)
        geoms = [shape(feat["geometry"]) for feat in gj["features"]]
        _BORDER_FEATURE = ShapelyFeature(geoms, ccrs.PlateCarree())
    return _BORDER_FEATURE


def draw_map(out_png, model, run_name, init_time, plot_hour, data, tracks_df,
             n_persistent, plot_windbarbs):

    cfg = MODELS[model]
    plot_index = data["hours"].index(plot_hour)  # WeatherNext: +24 h -> index 23
    valid_time = init_time + timedelta(hours=plot_hour)

    lat_sub, lon_sub = data["lat"], data["lon"]
    mslp_plot = data["mslp"][plot_index]

    selected_centres = tracks_df[tracks_df["hour"] == plot_index].sort_values("score", ascending=False)

    fig = plt.figure(figsize=(16, 10.5), facecolor="white")
    ax = fig.add_axes([0.045, 0.095, 0.845, 0.78], projection=ccrs.PlateCarree())
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    levels = np.arange(990, 1036, 1)

    cf = ax.contourf(lon_sub, lat_sub, mslp_plot, levels=levels, cmap="RdYlBu_r",
                     extend="both", transform=ccrs.PlateCarree())

    cs = ax.contour(lon_sub, lat_sub, mslp_plot, levels=levels, colors="black",
                    linewidths=0.55, alpha=0.75, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=7, fmt="%d", inline_spacing=3)

    ax.coastlines(resolution="50m", linewidth=0.8, color="black")

    # National borders — India as per Government of India
    ax.add_feature(india_view_borders(), facecolor="none", edgecolor="black", linewidth=0.45)

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="gray", alpha=0.35, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    if plot_windbarbs:
        step = max(1, int(round(1.0 / data.get("grid_deg", 0.1))))  # ~1° spacing on any grid
        ax.barbs(lon_sub[::step], lat_sub[::step],
                 data["u10"][plot_index][::step, ::step],
                 data["v10"][plot_index][::step, ::step],
                 transform=ccrs.PlateCarree(), length=5.2, linewidth=0.45, alpha=0.72,
                 pivot="middle", barb_increments={"half": 2.5, "full": 5, "flag": 25}, zorder=10)

    for _, row in selected_centres.iterrows():
        lon, lat = float(row["lon"]), float(row["lat"])
        ax.plot(lon, lat, marker="o", markersize=11, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=2.0,
                transform=ccrs.PlateCarree(), zorder=20)
        ax.text(lon + 0.25, lat + 0.15, f"T{int(row['track_id']):02d}\n{float(row['mslp_hpa']):.1f}",
                fontsize=8.5, fontweight="bold", color="black",
                transform=ccrs.PlateCarree(), zorder=21)

    ax.set_title(
        f"{cfg['title']}\n"
        f"Valid: {valid_time.strftime('%Y-%m-%d %H:%M UTC')} (T+{plot_hour} h)",
        fontsize=18, fontweight="bold", pad=12)

    info_text = (
        f"{cfg['info_head'].format(grid=grid_label(data.get('grid_deg', 0.1)))}\n"
        f"Run: {run_name}\n"
        f"Initialisation: {init_time.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Forecast: +{plot_hour} h\n"
        f"V3 persistent tracks: {n_persistent}\n"
        f"Centres at valid time: {len(selected_centres)}"
    )
    if plot_windbarbs:
        info_text += "\n" + cfg["barbs_label"]

    ax.text(0.012, 0.985, info_text, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="gray", alpha=0.82),
            zorder=30)

    cax = fig.add_axes([0.915, 0.18, 0.018, 0.58])
    cbar = fig.colorbar(cf, cax=cax, orientation="vertical")
    cbar.set_label("Mean Sea Level Pressure (hPa)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    for y, size, line in zip((0.045, 0.027, 0.010), (7.5, 7.0, 6.8), cfg["credits"]):
        fig.text(0.5, y, line, ha="center", va="bottom", fontsize=size, color="dimgray")

    plt.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)

    return selected_centres, valid_time


_RAIN_CMAP = None


def rain_cmap():
    """Discrete colour scale: one fixed colour per band."""
    global _RAIN_CMAP
    if _RAIN_CMAP is None:
        from matplotlib.colors import BoundaryNorm, ListedColormap
        assert len(RAIN_COLOURS) == len(RAIN_LEVELS) - 1
        cmap = ListedColormap(RAIN_COLOURS)
        cmap.set_under("white")
        cmap.set_over(RAIN_OVER)
        _RAIN_CMAP = (cmap, BoundaryNorm(RAIN_LEVELS, cmap.N))
    return _RAIN_CMAP


def draw_rain_map(out_png, model, run_name, init_time, plot_hour, data, period_h=None):
    """Rain chart. period_h=None -> total since T+0, else rain in the last period_h hours."""
    cfg = MODELS[model]
    i = data["hours"].index(plot_hour)
    valid_time = init_time + timedelta(hours=plot_hour)
    rain = data["rain"][i]
    if period_h:
        start = plot_hour - period_h
        rain = rain - (data["rain"][data["hours"].index(start)] if start > 0 else 0)
        what = f"{period_h}-h Precipitation (mm), T+{max(start, 0)} to T+{plot_hour}"
    else:
        what = "Total Precipitation (mm)"
    rain = np.maximum(rain, 0)
    cmap, norm = rain_cmap()

    fig = plt.figure(figsize=(13, 10.6), facecolor="white")
    ax = fig.add_axes([0.01, 0.135, 0.98, 0.805], projection=ccrs.PlateCarree())
    ax.set_extent([RAIN_LON_MIN, RAIN_LON_MAX, RAIN_LAT_MIN, RAIN_LAT_MAX], crs=ccrs.PlateCarree())
    ax.set_facecolor("#f4f6fb")
    ax.pcolormesh(data["rlon"], data["rlat"], np.where(rain < RAIN_LEVELS[0], np.nan, rain),
                  cmap=cmap, norm=norm, shading="nearest", transform=ccrs.PlateCarree(), zorder=1)
    ax.coastlines(resolution="50m", linewidth=0.7, color="black", zorder=5)
    ax.add_feature(india_view_borders(), facecolor="none", edgecolor="black", linewidth=0.5, zorder=6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--", zorder=7)
    gl.top_labels = False
    gl.left_labels = False
    gl.xlabel_style = {"size": 8.5}
    gl.ylabel_style = {"size": 8.5}

    grid = grid_label(data.get("grid_deg", 0.25)).split(" ")[0]
    fig.text(0.012, 0.965, f"{cfg.get('rain_label', cfg['label'])} {grid}  Init {init_time.strftime('%Hz %d %b %Y')} • {what}",
             ha="left", va="center", fontsize=12.5, fontweight="bold")
    fig.text(0.988, 0.965, f"Hour: {plot_hour} • Valid: {valid_time.strftime('%Hz %a %d %b %Y')}",
             ha="right", va="center", fontsize=12.5, fontweight="bold")

    # Horizontal colour bar with every band, labelled at the usual values
    cax = fig.add_axes([0.04, 0.055, 0.68, 0.022])
    n = len(RAIN_LEVELS) - 1
    cax.imshow(np.arange(n)[None, :], cmap=cmap, norm=plt.Normalize(-0.5, n - 0.5),
               aspect="auto", extent=(0, n, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([RAIN_LEVELS.index(t) for t in RAIN_TICKS])
    cax.set_xticklabels([f"{t:g}" for t in RAIN_TICKS], fontsize=8.5)
    cax.xaxis.set_ticks_position("top")
    cax.tick_params(length=0, pad=2)
    for sp in cax.spines.values():
        sp.set_linewidth(0.6)

    fig.text(0.988, 0.065, f"Max: {np.nanmax(rain):.0f}", ha="right", va="center",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.025, cfg["credits"][0] + " | Rain chart: MasRainman", ha="center",
             fontsize=7.5, color="dimgray")
    fig.text(0.5, 0.008, cfg["credits"][-1], ha="center", fontsize=7.2, color="dimgray")

    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    return valid_time


def make_gif(png_paths, out_gif):
    frames = []
    for p in png_paths:
        im = Image.open(p).convert("RGB")
        h = int(im.height * GIF_WIDTH_PX / im.width)
        im = im.resize((GIF_WIDTH_PX, h), Image.LANCZOS)
        frames.append(im.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=GIF_FRAME_MS, loop=0, optimize=True)


# ============================================================
# ONE RUN -> WEBSITE FILES
# ============================================================

def runs_root(model):
    return os.path.join(SITE_DIR, MODELS[model]["runs_dir"])


def render_run(model, run_name, data):
    cfg = MODELS[model]
    pre = cfg["file_prefix"]
    init_time = decode_init_time(run_name)
    run_dir = os.path.join(runs_root(model), run_name)
    tmp_dir = run_dir + ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)

    candidates_df, tracks_df, n_persistent = run_tracking(data)

    candidates_df.to_csv(os.path.join(tmp_dir, f"{pre}_{run_name}_V3_candidates.csv"), index=False)
    tracks_df.to_csv(os.path.join(tmp_dir, f"{pre}_{run_name}_V3_persistent_tracks.csv"), index=False)

    available = set(data["hours"])
    last_hour = min(FRAME_END_H, max(data["hours"]))
    hours = [h for h in range(FRAME_STEP_H, last_hour + 1, FRAME_STEP_H) if h in available]

    frames = []
    pngs = {"mslp": [], "barbs": []}
    has_rain = "rain" in data
    if has_rain:
        pngs.update({"rain": [], "rain24": []})

    for h in hours:
        print(f"[{cfg['label']}] Drawing T+{h:03d} ...")
        entry = {"hour": h}
        for layer, barbs in (("mslp", False), ("barbs", True)):
            fname = f"{layer}_T{h:03d}.png"
            centres, valid_time = draw_map(os.path.join(tmp_dir, fname), model, run_name, init_time, h,
                                           data, tracks_df, n_persistent, barbs)
            entry[layer] = fname
            pngs[layer].append(os.path.join(tmp_dir, fname))

        if has_rain:
            for layer, period in (("rain", None), ("rain24", 24)):
                if period and (h - period) not in available and h - period > 0:
                    continue
                fname = f"{layer}_T{h:03d}.png"
                draw_rain_map(os.path.join(tmp_dir, fname), model, run_name, init_time, h,
                              data, period)
                entry[layer] = fname
                pngs[layer].append(os.path.join(tmp_dir, fname))

        entry["valid"] = valid_time.strftime("%Y-%m-%dT%H:%MZ")
        entry["systems"] = [
            {
                "id": f"T{int(r['track_id']):02d}",
                "lat": round(float(r["lat"]), 1),
                "lon": round(float(r["lon"]), 1),
                "mslp": round(float(r["mslp_hpa"]), 1),
                "dep": round(float(r["pressure_depression_hpa"]), 2),
                "wind": round(float(r["max_wind_kt"]), 1),
                "circ": round(float(r["circulation"]), 3),
                "score": round(float(r["score"]), 3),
            }
            for _, r in centres.iterrows()
        ]
        frames.append(entry)

    print("Building GIFs ...")
    gifs = {}
    for layer in pngs:
        if not pngs[layer]:
            continue
        gname = f"{pre}_{run_name}_{layer}_loop.gif"
        make_gif(pngs[layer], os.path.join(tmp_dir, gname))
        gifs[layer] = gname

    run_info = {
        "model": model,
        "model_label": cfg["label"],
        "base": cfg["runs_dir"].replace(os.sep, "/"),
        "run": run_name,
        "init": init_time.strftime("%Y-%m-%dT%H:%MZ"),
        "last_hour": int(max(data["hours"])),
        "resol": data.get("resol"),
        "grid_deg": data.get("grid_deg"),
        "persistent_tracks": n_persistent,
        "frames": frames,
        "gifs": gifs,
        "csv": {
            "candidates": f"{pre}_{run_name}_V3_candidates.csv",
            "tracks": f"{pre}_{run_name}_V3_persistent_tracks.csv",
        },
        "rendered": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
    }
    with open(os.path.join(tmp_dir, "run.json"), "w") as f:
        json.dump(run_info, f, indent=1)

    shutil.rmtree(run_dir, ignore_errors=True)
    os.rename(tmp_dir, run_dir)
    print(f"Run {run_name} done -> {run_dir}")
    return run_info


# ============================================================
# MANIFEST + PRUNING
# ============================================================

def rendered_runs(model="wn3"):
    root = runs_root(model)
    if not os.path.isdir(root):
        return []
    pattern = MODELS[model]["pattern"]
    out = []
    for name in os.listdir(root):
        if re.match(pattern, name) and os.path.exists(os.path.join(root, name, "run.json")):
            out.append(name)
    return sorted(out, reverse=True)


def write_manifest(model="wn3"):
    cfg = MODELS[model]
    root = runs_root(model)
    runs = rendered_runs(model)

    # Keep only the latest KEEP_RUNS
    for old in runs[KEEP_RUNS:]:
        print(f"[{cfg['label']}] Removing old run", old)
        shutil.rmtree(os.path.join(root, old), ignore_errors=True)
    runs = runs[:KEEP_RUNS]

    manifest = {
        "model": model,
        "label": cfg["label"],
        "base": cfg["runs_dir"].replace(os.sep, "/"),
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "runs": [],
    }
    for r in runs:
        with open(os.path.join(root, r, "run.json")) as f:
            manifest["runs"].append(json.load(f))

    os.makedirs(os.path.join(SITE_DIR, "data"), exist_ok=True)
    with open(os.path.join(SITE_DIR, "data", cfg["manifest"]), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[{cfg['label']}] Manifest written with", len(runs), "runs")


def write_models_index():
    models = []
    for m in MODEL_ORDER:
        cfg = MODELS[m]
        path = os.path.join(SITE_DIR, "data", cfg["manifest"])
        n = 0
        if os.path.exists(path):
            with open(path) as f:
                n = len(json.load(f).get("runs", []))
        models.append({"id": m, "label": cfg["label"], "manifest": f"data/{cfg['manifest']}",
                       "base": cfg["runs_dir"].replace(os.sep, "/"), "runs": n})
    with open(os.path.join(SITE_DIR, "data", "models.json"), "w") as f:
        json.dump({"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                   "models": models}, f, indent=1)

    # Copy the viewer page
    for fn in ("index.html",):
        src = os.path.join(HERE, "web", fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(SITE_DIR, fn))
    open(os.path.join(SITE_DIR, ".nojekyll"), "w").close()


# ============================================================
# MAIN
# ============================================================

def process_wn3(force_run=None):
    session = get_session()
    if force_run:
        targets = [force_run]
    else:
        available = list_weathernext_runs(session)
        if not available:
            raise RuntimeError("No full-length WeatherNext 3 runs were found.")
        print("Latest available:", ", ".join(available[:6]))
        done = set(rendered_runs("wn3"))
        targets = [r for r in available[:KEEP_RUNS] if r not in done]

    if not targets:
        print("[WeatherNext 3] Nothing new to render.")
        return 0

    rendered = 0
    for run_name in targets:
        if rendered >= MAX_NEW_RUNS_PER_JOB:
            break
        data = load_run(run_name)
        if data is None:
            continue
        render_run("wn3", run_name, data)
        rendered += 1
        del data
    return rendered


def process_ecmwf(model, force_run=None):
    label = MODELS[model]["label"]
    http = _http()
    if force_run:
        src = ecmwf_best_source(model, decode_init_time(force_run), http)
        if not src:
            print(f"[{label}] {force_run} is not complete on data.ecmwf.int yet.")
            return 0
        targets = [(force_run, src)]
    else:
        available = list_ecmwf_runs(model)
        print("Latest available:", ", ".join(f"{r} ({s['resol']}, T+{s['step']})"
                                             for r, s in available) or "(none)")
        targets = []
        for r, src in available:
            have = rendered_rank(model, r)
            best = source_rank(model, src["step"], src["resol"])
            if have is None:
                targets.append((r, src))
            elif best > have:
                print(f"[{label}] {r}: better data now ({src['resol']}, T+{src['step']}) - redrawing")
                targets.append((r, src))

    if not targets:
        print(f"[{label}] Nothing new to render.")
        return 0

    rendered = 0
    for run_name, src in targets:
        if rendered >= MAX_NEW_RUNS_PER_JOB:
            break
        data = load_ecmwf_run(model, run_name, src)
        render_run(model, run_name, data)
        rendered += 1
        del data
    return rendered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="wn3,ifs,aifs",
                    help="Comma list of models to process (wn3, ifs, aifs)")
    ap.add_argument("--model", help="Model for --run (wn3, ifs or aifs)")
    ap.add_argument("--run", help="Render this run name (forces re-render)")
    ap.add_argument("--demo", action="store_true", help="Synthetic data test, no downloads")
    args = ap.parse_args()

    os.makedirs(SITE_DIR, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in MODELS:
            sys.exit(f"Unknown model '{m}'. Use: {', '.join(MODEL_ORDER)}")

    if args.demo:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        for m in models:
            if m == "wn3":
                render_run(m, today + "_00hr_01_preds", make_demo_data())
            else:
                render_run(m, today + "_00", make_demo_data(360, ECMWF_STEP_H, 0.25))
            write_manifest(m)
        write_models_index()
        return

    if args.run:
        model = args.model or ("wn3" if re.match(RUN_PATTERN, args.run) else None)
        if model not in MODELS:
            sys.exit("With --run, also give --model ifs or --model aifs for ECMWF runs.")
        models = [model]

    total = 0
    failures = []
    for m in models:
        try:
            if m == "wn3":
                total += process_wn3(args.run)
            else:
                total += process_ecmwf(m, args.run)
        except Exception as exc:
            # One model failing must not stop the others from updating
            print(f"\n!!! {MODELS[m]['label']} failed: {exc}")
            failures.append(m)
        write_manifest(m)

    write_models_index()
    print(f"\nRendered {total} new run(s).")
    if failures and len(failures) == len(models):
        sys.exit(1)


if __name__ == "__main__":
    main()
