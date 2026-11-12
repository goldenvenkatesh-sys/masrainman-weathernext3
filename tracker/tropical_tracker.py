# =============================================================================
# MASRAINMAN — INDIA TROPICAL TRACKER  (V87, from multitrack.py V86)
# ECMWF IFS HRES + AIFS Single + IFS ENS (ECMWF track BUFR) + NOAA AI-GFS (MSLP low centre) + NOAA OISST SST
#
# Two ways to run it:
#   python tracker/tropical_tracker.py --out site_tracker     (GitHub: writes index.html + data.json + sst.json
#                                                            + stamp.json; the web page refreshes itself)
#   python tracker/tropical_tracker.py --standalone --open    (PC: one self-contained HTML, opens the browser)
#   python tracker/tropical_tracker.py --standalone --every 2 (PC: keeps going, new check every 2 hours)
#
# Latest runs: the newest 4 IFS HRES runs (searched back 8 days) and, for the same runs, AIFS and IFS ENS;
# the newest 4 AI-GFS runs. Tracks already decoded are kept in <out>/state/ and reused, so a check with
# nothing new downloads almost nothing. Runs older than the newest 4 are dropped.
# Borders: India as per the Government of India (DataMeet India composite + states, GoI-view neighbours).
# =============================================================================

import argparse
import base64
import glob
import io
import json
import os
import shutil
import tempfile
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from PIL import Image
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from ecmwf.opendata import Client
from eccodes import (
    codes_bufr_new_from_file, codes_grib_new_from_file, codes_get, codes_get_array, codes_set,
    codes_release, CODES_MISSING_DOUBLE, CODES_MISSING_LONG, CodesInternalError,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# =============================================================================
# CONFIGURATION
# =============================================================================

APP_NAME = "MASRAINMAN INDIA TROPICAL TRACKER"

TRACK_WEST, TRACK_EAST = 35.0, 120.0
TRACK_SOUTH, TRACK_NORTH = -10.0, 40.0
MAP_WEST, MAP_EAST = 40.0, 115.0
MAP_SOUTH, MAP_NORTH = -5.0, 38.0

KEEP_RUNS = int(os.environ.get("TRACKER_KEEP_RUNS", "4"))
MAX_HOURS = 360
IFS_RESOLUTION = "0p25"
IFS_STEPS = {0: 360, 6: 144, 12: 360, 18: 144}
AIFS_STEPS = {0: 360, 6: 360, 12: 360, 18: 360}
AIGFS_MAX_HOURS = 384
AIGFS_STEP = 6
AIGFS_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod"
UA = {"User-Agent": "MASRAINMAN-India-Tropical-Tracker/87"}

# Boundaries (GoI): the PC's Survey of India shapefile if present, else DataMeet (downloaded once)
ALL_INDIA_DIR = Path(os.environ.get("MASRAINMAN_ALL_INDIA", r"D:\ALL INDIA"))
STATE_BOUNDARY_SHP = ALL_INDIA_DIR / "STATE_BOUNDARY.shp"
BOUNDARY_CACHE = Path(os.environ.get("TRACKER_BOUNDARY_DIR", HERE / "boundaries"))
DATAMEET = "https://raw.githubusercontent.com/datameet/maps/master/"
DATAMEET_FILES = {"india-composite.geojson": "Country/india-composite.geojson",
                  "Admin2.shp": "States/Admin2.shp", "Admin2.shx": "States/Admin2.shx",
                  "Admin2.dbf": "States/Admin2.dbf"}
NEIGHBOURS = REPO / "boundaries" / "borders_india_view.geojson"

# filled in main()
OUT_DIR = STATE_DIR = DATA_DIR = None


def utc_now():
    return datetime.now(timezone.utc)


def print_banner(text):
    print("\n" + "=" * 78 + "\n" + text + "\n" + "=" * 78, flush=True)


def normalize_lon(lon):
    lon = float(lon)
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def in_domain(df):
    if df.empty:
        return df
    return df[(df["lat"] >= TRACK_SOUTH) & (df["lat"] <= TRACK_NORTH) &
              (df["lon"] >= TRACK_WEST) & (df["lon"] <= TRACK_EAST)].copy()


def first_valid(a):
    for x in np.asarray(a).ravel():
        try:
            xf = float(x)
            if np.isfinite(xf) and xf not in (float(CODES_MISSING_LONG), float(CODES_MISSING_DOUBLE)):
                return xf
        except Exception:
            pass
    return np.nan


def arr(bufr, key):
    try:
        v = codes_get_array(bufr, key)
    except Exception:
        try:
            v = [codes_get(bufr, key)]
        except Exception:
            return np.array([], dtype=float)
    a = np.asarray(v)
    return a.reshape(1) if a.ndim == 0 else a


def align(a, n, fill=np.nan):
    a = np.asarray(a).ravel()
    if n <= 0:
        return np.array([], dtype=float)
    if a.size == n:
        return a
    if a.size == 0:
        return np.full(n, fill)
    if a.size == 1:
        return np.repeat(a, n)
    return np.resize(a, n)


def run_key(run_dt):
    return run_dt.strftime("%Y%m%d%H")


def key_dt(key):
    return datetime.strptime(str(key), "%Y%m%d%H").replace(tzinfo=timezone.utc)


def run_label(run_dt):
    return f"{run_dt:%d.%m.%Y} {run_dt:%H}Z"


def cycle_candidates(now=None, search_cycles=32):
    now = now or utc_now()
    anchor = datetime(now.year, now.month, now.day, (now.hour // 6) * 6, tzinfo=timezone.utc)
    return [anchor - timedelta(hours=6 * i) for i in range(search_cycles)]


# =============================================================================
# TRACK CACHE  (state/<product>_<YYYYMMDDHH>.json)
# =============================================================================

def cache_path(product, key):
    return STATE_DIR / f"{product}_{key}.json"


def cache_load(product, key):
    p = cache_path(product, key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def cache_save(product, key, df, complete=True, next_hour=None):
    rows = [] if df is None or df.empty else df_to_rows(df)
    cache_path(product, key).write_text(json.dumps(
        {"complete": complete, "next": next_hour, "rows": rows}, separators=(",", ":")))


def df_to_rows(df):
    out = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        out.append({k: (None if isinstance(v, float) and not np.isfinite(v) else
                        (v.item() if hasattr(v, "item") else v)) for k, v in d.items()})
    return out


def rows_to_df(rows):
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def prune_cache(keep):
    """Drop cached runs that are not in `keep` = {product: set(keys)}."""
    for p in STATE_DIR.glob("*_*.json"):
        prod, key = p.stem.rsplit("_", 1)
        if prod in keep and key not in keep[prod]:
            p.unlink()


# =============================================================================
# ECMWF DOWNLOADS (TC track BUFR)
# =============================================================================

def _retrieve(model, stream, run_dt, step, target):
    try:
        client = Client(source="ecmwf", model=model, resol=IFS_RESOLUTION, preserve_request_order=True)
        client.retrieve(date=run_dt.strftime("%Y-%m-%d"), time=run_dt.hour, stream=stream, type="tf",
                        step=step, target=str(target))
    except Exception:
        pass
    if not target.exists() or target.stat().st_size <= 100:
        try:
            target.unlink()
        except Exception:
            pass
        return None
    return target


def download_hres(run_dt):
    return _retrieve("ifs", "oper", run_dt, IFS_STEPS[run_dt.hour], DATA_DIR / f"ifs_tc_{run_key(run_dt)}.bufr")


def download_ens(run_dt):
    return _retrieve("ifs", "enfo", run_dt, IFS_STEPS[run_dt.hour], DATA_DIR / f"ifs_ens_tc_{run_key(run_dt)}.bufr")


def download_aifs(run_dt):
    return _retrieve("aifs-single", "oper", run_dt, AIFS_STEPS[run_dt.hour],
                     DATA_DIR / f"aifs_single_tc_{run_key(run_dt)}.bufr")


# =============================================================================
# ECMWF TC BUFR DECODER — HRES + ENS  (unchanged from V86)
# =============================================================================

def decode_ecmwf_tc_bufr(path, run_dt):
    rows = []

    def add_rows_direct(bufr, storm_id, storm_name, member_values):
        tp, lat, lon = arr(bufr, "timePeriod"), arr(bufr, "latitude"), arr(bufr, "longitude")
        p, wind = arr(bufr, "pressureReducedToMeanSeaLevel"), arr(bufr, "windSpeedAt10M")
        n = max(len(tp), len(lat), len(lon))
        if n == 0:
            return 0
        tp, lat, lon, p, wind = align(tp, n), align(lat, n), align(lon, n), align(p, n), align(wind, n)
        members = np.asarray(member_values).ravel()
        if members.size == 0:
            members = np.array([0])
        if members.size > 1 and n % members.size == 0:
            nper = n // members.size
            A = [x.reshape(members.size, nper) for x in (tp, lat, lon, p, wind)]
            for k, mem in enumerate(members):
                for j in range(nper):
                    rows.append({"lat": A[1][k, j], "lon": A[2][k, j], "tau": A[0][k, j], "mslp": A[3][k, j],
                                 "wind_kt": A[4][k, j], "storm_id": storm_id, "storm_name": storm_name,
                                 "member": mem})
            return n
        for j in range(n):
            rows.append({"lat": lat[j], "lon": lon[j], "tau": tp[j], "mslp": p[j], "wind_kt": wind[j],
                         "storm_id": storm_id, "storm_name": storm_name, "member": members[0]})
        return n

    with open(path, "rb") as f:
        while True:
            bufr = codes_bufr_new_from_file(f)
            if bufr is None:
                break
            try:
                codes_set(bufr, "unpack", 1)
                try:
                    storm_id = codes_get(bufr, "stormIdentifier")
                except Exception:
                    storm_id = "UNKNOWN"
                try:
                    storm_name = codes_get(bufr, "longStormName")
                except Exception:
                    storm_name = ""
                member_values = arr(bufr, "ensembleMemberNumber")
                if member_values.size == 0:
                    member_values = np.array([0])
                n_members = len(member_values)

                n_periods = 0
                while True:
                    try:
                        codes_get_array(bufr, f"#{n_periods + 1}#timePeriod")
                        n_periods += 1
                    except (CodesInternalError, Exception):
                        break

                if n_periods > 0:
                    periods = [0]
                    period_rows = [(align(arr(bufr, "#2#latitude"), n_members),
                                    align(arr(bufr, "#2#longitude"), n_members),
                                    align(arr(bufr, "#1#pressureReducedToMeanSeaLevel"), n_members),
                                    align(arr(bufr, "#3#latitude"), n_members),
                                    align(arr(bufr, "#3#longitude"), n_members),
                                    align(arr(bufr, "#1#windSpeedAt10M"), n_members))]
                    for i in range(1, n_periods):
                        tau = first_valid(arr(bufr, f"#{i}#timePeriod"))
                        if not np.isfinite(tau):
                            continue
                        rc, rw = i * 2 + 2, i * 2 + 3
                        period_rows.append((
                            align(arr(bufr, f"#{rc}#latitude"), n_members),
                            align(arr(bufr, f"#{rc}#longitude"), n_members),
                            align(arr(bufr, f"#{i + 1}#pressureReducedToMeanSeaLevel"), n_members),
                            align(arr(bufr, f"#{rw}#latitude"), n_members),
                            align(arr(bufr, f"#{rw}#longitude"), n_members),
                            align(arr(bufr, f"#{i + 1}#windSpeedAt10M"), n_members)))
                        periods.append(tau)
                    for k in range(n_members):
                        for tau, (lat, lon, pressure, lat_w, lon_w, wind) in zip(periods, period_rows):
                            rows.append({"lat": lat[k], "lon": lon[k], "tau": tau, "mslp": pressure[k],
                                         "wind_kt": wind[k], "storm_id": storm_id, "storm_name": storm_name,
                                         "member": member_values[k]})
                else:
                    add_rows_direct(bufr, storm_id, storm_name, member_values)
            except Exception:
                pass
            finally:
                codes_release(bufr)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    for c in ["lat", "lon", "tau", "mslp", "wind_kt"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([float(CODES_MISSING_DOUBLE), 1e20, -1e20, np.inf, -np.inf], np.nan)
    if out["mslp"].dropna().size and out["mslp"].dropna().median() > 2000:
        out["mslp"] /= 100.0
    if out["wind_kt"].dropna().size and out["wind_kt"].dropna().median() < 100:
        out["wind_kt"] *= 1.943844
    if out["lat"].dropna().size and out["lat"].abs().max() > 90:
        out["lat"] /= 10.0
    if out["lon"].dropna().size and out["lon"].abs().max() > 360:
        out["lon"] /= 10.0
    out["lon"] = out["lon"].map(normalize_lon)
    out = out.dropna(subset=["lat", "lon", "tau"])
    out = out[(out["tau"] >= 0) & (out["tau"] <= MAX_HOURS)].copy()
    out = in_domain(out)
    out["storm_id"] = out["storm_id"].astype(str)
    out["storm_name"] = out["storm_name"].astype(str)
    out["member"] = out["member"].astype(str)
    out = out.drop_duplicates(subset=["storm_id", "member", "tau", "lat", "lon"])
    return out.reset_index(drop=True)


def ecmwf_product(product, run_dt, downloader):
    """Cached track table of one ECMWF product/run, or None if the product isn't published (yet)."""
    key = run_key(run_dt)
    c = cache_load(product, key)
    if c is not None:
        return rows_to_df(c["rows"])
    path = downloader(run_dt)
    if path is None:
        return None
    df = decode_ecmwf_tc_bufr(path, run_dt)
    cache_save(product, key, df)
    try:
        path.unlink()
    except Exception:
        pass
    print(f"  {product.upper():5s} {run_label(run_dt)}: {len(df)} track points", flush=True)
    return df


def load_ecmwf():
    """Newest KEEP_RUNS HRES runs; AIFS + ENS for the same runs."""
    print_banner(f"ECMWF LATEST {KEEP_RUNS} RUNS — {utc_now():%d %b %Y %H:%M UTC}")
    hres, aifs, ens, meta = {}, {}, {}, {}
    for run_dt in cycle_candidates():
        if len(hres) >= KEEP_RUNS:
            break
        df = ecmwf_product("hres", run_dt, download_hres)
        if df is None:
            continue
        key = run_key(run_dt)
        hres[key] = df
        meta[key] = {"date": run_dt.strftime("%Y-%m-%d"), "hour": run_dt.hour, "label": run_label(run_dt)}
        print(f"  FOUND #{len(hres)}: {run_label(run_dt)}", flush=True)
    for key in hres:
        run_dt = key_dt(key)
        a = ecmwf_product("aifs", run_dt, download_aifs)
        if a is not None and not a.empty:
            aifs[key] = a
        e = ecmwf_product("ens", run_dt, download_ens)
        if e is not None and not e.empty:
            ens[key] = e
    if not hres:
        raise RuntimeError("No usable IFS HRES data found.")
    return hres, aifs, ens, meta


# =============================================================================
# NOAA AI-GFS — MSLP-DERIVED LOW-CENTER TRACK  (centre-finding unchanged from V86)
# =============================================================================

_HTTP = None


def http():
    global _HTTP
    if _HTTP is None:
        _HTTP = requests.Session()
        _HTTP.headers.update(UA)
    return _HTTP


def aigfs_file_url(run_dt, forecast_hour):
    cc = f"{run_dt.hour:02d}"
    return (f"{AIGFS_BASE_URL}/aigfs.{run_dt:%Y%m%d}/{cc}/model/atmos/grib2/"
            f"aigfs.t{cc}z.sfc.f{int(forecast_hour):03d}.grib2")


def aigfs_url_exists(url):
    try:
        return http().head(url, timeout=20, allow_redirects=True).status_code == 200
    except Exception:
        return False


def find_latest_aigfs_runs(max_runs=KEEP_RUNS):
    found = []
    for run_dt in cycle_candidates():
        if cache_load("aigfs", run_key(run_dt)) is not None or aigfs_url_exists(aigfs_file_url(run_dt, 6)):
            found.append(run_dt)
            if len(found) >= max_runs:
                break
    return found


AIGFS_WANTED = {("PRMSL", "mean sea level"), ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")}


def download_aigfs_sfc(run_dt, forecast_hour):
    """Only MSLP + 10-m wind: the .idx tells where they are, one ranged request (NOMADS limits requests)."""
    url = aigfs_file_url(run_dt, forecast_hour)
    target = DATA_DIR / f"aigfs_{run_key(run_dt)}_f{int(forecast_hour):03d}.grib2"
    for attempt in range(3):
        try:
            r = http().get(url + ".idx", timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            lines = [l.split(":") for l in r.text.strip().splitlines()]
            parts = []
            for k, l in enumerate(lines):
                if len(l) >= 5 and (l[3], l[4]) in AIGFS_WANTED:
                    end = int(lines[k + 1][1]) - 1 if k + 1 < len(lines) else None
                    parts.append((int(l[1]), end))
            headers = {}
            if len(parts) >= 3:
                lo = min(p[0] for p in parts)
                hi = None if any(p[1] is None for p in parts) else max(p[1] for p in parts)
                headers["Range"] = f"bytes={lo}-" + ("" if hi is None else str(hi))
            rr = http().get(url, timeout=120, headers=headers)
            if rr.status_code == 404:
                return None
            rr.raise_for_status()
            target.write_bytes(rr.content)
            return target
        except Exception:
            time.sleep(8 * (attempt + 1))
    return None


def decode_aigfs_sfc_prmsl(path):
    prmsl = u10 = v10 = lats = lons = None
    with open(path, "rb") as fh:
        while True:
            try:
                h = codes_grib_new_from_file(fh)
            except Exception:
                break
            if h is None:
                break
            try:
                sn = str(codes_get(h, "shortName"))
                if sn == "prmsl":
                    prmsl = np.asarray(codes_get_array(h, "values"), dtype=float).ravel()
                    lats = np.asarray(codes_get_array(h, "latitudes"), dtype=float).ravel()
                    lons = np.asarray(codes_get_array(h, "longitudes"), dtype=float).ravel()
                elif sn == "10u":
                    u10 = np.asarray(codes_get_array(h, "values"), dtype=float).ravel()
                elif sn == "10v":
                    v10 = np.asarray(codes_get_array(h, "values"), dtype=float).ravel()
            finally:
                codes_release(h)
    if prmsl is None:
        raise RuntimeError("PRMSL/grid not found")
    n = min(prmsl.size, lats.size, lons.size)
    lons = np.where(lons[:n] > 180, lons[:n] - 360, lons[:n])
    return lats[:n], lons, prmsl[:n], (u10[:n] if u10 is not None else None), (v10[:n] if v10 is not None else None)


def choose_aigfs_low_center(lats, lons, prmsl, u10, v10, prev=None):
    from scipy.ndimage import minimum_filter, maximum_filter
    lats, lons, p = np.asarray(lats).ravel(), np.asarray(lons).ravel(), np.asarray(prmsl).ravel()
    n = min(lats.size, lons.size, p.size)
    lats, lons, p = lats[:n], lons[:n], p[:n]
    good = (np.isfinite(lats) & np.isfinite(lons) & np.isfinite(p) & (lons >= TRACK_WEST) & (lons <= TRACK_EAST)
            & (lats >= TRACK_SOUTH) & (lats <= TRACK_NORTH))
    if np.unique(lats[good]).size < 20:
        return None
    order = np.where(good)[0]
    lat0, lon0, p0 = lats[order], lons[order], p[order]
    sort_idx = np.lexsort((lon0, lat0))
    lat_s, lon_s, p_s = lat0[sort_idx], lon0[sort_idx], p0[sort_idx]
    try:
        p_grid = p_s.reshape(np.unique(lat0).size, np.unique(lon0).size)
        local_min = minimum_filter(p_grid, size=11, mode="nearest")
        broad_max = maximum_filter(p_grid, size=31, mode="nearest")
    except Exception:
        return None
    depth = broad_max - p_grid
    lat_grid, lon_grid = lat_s.reshape(p_grid.shape), lon_s.reshape(p_grid.shape)
    cand = (np.isfinite(p_grid) & (p_grid <= local_min + 1e-6) & (depth >= 200.0)
            & (lat_grid >= TRACK_SOUTH) & (lat_grid <= TRACK_NORTH)
            & (lon_grid >= TRACK_WEST) & (lon_grid <= TRACK_EAST))
    cand_flat = np.where(cand.ravel())[0]
    if not cand_flat.size:
        return None
    cand_lat, cand_lon, cand_p = lat_s[cand_flat], lon_s[cand_flat], p_s[cand_flat]
    cand_depth = depth.ravel()[cand_flat]

    if prev is None:
        initial = ((cand_lat >= 5.0) & (cand_lat <= 25.0) & (cand_lon >= 45.0) & (cand_lon <= 100.0)
                   & (cand_p <= 101000.0))
        land = india_land_prepared()
        if land is not None and np.any(initial):
            idx = np.where(initial)[0]
            keep = np.array([not land.covers(Point(float(cand_lon[i]), float(cand_lat[i]))) for i in idx], bool)
            initial[:] = False
            initial[idx[keep]] = True
        if not np.any(initial):
            return None
        ii = np.where(initial)[0]
        bob = ii[(cand_lat[ii] >= 8.0) & (cand_lat[ii] <= 18.0) & (cand_lon[ii] >= 88.0) & (cand_lon[ii] <= 100.0)]
        j = bob[np.argmin(cand_p[bob])] if bob.size else ii[np.argmin(cand_p[ii])]
    else:
        plat, plon = prev
        distance = np.sqrt((cand_lat - plat) ** 2 + ((cand_lon - plon) * np.cos(np.deg2rad(plat))) ** 2)
        nearby = distance <= 4.0
        if not np.any(nearby):
            return None
        ii = np.where(nearby)[0]
        score = (distance[ii] / 4.0) - 0.15 * np.clip(cand_depth[ii] / 500.0, 0.0, 1.0)
        j = ii[np.argmin(score)]

    lat, lon, mslp = float(cand_lat[j]), float(cand_lon[j]), float(cand_p[j] / 100.0)
    wind_kt = np.nan
    if u10 is not None and v10 is not None:
        k = int(cand_flat[j])
        if k < sort_idx.size:
            orig = int(order[sort_idx[k]])
            uv = np.hypot(u10[orig], v10[orig])
            if np.isfinite(uv):
                wind_kt = float(uv * 1.94384449)
    return lat, lon, mslp, wind_kt


def load_aigfs_track(run_dt):
    """Resumes a run that was still coming in last time; a finished run comes straight from the cache."""
    key = run_key(run_dt)
    c = cache_load("aigfs", key) or {"complete": False, "next": 0, "rows": []}
    rows = c["rows"]
    if c["complete"]:
        return rows_to_df(rows)
    prev = (rows[-1]["lat"], rows[-1]["lon"]) if rows else None
    fhour, complete = int(c.get("next") or 0), False
    while fhour <= AIGFS_MAX_HOURS:
        path = download_aigfs_sfc(run_dt, fhour)
        if path is None:                      # not published yet: carry on from here next time
            break
        try:
            center = choose_aigfs_low_center(*decode_aigfs_sfc_prmsl(path), prev)
        except Exception as exc:
            print(f"  AI-GFS {run_label(run_dt)} f{fhour:03d}: {exc}")
            center = None
        finally:
            path.unlink(missing_ok=True)
        if center is None:                    # no low to follow any more: the track has ended
            complete = True
            break
        lat, lon, mslp, wind_kt = center
        rows.append({"storm_id": "AIGFS_LOW", "storm_name": "AI-GFS MSLP-DERIVED LOW", "member": "AI-GFS",
                     "tau": fhour, "lat": lat, "lon": lon, "mslp": mslp,
                     "wind_kt": None if not np.isfinite(wind_kt) else wind_kt})
        prev = (lat, lon)
        fhour += AIGFS_STEP
    if fhour > AIGFS_MAX_HOURS or utc_now() - run_dt > timedelta(hours=18):
        complete = True                       # end of the run, or NOMADS will not add more
    cache_path("aigfs", key).write_text(json.dumps({"complete": complete, "next": fhour, "rows": rows},
                                                   separators=(",", ":")))
    print(f"  AI-GFS {run_label(run_dt)}: {len(rows)} points{'' if complete else ' (run still coming in)'}",
          flush=True)
    return rows_to_df(rows)


def load_aigfs():
    print_banner("NOAA AI-GFS LATEST RUNS")
    data, meta = {}, {}
    for run_dt in find_latest_aigfs_runs():
        key = run_key(run_dt)
        df = load_aigfs_track(run_dt)
        meta[key] = {"date": run_dt.strftime("%Y-%m-%d"), "hour": run_dt.hour, "label": run_label(run_dt)}
        if not df.empty:
            data[key] = df
    return data, meta


# =============================================================================
# BOUNDARIES — India as per the Government of India
# =============================================================================

_LAND = None
_BOUNDARY_SEGMENTS = None


def _ensure_datameet():
    BOUNDARY_CACHE.mkdir(parents=True, exist_ok=True)
    for name, rel in DATAMEET_FILES.items():
        dst = BOUNDARY_CACHE / name
        if not dst.exists():
            r = requests.get(DATAMEET + rel, timeout=300, headers=UA)
            r.raise_for_status()
            dst.write_bytes(r.content)


def _state_shapes():
    import shapefile   # pyshp
    shp = STATE_BOUNDARY_SHP if STATE_BOUNDARY_SHP.exists() else None
    if shp is None:
        _ensure_datameet()
        shp = BOUNDARY_CACHE / "Admin2.shp"
    r = shapefile.Reader(str(shp))
    return [shape(s.__geo_interface__).buffer(0) for s in r.shapes()]


def india_land_prepared():
    global _LAND
    if _LAND is None:
        try:
            _ensure_datameet()
            with open(BOUNDARY_CACHE / "india-composite.geojson") as f:
                _LAND = prep(unary_union([shape(ft["geometry"]) for ft in json.load(f)["features"]]))
        except Exception:
            try:
                _LAND = prep(unary_union(_state_shapes()))
            except Exception:
                _LAND = False
    return _LAND or None


def _lines(geom):
    if geom.is_empty:
        return []
    if geom.geom_type in ("LineString", "LinearRing"):
        return [geom]
    if hasattr(geom, "geoms"):
        return [g for sub in geom.geoms for g in _lines(sub)]
    return _lines(geom.boundary)


def load_boundary_segments():
    """State lines + India outline (GoI) + neighbour borders (GoI view), as lon/lat line lists."""
    global _BOUNDARY_SEGMENTS
    if _BOUNDARY_SEGMENTS is not None:
        return _BOUNDARY_SEGMENTS
    geoms = []
    try:
        geoms += [g.simplify(0.01, preserve_topology=True).boundary for g in _state_shapes()]
    except Exception as exc:
        print(f"  (state boundaries unavailable: {exc})")
    try:
        with open(NEIGHBOURS) as f:
            geoms += [shape(ft["geometry"]).simplify(0.01) for ft in json.load(f)["features"]]
    except Exception as exc:
        print(f"  (neighbour borders unavailable: {exc})")
    segs = []
    for g in geoms:
        for line in _lines(g):
            xy = np.asarray(line.coords)
            if len(xy) < 2:
                continue
            segs.append({"lon": np.round(xy[:, 0], 3).tolist(), "lat": np.round(xy[:, 1], 3).tolist()})
    _BOUNDARY_SEGMENTS = segs
    return segs


# =============================================================================
# NOAA OISST v2.1 — DAILY OCEAN SST BACKGROUND  (raster unchanged from V86)
# =============================================================================

OISST_DAILY_BASE = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"
SST_DISPLAY_RES = 0.125
SST_GAUSSIAN_SIGMA = 2.8


def _oisst_daily_urls(date_obj):
    base = f"{OISST_DAILY_BASE}/{date_obj:%Y%m}/"
    return [("NRT", f"{base}oisst-avhrr-v02r01.{date_obj:%Y%m%d}_preliminary.nc"),
            ("FINAL", f"{base}oisst-avhrr-v02r01.{date_obj:%Y%m%d}.nc")]


def _sst_palette_rgb(values):
    """Muted Turbo-style palette: easy on the eyes, professional oceanic tones."""
    v = np.asarray(values, dtype=float)
    stops = np.array([[0.00, 48, 18, 59], [0.12, 70, 100, 220], [0.25, 40, 180, 200], [0.40, 90, 210, 140],
                      [0.55, 180, 220, 90], [0.70, 240, 190, 60], [0.85, 220, 90, 50], [0.95, 150, 30, 30],
                      [1.00, 90, 10, 15]], dtype=float)
    x = np.clip((v - 20.0) / 13.0, 0.0, 1.0)
    rgb = np.stack([np.interp(x, stops[:, 0], stops[:, i]) for i in (1, 2, 3)], axis=-1)
    out = np.rint(rgb).astype(np.uint8)
    out[np.isnan(v)] = [0, 0, 0]
    return out


def _mercator_y(lat):
    lat = np.clip(np.asarray(lat, dtype=float), -85.0, 85.0)
    return np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0))


def _read_oisst_daily_netcdf(nc_path):
    from scipy.ndimage import gaussian_filter, binary_dilation
    ds = xr.open_dataset(nc_path)
    try:
        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        da = ds["sst"].squeeze(drop=True).transpose(lat_name, lon_name)
        lat_values, lon_values = np.asarray(da[lat_name].values, float), np.asarray(da[lon_name].values, float)
        da = da.isel({lat_name: np.where((lat_values >= MAP_SOUTH) & (lat_values <= MAP_NORTH))[0],
                      lon_name: np.where((lon_values >= MAP_WEST) & (lon_values <= MAP_EAST))[0]})
        da = da.sortby(lat_name).sortby(lon_name).astype("float32")
        da = da.where(np.isfinite(da)).where((da > -3.0) & (da < 45.0))
        lat_native, lon_native = np.asarray(da[lat_name].values, float), np.asarray(da[lon_name].values, float)
        native = np.asarray(da.values, dtype=np.float32)

        fine_lat = np.arange(max(MAP_SOUTH, lat_native.min()), min(MAP_NORTH, lat_native.max()) + 1e-4, SST_DISPLAY_RES)
        fine_lon = np.arange(max(MAP_WEST, lon_native.min()), min(MAP_EAST, lon_native.max()) + 1e-4, SST_DISPLAY_RES)
        coords = {lat_name: lat_native, lon_name: lon_native}
        grid = np.asarray(xr.DataArray(native, coords=coords, dims=(lat_name, lon_name))
                          .interp({lat_name: fine_lat, lon_name: fine_lon}, method="linear").values, np.float32)
        valid_fine = np.asarray(xr.DataArray(np.isfinite(native).astype(np.float32), coords=coords,
                                             dims=(lat_name, lon_name))
                                .interp({lat_name: fine_lat, lon_name: fine_lon}, method="nearest").values) > 0.5
        valid_exp = binary_dilation(valid_fine, iterations=15)
        valid = valid_fine & np.isfinite(grid)
        sraw = gaussian_filter(np.where(valid, grid, 0.0).astype(np.float32), sigma=SST_GAUSSIAN_SIGMA, mode="nearest")
        sw = gaussian_filter(valid.astype(np.float32), sigma=SST_GAUSSIAN_SIGMA, mode="nearest")
        smooth = np.divide(sraw, sw, out=np.full_like(sraw, np.nan), where=sw > 0.001)
        smooth[~valid_exp] = np.nan

        hover_lat, hover_lon, hover_sst = [], [], []
        for i in range(0, len(fine_lat), 3):
            for j in range(0, len(fine_lon), 3):
                if np.isfinite(smooth[i, j]) and valid_exp[i, j]:
                    hover_lat.append(round(float(fine_lat[i]), 3))
                    hover_lon.append(round(float(fine_lon[j]), 3))
                    hover_sst.append(round(float(smooth[i, j]), 1))

        y0, y1 = float(_mercator_y(MAP_SOUTH)), float(_mercator_y(MAP_NORTH))
        W = 1400
        H = int(W * ((y1 - y0) / np.deg2rad(MAP_EAST - MAP_WEST)))
        x_lon = np.linspace(MAP_WEST, MAP_EAST, W)
        lat_rows = np.rad2deg(2.0 * np.arctan(np.exp(np.linspace(y1, y0, H))) - np.pi / 2.0)
        lon_stage = np.full((smooth.shape[0], W), np.nan, np.float32)
        mask_stage = np.zeros((smooth.shape[0], W), np.float32)
        for i in range(smooth.shape[0]):
            good = np.isfinite(smooth[i])
            if good.sum() >= 2:
                lon_stage[i] = np.interp(x_lon, fine_lon[good], smooth[i][good], left=np.nan, right=np.nan)
            mask_stage[i] = np.interp(x_lon, fine_lon, valid_exp[i].astype(np.float32), left=0.0, right=0.0)
        raster = np.full((H, W), np.nan, np.float32)
        rmask = np.zeros((H, W), bool)
        for j in range(W):
            good = np.isfinite(lon_stage[:, j])
            if good.sum() >= 2:
                raster[:, j] = np.interp(lat_rows, fine_lat[good], lon_stage[:, j][good], left=np.nan, right=np.nan)
            rmask[:, j] = np.interp(lat_rows, fine_lat, mask_stage[:, j], left=0.0, right=0.0) > 0.5
        ok = np.isfinite(raster) & rmask
        rgba = np.zeros((H, W, 4), np.uint8)
        rgb = _sst_palette_rgb(np.clip(np.nan_to_num(raster, nan=20.0), 20.0, 33.0))
        rgba[..., :3][ok] = rgb[ok]
        rgba[..., 3][ok] = 225
        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
        return {"date": None, "source": None,
                "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
                "min": float(np.nanmin(raster[ok])), "max": float(np.nanmax(raster[ok])),
                "hover_lat": hover_lat, "hover_lon": hover_lon, "hover_sst": hover_sst}
    finally:
        ds.close()


def load_latest_oisst_sst(previous=None):
    """Newest OISST day (preliminary first). Reuses the previous picture when the day hasn't changed."""
    today = utc_now().date()
    for back in range(15):
        d = today - timedelta(days=back)
        for source_name, url in _oisst_daily_urls(d):
            try:
                if requests.head(url, timeout=30, headers=UA, allow_redirects=True).status_code == 404:
                    continue
            except Exception:
                continue
            src = f"NOAA/NCEI OISST v2.1 {source_name}"
            if previous and previous.get("date") == d.isoformat() and previous.get("source") == src \
                    and previous.get("image"):
                print(f"  SST: {d} unchanged")
                return previous
            try:
                r = requests.get(url, timeout=120, headers=UA)
                r.raise_for_status()
                nc = DATA_DIR / Path(url).name
                nc.write_bytes(r.content)
                sst = _read_oisst_daily_netcdf(nc)
                nc.unlink(missing_ok=True)
                sst["date"], sst["source"] = d.isoformat(), src
                print(f"  SST: {d} ({source_name})")
                return sst
            except Exception as exc:
                print(f"  SST {d} {source_name} failed: {exc}")
    return previous or {"date": None, "source": None, "image": None, "hover_lat": [], "hover_lon": [], "hover_sst": []}


# =============================================================================
# OUTPUT
# =============================================================================

def _num(v, nd):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def compact(dataset):
    """{run: [{id, n, m, p: [[tau, lat, lon, mslp, wind], ...]}, ...]} — small JSON for the page."""
    out = {}
    for key, df in dataset.items():
        tracks = []
        if df is not None and not df.empty:
            df = df.copy()
            for c in ("storm_name", "member", "storm_id"):
                if c not in df:
                    df[c] = ""
            df["storm_name"] = df["storm_name"].fillna("").astype(str)
            for (sid, mem), g in df.groupby([df["storm_id"].astype(str), df["member"].astype(str)], sort=False):
                g = g.sort_values("tau")
                name = str(g["storm_name"].iloc[0]).strip()
                tracks.append({"id": sid, "n": "" if name in ("", "nan", "None") else name, "m": mem,
                               "p": [[int(round(float(r.tau))), _num(r.lat, 2), _num(r.lon, 2), _num(r.mslp, 1),
                                      _num(r.wind_kt, 1)] for r in g.itertuples(index=False)]})
        out[str(key)] = tracks
    return out


def build_payload(hres, aifs, ens, aigfs, meta):
    products = {"hres": compact(hres), "aifs": compact(aifs), "ens": compact(ens), "aigfs": compact(aigfs)}
    stamp = "|".join(f"{p}:{','.join(sorted(k for k, v in d.items() if v))}:"
                     f"{sum(len(t['p']) for v in d.values() for t in v)}" for p, d in products.items())
    return {"generated": utc_now().isoformat(timespec="seconds"), "stamp": stamp,
            "runs": {k: meta[k] for k in sorted(meta)}, "products": products,
            "boundary": load_boundary_segments()}


def write_outputs(payload, sst, standalone):
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    if standalone:
        boot = dict(payload, sst=sst)
        out = OUT_DIR / "MASRAINMAN_Tropical_Tracker.html"
        out.write_text(tpl.replace("__BOOT__", json.dumps(boot, separators=(",", ":"))), encoding="utf-8")
        return out
    (OUT_DIR / "index.html").write_text(tpl.replace("__BOOT__", "null"), encoding="utf-8")
    (OUT_DIR / "data.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    (OUT_DIR / "sst.json").write_text(json.dumps(sst, separators=(",", ":")), encoding="utf-8")
    (OUT_DIR / "stamp.json").write_text(json.dumps(
        {"stamp": payload["stamp"], "generated": payload["generated"], "sst_date": sst.get("date")}),
        encoding="utf-8")
    return OUT_DIR / "index.html"


def run_once(standalone=False, open_browser=False):
    print_banner(APP_NAME)
    old_stamp = None
    try:
        old_stamp = json.loads((OUT_DIR / "stamp.json").read_text())
    except Exception:
        pass
    hres, aifs, ens, meta = load_ecmwf()
    aigfs, aigfs_meta = load_aigfs()
    meta.update(aigfs_meta)
    prune_cache({"hres": set(hres), "aifs": set(hres), "ens": set(hres),
                 "aigfs": set(aigfs_meta)})

    print_banner("SST")
    previous_sst = None
    try:
        previous_sst = json.loads((OUT_DIR / "sst.json").read_text())
    except Exception:
        pass
    sst = load_latest_oisst_sst(previous_sst)

    payload = build_payload(hres, aifs, ens, aigfs, meta)
    page = write_outputs(payload, sst, standalone)
    changed = (old_stamp is None or old_stamp.get("stamp") != payload["stamp"]
               or old_stamp.get("sst_date") != sst.get("date"))
    with open(OUT_DIR / "changed.txt", "w") as f:
        f.write("true" if changed else "false")
    print_banner(f"TRACKER READY — {'new runs' if changed else 'no new runs'} — {page}")
    if open_browser:
        try:
            webbrowser.open(page.resolve().as_uri())
        except Exception:
            print(f"Open this file manually: {page}")
    return changed


def main():
    global OUT_DIR, STATE_DIR, DATA_DIR
    ap = argparse.ArgumentParser(description=APP_NAME)
    ap.add_argument("--out", default=os.environ.get("MASRAINMAN_TRACK_ROOT", "site_tracker"),
                    help="output folder (page, data, state)")
    ap.add_argument("--standalone", action="store_true", help="one self-contained HTML (for the PC)")
    ap.add_argument("--open", action="store_true", help="open the page in the browser when done")
    ap.add_argument("--every", type=float, default=0, help="keep running: new check every N hours")
    a = ap.parse_args()
    OUT_DIR = Path(a.out)
    STATE_DIR = OUT_DIR / "state"
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = Path(tempfile.mkdtemp(prefix="mr_tracker_"))
    try:
        first = True
        while True:
            try:
                run_once(a.standalone, a.open and first)
            except Exception as exc:
                print(f"Tracker run failed: {type(exc).__name__}: {exc}")
                if not a.every:
                    raise
            first = False
            if not a.every:
                break
            nxt = utc_now() + timedelta(hours=a.every)
            print(f"Next check {nxt.astimezone(timezone(timedelta(hours=5, minutes=30))):%d %b %H:%M} IST")
            time.sleep(a.every * 3600)
    finally:
        shutil.rmtree(DATA_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
