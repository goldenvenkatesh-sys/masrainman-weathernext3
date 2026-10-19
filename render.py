# ============================================================
# WEATHERNext 3 — V3 SURFACE SYSTEM TRACKING — AUTO RENDERER
# ============================================================
#
# GitHub Actions version of the MasRainman Colab script.
#
# Every time it runs it:
#   1. Lists WeatherNext 3 full-length 00/06/12/18 UTC runs
#   2. Picks the newest runs that are not yet on the website
#   3. Checks the run is complete (all 360 h written)
#   4. Loads regional MSLP / U10 / V10 (55–100E, 2–28N)
#   5. Runs V3 surface-system detection + persistent tracking
#   6. Draws T+24, T+48 … T+360 maps, with and without
#      mean 10-m wind barbs (same design as the Colab map)
#   7. Builds an animated GIF for each layer
#   8. Writes CSVs + systems JSON + site/data/manifest.json
#   9. Keeps only the latest KEEP_RUNS runs on the website
#
# Usage:
#   python render.py                     # automatic (latest runs)
#   python render.py --run 20261019_00hr_01_preds   # force one run
#   python render.py --demo              # synthetic data test
#
# ============================================================

import os
import re
import sys
import json
import shutil
import argparse
import warnings

from datetime import datetime, timedelta, timezone

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

# Website output folder (published to GitHub Pages)
SITE_DIR = os.environ.get("SITE_DIR", os.path.join(HERE, "site"))

# How many runs stay selectable on the website
KEEP_RUNS = int(os.environ.get("KEEP_RUNS", "4"))

# Max new runs rendered in one job (protects job time)
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

# India national boundary as per Government of India
# (land borders only, from Natural Earth "India point-of-view" countries)
BOUNDARY_FILE = os.path.join(HERE, "boundaries", "borders_india_view.geojson")


# ============================================================
# V3 DETECTION PARAMETERS (unchanged from the Colab script)
# ============================================================

MINIMUM_FILTER_SIZE = 21
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


# ============================================================
# GOOGLE CLOUD ACCESS
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
    m = re.match(RUN_PATTERN, run_name)
    if not m:
        raise RuntimeError(f"Unable to decode run time: {run_name}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)


# ============================================================
# DATA LOADING
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

    return {
        "lat": lat_full[lat_start:lat_end],
        "lon": lon_full[lon_start:lon_end],
        "mslp": mslp,
        "u10": u10,
        "v10": v10,
        "n_hours": n_hours,
    }


def make_demo_data(n_hours=360):
    """Synthetic fields for testing the pipeline without Google Cloud."""
    lat = np.round(np.arange(LAT_MIN, LAT_MAX + 0.001, 0.1), 1)
    lon = np.round(np.arange(LON_MIN, LON_MAX + 0.001, 0.1), 1)
    LON2, LAT2 = np.meshgrid(lon, lat)
    mslp = np.empty((n_hours, len(lat), len(lon)), np.float32)
    u10 = np.empty_like(mslp)
    v10 = np.empty_like(mslp)
    for t in range(n_hours):
        clat = 11.0 + 0.02 * t
        clon = 90.0 - 0.045 * t
        r2 = ((LAT2 - clat) ** 2 + ((LON2 - clon) * np.cos(np.radians(clat))) ** 2)
        depth = 4 + 10 * np.sin(np.pi * t / n_hours)
        p = 1008 + 0.25 * (LAT2 - 15) - depth * np.exp(-r2 / 6.0)
        mslp[t] = p
        dy = LAT2 - clat
        dx = (LON2 - clon) * np.cos(np.radians(clat))
        speed = 2.2 * depth * np.sqrt(r2) * np.exp(-r2 / 6.0)
        rr = np.sqrt(r2) + 1e-6
        u10[t] = -speed * dy / rr - 0.3 * dx / rr * speed - 4
        v10[t] = speed * dx / rr - 0.3 * dy / rr * speed
    return {"lat": lat, "lon": lon, "mslp": mslp, "u10": u10, "v10": v10, "n_hours": n_hours}


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


def detect_v3_candidates(hour, mslp, u, v, LAT2D, LON2D):
    candidates = []

    local_min = mslp == minimum_filter(mslp, size=MINIMUM_FILTER_SIZE, mode="nearest")

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
    lat_sub, lon_sub = data["lat"], data["lon"]
    LON2D, LAT2D = np.meshgrid(lon_sub, lat_sub)
    n_hours = data["n_hours"]

    print("\nRUNNING V3 SURFACE SYSTEM DETECTION")
    by_hour = []
    for hour in range(n_hours):
        if hour % 48 == 0:
            print(f"  forecast hour {hour:03d}/{n_hours - 1:03d}")
        by_hour.append(detect_v3_candidates(
            hour, data["mslp"][hour], data["u10"][hour], data["v10"][hour], LAT2D, LON2D))

    all_candidates = [c for h in by_hour for c in h]
    print("V3 candidate records:", len(all_candidates))

    active_tracks, finished_tracks, next_track_id = {}, [], 1

    for hour in range(n_hours):
        hour_candidates = by_hour[hour]

        if hour == 0:
            for c in hour_candidates:
                c = dict(c, track_id=next_track_id)
                active_tracks[next_track_id] = [c]
                next_track_id += 1
            continue

        previous_tracks = dict(active_tracks)
        active_tracks = {}
        assigned_ci = set()
        possible = []

        for tid, pts in previous_tracks.items():
            last = pts[-1]
            for ci, c in enumerate(hour_candidates):
                d = haversine_km(last["lat"], last["lon"], c["lat"], c["lon"])
                if d <= MAX_MOVEMENT_KM_PER_HOUR:
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

    persistent = [t for t in finished_tracks if len(t) >= MIN_TRACK_HOURS]
    print("Raw tracks:", len(finished_tracks), "| Persistent tracks:", len(persistent))

    candidates_df = pd.DataFrame(all_candidates)
    tracks_df = pd.DataFrame([p for t in persistent for p in t])
    if tracks_df.empty:
        tracks_df = pd.DataFrame(columns=[
            "hour", "lat", "lon", "mslp_hpa", "pressure_depression_hpa",
            "circulation", "max_wind_kt", "convergence", "score", "track_id"])
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


def draw_map(out_png, run_name, init_time, plot_hour, data, tracks_df,
             n_persistent, plot_windbarbs):

    plot_index = plot_hour - 1  # +24 h -> array index 23
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
        step = 10  # 0.1° grid -> ~1°
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
        "WeatherNext 3 — Mean MSLP with V3 Low Centres\n"
        f"Valid: {valid_time.strftime('%Y-%m-%d %H:%M UTC')} (T+{plot_hour} h)",
        fontsize=18, fontweight="bold", pad=12)

    info_text = (
        "WeatherNext 3 Statistics Mean\n"
        f"Run: {run_name}\n"
        f"Initialisation: {init_time.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Forecast: +{plot_hour} h\n"
        f"V3 persistent tracks: {n_persistent}\n"
        f"Centres at valid time: {len(selected_centres)}"
    )
    if plot_windbarbs:
        info_text += "\nMean 10-m wind barbs: ON"

    ax.text(0.012, 0.985, info_text, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="gray", alpha=0.82),
            zorder=30)

    cax = fig.add_axes([0.915, 0.18, 0.018, 0.58])
    cbar = fig.colorbar(cf, cax=cax, orientation="vertical")
    cbar.set_label("Mean Sea Level Pressure (hPa)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.text(0.5, 0.045, "WeatherNext 3 © 2026 DeepMind Technologies Limited | Experimental model output",
             ha="center", va="bottom", fontsize=7.5, color="dimgray")
    fig.text(0.5, 0.027,
             "Historical WeatherNext 3 data: CC BY 4.0 | "
             "Future/real-time data: applicable GDM experimental-data terms | "
             "V3 analysis & visualisation: MasRainman",
             ha="center", va="bottom", fontsize=7.0, color="dimgray")
    fig.text(0.5, 0.010,
             "WeatherNext 3 predictions are experimental and are not official weather warnings or forecasts.",
             ha="center", va="bottom", fontsize=6.8, color="dimgray")

    plt.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)

    return selected_centres, valid_time


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

def render_run(run_name, data):
    init_time = decode_init_time(run_name)
    run_dir = os.path.join(SITE_DIR, "runs", run_name)
    tmp_dir = run_dir + ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)

    candidates_df, tracks_df, n_persistent = run_tracking(data)

    candidates_df.to_csv(os.path.join(tmp_dir, f"weathernext3_{run_name}_V3_candidates.csv"), index=False)
    tracks_df.to_csv(os.path.join(tmp_dir, f"weathernext3_{run_name}_V3_persistent_tracks.csv"), index=False)

    last_hour = min(FRAME_END_H, data["n_hours"])
    hours = list(range(FRAME_STEP_H, last_hour + 1, FRAME_STEP_H))

    frames = []
    pngs = {"mslp": [], "barbs": []}

    for h in hours:
        print(f"Drawing T+{h:03d} ...")
        entry = {"hour": h}
        for layer, barbs in (("mslp", False), ("barbs", True)):
            fname = f"{layer}_T{h:03d}.png"
            centres, valid_time = draw_map(os.path.join(tmp_dir, fname), run_name, init_time, h,
                                           data, tracks_df, n_persistent, barbs)
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
        gname = f"weathernext3_{run_name}_{layer}_loop.gif"
        make_gif(pngs[layer], os.path.join(tmp_dir, gname))
        gifs[layer] = gname

    run_info = {
        "run": run_name,
        "init": init_time.strftime("%Y-%m-%dT%H:%MZ"),
        "persistent_tracks": n_persistent,
        "frames": frames,
        "gifs": gifs,
        "csv": {
            "candidates": f"weathernext3_{run_name}_V3_candidates.csv",
            "tracks": f"weathernext3_{run_name}_V3_persistent_tracks.csv",
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

def rendered_runs():
    runs_root = os.path.join(SITE_DIR, "runs")
    if not os.path.isdir(runs_root):
        return []
    out = []
    for name in os.listdir(runs_root):
        if re.match(RUN_PATTERN, name) and os.path.exists(os.path.join(runs_root, name, "run.json")):
            out.append(name)
    return sorted(out, reverse=True)


def write_manifest():
    runs_root = os.path.join(SITE_DIR, "runs")
    runs = rendered_runs()

    # Keep only the latest KEEP_RUNS
    for old in runs[KEEP_RUNS:]:
        print("Removing old run", old)
        shutil.rmtree(os.path.join(runs_root, old), ignore_errors=True)
    runs = runs[:KEEP_RUNS]

    manifest = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "runs": [],
    }
    for r in runs:
        with open(os.path.join(runs_root, r, "run.json")) as f:
            manifest["runs"].append(json.load(f))

    os.makedirs(os.path.join(SITE_DIR, "data"), exist_ok=True)
    with open(os.path.join(SITE_DIR, "data", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    # Copy the viewer page
    for fn in ("index.html",):
        src = os.path.join(HERE, "web", fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(SITE_DIR, fn))
    open(os.path.join(SITE_DIR, ".nojekyll"), "w").close()
    print("Manifest written with", len(runs), "runs")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Render this run name (forces re-render)")
    ap.add_argument("--demo", action="store_true", help="Synthetic data test, no Google Cloud")
    args = ap.parse_args()

    os.makedirs(SITE_DIR, exist_ok=True)

    if args.demo:
        name = datetime.now(timezone.utc).strftime("%Y%m%d") + "_00hr_01_preds"
        render_run(name, make_demo_data())
        write_manifest()
        return

    session = get_session()

    if args.run:
        targets = [args.run]
    else:
        available = list_weathernext_runs(session)
        if not available:
            raise RuntimeError("No full-length WeatherNext 3 runs were found.")
        print("Latest available:", ", ".join(available[:6]))
        done = set(rendered_runs())
        targets = [r for r in available[:KEEP_RUNS] if r not in done]

    if not targets:
        print("Nothing new to render.")
        write_manifest()
        return

    rendered = 0
    for run_name in targets:
        if rendered >= MAX_NEW_RUNS_PER_JOB:
            break
        data = load_run(run_name)
        if data is None:
            continue
        render_run(run_name, data)
        rendered += 1
        del data

    write_manifest()
    print(f"\nRendered {rendered} new run(s).")


if __name__ == "__main__":
    main()
