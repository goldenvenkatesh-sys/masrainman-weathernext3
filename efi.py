# ============================================================
# MASRAINMAN — EXTREME FORECAST INDEX (EFI) FOR INDIA, DAY 1 TO DAY 7
#   WeatherNext 3 statistics  vs  ERA5 daily climate (1991–2020)
# ============================================================
#
# ECMWF-style Extreme Forecast Index and Shift of Tails for:
#
#   rain24  24-h rainfall               heavy-rain signal  (EFI > 0, SOT upper)
#   tmax    daily maximum temperature   heat / cool days   (EFI ±, SOT upper)
#   tmin    daily minimum temperature   warm / cold nights (EFI ±, SOT upper + lower)
#   tmean   daily mean temperature      (EFI ±)
#
# Day n = the 24 h from T+24(n-1) to T+24n, n = 1..7.
#
# How the forecast distribution is built
#   WeatherNext 3 publishes, for every hour, the ensemble mean and the
#   10/25/50/75/90th percentiles — not the members. For each percentile
#   the 24 hourly values are summed (rain), or the max / min / mean is taken
#   (temperature). This treats the members' ranks as fixed through the day
#   ("comonotonic"): exact for the mean, good for temperature, and for rain
#   it widens the spread somewhat (the wet percentiles come out a little
#   high). Between percentiles the CDF is linear, beyond 10 % / 90 % the
#   end slopes continue to 0 % / 100 %.
#
# How the climate is built: see efi_climate.py (ERA5 0.25°, ±15-day window,
# 30 years -> 930 days per point). The forecast is sampled to that grid.
#
# Output (site/efi/<run>/):
#   <var>_D<n>.png      maps, EFI shaded, SOT contoured
#   <var>_loop.gif      Day 1..7 loop for each variable
#   efi_<run>.npz       EFI / SOT grids (float16) for your own analysis
#   run.json            what was drawn + the biggest signals each day
# site/data/manifest_efi.json and site/efi.html (viewer).
#
# Usage:
#   python efi.py                                 # latest WeatherNext 3 runs
#   python efi.py --run 20261022_00hr_01_preds    # one run
#   python efi.py --demo                          # synthetic test, no WeatherNext login
# ============================================================

import os
import re
import sys
import json
import shutil
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs

import efi_core
import efi_climate
from render import (SITE_DIR, KEEP_RUNS, RUN_PATTERN, HERE, get_session, list_weathernext_runs,
                    open_run_zarr, decode_init_time, india_view_borders, make_gif, MODELS)

EFI_DIR = os.path.join(SITE_DIR, "efi")
EFI_MANIFEST = "manifest_efi.json"
EFI_DAYS = 7
MAX_NEW_RUNS = int(os.environ.get("MAX_NEW_RUNS", "2"))

WN3_PROBS = [0.10, 0.25, 0.50, 0.75, 0.90]
WN3_PCT = ["p10", "p25", "p50", "p75", "p90"]
# Hourly WeatherNext 3 fields behind each EFI variable
WN3_T2M = "temperature_2m"
WN3_RAIN = os.environ.get("WN3_EFI_RAIN", "total_precipitation_1hr")  # or imerg_tp_1hr

EFI_VARS = {
    "rain24": {"label": "24-h rainfall", "short": "Rain 24 h", "units": "mm",
               "lower_bound": 0.0, "two_sided": False, "sot": ("upper",)},
    "tmax": {"label": "Maximum temperature", "short": "Tmax", "units": "°C",
             "lower_bound": None, "two_sided": True, "sot": ("upper",)},
    "tmin": {"label": "Minimum temperature", "short": "Tmin", "units": "°C",
             "lower_bound": None, "two_sided": True, "sot": ("upper", "lower")},
    "tmean": {"label": "Mean temperature", "short": "Tmean", "units": "°C",
              "lower_bound": None, "two_sided": True, "sot": ()},
}
EFI_ORDER = ["rain24", "tmax", "tmin", "tmean"]

# ECMWF-like EFI colours: |EFI| < 0.5 left blank
EFI_LEVELS = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EFI_COLOURS = ["#08306b", "#1c5aa6", "#3f8fd2", "#7cbbe8", "#bfdff5",   # cold / dry side
               (0, 0, 0, 0),                                             # -0.5 .. 0.5: none
               "#fff07a", "#ffbf3a", "#ff7a1a", "#e3261b", "#9e0a5e"]    # warm / wet side
SOT_LEVELS = [0, 1, 2, 4]
STRONG_EFI = 0.8
RAIN_MIN_MM = 1.0          # wet EFI needs forecast 90th percentile >= this (mm / 24 h)


# ============================================================
# FORECAST: WeatherNext 3 percentiles -> daily percentiles
# ============================================================

def _nearest(src, targets):
    src = np.asarray(src, dtype=float)
    return np.array([int(np.argmin(np.abs(src - t))) for t in targets])


def load_wn3_daily(run_name, clim_lat, clim_lon):
    """{var: (5 percentiles, 7 days, ny, nx)} on the climate grid, or None if incomplete."""
    from scipy.ndimage import uniform_filter

    print(f"\nOpening WeatherNext 3 Zarr for {run_name} ...")
    root = open_run_zarr(run_name)
    lat_full = np.asarray(root["lat_0p1"][:])
    lon_full = np.asarray(root["lon_0p1"][:])
    pad = 0.3
    iy = np.where((lat_full >= clim_lat.min() - pad) & (lat_full <= clim_lat.max() + pad))[0]
    ix = np.where((lon_full >= clim_lon.min() - pad) & (lon_full <= clim_lon.max() + pad))[0]
    lat, lon = lat_full[iy[0]:iy[-1] + 1], lon_full[ix[0]:ix[-1] + 1]
    sy, sx = _nearest(lat, clim_lat), _nearest(lon, clim_lon)
    n_h = 24 * EFI_DAYS

    probe_arr = root[f"{WN3_T2M}_p50"]
    if probe_arr.shape[0] < n_h:
        print(f"Run has only {probe_arr.shape[0]} hours — skipping for now.")
        return None
    probe = float(np.asarray(probe_arr[n_h - 1, (iy[0] + iy[-1]) // 2, (ix[0] + ix[-1]) // 2]))
    if not np.isfinite(probe) or not 150 < probe < 350:
        print(f"Run not complete yet (T+{n_h} probe = {probe}) — skipping for now.")
        return None

    def daily(name, how):
        sl = (slice(0, n_h), slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1))
        hourly = np.asarray(root[name][sl], dtype=np.float32).reshape(EFI_DAYS, 24, len(lat), len(lon))
        day = how(hourly)                                   # (7, lat, lon)
        # ~0.3° box average, then sample at the 0.25° climate points
        day = uniform_filter(np.nan_to_num(day), size=(1, 3, 3), mode="nearest")
        return day[:, sy][:, :, sx]

    out = {v: [] for v in EFI_ORDER}
    for pct in WN3_PCT:
        name = f"{WN3_RAIN}_{pct}"
        print(f"  {name} -> daily totals")
        out["rain24"].append(daily(name, lambda h: np.maximum(h, 0).sum(axis=1) * 1000.0))
        name = f"{WN3_T2M}_{pct}"
        print(f"  {name} -> daily max / min / mean")
        sl = (slice(0, n_h), slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1))
        hourly = np.asarray(root[name][sl], dtype=np.float32).reshape(EFI_DAYS, 24, len(lat), len(lon)) - 273.15
        for var, how in (("tmax", np.max), ("tmin", np.min), ("tmean", np.mean)):
            day = uniform_filter(np.nan_to_num(how(hourly, axis=1)), size=(1, 3, 3), mode="nearest")
            out[var].append(day[:, sy][:, :, sx])
        del hourly
    return {v: np.stack(out[v]).astype(np.float32) for v in EFI_ORDER}


# ============================================================
# EFI / SOT FOR ONE RUN
# ============================================================

def compute_run(init_time, fc, climates):
    """fc {var: (5, 7, ny, nx)}; climates {(var, window): (q, lat, lon)}.
    Returns {var: {"efi": (7, ny, nx), "sot_upper": ..., "sot_lower": ..., "windows": [...]}}."""
    result = {}
    for var in EFI_ORDER:
        cfg = EFI_VARS[var]
        efi_days, sotu, sotl, windows = [], [], [], []
        for d in range(EFI_DAYS):
            mid = (init_time + timedelta(hours=24 * d + 12)).date()
            w = efi_climate.nearest_window(mid)
            q = climates[(var, w)][0]
            f = fc[var][:, d]
            e = efi_core.efi_from_percentiles(q, f, WN3_PROBS, cfg["lower_bound"])
            if var == "rain24":
                # In a near-rainless climate a trace of rain is "extreme"; do not flag
                # a wet signal unless the forecast 90th percentile is at least RAIN_MIN_MM.
                e = np.where(f[4] < RAIN_MIN_MM, np.minimum(e, 0), e)
            efi_days.append(e)
            min_spread = 1.0 if var == "rain24" else 0.1
            sotu.append(efi_core.sot_upper(q, f[4], min_spread))
            sotl.append(efi_core.sot_lower(q, f[0], min_spread))
            windows.append(f"{w[0]:02d}-{w[1]:02d}")
        result[var] = {"efi": np.stack(efi_days), "sot_upper": np.stack(sotu),
                       "sot_lower": np.stack(sotl), "windows": windows}
    return result


def needed_windows(init_time):
    return sorted({efi_climate.nearest_window((init_time + timedelta(hours=24 * d + 12)).date())
                   for d in range(EFI_DAYS)})


def load_climates(init_time):
    out, missing = {}, []
    for var in EFI_ORDER:
        for w in needed_windows(init_time):
            c = efi_climate.load_clim(var, w)
            if c is None:
                missing.append(os.path.basename(efi_climate.clim_path(var, w)))
            else:
                out[(var, w)] = c
    return out, missing


# ============================================================
# MAPS
# ============================================================

def efi_cmap(two_sided):
    colours = list(EFI_COLOURS)
    if not two_sided:
        colours[:5] = [(0, 0, 0, 0)] * 5
    cmap = ListedColormap(colours)
    return cmap, BoundaryNorm(EFI_LEVELS, cmap.N)


def draw_efi_map(out_png, var, day, run_name, init_time, lat, lon, efi, sotu, sotl, window):
    cfg = EFI_VARS[var]
    t0 = init_time + timedelta(hours=24 * day)
    t1 = t0 + timedelta(hours=24)
    cmap, norm = efi_cmap(cfg["two_sided"])

    fig = plt.figure(figsize=(12, 11), facecolor="white")
    ax = fig.add_axes([0.02, 0.15, 0.96, 0.77], projection=ccrs.PlateCarree())
    ax.set_extent([float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
                  crs=ccrs.PlateCarree())
    ax.contourf(lon, lat, efi, levels=EFI_LEVELS, cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(), zorder=1)
    if "upper" in cfg["sot"]:
        cs = ax.contour(lon, lat, np.nan_to_num(sotu, nan=-9), levels=SOT_LEVELS, colors="black",
                        linewidths=[0.9, 1.3, 1.7, 2.1], transform=ccrs.PlateCarree(), zorder=8)
        ax.clabel(cs, fontsize=7.5, fmt="%g", inline_spacing=2)
    if "lower" in cfg["sot"]:
        cs = ax.contour(lon, lat, np.nan_to_num(sotl, nan=-9), levels=SOT_LEVELS, colors="#0b2e7a",
                        linewidths=[0.9, 1.3, 1.7, 2.1], linestyles="dashed",
                        transform=ccrs.PlateCarree(), zorder=8)
        ax.clabel(cs, fontsize=7.5, fmt="%g", inline_spacing=2)
    ax.coastlines(resolution="50m", linewidth=0.7, color="black", zorder=5)
    ax.add_feature(india_view_borders(), facecolor="none", edgecolor="black", linewidth=0.5, zorder=6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="gray", alpha=0.4, linestyle="--", zorder=7)
    gl.top_labels = False
    gl.right_labels = False

    fig.text(0.02, 0.965, f"WeatherNext 3 · Extreme Forecast Index · {cfg['label']}",
             ha="left", va="center", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.940,
             f"Day {day + 1}: {t0.strftime('%Hz %a %d %b')} to {t1.strftime('%Hz %a %d %b %Y')}"
             f"  (T+{24 * day} to T+{24 * day + 24})   Init {init_time.strftime('%Hz %d %b %Y')}",
             ha="left", va="center", fontsize=11.5)
    lines = ["Shaded: EFI (blank between -0.5 and 0.5)"]
    if "upper" in cfg["sot"]:
        lines.append("Black contours: Shift of Tails (90th) 0, 1, 2, 4")
    if "lower" in cfg["sot"]:
        lines.append("Blue dashed: Shift of Tails (10th, cold side)")
    lines.append(f"Climate: ERA5 1991–2020, window centred {window}")
    ax.text(0.012, 0.985, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85), zorder=30)

    cax = fig.add_axes([0.12, 0.095, 0.76, 0.02])
    sm = plt.cm.ScalarMappable(cmap=ListedColormap([c if c != (0, 0, 0, 0) else "#ffffff"
                                                    for c in efi_cmap(cfg["two_sided"])[0].colors]),
                               norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=EFI_LEVELS)
    cb.set_label("Extreme Forecast Index", fontsize=10)
    cb.ax.tick_params(labelsize=8.5)
    fig.text(0.5, 0.030, MODELS["wn3"]["credits"][0] + " | Climate: ERA5 (Copernicus) | EFI: MasRainman",
             ha="center", fontsize=7.3, color="dimgray")
    fig.text(0.5, 0.013, MODELS["wn3"]["credits"][-1], ha="center", fontsize=7.0, color="dimgray")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def hotspot(lat, lon, efi, sotu):
    """Biggest signal of one map: max EFI, where, and how many 0.25° points reach STRONG_EFI."""
    i = np.unravel_index(int(np.nanargmax(efi)), efi.shape)
    return {
        "max_efi": round(float(efi[i]), 2),
        "lat": round(float(lat[i[0]]), 2),
        "lon": round(float(lon[i[1]]), 2),
        "sot": None if not np.isfinite(sotu[i]) else round(float(sotu[i]), 2),
        "points_ge_%g" % STRONG_EFI: int(np.sum(efi >= STRONG_EFI)),
        "min_efi": round(float(np.nanmin(efi)), 2),
    }


# ============================================================
# ONE RUN -> WEBSITE FILES
# ============================================================

def render_efi_run(run_name, init_time, lat, lon, res):
    run_dir = os.path.join(EFI_DIR, run_name)
    tmp = run_dir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    variables = []
    for var in EFI_ORDER:
        r = res[var]
        pngs, days = [], []
        for d in range(EFI_DAYS):
            fname = f"{var}_D{d + 1}.png"
            print(f"[EFI] {var} Day {d + 1}")
            draw_efi_map(os.path.join(tmp, fname), var, d, run_name, init_time, lat, lon,
                         r["efi"][d], r["sot_upper"][d], r["sot_lower"][d], r["windows"][d])
            pngs.append(os.path.join(tmp, fname))
            t0 = init_time + timedelta(hours=24 * d)
            days.append({"day": d + 1, "png": fname,
                         "start": t0.strftime("%Y-%m-%dT%H:%MZ"),
                         "end": (t0 + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%MZ"),
                         "climate_window": r["windows"][d],
                         **hotspot(lat, lon, r["efi"][d], r["sot_upper"][d])})
        gif = f"{var}_loop.gif"
        make_gif(pngs, os.path.join(tmp, gif))
        variables.append({"id": var, "label": EFI_VARS[var]["label"], "short": EFI_VARS[var]["short"],
                          "gif": gif, "days": days})

    npz = f"efi_{run_name}.npz"
    np.savez_compressed(os.path.join(tmp, npz), lat=lat, lon=lon,
                        **{f"{v}_{k}": res[v][k].astype(np.float16)
                           for v in EFI_ORDER for k in ("efi", "sot_upper", "sot_lower")})

    info = {
        "run": run_name,
        "init": init_time.strftime("%Y-%m-%dT%H:%MZ"),
        "model": "WeatherNext 3 statistics (10/25/50/75/90th percentiles)",
        "climate": "ERA5 daily 1991-2020, ±15-day window, 0.25°",
        "variables": variables,
        "npz": npz,
        "rendered": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
    }
    with open(os.path.join(tmp, "run.json"), "w") as f:
        json.dump(info, f, indent=1)
    shutil.rmtree(run_dir, ignore_errors=True)
    os.rename(tmp, run_dir)
    print(f"[EFI] {run_name} done -> {run_dir}")
    return info


def rendered_efi_runs():
    if not os.path.isdir(EFI_DIR):
        return []
    return sorted((n for n in os.listdir(EFI_DIR) if re.match(RUN_PATTERN, n)
                   and os.path.exists(os.path.join(EFI_DIR, n, "run.json"))), reverse=True)


def write_efi_manifest():
    runs = rendered_efi_runs()
    for old in runs[KEEP_RUNS:]:
        print("[EFI] Removing old run", old)
        shutil.rmtree(os.path.join(EFI_DIR, old), ignore_errors=True)
    manifest = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "runs": []}
    for r in runs[:KEEP_RUNS]:
        with open(os.path.join(EFI_DIR, r, "run.json")) as f:
            manifest["runs"].append(json.load(f))
    os.makedirs(os.path.join(SITE_DIR, "data"), exist_ok=True)
    with open(os.path.join(SITE_DIR, "data", EFI_MANIFEST), "w") as f:
        json.dump(manifest, f, indent=1)
    page = os.path.join(HERE, "web", "efi.html")
    if os.path.exists(page):
        shutil.copy(page, os.path.join(SITE_DIR, "efi.html"))
    print(f"[EFI] Manifest written with {len(manifest['runs'])} runs")


# ============================================================
# DEMO (no downloads): synthetic climate + a heavy-rain / heat event
# ============================================================

def make_demo(init_time):
    lat = np.arange(efi_climate.EFI_LAT_MIN, efi_climate.EFI_LAT_MAX + 0.01, 0.25)
    lon = np.arange(efi_climate.EFI_LON_MIN, efi_climate.EFI_LON_MAX + 0.01, 0.25)
    LON, LAT = np.meshgrid(lon, lat)
    p = np.clip(efi_climate.np.linspace(0, 1, 101), 0.002, 0.998)[:, None, None]
    from scipy.stats import norm, gamma

    # Climate: rain gamma (wetter on the west coast / northeast), temperatures normal
    wet = 4 + 10 * np.exp(-((LON - 74) ** 2) / 4) * (LAT < 20) + 8 * np.exp(-((LON - 92) ** 2 + (LAT - 25) ** 2) / 8)
    dry_frac = np.clip(0.75 - wet / 30, 0.2, 0.9)
    pr = np.clip((p - dry_frac) / (1 - dry_frac), 0, 0.999)
    rain_q = np.where(p > dry_frac, gamma.ppf(pr, 0.8, scale=wet), 0)
    tbase = 34 - 0.35 * (LAT - 20) ** 2 / 10
    clim = {"rain24": rain_q, "tmax": tbase + 2.5 * norm.ppf(p),
            "tmin": tbase - 10 + 2.0 * norm.ppf(p), "tmean": tbase - 5 + 2.0 * norm.ppf(p)}
    climates = {(v, w): (clim[v].astype(np.float32), lat, lon)
                for v in EFI_ORDER for w in efi_climate.WINDOW_CENTRES}

    # Forecast: a depression moving NW across Odisha with heavy rain; heat over the NW
    z = norm.ppf(WN3_PROBS)[:, None, None, None]
    days = np.arange(EFI_DAYS)[None, :, None, None]
    cy, cx = 19 + 0.8 * days, 87 - 1.2 * days
    blob = np.exp(-((LON - cx) ** 2 + (LAT - cy) ** 2) / 6)
    rain_mu = wet * 0.6 + 120 * blob
    fc_rain = np.maximum(rain_mu * (1 + 0.45 * z), 0)
    heat = 5 * np.exp(-((LON - 73) ** 2 + (LAT - 27) ** 2) / 20) * np.minimum(days + 1, 4) / 4
    cold = -4 * np.exp(-((LON - 78) ** 2 + (LAT - 33) ** 2) / 10)
    fc = {"rain24": fc_rain,
          "tmax": tbase + heat + 1.2 * z + 0 * days,
          "tmin": tbase - 10 + heat * 0.8 + cold + 1.0 * z + 0 * days,
          "tmean": tbase - 5 + heat * 0.9 + 1.0 * z + 0 * days}
    return lat, lon, {v: np.broadcast_to(fc[v], (5, EFI_DAYS) + LAT.shape).astype(np.float32)
                      for v in EFI_ORDER}, climates


# ============================================================
# MAIN
# ============================================================

def process_run(run_name):
    init_time = decode_init_time(run_name)
    climates, missing = load_climates(init_time)
    if missing:
        raise RuntimeError(f"Climate files missing in {efi_climate.CLIM_DIR}: {', '.join(missing)} "
                           "— run the 'EFI climate (one-time)' workflow first.")
    any_c = next(iter(climates.values()))
    lat, lon = any_c[1], any_c[2]
    fc = load_wn3_daily(run_name, lat, lon)
    if fc is None:
        return False
    res = compute_run(init_time, fc, climates)
    render_efi_run(run_name, init_time, lat, lon, res)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="WeatherNext 3 run, e.g. 20261022_00hr_01_preds")
    ap.add_argument("--demo", action="store_true", help="Synthetic data, no downloads")
    args = ap.parse_args()
    os.makedirs(EFI_DIR, exist_ok=True)

    if args.demo:
        init = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        run_name = init.strftime("%Y%m%d") + "_00hr_01_preds"
        lat, lon, fc, climates = make_demo(init)
        render_efi_run(run_name, init, lat, lon, compute_run(init, fc, climates))
        write_efi_manifest()
        return

    if args.run:
        targets = [args.run]
    else:
        available = list_weathernext_runs(get_session())
        done = set(rendered_efi_runs())
        targets = [r for r in available[:KEEP_RUNS] if r not in done]
        print("[EFI] To do:", ", ".join(targets) or "(nothing new)")

    made = 0
    for run_name in targets:
        if made >= MAX_NEW_RUNS:
            break
        if process_run(run_name):
            made += 1
    write_efi_manifest()
    print(f"\n[EFI] Rendered {made} new run(s).")


if __name__ == "__main__":
    sys.exit(main())
