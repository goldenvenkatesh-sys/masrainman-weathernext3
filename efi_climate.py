# ============================================================
# EFI CLIMATE — ERA5 daily percentiles for India (1991–2020)
# ============================================================
#
# The EFI compares a forecast with a climate. ECMWF uses its own
# re-forecasts ("M-climate"); WeatherNext 3 has none, so we use ERA5,
# the reanalysis WeatherNext is trained on. Source: the public
# WeatherBench 2 daily ERA5 store on Google Cloud (no login needed):
#
#   gs://weatherbench2/datasets/era5_daily/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr
#
# For every climate window (centred on the 1st and 16th of each month,
# ±WINDOW_DAYS days, all years) and every 0.25° point of the EFI domain
# it stores the 0,1,...,100 % percentiles of:
#
#   rain24  24-h rainfall (mm)
#   tmax    daily maximum 2-m temperature (°C)
#   tmin    daily minimum 2-m temperature (°C)
#   tmean   daily mean 2-m temperature (°C)
#
# Output: one file per variable and window, climate/<var>_<MMDD>.npz
# (q: float16 (101, ny, nx), lat, lon, years, window). About 3 MB each,
# 96 files. They are kept as assets of the GitHub release "efi-climate"
# (workflow "EFI climate (one-time)") so the repository stays small.
#
# Usage:
#   python efi_climate.py --vars rain24,tmax       # build (≈ 11,000 days per variable)
#   python efi_climate.py --years 2011-2020        # quicker, shorter climate
# ============================================================

import os
import argparse
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLIM_DIR = os.environ.get("EFI_CLIM_DIR", os.path.join(HERE, "climate"))

ERA5_DAILY = ("https://storage.googleapis.com/weatherbench2/datasets/era5_daily/"
              "1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr")
ERA5_T0 = date(1959, 1, 1)              # time index 0 (one chunk per day)
ERA5_SHAPE = (721, 1440)                # 90N..90S, 0..359.75E

# EFI domain (India and the seas around it), 0.25°
EFI_LAT_MIN, EFI_LAT_MAX = 5.0, 38.0
EFI_LON_MIN, EFI_LON_MAX = 60.0, 100.0

CLIM_YEARS = (1991, 2020)
WINDOW_DAYS = 15                        # ± days around each window centre
WINDOW_CENTRES = [(m, d) for m in range(1, 13) for d in (1, 16)]

# variable -> (ERA5 daily field, convert to plotting units)
CLIM_VARS = {
    "rain24": ("total_precipitation_24hr", lambda a: np.maximum(a, 0) * 1000.0),   # m -> mm
    "tmax": ("2m_temperature_max", lambda a: a - 273.15),
    "tmin": ("2m_temperature_min", lambda a: a - 273.15),
    "tmean": ("2m_temperature", lambda a: a - 273.15),
}

WORKERS = int(os.environ.get("EFI_CLIM_WORKERS", "24"))


def _decode(raw, shape, dtype="<f4"):
    import numcodecs
    return np.frombuffer(numcodecs.Blosc().decode(raw), dtype).reshape(shape)


def era5_grid(http):
    lat = _decode(http.get(f"{ERA5_DAILY}/latitude/0", timeout=60).content, (721,))
    lon = _decode(http.get(f"{ERA5_DAILY}/longitude/0", timeout=60).content, (1440,))
    iy = np.where((lat >= EFI_LAT_MIN) & (lat <= EFI_LAT_MAX))[0]
    ix = np.where((lon >= EFI_LON_MIN) & (lon <= EFI_LON_MAX))[0]
    # flip to south -> north so maps and WeatherNext sampling are simple
    iy = iy[::-1]
    return lat[iy], lon[ix], iy, ix


def window_dates(month, day, years=CLIM_YEARS):
    """All dates within ±WINDOW_DAYS of month/day in the climate years."""
    out = []
    for y in range(years[0], years[1] + 1):
        try:
            c = date(y, month, day)
        except ValueError:          # 29 Feb never used as a centre, but be safe
            c = date(y, month, 28)
        out += [c + timedelta(days=k) for k in range(-WINDOW_DAYS, WINDOW_DAYS + 1)]
    return out


def nearest_window(d):
    """Climate window (month, day) whose centre is closest to date d (any year)."""
    best, best_gap = None, 999
    for m, dd in WINDOW_CENTRES:
        for y in (d.year - 1, d.year, d.year + 1):
            gap = abs((date(y, m, dd) - d).days)
            if gap < best_gap:
                best, best_gap = (m, dd), gap
    return best


def clim_path(var, window):
    return os.path.join(CLIM_DIR, f"{var}_{window[0]:02d}{window[1]:02d}.npz")


def load_clim(var, window):
    """(q (101, ny, nx) float32, lat, lon) or None when the file is missing."""
    path = clim_path(var, window)
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z["q"].astype(np.float32), z["lat"], z["lon"]


def build(var_names, years=CLIM_YEARS):
    import requests

    http = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=WORKERS, pool_maxsize=WORKERS)
    http.mount("https://", adapter)
    lat, lon, iy, ix = era5_grid(http)
    print(f"EFI domain {len(lat)} x {len(lon)} points, years {years[0]}-{years[1]}")
    os.makedirs(CLIM_DIR, exist_ok=True)

    # Every calendar day used by any window (each read once)
    all_dates = sorted({d for w in WINDOW_CENTRES for d in window_dates(*w, years)})
    index = {d: i for i, d in enumerate(all_dates)}

    for var in var_names:
        field, convert = CLIM_VARS[var]
        print(f"\n[{var}] reading {len(all_dates)} days of ERA5 {field} ...")
        cube = np.empty((len(all_dates), len(lat), len(lon)), dtype=np.float32)

        def fetch(d, cube=cube):
            t = (d - ERA5_T0).days
            for attempt in range(5):
                try:
                    r = http.get(f"{ERA5_DAILY}/{field}/{t}.0.0", timeout=120)
                    r.raise_for_status()
                    a = _decode(r.content, ERA5_SHAPE)
                    cube[index[d]] = convert(a[np.ix_(iy, ix)])
                    return
                except Exception as exc:
                    if attempt == 4:
                        raise RuntimeError(f"{field} {d}: {exc}")

        with ThreadPoolExecutor(WORKERS) as pool:
            for n, _ in enumerate(pool.map(fetch, all_dates), 1):
                if n % 1000 == 0:
                    print(f"  {n}/{len(all_dates)}")

        for w in WINDOW_CENTRES:
            rows = [index[d] for d in window_dates(*w, years)]
            q = np.percentile(cube[rows], np.arange(101), axis=0).astype(np.float16)
            np.savez_compressed(clim_path(var, w), q=q, lat=lat, lon=lon,
                                years=np.array(years), window=np.array(w))
        print(f"[{var}] wrote {len(WINDOW_CENTRES)} climate files to {CLIM_DIR}")
        del cube


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vars", default=",".join(CLIM_VARS), help="rain24,tmax,tmin,tmean")
    ap.add_argument("--years", default=f"{CLIM_YEARS[0]}-{CLIM_YEARS[1]}")
    args = ap.parse_args()
    y0, y1 = (int(y) for y in args.years.split("-"))
    names = [v.strip() for v in args.vars.split(",") if v.strip()]
    for v in names:
        if v not in CLIM_VARS:
            raise SystemExit(f"Unknown variable {v}. Use: {', '.join(CLIM_VARS)}")
    build(names, (y0, y1))


if __name__ == "__main__":
    main()
