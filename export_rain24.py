"""
Save WeatherNext 3 24-hour rain over Tamil Nadu for the MasRainman rain report.

The rain report's "Past 24 hours" video (live24) compares the models' forecasts with the rain gauges for
the rain-day 08:30 IST to 08:30 IST (03 UTC to 03 UTC), using the run issued at 12 UTC two days before.
This script writes, for each recent 12 UTC run, the 24-h rain of the first windows ending at 03 UTC:

  site/rain24/wn3_<YYYYMMDD>_12.json
  {"run": "<run name>", "windows": {"<YYYY-MM-DD>": {"lat": [...], "lon": [...], "mm": [[...]]}}}

(mm = 24-h total of total_precipitation_1hr_mean, rows = lat). The rain report reads it from the
site-data branch. Runs that already have a complete file are skipped, so the first run backfills the
last DAYS days and later runs only add the newest one.

  python export_rain24.py site            # needs GOOGLE_APPLICATION_CREDENTIALS like render.py
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

import render

BOX = (7.0, 15.0, 74.5, 81.5)          # lat0, lat1, lon0, lon1
DAYS = int(os.environ.get("RAIN24_DAYS", "10"))
WINDOWS = 3                            # windows ending 03 UTC on the 2nd, 3rd and 4th day after the run


def export(run, out_dir):
    name = f"{run:%Y%m%d}_{run:%H}hr_01_preds"
    path = os.path.join(out_dir, f"wn3_{run:%Y%m%d_%H}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            if len(json.load(f).get("windows", {})) >= WINDOWS:
                return False
    root = render.open_run_zarr(name)
    tp = root[render.WN3_RAIN_VAR]
    lat = np.asarray(root["lat_0p1"][:])
    lon = np.asarray(root["lon_0p1"][:])
    lon180 = (lon + 180) % 360 - 180
    li = np.where((lat >= BOX[0]) & (lat <= BOX[1]))[0]
    lo = np.where((lon180 >= BOX[2]) & (lon180 <= BOX[3]))[0]
    n = int(tp.shape[0])                 # array index i = rain in the hour ending T+(i+1)
    windows = {}
    for w in range(WINDOWS):
        end = datetime(run.year, run.month, run.day, 3, tzinfo=timezone.utc) + timedelta(days=2 + w)
        lead = int((end - run).total_seconds() // 3600)          # hours from the run to the window end
        if lead > n:
            break
        sl = (slice(lead - 24, lead), slice(int(li[0]), int(li[-1]) + 1), slice(int(lo[0]), int(lo[-1]) + 1))
        hourly = np.asarray(tp[sl], dtype=np.float32) * 1000.0  # m -> mm
        if np.mean(np.isfinite(hourly)) < 0.99:                 # run not complete that far yet
            break
        mm = np.maximum(np.nan_to_num(hourly), 0).sum(axis=0)
        windows[end.date().isoformat()] = {
            "lat": [round(float(x), 3) for x in lat[sl[1]]],
            "lon": [round(float(x), 3) for x in lon180[sl[2]]],
            "mm": [[round(float(v), 1) for v in row] for row in mm],
        }
    if not windows:
        print(f"  {name}: no complete window yet")
        return False
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"run": name, "windows": windows}, f, separators=(",", ":"))
    top = {d: round(float(np.max(v["mm"])), 1) for d, v in windows.items()}
    print(f"  {name}: saved windows {top} (max mm)")
    return True


def main(site):
    out_dir = os.path.join(site, "rain24")
    os.makedirs(out_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    new = False
    print("WeatherNext 3 24-h rain for the rain report:")
    for k in range(DAYS, -1, -1):
        d = (now - timedelta(days=k)).date()
        run = datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc)
        if run + timedelta(hours=8) > now:                      # not out yet
            continue
        try:
            new |= export(run, out_dir)
        except Exception as e:                                  # missing run: carry on
            print(f"  {run:%Y%m%d}_12: skipped ({type(e).__name__}: {e})")
    # drop files older than DAYS + 5 days
    cut = (now - timedelta(days=DAYS + 5)).strftime("%Y%m%d")
    for fn in os.listdir(out_dir):
        if fn.startswith("wn3_") and fn[4:12] < cut:
            os.remove(os.path.join(out_dir, fn))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"new={'true' if new else 'false'}\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "site")
