# Quick check (needs only google-auth + requests):
# is there a new run, among the latest KEEP_RUNS, that the site does not
# have yet - for WeatherNext 3, ECMWF IFS (operational HRES) or ECMWF AIFS?
#
# Writes to $GITHUB_OUTPUT:
#   new=true/false          anything at all to render
#   wn3=true/false          WeatherNext 3 has a new run
#   ifs=true/false          ECMWF IFS has a new run
#   aifs=true/false         ECMWF AIFS has a new run
#   ifs_runs=YYYYMMDD_HH,.. new ECMWF IFS runs (complete only, or now
#                            available at 0.1° after being drawn at 0.25°)
#   aifs_runs=...           new ECMWF AIFS runs (complete only)
# so the workflow can skip the heavy install + render when nothing is new.
#
# Site layout:
#   site/runs/<YYYYMMDD_HHhr_01_preds>/   WeatherNext 3 (unchanged)
#   site/ecmwf/ifs/<YYYYMMDD_HH>/         ECMWF IFS
#   site/ecmwf/aifs/<YYYYMMDD_HH>/        ECMWF AIFS

import os
import re
from datetime import datetime, timedelta, timezone

import requests

SITE_DIR = os.environ.get("SITE_DIR", "site")
KEEP_RUNS = int(os.environ.get("KEEP_RUNS", "4"))
TIMEOUT = (5, 15)  # seconds: connect, read - so a bad connection can't hang it
# Which ECMWF runs to draw (must match render.py). "00,12" saves GitHub minutes.
ECMWF_CYCLES = [int(c) for c in os.environ.get("ECMWF_CYCLES", "00,06,12,18").split(",") if c.strip()]


def site_has(*parts):
    d = os.path.join(SITE_DIR, *parts)
    return set(os.listdir(d)) if os.path.isdir(d) else set()


# ---------------------------------------------------------------- WeatherNext 3
def check_wn3():
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    bucket = "weathernext3_statistics_spatial"
    prefix = "weathernext_3_0_0_statistics/zarr/2026_to_present/"
    pattern = r"^\d{8}_(00|06|12|18)hr_01_preds$"

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(creds)

    runs, params = [], {"prefix": prefix, "delimiter": "/", "maxResults": 1000}
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    while True:
        r = session.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        for p in data.get("prefixes", []):
            name = p.rstrip("/").split("/")[-1]
            if re.match(pattern, name):
                runs.append(name)
        if not data.get("nextPageToken"):
            break
        params["pageToken"] = data["nextPageToken"]

    latest = sorted(set(runs), reverse=True)[:KEEP_RUNS]
    have = site_has("runs")
    missing = [r for r in latest if r not in have]
    print("[WN3 ] latest :", ", ".join(latest))
    print("[WN3 ] to draw:", ", ".join(missing) or "(nothing new)")
    return missing


# ------------------------------------------------------------- ECMWF open data
# Files live at https://data.ecmwf.int/forecasts/YYYYMMDD/HHz/<model>/<resol>/oper/
# 00/12Z runs go to 360 h; 06/18Z go to 144 h. All four runs use stream "oper".
# Since Oct 2026 ECMWF also gives free 0.1° (~9 km) IFS data in .../ifs/0p1/oper/
# with files named *-oper-fc-sfc.index; the 0.25° files are *-oper-fc.index.
# IFS uses 0.1° when it is fully published, else 0.25° (must match render.py).
import json

ECMWF_ROOT = os.environ.get("ECMWF_ROOT", "https://data.ecmwf.int/forecasts")
ECMWF = {
    "ifs":  {"model": "ifs",
             "resols": [r for r in os.environ.get("IFS_RESOLS", "0p1,0p25").split(",") if r.strip()]},
    "aifs": {"model": "aifs-single", "resols": ["0p25"]},
}
FILE_KINDS = ["oper-fc-sfc", "oper-fc"]


def ecmwf_best_source(key, run_time):
    """(step, resol) of the best fully published version, or None.
    Longest forecast first, then the finest grid."""
    cfg = ECMWF[key]
    hh = run_time.hour
    ymd = run_time.strftime("%Y%m%d")
    wanted = [360] if hh in (0, 12) else [360, 144]
    print(f"  checking {key.upper()} {ymd} {hh:02d}Z ... ", end="", flush=True)
    for step in wanted:
        for resol in cfg["resols"]:
            base = f"{ECMWF_ROOT}/{ymd}/{hh:02d}z/{cfg['model']}/{resol}/oper"
            for kind in FILE_KINDS:
                try:
                    r = requests.head(f"{base}/{ymd}{hh:02d}0000-{step}h-{kind}.index",
                                      timeout=TIMEOUT, allow_redirects=True)
                except requests.RequestException as e:
                    print(f"[{resol} failed: {type(e).__name__}] ", end="", flush=True)
                    continue
                if r.status_code == 200:
                    print(f"complete ({resol}, T+{step})", flush=True)
                    return step, resol
    print("not ready", flush=True)
    return None


def rank(key, step, resol):
    order = ECMWF[key]["resols"]
    return (int(step), len(order) - order.index(resol) if resol in order else 0)


def rendered_rank(key, run_name):
    path = os.path.join(SITE_DIR, "ecmwf", key, run_name, "run.json")
    try:
        with open(path) as f:
            info = json.load(f)
    except Exception:
        return None
    return rank(key, info.get("last_hour", 0), info.get("resol") or "0p25")


def check_ecmwf(key):
    # Walk back over the last ~2 days of 6-hourly cycles, newest first,
    # and keep the KEEP_RUNS newest complete runs.
    now = datetime.now(timezone.utc)
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 6)
    latest = []
    for _ in range(16):
        if t.hour not in ECMWF_CYCLES:
            t -= timedelta(hours=6)
            continue
        src = ecmwf_best_source(key, t)
        if src:
            latest.append((t.strftime("%Y%m%d_%H"), src))
            if len(latest) == KEEP_RUNS:
                break
        t -= timedelta(hours=6)

    missing = []
    for run, (step, resol) in latest:
        have = rendered_rank(key, run)
        if have is None or rank(key, step, resol) > have:
            missing.append(run)
    print(f"[{key.upper():4}] latest :", ", ".join(f"{r} ({s[1]})" for r, s in latest) or "(none found)")
    print(f"[{key.upper():4}] to draw:", ", ".join(missing) or "(nothing new)")
    return missing


# ------------------------------------------------------------------- NOAA GFS / AIGFS
# 00/12 UTC runs only (must match render.py NOAA_CYCLES); complete when the T+360 .idx exists.
NOAA_CYCLES = [int(c) for c in os.environ.get("NOAA_CYCLES", "00,12").split(",") if c.strip()]


def noaa_url(key, t, step):
    ymd, hh = t.strftime("%Y%m%d"), t.strftime("%H")
    if key == "gfs":
        return f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{step:03d}"
    return (f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod/aigfs.{ymd}/{hh}/model/atmos/grib2/"
            f"aigfs.t{hh}z.sfc.f{step:03d}.grib2")


def check_noaa(key):
    t = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 6)
    latest = []
    for _ in range(16):
        if t.hour in NOAA_CYCLES:
            try:
                r = requests.head(noaa_url(key, t, 360) + ".idx", timeout=TIMEOUT, allow_redirects=True)
                if r.status_code == 200:
                    latest.append(t.strftime("%Y%m%d_%H"))
                    if len(latest) == int(os.environ.get("NOAA_KEEP_RUNS", "2")):
                        break
            except requests.RequestException as e:
                print(f"[{key.upper()}] check failed: {type(e).__name__}")
        t -= timedelta(hours=6)
    have = site_has("noaa", key)
    missing = [r for r in latest if not os.path.exists(os.path.join(SITE_DIR, "noaa", key, r, "run.json"))]
    print(f"[{key.upper():5}] latest :", ", ".join(latest) or "(none found)")
    print(f"[{key.upper():5}] to draw:", ", ".join(missing) or "(nothing new)")
    return missing


# ------------------------------------------------------------------------ main
results = {}
for key in ("ifs", "aifs"):
    print(f"--- ECMWF {key.upper()} ---", flush=True)
    results[key] = check_ecmwf(key)
for key in ("gfs", "aigfs"):
    print(f"--- NOAA {key.upper()} ---", flush=True)
    try:
        results[key] = check_noaa(key)
    except Exception as e:  # never let one source block the others
        print(f"[{key.upper()}] check failed:", e)
        results[key] = []

# WeatherNext 3 needs the Google login, which exists only on GitHub
# (GOOGLE_APPLICATION_CREDENTIALS). Skip it on a PC without it, instead of
# letting google-auth sit waiting for a cloud server that isn't there.
print("--- WeatherNext 3 ---", flush=True)
if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GITHUB_ACTIONS"):
    try:
        results["wn3"] = check_wn3()
    except Exception as e:  # never let one source block the others
        print("[WN3 ] check failed:", e)
        results["wn3"] = []
else:
    print("[WN3 ] skipped (no Google login on this computer - works on GitHub)")
    results["wn3"] = []

any_new = any(results.values())
out = os.environ.get("GITHUB_OUTPUT")
if out:
    with open(out, "a") as f:
        f.write(f"new={'true' if any_new else 'false'}\n")
        for key, missing in results.items():
            f.write(f"{key}={'true' if missing else 'false'}\n")
        f.write(f"ifs_runs={','.join(results['ifs'])}\n")
        f.write(f"aifs_runs={','.join(results['aifs'])}\n")
