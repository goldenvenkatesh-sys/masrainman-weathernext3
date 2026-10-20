# Quick check (needs only google-auth + requests):
# is there a new run, among the latest KEEP_RUNS, that the site does not
# have yet - for WeatherNext 3, ECMWF IFS (operational HRES) or ECMWF AIFS?
#
# Writes to $GITHUB_OUTPUT:
#   new=true/false          anything at all to render
#   wn3=true/false          WeatherNext 3 has a new run
#   ifs=true/false          ECMWF IFS has a new run
#   aifs=true/false         ECMWF AIFS has a new run
#   ifs_runs=YYYYMMDD_HH,.. new ECMWF IFS runs (complete only)
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
# Files live at https://data.ecmwf.int/forecasts/YYYYMMDD/HHz/<model>/0p25/<stream>/
# 00/12Z runs go to 360 h; 06/18Z go to 144 h. All four runs use stream "oper"
# (checked on data.ecmwf.int, Oct 2026 - the old "scda" name is no longer used).
ECMWF_ROOT = "https://data.ecmwf.int/forecasts"
ECMWF = {
    "ifs":  {"model": "ifs",         "streams": {0: "oper", 6: "oper", 12: "oper", 18: "oper"}},
    "aifs": {"model": "aifs-single", "streams": {0: "oper", 6: "oper", 12: "oper", 18: "oper"}},
}


def ecmwf_last_step(day, hh):
    return 360 if hh in (0, 12) else 144


def ecmwf_run_complete(key, run_time):
    """A run counts as complete once its last-step .index file exists."""
    cfg = ECMWF[key]
    hh = run_time.hour
    stream = cfg["streams"][hh]
    step = ecmwf_last_step(run_time, hh)
    ymd = run_time.strftime("%Y%m%d")
    base = f"{ECMWF_ROOT}/{ymd}/{hh:02d}z/{cfg['model']}/0p25/{stream}"
    name = f"{ymd}{hh:02d}0000-{step}h-{stream}-fc.index"
    print(f"  checking {key.upper()} {ymd} {hh:02d}Z ... ", end="", flush=True)
    try:
        r = requests.head(f"{base}/{name}", timeout=TIMEOUT, allow_redirects=True)
        ok = r.status_code == 200
        # AIFS 06/18Z may run to 360 h too; accept either length.
        if not ok and key == "aifs" and step == 144:
            name360 = f"{ymd}{hh:02d}0000-360h-{stream}-fc.index"
            r = requests.head(f"{base}/{name360}", timeout=TIMEOUT, allow_redirects=True)
            ok = r.status_code == 200
        print("complete" if ok else f"not ready (HTTP {r.status_code})", flush=True)
        return ok
    except requests.RequestException as e:
        print(f"FAILED: {type(e).__name__}", flush=True)
    return False


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
        if ecmwf_run_complete(key, t):
            latest.append(t.strftime("%Y%m%d_%H"))
            if len(latest) == KEEP_RUNS:
                break
        t -= timedelta(hours=6)

    have = site_has("ecmwf", key)
    missing = [r for r in latest if r not in have]
    print(f"[{key.upper():4}] latest :", ", ".join(latest) or "(none found)")
    print(f"[{key.upper():4}] to draw:", ", ".join(missing) or "(nothing new)")
    return missing


# ------------------------------------------------------------------------ main
results = {}
for key in ("ifs", "aifs"):
    print(f"--- ECMWF {key.upper()} ---", flush=True)
    results[key] = check_ecmwf(key)

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
