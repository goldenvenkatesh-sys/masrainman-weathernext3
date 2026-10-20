# List the variables inside the latest WeatherNext 3 run (names, shape, units).
# Needs only google-auth + requests and the same Google login as render.py.
# Run on GitHub: Actions -> "WeatherNext 3 variable list" -> Run workflow.

import json
import os
import re

import google.auth
from google.auth.transport.requests import AuthorizedSession

BUCKET = "weathernext3_statistics_spatial"
ROOT = "weathernext_3_0_0_statistics/zarr/2026_to_present/"
RUN_PATTERN = r"^\d{8}_(00|06|12|18)hr_01_preds$"
RAIN_WORDS = ("precip", "rain", "tp", "accum", "convective")

creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
s = AuthorizedSession(creds)
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"


def prefixes(prefix):
    out, params = [], {"prefix": prefix, "delimiter": "/", "maxResults": 1000}
    while True:
        r = s.get(API, params=params)
        r.raise_for_status()
        d = r.json()
        out += d.get("prefixes", [])
        if not d.get("nextPageToken"):
            return out
        params["pageToken"] = d["nextPageToken"]


def get_json(name):
    r = s.get(f"https://storage.googleapis.com/{BUCKET}/{name}")
    return r.json() if r.status_code == 200 else None


runs = sorted(p.rstrip("/").split("/")[-1] for p in prefixes(ROOT)
              if re.match(RUN_PATTERN, p.rstrip("/").split("/")[-1]))
run = os.environ.get("WN3_RUN") or runs[-1]
zroot = f"{ROOT}{run}/predictions.zarr/"
print(f"Run: {run}  (of {len(runs)} runs)\n")

rows = []
for p in prefixes(zroot):
    var = p.rstrip("/").split("/")[-1]
    meta = get_json(f"{p}zarr.json") or get_json(f"{p}.zarray") or {}
    attrs = meta.get("attributes") or get_json(f"{p}.zattrs") or {}
    shape = meta.get("shape")
    dims = meta.get("dimension_names") or attrs.get("_ARRAY_DIMENSIONS")
    units = attrs.get("units", "")
    long = attrs.get("long_name") or attrs.get("standard_name") or ""
    rows.append((var, shape, dims, units, long))

print(f"{len(rows)} variables:\n")
for var, shape, dims, units, long in rows:
    print(f"  {var:55s} shape={shape} dims={dims} units={units!s} {long}")

rain = [r for r in rows if any(w in r[0].lower() or w in str(r[4]).lower() for w in RAIN_WORDS)]
print("\nRAIN-LIKE VARIABLES:")
for var, shape, dims, units, long in rain or [("(none found)", "", "", "", "")]:
    print(f"  >> {var}  shape={shape} units={units} {long}")

if rain:
    # Peek at attrs of the first one in full, to see accumulation period
    var = rain[0][0]
    print(f"\nFull metadata of {var}:")
    print(json.dumps(get_json(f"{zroot}{var}/zarr.json") or get_json(f"{zroot}{var}/.zattrs"), indent=1)[:3000])
