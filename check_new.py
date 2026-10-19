# Quick check (needs only google-auth + requests):
# is there a WeatherNext 3 run among the latest KEEP_RUNS that the site
# does not have yet? Writes new=true/false to $GITHUB_OUTPUT so the
# workflow can skip the heavy install + render when nothing is new.

import os
import re

import google.auth
from google.auth.transport.requests import AuthorizedSession

BUCKET = "weathernext3_statistics_spatial"
PREFIX = "weathernext_3_0_0_statistics/zarr/2026_to_present/"
PATTERN = r"^\d{8}_(00|06|12|18)hr_01_preds$"
SITE_DIR = os.environ.get("SITE_DIR", "site")
KEEP_RUNS = int(os.environ.get("KEEP_RUNS", "4"))

creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
session = AuthorizedSession(creds)

runs, params = [], {"prefix": PREFIX, "delimiter": "/", "maxResults": 1000}
url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
while True:
    r = session.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    for p in data.get("prefixes", []):
        name = p.rstrip("/").split("/")[-1]
        if re.match(PATTERN, name):
            runs.append(name)
    if not data.get("nextPageToken"):
        break
    params["pageToken"] = data["nextPageToken"]

latest = sorted(set(runs), reverse=True)[:KEEP_RUNS]
runs_dir = os.path.join(SITE_DIR, "runs")
have = set(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else set()
missing = [r for r in latest if r not in have]

print("Latest runs :", ", ".join(latest))
print("On the site :", ", ".join(sorted(have, reverse=True)) or "(none)")
print("To render   :", ", ".join(missing) or "(nothing new)")

out = os.environ.get("GITHUB_OUTPUT")
if out:
    with open(out, "a") as f:
        f.write(f"new={'true' if missing else 'false'}\n")
