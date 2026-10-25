#!/usr/bin/env bash
# Put the latest MasRainman rainfall reports into site/rain/ before publishing the site.
# They come from the `reports` branch of the public repo goldenvenkatesh-sys/masrainman-rain-report
# (saved there by its rain-report workflows), so no token is needed to read them.
set -uo pipefail
SITE=${1:-site}
REPO=${RAIN_REPO:-https://github.com/goldenvenkatesh-sys/masrainman-rain-report}
RAW=https://raw.githubusercontent.com/goldenvenkatesh-sys/masrainman-rain-report/reports

cp web/index.html web/rain.html "$SITE"/ 2>/dev/null || true
rm -rf "$SITE/rain" && mkdir -p "$SITE/rain"
if ! git fetch -q --depth 1 "$REPO" reports 2>/dev/null; then
  echo "No rain reports yet (the rain-report repo has no 'reports' branch)."
  echo '{"updated":null,"reports":[]}' > "$SITE/rain/index.json"
  exit 0
fi
git --work-tree="$SITE/rain" checkout -q FETCH_HEAD -- index.json files
git reset -q

# Cloudflare Pages takes files up to 25 MB: link bigger ones to GitHub instead
python3 - "$SITE/rain" "$RAW" <<'PY'
import json, os, sys
d, raw = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(d, "index.json"), encoding="utf-8"))
moved = 0
for r in idx["reports"]:
    for e, p in list(r["files"].items()):
        fp = os.path.join(d, p) if isinstance(p, str) else ""
        if fp and os.path.isfile(fp) and os.path.getsize(fp) > 24 * 1024 * 1024:
            os.remove(fp); r["files"][e] = f"{raw}/{p}"; moved += 1
json.dump(idx, open(os.path.join(d, "index.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"Rain reports: {len(idx['reports'])} listed" + (f", {moved} big files linked to GitHub" if moved else ""))
PY
