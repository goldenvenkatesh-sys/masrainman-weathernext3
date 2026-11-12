#!/usr/bin/env bash
# Put the latest tropical tracker (IFS HRES + AIFS + IFS ENS + AI-GFS tracks, OISST SST) into site/tropical/
# before publishing. It is saved on the `tracker-data` branch by the "Tropical tracker" workflow (every 2 h).
set -uo pipefail
SITE=${1:-site}
rm -rf "$SITE/tropical" && mkdir -p "$SITE/tropical"
if git fetch -q --depth 1 origin tracker-data 2>/dev/null; then
  git --work-tree="$SITE/tropical" checkout -q FETCH_HEAD -- . && git reset -q
  rm -rf "$SITE/tropical/state" "$SITE/tropical/changed.txt"
  echo "Tropical tracker: $(python3 -c "import json;print(json.load(open('$SITE/tropical/stamp.json'))['generated'])" 2>/dev/null || echo '?')"
else
  echo "No tropical tracker yet (no tracker-data branch)."
fi
