#!/usr/bin/env bash
# Put the 3D cyclone view (web/cyclone3d.html and its pictures) into the site folder before publishing.
# The pictures are kept on this repo's `cyclone3d` branch by the cyclone3d.yml workflow.
set -uo pipefail
SITE=${1:-site}
cp web/cyclone3d.html "$SITE"/ 2>/dev/null || true
rm -rf "$SITE/cyclone3d" && mkdir -p "$SITE/cyclone3d"
if git fetch -q --depth 1 origin cyclone3d 2>/dev/null; then
  git --work-tree="$SITE/cyclone3d" checkout -q FETCH_HEAD -- . && git reset -q
  echo "3D cyclone view: $(ls "$SITE/cyclone3d"/*.mp4 2>/dev/null | wc -l) pass(es)"
else
  echo '{"updated":null,"passes":[]}' > "$SITE/cyclone3d/index.json"
  echo "3D cyclone view: no pictures yet"
fi
