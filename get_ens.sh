#!/usr/bin/env bash
# Put the latest ensemble district rain charts into site/ens/ before publishing the site.
# They are saved on the `ens-data` branch by the "Ensemble rain charts" workflow.
set -uo pipefail
SITE=${1:-site}
cp web/ens.html "$SITE"/ 2>/dev/null || true
rm -rf "$SITE/ens" && mkdir -p "$SITE/ens"
if git fetch -q --depth 1 origin ens-data 2>/dev/null; then
  git --work-tree="$SITE/ens" checkout -q FETCH_HEAD -- . && git reset -q
  echo "Ensemble charts: $(find "$SITE/ens" -name '*.png' | wc -l) PNGs"
else
  echo "No ensemble charts yet (no ens-data branch)."
fi

# Tropical tracker page (tracker-data branch)
bash "$(dirname "$0")/get_tracker.sh" "$SITE" || true
