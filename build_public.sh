#!/usr/bin/env bash
# Builds ./public for the public Cloudflare project (masrainman-ai, no login, embedded on the Blogger page).
#   bash build_public.sh <site_dir>
# Public pages: MasRainMan AI forecast (/), verification (/verify/), model maps (/models/: WeatherNext 3, ECMWF IFS,
# ECMWF AIFS), landfall maps (/landfall/), cyclone outlook (/cyclone/), LPA + cyclone tracker (/tracks/).
# Rain products never go here (public_site.py copies only allow-lists and stops if it sees rain).
# A Cloudflare deploy replaces the whole project, so BOTH workflows (AI forecast and model maps) call this
# script and publish all the pages together.
set -euo pipefail
site="${1:-site}"

rm -rf public && cp -r "$site/ai" public

# AI + verification pages: links back to the private dashboard are taken out, links to the public pages
# and the blog are added instead.
blog='<a class="btn" href="https://masrainman.blogspot.com/" target="_top">🏠 MasRainman blog</a>'
links='<a class="btn" href="raintotals/">🌧 Rain totals</a>\n    <a class="btn" href="ensrain/">📊 District rain (ensemble)</a>\n    <a class="btn" href="models/">🗺 Forecast models</a>\n    <a class="btn" href="landfall/">🌀 Landfall maps</a>\n    <a class="btn" href="cyclone/">📈 Cyclone outlook</a>\n    <a class="btn" href="tracks/">🌐 LPA \&amp; cyclone tracker</a>\n    <a class="btn" href="tropical/">🌀 Tropical tracker</a>'
sed -i -e '\#href="\.\./mme/"#d' -e "s#<a class=\"btn\" href=\"\.\./\">[^<]*</a>#${links}\n    ${blog}#" public/index.html
sed -i -e "s#<a class=\"btn\" href=\"\.\./\.\./\">[^<]*</a>#${blog}#" public/verify/index.html
grep -c 'href="\.\./\.\./"\|href="\.\./mme/"' public/index.html public/verify/index.html || true

# Model maps, landfall, cyclone outlook and tracker pages (no rain)
python3 public_site.py "$site" public

# Interactive tropical tracker (IFS HRES + AIFS + IFS ENS + AI-GFS tracks, OISST SST), refreshed every 2 h
# by the "Tropical tracker" workflow on the tracker-data branch. No rain in it. Page at /tropical/.
bash get_tracker.sh public

# Ensemble 24-h rain charts by district (ECMWF ENS + WeatherNext 3) from the ens-data branch.
# The owner chose to show these on the blog (Nov 2026). Page at /ensrain/, charts at /ensrain/ens/.
mkdir -p public/ensrain/ens
if git fetch -q --depth 1 origin ens-data 2>/dev/null; then
  git --work-tree=public/ensrain/ens checkout -q FETCH_HEAD -- . && git reset -q
  echo "  ensrain: $(find public/ensrain/ens -name '*.png' | wc -l) charts"
else
  echo "  ensrain: no charts yet"
fi
python3 - <<'PY'
import re
html = open("web/ens.html", encoding="utf-8").read()
nav = ('<a class="btn" href="../">🤖 MasRainMan AI</a>\n  <a class="btn" href="../raintotals/">🌧 Rain totals</a>\n'
       '  <a class="btn" href="../models/">🗺 Forecast models</a>\n'
       '  <a class="btn" href="https://masrainman.blogspot.com/" target="_top">🏠 MasRainman blog</a>')
html, n = re.subn(r'<a class="btn" href="index.html">[^<]*</a>\s*<a class="btn" href="rain.html">[^<]*</a>', nav, html)
assert n == 1, "ens.html nav not found"
fit = """<script>
// Embedded on the blog: send the page height so the frame fits the page exactly
if (window.parent !== window) {
  const send = () => parent.postMessage({masrainmanHeight: document.documentElement.scrollHeight}, "*");
  addEventListener("load", send); addEventListener("resize", send);
  new ResizeObserver(send).observe(document.body);
  document.addEventListener("load", send, true);
}
</script>
</body>"""
html = html.replace("</body>", fit, 1)
open("public/ensrain/index.html", "w", encoding="utf-8").write(html)
PY
