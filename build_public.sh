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
links='<a class="btn" href="models/">🗺 Model maps</a>\n    <a class="btn" href="landfall/">🌀 Landfall maps</a>\n    <a class="btn" href="cyclone/">📈 Cyclone outlook</a>\n    <a class="btn" href="tracks/">🌐 LPA \&amp; cyclone tracker</a>'
sed -i -e '\#href="\.\./mme/"#d' -e "s#<a class=\"btn\" href=\"\.\./\">[^<]*</a>#${links}\n    ${blog}#" public/index.html
sed -i -e "s#<a class=\"btn\" href=\"\.\./\.\./\">[^<]*</a>#${blog}#" public/verify/index.html
grep -c 'href="\.\./\.\./"\|href="\.\./mme/"' public/index.html public/verify/index.html || true

# Model maps, landfall, cyclone outlook and tracker pages (no rain)
python3 public_site.py "$site" public
