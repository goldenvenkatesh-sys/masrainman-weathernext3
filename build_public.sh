#!/usr/bin/env bash
# Builds ./public for the public Cloudflare project (masrainman-ai, no login, embedded on the Blogger page).
#   bash build_public.sh <site_dir>
# Public pages: MasRainMan AI forecast (/), verification (/verify/) and WeatherNext 3 maps (/weathernext3/).
# Rain products never go here (public_wn3.py copies only an allow-list and stops if it sees rain).
# A Cloudflare deploy replaces the whole project, so BOTH workflows (AI forecast and model maps) call this
# script and publish all three pages together.
set -euo pipefail
site="${1:-site}"

rm -rf public && cp -r "$site/ai" public

# AI + verification pages: links back to the private dashboard are taken out, the blog link is added instead.
blog='<a class="btn" href="https://masrainman.blogspot.com/" target="_top">🏠 MasRainman blog</a>'
wn3='<a class="btn" href="weathernext3/">🌐 WeatherNext 3</a>'
sed -i -e '\#href="\.\./mme/"#d' -e "s#<a class=\"btn\" href=\"\.\./\">[^<]*</a>#${wn3}\n    ${blog}#" public/index.html
sed -i -e "s#<a class=\"btn\" href=\"\.\./\.\./\">[^<]*</a>#${blog}#" public/verify/index.html
grep -c 'href="\.\./\.\./"\|href="\.\./mme/"' public/index.html public/verify/index.html || true

# WeatherNext 3 maps (MSLP + wind barbs only, no rain)
python3 public_wn3.py "$site" public/weathernext3
