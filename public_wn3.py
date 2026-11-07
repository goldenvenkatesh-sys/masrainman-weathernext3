#!/usr/bin/env python3
"""Build the PUBLIC WeatherNext 3 page (no rain) for the masrainman-ai Cloudflare project.

    python public_wn3.py <site_dir> <out_dir>

<site_dir>  the full private site (site-data branch: data/manifest.json, runs/<run>/...)
<out_dir>   where the public page goes, e.g. public/weathernext3

Only an allow-list of files is copied: the MSLP and MSLP + wind-barb maps and their GIFs, plus the V3
low-centre CSVs. Rain maps/GIFs/JSON (rain_T*, rain24_T*, *_rain_loop.gif, run.json, rain reports ...)
are never copied, the manifest is rebuilt with an allow-list of keys, and the script stops with an error
if any rain file or rain link would end up in the output.
"""
import json
import os
import re
import shutil
import sys

KEEP_RUNS = int(os.environ.get("PUBLIC_WN3_RUNS", "4"))
LAYERS = ("mslp", "barbs")            # public layers: no rain

FRAME_KEYS = ("hour", "valid", "systems", *LAYERS)
RUN_KEYS = ("model", "model_label", "base", "run", "init", "last_hour", "resol", "grid_deg",
            "persistent_tracks", "rendered")


def die(msg):
    sys.exit(f"public_wn3: {msg}")


def public_manifest(man):
    runs = []
    for r in man["runs"][:KEEP_RUNS]:
        pr = {k: r[k] for k in RUN_KEYS if k in r}
        pr["frames"] = [{k: f[k] for k in FRAME_KEYS if k in f} for f in r["frames"]]
        pr["gifs"] = {k: v for k, v in r.get("gifs", {}).items() if k in LAYERS}
        pr["csv"] = dict(r.get("csv", {}))
        runs.append(pr)
    return {"model": man.get("model", "wn3"), "label": man.get("label", "WeatherNext 3"),
            "base": man.get("base", "runs"), "updated": man.get("updated"), "runs": runs}


def public_page(src_html):
    html = src_html
    # 1) title
    html, n = re.subn(r"<title>.*?</title>", "<title>WeatherNext 3 · MasRainman</title>", html, count=1)
    if n != 1:
        die("title not found in web/index.html")
    # 2) no rain layer buttons
    html, n = re.subn(r'[ \t]*<button class="pill" data-layer="rain(?:24)?"[^\n]*</button>\n', "", html)
    if n != 2:
        die(f"expected 2 rain layer buttons in web/index.html, found {n}")
    # 3) header links: only public pages
    nav = ('<nav class="mr-nav">\n'
           '    <a class="btn" href="../">🤖 MasRainMan AI (10-day)</a>\n'
           '    <a class="btn" href="../verify/">✅ Verification</a>\n'
           '    <a class="btn" href="https://masrainman.blogspot.com/" target="_top">🏠 MasRainman blog</a>\n'
           '  </nav>')
    html, n = re.subn(r'<nav class="mr-nav">.*?</nav>', nav, html, count=1, flags=re.S)
    if n != 1:
        die("nav block not found in web/index.html")
    # 4) the public page lists only WeatherNext 3
    html, n = re.subn(r"(const FALLBACK_MODELS = \[).*?(\n  \];)",
                      r"\1\n    {id:'wn3',  label:'WeatherNext 3', manifest:'data/manifest.json', base:'runs'}\2",
                      html, count=1, flags=re.S)
    if n != 1:
        die("FALLBACK_MODELS not found in web/index.html")
    return html


def main():
    if len(sys.argv) != 3:
        die("usage: public_wn3.py <site_dir> <out_dir>")
    site, out = sys.argv[1], sys.argv[2]

    with open(os.path.join(site, "data", "manifest.json")) as f:
        man = json.load(f)
    pub = public_manifest(man)
    if not pub["runs"]:
        die("no WeatherNext 3 runs on the site yet")

    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "data"))

    copied = 0
    for r in pub["runs"]:
        src_dir = os.path.join(site, "runs", r["run"])
        dst_dir = os.path.join(out, "runs", r["run"])
        os.makedirs(dst_dir, exist_ok=True)
        names = [fr[l] for fr in r["frames"] for l in LAYERS if l in fr]
        names += list(r["gifs"].values()) + list(r["csv"].values())
        for name in names:
            s = os.path.join(src_dir, name)
            if not os.path.isfile(s):
                die(f"missing file listed in the manifest: {s}")
            shutil.copy2(s, os.path.join(dst_dir, name))
            copied += 1

    with open(os.path.join(out, "data", "manifest.json"), "w") as f:
        json.dump(pub, f, indent=1)
    with open(os.path.join(out, "data", "models.json"), "w") as f:
        json.dump({"updated": pub["updated"],
                   "models": [{"id": "wn3", "label": "WeatherNext 3", "manifest": "data/manifest.json",
                               "base": "runs", "runs": len(pub["runs"])}]}, f, indent=1)

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "web", "index.html"), encoding="utf-8") as f:
        page = public_page(f.read())
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)

    # ---- final safety check: nothing rain-related may be in the public output ----
    bad = []
    for root, _, files in os.walk(out):
        for fn in files:
            p = os.path.join(root, fn)
            if re.search(r"rain|precip", fn, re.I):
                bad.append(p)
            elif fn.endswith((".json", ".html")):
                txt = open(p, encoding="utf-8", errors="ignore").read()
                if re.search(r'data-layer="rain|rain24|rain\.html|"rain"\s*:', txt):
                    bad.append(p)
    if bad:
        shutil.rmtree(out, ignore_errors=True)
        die("rain content found in the public output, nothing published:\n  " + "\n  ".join(bad))

    print(f"public WeatherNext 3: {len(pub['runs'])} runs, {copied} files -> {out}")


if __name__ == "__main__":
    main()
