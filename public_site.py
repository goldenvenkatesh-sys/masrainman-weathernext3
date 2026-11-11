#!/usr/bin/env python3
"""Build the PUBLIC pages (no rain) for the masrainman-ai Cloudflare project.

    python public_site.py <site_dir> <out_dir>

<site_dir>  the full private site (the site-data branch)
<out_dir>   the public folder, e.g. public/   (the AI forecast + verification pages are already in it)

Pages written (each one a folder with its own index.html):
  models/    WeatherNext 3 + ECMWF IFS + ECMWF AIFS + NCEP GFS + NOAA AIGFS maps: MSLP, MSLP + wind barbs,
             model rain total since the start of the run and 24-h model rain
  landfall/  ensemble landfall maps
  cyclone/   NEM & cyclone outlook: genesis, cyclone tracks, 30-day NEM rain forecast, active/break, consistency, PDI,
             environment, cyclone verification (the observed station-rainfall verification stays private)
  tracks/    multi-model LPA tracker + cyclone tracker
  weathernext3/  small redirect to models/ (the first public address)
  raintotals/  single-model rain totals (24 h, 7 days, 15 days: WeatherNext 3, ECMWF IFS, AIFS, GFS, AIGFS).
             The owner chose to publish these totals (Nov 2026); they are the only rain maps here, copied
             from an allow-list of file names (mme/rain/<cycle>/models/<model>/<region>_<period>.png).

Observed rain (rain reports, TN-SMART station rainfall) and model rain maps are never published. Files are copied from allow-lists, the JSON files are rebuilt from
allow-lists of keys, the pages are cut down (rain layers, rain tabs, rain verification), and at the end the
whole output is scanned: if a rain file or rain link is found, the pages are deleted and the script fails,
so the previous public site stays as it was.
"""
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEEP_RUNS = int(os.environ.get("PUBLIC_RUNS", "4"))        # model-map runs per model
BLOG = "https://masrainman.blogspot.com/"
# public model-map layers: the owner chose (Nov 2026) to show the model rain charts too
# (Rain total since the start of the run, Rain 24 h). Observed rain stays private.
LAYERS = ("mslp", "barbs", "w850", "rain", "rain24")
MODELS = ("wn3", "ifs", "aifs", "gfs", "aigfs")

# (key, label, link from a sub-page)
PAGES = [("models", "🗺 Forecast models", "../models/"),
         ("landfall", "🌀 Landfall maps", "../landfall/"),
         ("cyclone", "📈 NEM &amp; cyclone outlook", "../cyclone/"),
         ("tracks", "🌐 LPA &amp; cyclone tracker", "../tracks/"),
         ("raintotals", "🌧 Rain totals", "../raintotals/"),
         ("ensrain", "📊 District rain", "../ensrain/"),
         ("ai", "🤖 MasRainMan AI (10-day)", "../")]


def die(msg):
    sys.exit(f"public_site: {msg}")


def read(*parts):
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return f.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path, obj):
    write(path, json.dumps(obj, indent=1, ensure_ascii=False))


def copy(site, rel, out, rel_out=None):
    src = os.path.join(site, rel)
    if not os.path.isfile(src):
        die(f"file listed in a manifest is missing: {src}")
    dst = os.path.join(out, rel_out or rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def sub1(pattern, repl, text, what, flags=0, count=1):
    new, n = re.subn(pattern, repl, text, count=count, flags=flags)
    if n < 1:
        die(f"{what} not found in the page: the page changed, update public_site.py")
    return new


def nav_buttons(current):
    links = [f'<a class="btn" href="{h}">{lab}</a>' for k, lab, h in PAGES if k != current]
    links.append(f'<a class="btn" href="{BLOG}" target="_top">🏠 MasRainman blog</a>')
    return '<nav class="mr-nav">\n    ' + "\n    ".join(links) + "\n  </nav>"


# ---------------------------------------------------------------- model maps (WN3 / IFS / AIFS)
def build_models(site, out):
    dest = os.path.join(out, "models")
    with open(os.path.join(site, "data", "models.json")) as f:
        listed = {m["id"]: m for m in json.load(f)["models"]}
    pub_models, copied, updated = [], 0, None
    for mid in MODELS:
        m = listed.get(mid)
        if not m or not os.path.isfile(os.path.join(site, m["manifest"])):
            print(f"public_site: model {mid} is not on the site yet, skipped")
            continue
        with open(os.path.join(site, m["manifest"])) as f:
            man = json.load(f)
        runs = []
        for r in man["runs"][:KEEP_RUNS]:
            pr = {k: r[k] for k in ("model", "model_label", "base", "run", "init", "last_hour", "resol",
                                    "grid_deg", "persistent_tracks", "rendered") if k in r}
            pr["frames"] = [{k: f[k] for k in ("hour", "valid", "systems", *LAYERS) if k in f}
                            for f in r["frames"]]
            pr["gifs"] = {k: v for k, v in r.get("gifs", {}).items() if k in LAYERS}
            pr["csv"] = dict(r.get("csv", {}))
            base = man.get("base") or m["base"]
            names = [fr[l] for fr in pr["frames"] for l in LAYERS if l in fr]
            names += list(pr["gifs"].values()) + list(pr["csv"].values())
            for n in names:
                copy(site, f"{base}/{r['run']}/{n}", dest)
                copied += 1
            runs.append(pr)
        if not runs:
            continue
        pub = {"model": man.get("model", mid), "label": man.get("label", m["label"]), "base": man.get("base", m["base"]),
               "updated": man.get("updated"), "runs": runs}
        write_json(os.path.join(dest, m["manifest"]), pub)
        updated = max(filter(None, [updated, pub["updated"]]), default=None)
        pub_models.append({"id": mid, "label": m["label"], "manifest": m["manifest"], "base": m["base"],
                           "runs": len(runs)})
    if not pub_models:
        die("no model-map runs on the site yet")
    write_json(os.path.join(dest, "data", "models.json"), {"updated": updated, "models": pub_models})

    html = read("web", "index.html")
    html = sub1(r"<title>.*?</title>", "<title>Forecast models · MasRainman</title>", html, "title")
    html = sub1(r'<nav class="mr-nav">.*?</nav>', lambda _: nav_buttons("models"), html, "nav", re.S)
    fb = ",\n".join(f"    {{id:'{m['id']}', label:{json.dumps(m['label'])}, manifest:'{m['manifest']}', base:'{m['base']}'}}"
                    for m in pub_models)
    html = sub1(r"(const FALLBACK_MODELS = \[).*?(\n  \];)", lambda mo: mo.group(1) + "\n" + fb + mo.group(2),
                html, "FALLBACK_MODELS", re.S)
    write(os.path.join(dest, "index.html"), html)
    print(f"  models:   {len(pub_models)} models, {copied} files")


# ---------------------------------------------------------------- landfall
def build_landfall(site, out):
    dest = os.path.join(out, "landfall")
    with open(os.path.join(site, "data", "manifest_landfall.json")) as f:
        man = json.load(f)
    n = 0
    for mdl in man.get("models", {}).values():
        for run in mdl.get("runs", []):
            for s in run.get("systems", []):
                for k in ("png", "csv"):
                    if s.get(k):
                        copy(site, s[k], dest)
                        n += 1
    write_json(os.path.join(dest, "data", "manifest_landfall.json"), man)
    html = read("web", "landfall.html")
    html = sub1(r'<nav class="mr-nav">.*?</nav>', lambda _: nav_buttons("landfall"), html, "nav", re.S)
    write(os.path.join(dest, "index.html"), html)
    print(f"  landfall: {n} files")


# ---------------------------------------------------------------- cyclone outlook
OUTLOOK_VERIF_JS = r'''function verif(){
  const q = S.verification; if (!q) return `<div class="empty">Verification not available yet.</div>`;
  const d = q.detection;
  let h = `<div class="grid g6" style="margin-bottom:14px">
    ${d.bins.map(b=>`<div class="kpi"><div class="k">Detection rate ${esc(b.lead)}</div><div class="v">${v(b.pod,0,"%")}</div><div class="s">${b.hits}/${b.opportunities} chances</div></div>`).join("")}
    <div class="kpi"><div class="k">False-alarm ratio</div><div class="v">${v(d.far,0,"%")}</div><div class="s">${d.false_alarms}/${d.predictions_verified} predictions</div></div>
    <div class="kpi"><div class="k">Track error 72 h</div><div class="v">${v((q.track.find(t=>t.lead_h===72)||{}).ens_km,0," km")}</div><div class="s">ENS mean · n=${(q.track.find(t=>t.lead_h===72)||{}).ens_n||0}</div></div>
    </div>
    <div class="grid g2" style="margin-top:14px"><div class="card"><h3>Track and intensity</h3><div class="tw"><table>
    <tr><th>Lead</th><th class="n">ENS track</th><th class="n">HRES track</th><th class="n">ENS int. MAE / bias</th><th class="n">HRES int. MAE / bias</th></tr>
    ${q.track.map(t=>`<tr><td>${t.lead_h} h</td><td class="n">${v(t.ens_km,0," km")} <span class="note">n=${t.ens_n}</span></td><td class="n">${v(t.hres_km,0," km")} <span class="note">n=${t.hres_n}</span></td><td class="n">${v(t.ens_int_mae,1)} / ${v(t.ens_int_bias,1)} kt</td><td class="n">${v(t.hres_int_mae,1)} / ${v(t.hres_int_bias,1)} kt</td></tr>`).join("")}</table></div></div>
    <div class="card"><h3>Observed systems in the archive period</h3>${d.observed.length?`<div class="tw"><table><tr><th>System</th><th>Source</th><th>Reached depression</th><th>Position</th><th class="n">Peak</th><th class="n">Runs that predicted it</th></tr>
    ${d.observed.map(o=>`<tr><td>${esc(o.id)}</td><td>${esc(o.src)}</td><td>${esc(o.time)}</td><td>${ll(o.lat,o.lon)}</td><td class="n">${v(o.peak_kt)} kt</td><td class="n">${o.detected_runs}</td></tr>`).join("")}</table></div>`:`<div class="empty">None yet.</div>`}
    ${d.false_list.length?`<p class="note">Recent false alarms: ${d.false_list.map(f=>`${esc(f.storm)} (${fmtRun(f.run)}, ${v(f.p)}%)`).join(", ")}</p>`:""}</div></div>
    <p class="note">Tracks and intensity: IBTrACS (IMD) where published, else ECMWF analysis (T+0). Period ${esc(q.period)}; ${q.runs_archived} runs archived. Detection: a run "detects" an observed system if ≥ 30% of members form a depression within 500 km and 72 h of the observed genesis.
    False alarm: a ≥ 30% prediction with no observed system within 500 km and 72 h. Samples are small early in the season — read with care.</p>`;
  return h;
}

'''


def public_outlook_page(html):
    html = sub1(r"<title>.*?</title>", "<title>NEM &amp; Cyclone Outlook · MasRainman</title>", html, "title")
    html = sub1(r'<nav class="mr-nav">.*?</nav>', lambda _: nav_buttons("cyclone"), html, "nav", re.S)
    # verification: cyclone tracks/detection only. The observed station rainfall (TN-SMART) and the
    # active/break check against it stay private.
    html, n = re.subn(r"function verif\(\)\{.*?(?=\nfunction methods\(\))", lambda _: OUTLOOK_VERIF_JS.rstrip("\n"), html, flags=re.S)
    if n != 1:
        die("verif() not found in web/outlook.html")
    html = sub1(r"open data \(ECMWF ENS / IFS, NOAA CFSv2, NOAA CPC, IBTrACS, BoM RMM, TN-SMART\)",
                "open data (ECMWF ENS / IFS, NOAA CFSv2, NOAA CPC, IBTrACS, BoM RMM)", html, "footer sources")
    html = sub1(r"\n  <li><b>Verification</b>.*?</li>",
                "\n  <li><b>Verification</b>: each run's ENS-mean and HRES tracks and predicted geneses are archived for 120 days and scored against IBTrACS / ECMWF analyses.</li>",
                html, "verification method line", re.S)
    return html



def build_outlook(site, out):
    dest = os.path.join(out, "cyclone")
    with open(os.path.join(site, "data", "manifest_outlook.json")) as f:
        man = json.load(f)
    runs = man["runs"][:KEEP_RUNS + 2]
    pngs = 0
    for r in runs:
        with open(os.path.join(site, r["summary"])) as f:
            s = json.load(f)
        ver = s.get("verification") or {}
        pub = {k: s[k] for k in ("run", "init", "generated", "demo", "genesis", "cyclones", "pdi", "consistency",
                                 "environment", "nem", "active_break", "names") if k in s}
        pub["errors"] = {k: v for k, v in (s.get("errors") or {}).items() if k != "verification"}
        if ver:
            pub["verification"] = {k: ver[k] for k in ("track", "detection", "runs_archived", "period") if k in ver}
        pub["takeaways"] = list(s.get("takeaways", []))
        # every picture the public summary points to, and nothing else
        found = set()

        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
            elif isinstance(o, str) and o.startswith("outlook/") and o.endswith(".png"):
                found.add(o)
        walk(pub)
        for p in sorted(found):
            if re.search(r"verification", p):
                die(f"the verification picture (station rainfall) must stay private: {p}")
            copy(site, p, dest)
            pngs += 1
        write_json(os.path.join(dest, r["summary"]), pub)
    write_json(os.path.join(dest, "data", "manifest_outlook.json"), {"updated": man.get("updated"), "runs": runs})
    write(os.path.join(dest, "index.html"), public_outlook_page(read("web", "outlook.html")))
    print(f"  cyclone:  {len(runs)} runs, {pngs} pictures")


# ---------------------------------------------------------------- multi-model tracks (LPA + cyclone)
def build_tracks(site, out):
    dest = os.path.join(out, "tracks")
    with open(os.path.join(site, "mme", "manifest.json")) as f:
        man = json.load(f)
    tracks = man.get("tracks", [])[:KEEP_RUNS]
    n = 0
    for t in tracks:
        src = os.path.join(site, "mme", "tracks", t["id"])
        if not os.path.isdir(src):
            die(f"tracker folder missing: {src}")
        for fn in sorted(os.listdir(src)):
            if re.search(r"rain|precip", fn, re.I):
                continue
            copy(site, f"mme/tracks/{t['id']}/{fn}", dest, f"tracks/{t['id']}/{fn}")
            n += 1
    write_json(os.path.join(dest, "manifest.json"), {"updated": man.get("updated"), "tracks": tracks, "rain": []})

    html = read("multimodel", "web", "index.html")
    html = sub1(r"<title>.*?</title>", "<title>LPA &amp; Cyclone Tracker · MasRainman</title>", html, "title")
    html = sub1(r"<h1>MasRainman · <span>Multi-model Ensemble Guidance</span></h1>",
                "<h1>MasRainman · <span>LPA &amp; Cyclone Tracker</span></h1>", html, "h1")
    links = "\n".join(f'  <a href="{h}">{lab}</a>' for k, lab, h in PAGES if k != "tracks")
    links += f'\n  <a href="{BLOG}" target="_top">🏠 MasRainman blog</a>\n'
    html = sub1(r'  <a data-tab="rain">.*?</a>\n(?:  <a href=.*?</a>\n)+(?=</nav>)', lambda _: links, html, "tab links", re.S)
    html = sub1(r"<!-- =+ RAINFALL =+ -->\n<section id=\"rain\">.*?</section>\n", "", html, "rain section", re.S)
    html = sub1(r"  // -{10,} RAIN -{10,}\n.*?seg\('regions'.*?\n", "", html, "rain script", re.S)
    html = sub1(r"buildLpa\(\); buildCyclone\(\); buildRain\(\);", "buildLpa(); buildCyclone();", html, "init call")
    write(os.path.join(dest, "index.html"), html)
    print(f"  tracks:   {len(tracks)} cycles, {n} files")


# ---------------------------------------------------------------- rain totals
TOTAL_FILE = re.compile(r"(south_india|tamilnadu|kerala)_(24h|7day|15day)\.png")


def build_raintotals(site, out):
    """Newest multi-model cycle that has single-model totals -> raintotals/ (page + manifest + PNGs)."""
    dest = os.path.join(out, "raintotals")
    root = os.path.join(site, "mme", "rain")
    cycles = sorted((c for c in (os.listdir(root) if os.path.isdir(root) else []) if re.fullmatch(r"\d{10}", c)),
                    reverse=True)
    info, cyc = None, None
    for c in cycles:
        try:
            with open(os.path.join(root, c, "run.json")) as f:
                j = json.load(f)
        except (OSError, ValueError):
            continue
        if j.get("models"):
            info, cyc = j, c
            break
    models, n = {}, 0
    for mid, m in ((info or {}).get("models") or {}).items():
        maps = {}
        for region, keys in m.get("maps", {}).items():
            for k in keys:
                name = f"{region}_{k}.png"
                if not TOTAL_FILE.fullmatch(name) or not os.path.isfile(os.path.join(root, cyc, "models", mid, name)):
                    continue
                copy(site, f"mme/rain/{cyc}/models/{mid}/{name}", dest, f"{cyc}/{mid}/{name}")
                maps.setdefault(region, []).append(k)
                n += 1
        if maps:
            models[mid] = {"name": m["name"], "run": m["run"], "maps": maps,
                           "periods": [p for p in m.get("periods", []) if any(p["key"] in v for v in maps.values())]}
    write_json(os.path.join(dest, "manifest.json"),
               {"cycle": cyc, "base": cyc, "rendered": (info or {}).get("rendered", "").replace("Z", "+00:00"),
                "models": models})
    html = read("web", "raintotals.html")
    html = sub1(r'<nav class="mr-nav">.*?</nav>', lambda _: nav_buttons("raintotals"), html, "nav", re.S)
    write(os.path.join(dest, "index.html"), html)
    print(f"  raintotals: {len(models)} models, {n} files (cycle {cyc})")


# ---------------------------------------------------------------- final scan
BAD_NAME = re.compile(r"rain|precip|verification\.png|tnsmart|tn-smart", re.I)
BAD_TEXT = re.compile(r"rain\.html|TN-SMART|tnsmart|Rain reports|data-layer=\"rain|rain24"
                      r"|\"rain\"\s*:\s*\[\s*\{|Rainfall Forecast|Station rainfall", re.I)


def scan(out):
    bad = []
    for sub in ("landfall", "cyclone", "tracks", "weathernext3"):   # models/ now has model rain on purpose
        for root, _, files in os.walk(os.path.join(out, sub)):
            for fn in files:
                p = os.path.join(root, fn)
                if BAD_NAME.search(fn):
                    bad.append(p)
                elif fn.endswith((".json", ".html", ".csv")):
                    txt = open(p, encoding="utf-8", errors="ignore").read()
                    if BAD_TEXT.search(txt):
                        bad.append(p)
    return bad


def main():
    if len(sys.argv) != 3:
        die("usage: public_site.py <site_dir> <out_dir>")
    site, out = sys.argv[1], sys.argv[2]
    for sub in ("models", "landfall", "cyclone", "tracks", "weathernext3", "raintotals"):
        shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
    print("public pages:")
    build_models(site, out)
    build_landfall(site, out)
    build_outlook(site, out)
    build_tracks(site, out)
    write(os.path.join(out, "weathernext3", "index.html"),
          '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../models/">'
          '<title>WeatherNext 3 · MasRainman</title><a href="../models/">Forecast models</a>\n')
    bad = scan(out)
    try:   # after the scan: these totals are the one allowed rain product; never stop the other pages
        build_raintotals(site, out)
    except Exception as exc:
        print(f"public_site: rain totals skipped ({type(exc).__name__}: {exc})")
    if bad:
        for sub in ("models", "landfall", "cyclone", "tracks", "weathernext3"):
            shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
        die("rain content found in the public output, nothing published:\n  " + "\n  ".join(bad))
    print("public pages: rain scan clean")


if __name__ == "__main__":
    main()
