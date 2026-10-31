# ============================================================
# MASRAINMAN — ENSEMBLE LANDFALL MAPS (North Indian Ocean)
# ============================================================
#
# For each ensemble run it:
#   1. Gets the model's cyclone tracks
#        ECMWF ENS : tropical-cyclone track BUFR (…-enfo-tf.bufr) from ECMWF open data,
#                    which also tracks genesis systems (ids 70B, 71B …) before they are named
#        GFS (GEFS), UKMET : tracked here from the gridded fields (gridtrack.py),
#                    starting from ECMWF's position of each system
#   2. Keeps the North Indian Ocean systems (Bay of Bengal "B", Arabian Sea "A")
#   3. Finds each member's first landfall (sea -> land on Natural Earth 10 m land)
#      and its intensity there (TD / TS / Cat 1-5 from the member's 10-m wind)
#   4. Groups the landfalls into areas along the coast and ranks the top 5
#      (share of all members in the ensemble)
#   5. Draws the map: member landfall points, numbered top areas, ensemble mean track
#   6. Writes a PNG + CSV per system and site/data/manifest_landfall.json
#
# Website layout:
#   site/landfall/<model>/<YYYYMMDD_HH>/<storm>.png / .csv
#   site/data/manifest_landfall.json
#
# Usage:
#   python landfall.py                         # newest complete run of each model, if new
#   python landfall.py --models gfs --run 20261028_06   # one model, one run
#   python landfall.py --bufr file.bufr --run 20261028_06   # a local track file
#
# ============================================================

import os
import sys
import io
import json
import functools
import time
import shutil
import argparse
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import masrainman_logo  # noqa: F401  (adds the MasRainman logo to every saved figure)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# SETTINGS
# ============================================================

SITE_DIR = os.environ.get("SITE_DIR", "site")
KEEP_RUNS = int(os.environ.get("KEEP_RUNS", "4"))
BASINS = ("B", "A")                      # ECMWF basin letters: Bay of Bengal, Arabian Sea
MIN_LANDFALLS = 3                        # skip systems with fewer member landfalls
AREA_RADIUS_KM = 120                     # landfalls this close join the same area
MIN_LAND_DEG2 = 2.0                      # ignore islands smaller than this (Andaman & Nicobar, Maldives…);
                                         # Sri Lanka (~5.5 deg²) still counts
TOP_AREAS = 5
BUDGET_MIN = float(os.environ.get("LANDFALL_BUDGET_MIN", "45"))   # stop starting new models after this

ECMWF_URLS = [
    "https://data.ecmwf.int/forecasts/{d}/{h}z/ifs/0p25/enfo/{d}{h}0000-{s}h-enfo-tf.bufr",
    "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/{d}/{h}z/ifs/0p25/enfo/{d}{h}0000-{s}h-enfo-tf.bufr",
]

MODELS = {
    "ecmwf": {"label": "ECMWF ENS", "members": 51},
    "gfs": {"label": "GFS ensemble (GEFS)", "members": 31,
            "note": "Tracks found by MasRainman in the GEFS 0.5° MSLP and 10-m wind (31 members), "
                    "starting from ECMWF's position of each system."},
    "deepmind": {"label": "DeepMind FNV3 ensemble", "members": 50,
                 "note": "Google DeepMind Weather Lab cyclone ensemble (experimental research data). "
                         "Systems are matched to ECMWF's by position."},
    "icon": {"label": "ICON-EPS", "members": 40,
             "note": "Tracks found by MasRainman in the ICON-EPS MSLP and 10-m wind (40 members + control), "
                     "starting from ECMWF's position of each system."},
    "gem": {"label": "GEM ensemble (GEPS)", "members": 21,
            "note": "Tracks found by MasRainman in the GEPS 0.5° MSLP and 10-m wind (20 members + control), "
                    "starting from ECMWF's position of each system."},
    "ukmet": {"label": "UKMET global (deterministic)", "members": 1,
              "note": "One track found by MasRainman in the Met Office global 10 km MSLP and 10-m wind, "
                      "starting from ECMWF's position. The MOGREPS-G ensemble is not openly published."},
}

# Saffir-Simpson classes from the member's max 10-m wind (kt)
CLASSES = [  # (name, lower bound kt, colour)
    ("TD", 0, "#4f8ff7"),
    ("TS", 34, "#22c38e"),
    ("Cat 1", 64, "#e5b52a"),
    ("Cat 2", 83, "#f08a3a"),
    ("Cat 3", 96, "#ef6461"),
    ("Cat 4", 113, "#e0405a"),
    ("Cat 5", 137, "#c03bd6"),
]
MS_TO_KT = 1.943844

BG = "#121826"
LAND = "#141b2a"
COAST = "#8a94a8"
TEXT = "#e6ecf7"


def wind_class(kt):
    i = 0
    for k, (_, lo, _) in enumerate(CLASSES):
        if kt >= lo:
            i = k
    return i


# ============================================================
# TRACK SOURCES
# ============================================================

DEEPMIND_URL = ("https://deepmind.google.com/science/weatherlab/download/cyclones/{m}/ensemble/paired/csv/"
                "{m}_{t}_paired.csv")
DEEPMIND_MODELS = os.environ.get("DEEPMIND_MODELS", "OPER,FNV3_LARGE_ENSEMBLE").split(",")


@functools.lru_cache(maxsize=8)
def deepmind_csv(run):
    """Weather Lab ensemble track CSV for a run (text), trying each DeepMind model name; None if not online."""
    import gridtrack
    t = datetime.strptime(run, "%Y%m%d_%H").strftime("%Y_%m_%dT%H_00")
    for m in DEEPMIND_MODELS:
        data = gridtrack._get(DEEPMIND_URL.format(m=m, t=t))
        if data and b"," in data[:500]:
            return data.decode("utf-8", "replace")
    return None


def _col(df, *names, contains=None):
    cols = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in cols:
            return cols[n]
    if contains:
        for lc, c in cols.items():
            if all(w in lc for w in contains):
                return c
    return None


def read_deepmind_csv(text, run, ec):
    """Weather Lab paired CSV -> the read_ecmwf_tf table. The columns are found by name, so small format
    changes don't matter. Each track is given the id of the ECMWF system it starts near (within 300 km of
    an ECMWF member's position of that system at the same step); other North Indian Ocean tracks keep their own id."""
    df = pd.read_csv(io.StringIO(text), comment="#")
    c_mem = _col(df, "sample", "ensemble_member", "member", "sample_id", "ensemble_member_id")
    c_trk = _col(df, "track_id", "storm_id", "track", "id")
    c_lat = _col(df, "lat", "latitude", contains=("lat",))
    c_lon = _col(df, "lon", "longitude", contains=("lon",))
    c_p = _col(df, "minimum_sea_level_pressure_hpa", "mslp", contains=("pressure",))
    c_w = _col(df, "maximum_sustained_wind_speed_knots", "vmax", contains=("wind",))
    c_lead = _col(df, "lead_time", "lead_time_hours", "forecast_hour", "tau")
    c_valid = _col(df, "valid_time", "time")
    if not (c_lat and c_lon and c_trk and (c_lead or c_valid)):
        raise ValueError(f"unexpected DeepMind CSV columns: {list(df.columns)}")
    init = datetime.strptime(run, "%Y%m%d_%H")
    if c_lead:
        lead = df[c_lead]
        step = (pd.to_timedelta(lead).dt.total_seconds() / 3600 if lead.dtype == object else lead.astype(float))
    else:
        step = (pd.to_datetime(df[c_valid]).dt.tz_localize(None) - init).dt.total_seconds() / 3600
    wind = df[c_w].astype(float) if c_w else np.nan
    if c_w and ("knot" in c_w.lower() or "kt" in c_w.lower()):
        wind = wind / MS_TO_KT
    t = pd.DataFrame({"track": df[c_trk].astype(str),
                      "member": df[c_mem].astype(int) if c_mem else 0,
                      "step": step.round().astype(int), "lat": df[c_lat].astype(float),
                      "lon": ((df[c_lon].astype(float) + 180) % 360) - 180,
                      "mslp": df[c_p].astype(float) if c_p else np.nan, "wind": wind})
    t = t[t.step % 6 == 0]
    rows = []
    for (trk, m), g in t.groupby(["track", "member"]):
        g = g.sort_values("step")
        f = g.iloc[0]
        if not (45 <= f.lon <= 100 and -5 <= f.lat <= 32):
            continue
        r = ec[ec.step == f.step]
        storm = "FN" + trk[-4:]
        if len(r):
            d = haversine_km(f.lat, f.lon, r.lat.values, r.lon.values)
            if d.min() < 300:
                storm = r.storm.values[d.argmin()]
        rows.append(g.assign(storm=storm))
    if not rows:
        return pd.DataFrame(columns=["storm", "member", "step", "lat", "lon", "mslp", "wind"])
    out = pd.concat(rows)
    # one track per member and system: the longest
    keep = out.groupby(["storm", "member", "track"]).size().reset_index(name="n") \
              .sort_values("n").groupby(["storm", "member"]).tail(1)
    out = out.merge(keep[["storm", "member", "track"]])
    return out[["storm", "member", "step", "lat", "lon", "mslp", "wind"]]


def ecmwf_tf_path(run):
    """The ENS track BUFR for run 'YYYYMMDD_HH' (00/12Z run to 360 h, 06/18Z to 144 h), downloaded once
    into RUNNER_TEMP. None if it is not online."""
    path = os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), f"ecmwf_tf_{run}.bufr")
    if os.path.exists(path):
        return path
    d, h = run.split("_")
    s = 360 if h in ("00", "12") else 144
    for attempt in range(4):
        missing = 0
        for url in ECMWF_URLS:
            try:
                r = requests.get(url.format(d=d, h=h, s=s), timeout=120)
                if r.status_code == 200 and r.content[:4] == b"BUFR":
                    open(path, "wb").write(r.content)
                    return path
                missing += r.status_code == 404
            except requests.RequestException:
                pass
        if missing == len(ECMWF_URLS):
            return None
        time.sleep(10 * (attempt + 1))      # busy mirror (S3 "SlowDown"): wait and retry
    return None


def read_ecmwf_tf(path):
    """ECMWF track BUFR -> rows (storm, member, step h, lat, lon, mslp hPa, wind m/s)."""
    import eccodes

    def arr(h, key, n):
        try:
            a = np.array(eccodes.codes_get_array(h, key), dtype=float)
        except Exception:
            return np.full(n, np.nan)
        if a.size == 1:
            a = np.repeat(a, n)
        a[np.abs(a) > 1e90] = np.nan
        return a

    rows = []
    with open(path, "rb") as f:
        while True:
            h = eccodes.codes_bufr_new_from_file(f)
            if h is None:
                break
            try:
                eccodes.codes_set(h, "unpack", 1)
                sid = eccodes.codes_get(h, "stormIdentifier").strip()
                if not sid.endswith(BASINS):
                    continue
                n = eccodes.codes_get(h, "numberOfSubsets")
                mem = arr(h, "ensembleMemberNumber", n)
                steps = [0] + [int(x) for x in eccodes.codes_get_array(h, "timePeriod")]
                # #1# latitude = observed centre; then per step: centre (#2k+2#), max-wind place (#2k+3#)
                for k, step in enumerate(steps):
                    la = arr(h, f"#{2 * k + 2}#latitude", n)
                    lo = arr(h, f"#{2 * k + 2}#longitude", n)
                    p = arr(h, f"#{k + 1}#pressureReducedToMeanSeaLevel", n)
                    w = arr(h, f"#{k + 1}#windSpeedAt10M", n)
                    for i in range(n):
                        if np.isfinite(la[i]) and np.isfinite(mem[i]):
                            rows.append((sid, int(mem[i]), step, la[i], lo[i], p[i] / 100.0, w[i]))
            finally:
                eccodes.codes_release(h)
    return pd.DataFrame(rows, columns=["storm", "member", "step", "lat", "lon", "mslp", "wind"])


# ============================================================
# LANDFALL
# ============================================================

_land = None


def land():
    global _land
    if _land is None:
        shp = shpreader.natural_earth(resolution="10m", category="physical", name="land")
        polys = []
        for g in shpreader.Reader(shp).geometries():
            polys += list(getattr(g, "geoms", [g]))
        _land = prep(unary_union([g for g in polys if g.area >= MIN_LAND_DEG2]))
    return _land


def first_landfall(track, substeps=12):
    """First sea -> land crossing of a member track (sorted by step), interpolated every 6h/substeps.
    Returns (lat, lon, step, wind m/s at the last sea point) or None."""
    L = land()
    pts = track[["lat", "lon", "step", "wind"]].to_numpy()
    if len(pts) < 2 or L.contains(Point(pts[0][1], pts[0][0])):
        return None
    prev = pts[0]
    for a, b in zip(pts[:-1], pts[1:]):
        for f in np.linspace(0, 1, substeps + 1)[1:]:
            q = a + (b - a) * f
            if np.isnan(q[3]):
                q[3] = a[3] if np.isfinite(a[3]) else b[3]
            if L.contains(Point(q[1], q[0])):
                return float(q[0]), float(q[1]), float(q[2]), float(prev[3])
            prev = q
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def top_areas(lf, n_members):
    """Greedy grouping: the densest point (most landfalls within AREA_RADIUS_KM) starts an area,
    which takes those landfalls; repeat. Returns [(lat, lon, count, pct)] best first."""
    left = lf[["lat", "lon"]].to_numpy()
    areas = []
    while len(left) and len(areas) < TOP_AREAS:
        d = haversine_km(left[:, None, 0], left[:, None, 1], left[None, :, 0], left[None, :, 1])
        near = d <= AREA_RADIUS_KM
        c = near.sum(1).argmax()
        grp = left[near[c]]
        areas.append((grp[:, 0].mean(), grp[:, 1].mean(), len(grp), 100.0 * len(grp) / n_members))
        left = left[~near[c]]
    return areas


def mean_track(tr, n_members):
    """Mean position per step over the members still tracked (at least half of those that track it)."""
    g = tr.groupby("step").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("member", "nunique"))
    return g[g.n >= max(min(2, n_members), tr.member.nunique() / 2)].reset_index()


# ============================================================
# MAP
# ============================================================

_borders = None


def india_borders():
    global _borders
    if _borders is None:
        with open(os.path.join(HERE, "boundaries", "borders_india_view.geojson")) as f:
            gj = json.load(f)
        _borders = ShapelyFeature([shape(ft["geometry"]) for ft in gj["features"]], ccrs.PlateCarree(),
                                  facecolor="none", edgecolor=COAST, linewidth=0.5)
    return _borders


def seaward(lat, lon, d=0.6):
    """The first of 8 directions, d degrees from (lat, lon), that is over sea (else the point itself)."""
    L = land()
    for ang in (0, 45, -45, 90, -90, 135, -135, 180):   # east first (east coast), then around
        a = np.radians(ang)
        la, lo = lat + d * np.sin(a), lon + d * np.cos(a)
        if not L.contains(Point(lo, la)):
            return la, lo
    return lat, lon


def draw(model, run, storm, tr, lf, areas, mt, n_members, out_png):
    label = MODELS[model]["label"]
    init = datetime.strptime(run, "%Y%m%d_%H").strftime("%d %b %HZ").lstrip("0")

    lats = list(lf.lat) + list(mt.lat)
    lons = list(lf.lon) + list(mt.lon)
    lat0, lat1 = min(lats) - 2, max(lats) + 2
    lon0, lon1 = min(lons) - 2, max(lons) + 2
    # at least 10° x 10°, centred
    if lat1 - lat0 < 10:
        c = (lat0 + lat1) / 2; lat0, lat1 = c - 5, c + 5
    if lon1 - lon0 < 10:
        c = (lon0 + lon1) / 2; lon0, lon1 = c - 5, c + 5

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9, 9 * (lat1 - lat0) / (lon1 - lon0) + 0.6), facecolor=BG)
    ax = fig.add_axes([0.02, 0.07, 0.96, 0.91], projection=proj)
    ax.set_extent([lon0, lon1, lat0, lat1], crs=proj)
    ax.set_facecolor(BG)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor=LAND, edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor=COAST, linewidth=0.6)
    ax.add_feature(india_borders())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#2a3550", x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"color": "#8fa0bf", "size": 8}

    # ensemble mean track (white double line, like the TC-ATLAS style)
    ax.plot(mt.lon, mt.lat, color="white", lw=3.2, transform=proj, zorder=4)
    ax.plot(mt.lon, mt.lat, color=BG, lw=1.2, transform=proj, zorder=5)
    for _, r in mt[mt.step % 24 == 0].iterrows():
        ax.text(r.lon + 0.15, r.lat + 0.15, f"{int(r.step)}h", color="#cfd8ea", fontsize=7,
                transform=proj, zorder=6)

    # member landfall points by intensity
    for k, (name, _, col) in enumerate(CLASSES):
        s = lf[lf.cls == k]
        if len(s):
            ax.scatter(s.lon, s.lat, s=26, color=col, edgecolor="none", transform=proj, zorder=7)

    # numbered top areas, set a little out to sea so they don't hide the points
    for i, (la, lo, _, _) in enumerate(areas, 1):
        la, lo = seaward(la, lo)
        ax.scatter([lo], [la], s=330, facecolor=BG, edgecolor="white", linewidth=1.4,
                   transform=proj, zorder=8)
        ax.text(lo, la, str(i), color="white", ha="center", va="center", fontsize=10,
                fontweight="bold", transform=proj, zorder=9)

    # legend box
    used = sorted(set(lf.cls))
    handles = [Line2D([], [], ls="", marker="o", color=CLASSES[k][2], label=CLASSES[k][0]) for k in used]
    handles.append(Line2D([], [], color="white", lw=2, label="ensemble mean" if n_members > 1 else "model track"))
    leg = ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True, ncol=1,
                    facecolor="#0f1626", edgecolor="#2e4a6a", labelcolor=TEXT,
                    title=f"Not an official forecast\n{label} · {len(lf)}/{n_members} members land\n"
                          "member landfall by intensity", title_fontsize=8)
    leg.get_title().set_color(TEXT)
    leg.set_zorder(10)
    tops = "   ".join(f"{i} · {p:.0f}%" for i, (_, _, _, p) in enumerate(areas, 1))
    ax.text(0.99, 0.01, "Top landfall areas\n" + tops, transform=ax.transAxes, ha="right", va="bottom",
            color=TEXT, fontsize=8.5, zorder=10,
            bbox=dict(facecolor="#0f1626", edgecolor="#2e4a6a", boxstyle="round,pad=0.4"))

    fig.text(0.02, 0.025, f"System {storm} · {label} · init {init} · member landfall points · "
                          "NOT an official forecast", color=TEXT, fontsize=10.5, va="center")
    fig.text(0.98, 0.025, "MasRainman", color="#8fa0bf", fontsize=9, ha="right", va="center")
    fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# RUN
# ============================================================

def drop_duplicates(tracks):
    """Genesis ids (70B, 71B …) that start on a named storm are the same system: keep the named one."""
    first = tracks[tracks.step == tracks.groupby("storm").step.transform("min")]
    starts = first.groupby("storm")[["lat", "lon"]].median()
    named = starts[~starts.index.str.startswith("7")]
    for storm, (la, lo) in starts.iterrows():
        if storm.startswith("7") and len(named) and \
                haversine_km(la, lo, named.lat.values, named.lon.values).min() < 250:
            print(f"  {storm}: same system as a named storm, skipped")
            tracks = tracks[tracks.storm != storm]
    return tracks


def seeds_from_ecmwf(tracks, min_share=0.25):
    """Starting positions (median of the ECMWF members at T+0) of the systems most members have at T+0."""
    t0 = tracks[tracks.step == 0]
    n = max(MODELS["ecmwf"]["members"], tracks.member.nunique())
    return {s: (float(g.lat.median()), float(g.lon.median()))
            for s, g in t0.groupby("storm") if g.member.nunique() >= min_share * n}


def process(model, run, tracks):
    out_dir = os.path.join(SITE_DIR, "landfall", model, run)
    os.makedirs(out_dir, exist_ok=True)
    systems = []
    n_members = max(MODELS[model]["members"], int(tracks.member.nunique()))
    for storm, tr in tracks.groupby("storm"):
        tr = tr.sort_values(["member", "step"])
        rows = []
        for m, t in tr.groupby("member"):
            hit = first_landfall(t)
            if hit:
                la, lo, step, w = hit
                kt = w * MS_TO_KT if np.isfinite(w) else 0.0
                rows.append((m, la, lo, step, kt, wind_class(kt)))
        lf = pd.DataFrame(rows, columns=["member", "lat", "lon", "step", "wind_kt", "cls"])
        if len(lf) < min(MIN_LANDFALLS, n_members):
            print(f"  {model} {run} {storm}: {len(lf)} landfalls, skipped")
            continue
        areas = top_areas(lf, n_members)
        mt = mean_track(tr, n_members)
        png = os.path.join(out_dir, f"{storm}.png")
        draw(model, run, storm, tr, lf, areas, mt, n_members, png)
        lf.assign(cls=[CLASSES[k][0] for k in lf.cls]).round(2).to_csv(
            os.path.join(out_dir, f"{storm}.csv"), index=False)
        systems.append({
            "storm": storm, "png": f"landfall/{model}/{run}/{storm}.png",
            "csv": f"landfall/{model}/{run}/{storm}.csv",
            "start": {"lat": round(float(tr[tr.step == tr.step.min()].lat.mean()), 1),
                      "lon": round(float(tr[tr.step == tr.step.min()].lon.mean()), 1),
                      "step": int(tr.step.min())},
            "members": n_members, "tracked": int(tr.member.nunique()), "landfalls": len(lf),
            "areas": [{"lat": round(a, 2), "lon": round(o, 2), "n": n, "pct": round(p, 1)}
                      for a, o, n, p in areas],
        })
        print(f"  {model} {run} {storm}: {len(lf)}/{n_members} landfalls -> {png}")
    return systems


def update_manifest(model, run, systems):
    path = os.path.join(SITE_DIR, "data", "manifest_landfall.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    man = json.load(open(path)) if os.path.exists(path) else {"models": {}}
    entry = man["models"].setdefault(model, {"runs": []})
    entry.update(label=MODELS[model]["label"], note=MODELS[model].get("note", ""))
    runs = entry["runs"]
    runs[:] = [r for r in runs if r["run"] != run]
    runs.append({"run": run, "systems": systems})
    runs.sort(key=lambda r: r["run"], reverse=True)
    for old in runs[KEEP_RUNS:]:
        d = os.path.join(SITE_DIR, "landfall", model, old["run"])
        if os.path.isdir(d):
            shutil.rmtree(d)
    del runs[KEEP_RUNS:]
    man["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    json.dump(man, open(path, "w"), indent=1)


def ecmwf_tracks(run, bufr=None):
    """ECMWF ENS tracks for a run (North Indian Ocean, duplicates dropped), or None if not online."""
    path = bufr or ecmwf_tf_path(run)
    return drop_duplicates(read_ecmwf_tf(path)) if path else None


def candidate_runs(model, now=None):
    """Recent runs, newest first: every 6 h (UKMET and GEM: 00Z and 12Z, their long runs)."""
    t = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 6)
    out = []
    for _ in range(8):
        if model not in ("ukmet", "gem") or t.hour in (0, 12):
            out.append(t.strftime("%Y%m%d_%H"))
        t -= timedelta(hours=6)
    return out


def done_runs(model):
    man = os.path.join(SITE_DIR, "data", "manifest_landfall.json")
    if not os.path.exists(man):
        return set()
    return {r["run"] for r in json.load(open(man))["models"].get(model, {}).get("runs", [])}


def do_model(model, run, bufr=None):
    """Draw one run of one model. Returns True when the site changed."""
    import gridtrack
    ec = ecmwf_tracks(run, bufr)
    if ec is None:
        print(f"{model} {run}: no ECMWF ENS tracks for this run")
        return False
    if model == "ecmwf":
        tracks = ec
    elif model == "deepmind":
        text = deepmind_csv(run)
        if text is None:
            print(f"deepmind {run}: Weather Lab CSV not online")
            return False
        tracks = read_deepmind_csv(text, run, ec)
    else:
        seeds = seeds_from_ecmwf(ec)
        print(f"{model} {run}: tracking {', '.join(seeds) or 'nothing'} from ECMWF starting positions")
        tracks = gridtrack.tracks(model, run, seeds, lambda la, lo: land().contains(Point(lo, la)))
    print(f"{MODELS[model]['label']} {run}: {tracks.storm.nunique()} North Indian Ocean systems")
    update_manifest(model, run, process(model, run, tracks))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ecmwf,deepmind,ukmet,gfs,icon,gem",   # quickest first
                    help="e.g. ecmwf,gfs,ukmet")
    ap.add_argument("--run", help="YYYYMMDD_HH (default: newest complete run not on the site yet)")
    ap.add_argument("--bufr", help="use a local ECMWF track BUFR for --run instead of downloading")
    a = ap.parse_args()

    import gridtrack
    complete = {"ecmwf": lambda r: ecmwf_tf_path(r) is not None,
                "gfs": gridtrack.gefs_complete, "ukmet": gridtrack.ukmo_complete,
                "icon": gridtrack.icon_complete, "gem": gridtrack.geps_complete,
                "deepmind": lambda r: deepmind_csv(r) is not None}
    changed = False
    t0 = time.time()
    for model in a.models.split(","):
        if time.time() - t0 > BUDGET_MIN * 60:
            print(f"{model}: skipped, {BUDGET_MIN} min time budget used up (drawn next job)")
            continue
        try:
            if a.run:
                run = a.run
            else:
                run = next((r for r in candidate_runs(model) if complete[model](r)), None)
                if run is None or run in done_runs(model):
                    print(f"{model}: nothing new ({run or 'no complete run'})")
                    continue
            changed |= do_model(model, run, a.bufr)
        except Exception as e:      # one model failing must not stop the others
            print(f"{model}: failed: {e!r}")
    if changed:
        shutil.copy(os.path.join(HERE, "web", "landfall.html"), SITE_DIR)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("new=true\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
