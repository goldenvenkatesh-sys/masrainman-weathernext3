# ============================================================
# MASRAINMAN — NEM & CYCLONE OUTLOOK (North Indian Ocean, days 1-30)
# ============================================================
#
# For each ECMWF 00Z / 12Z run this builds:
#   A. Cyclone genesis probability maps (LPA / depression / cyclonic storm) for days 1-5, 6-10,
#      11-15 and 1-15 from the 51 ECMWF ENS member tracks, and a run-to-run consistency map
#   B. Detailed cyclone analysis for days 1-15 per system: strike probability + member tracks,
#      intensity plume, day-by-day table, landfall / recurve / dissipate scenarios
#   C. 30-day Northeast Monsoon rainfall outlook: weekly totals and anomalies (weeks 1-2 ECMWF ENS,
#      weeks 3-4 and days 29-30 NOAA CFSv2 lagged ensemble) against the CPC 1991-2020 gauge normal,
#      Tamil Nadu (state + 4 regions) and state-wise summaries with category probabilities
#   D. Active / break NEM detector for Tamil Nadu (ECMWF HRES days 0-10 + observed MJO)
#   E. Track consistency monitor across successive runs
#   F. Forecast-period PDI against the IBTrACS 1991-2025 record (same calendar window and OND seasons)
#   G. Verification scoreboard: TN-SMART station rainfall (MAE, RMSE, Brier), cyclone detection rate,
#      false alarms, track and intensity error
#
# Every probability is a share of ensemble members; nothing is read off a picture.
#
# Website layout:
#   site/outlook/<YYYYMMDD_HH>/...png, summary.json
#   site/outlook/state/...   (history, verification archive, climatology cache — kept between runs)
#   site/data/manifest_outlook.json
#
# Usage:
#   python nem_outlook.py                       # newest complete 00Z/12Z ENS run, if not done yet
#   python nem_outlook.py --run 20261031_00     # one run
#   python nem_outlook.py --backfill-days 20    # also archive the last 20 days of ENS tracks
#   python nem_outlook.py --demo                # synthetic data, no downloads
# ============================================================

import os
import sys
import json
import glob
import math
import shutil
import argparse
import warnings
import traceback
from datetime import datetime, timedelta, timezone, date

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import masrainman_logo  # noqa: F401  (MasRainman logo on every saved figure)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep

import outlook_data as od

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# SETTINGS
# ============================================================

SITE_DIR = os.environ.get("SITE_DIR", "site")
OUT_ROOT = os.path.join(SITE_DIR, "outlook")
STATE = os.path.join(OUT_ROOT, "state")
KEEP_RUNS = int(os.environ.get("KEEP_RUNS", "4"))
HISTORY_RUNS = 10                 # runs kept for the consistency monitor (5 days of 00Z/12Z)
ARCHIVE_DAYS = 120                # verification archive length
N_ENS = 51
MS_TO_KT = 1.943844
DPI = 120

GEN_RADIUS_KM = 300               # genesis probability: event within 300 km of the point
STRIKE_RADIUS_KM = 120            # strike probability (ECMWF convention)
GEN_WINDOWS = [("d1_5", "Days 1-5", 0, 120), ("d6_10", "Days 6-10", 120, 240),
               ("d11_15", "Days 11-15", 240, 360), ("d1_15", "Days 1-15", 0, 360)]
GEN_LEVELS = [("lpa", "Low-pressure area", None), ("dep", "Depression (≥17 kt)", 17),
              ("cs", "Cyclonic storm (≥34 kt)", 34)]
CONSIST_PROB = 0.20               # a run 'shows' development where P(depression) >= 20 %

# IMD intensity classes from the 10-m wind (kt)
CLASSES = [("LPA", 0, "#7f8fa6"), ("D", 17, "#4f8ff7"), ("DD", 28, "#22c3c3"), ("CS", 34, "#22c38e"),
           ("SCS", 48, "#e5b52a"), ("VSCS", 64, "#f08a3a"), ("ESCS", 90, "#ef4f5f"), ("SuCS", 120, "#c03bd6")]
CLASS_NAMES = {"LPA": "Low", "D": "Depression", "DD": "Deep depression", "CS": "Cyclonic storm",
               "SCS": "Severe cyclonic storm", "VSCS": "Very severe CS", "ESCS": "Extremely severe CS",
               "SuCS": "Super cyclonic storm"}

# 30-day outlook weeks (hours from the run)
WEEKS = [("w1", "Week 1", "Days 1-7", 0, 168, "ens"), ("w2", "Week 2", "Days 8-14", 168, 336, "ens"),
         ("w3", "Week 3", "Days 15-21", 336, 504, "cfs"), ("w4", "Week 4", "Days 22-28", 504, 672, "cfs"),
         ("w5", "Days 29-30", "Days 29-30 (partial)", 672, 720, "cfs")]
CAT_EDGE = 19.0                   # IMD: normal = departure within ±19 %

# Active / break detector boxes and ramps (value at 0 -> value at 1)
AB_BOX_BOB = (8, 16, 80, 88)      # lat0, lat1, lon0, lon1
AB_BOX_EAST = (10, 14, 80, 86)
AB_BOX_COAST = (9, 14, 79.8, 82)
AB_BOX_TN = (8, 13.5, 76.5, 80.3)
AB_BOX_NORTH = (20, 24, 76, 82)
AB_INDICATORS = [  # key, label, units, quiet, active, weight
    ("vort", "SW Bay of Bengal 850 hPa vorticity", "10⁻⁵ s⁻¹", 0.0, 3.0, 0.20),
    ("east", "Easterly wind 850 hPa (10-14N, 80-86E)", "m/s", 2.0, 8.0, 0.20),
    ("moist", "Moisture flux to TN coast (TCWV × easterly)", "kg m⁻¹ s⁻¹", 100.0, 400.0, 0.25),
    ("rh", "Mid-level humidity over TN (700/500 hPa)", "%", 45.0, 78.0, 0.20),
    ("grad", "Pressure gradient central India − SW Bay", "hPa", 0.0, 6.0, 0.15),
]
MJO_GOOD = {2, 3, 4}              # phases favouring NEM rain over south peninsular India
MJO_BAD = {6, 7, 8}

# Rain verification
VERIF_BOX = (7.5, 14.0, 75.8, 80.6)
RAIN_THRESH = [2.5, 15.6, 64.5]   # IMD rainy day, moderate, heavy (mm/24 h)

# Colours (same dark theme as the landfall maps)
BG, PANEL, LAND, COAST, GRID = "#121826", "#0f1626", "#141b2a", "#8a94a8", "#2a3550"
TEXT, MUTED, ACCENT = "#e6ecf7", "#8fa0bf", "#2fb5ff"

plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": "#2e4a6a", "axes.facecolor": PANEL,
                     "figure.facecolor": BG, "savefig.facecolor": BG, "legend.facecolor": PANEL,
                     "legend.edgecolor": "#2e4a6a", "legend.labelcolor": TEXT})


def wind_class(kt):
    i = 0
    for k, (_, lo, _) in enumerate(CLASSES):
        if np.isfinite(kt) and kt >= lo:
            i = k
    return i


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def run_time(run):
    return datetime.strptime(run, "%Y%m%d_%H").replace(tzinfo=timezone.utc)


def fmt_run(run):
    return run_time(run).strftime("%d %b %Y %HZ")


def r1(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# GEOGRAPHY
# ============================================================

_geo = {}


def india_borders():
    if "borders" not in _geo:
        with open(os.path.join(HERE, "boundaries", "borders_india_view.geojson")) as f:
            gj = json.load(f)
        _geo["borders"] = ShapelyFeature([shape(ft["geometry"]) for ft in gj["features"]], ccrs.PlateCarree(),
                                         facecolor="none", edgecolor="#b8c2d6", linewidth=0.7)
    return _geo["borders"]


def admin1():
    """Natural Earth states/provinces around the North Indian Ocean: [(name, country, geometry)]."""
    if "admin1" not in _geo:
        rows = []
        try:
            shp = shpreader.natural_earth(resolution="10m", category="cultural",
                                          name="admin_1_states_provinces")
            for rec in shpreader.Reader(shp).records():
                a = rec.attributes
                c = a.get("admin", "")
                if c in ("India", "Sri Lanka", "Bangladesh", "Myanmar", "Pakistan", "Oman", "Yemen",
                         "Somalia", "Iran", "Thailand"):
                    rows.append((a.get("name", ""), c, rec.geometry))
        except Exception as e:
            print(f"(states not available: {e})")
        _geo["admin1"] = rows
    return _geo["admin1"]


def state_feature():
    if "states" not in _geo:
        geoms = [g for n, c, g in admin1() if c == "India"]
        _geo["states"] = ShapelyFeature(geoms, ccrs.PlateCarree(), facecolor="none",
                                        edgecolor="#4a5875", linewidth=0.35)
    return _geo["states"]


def region_of(lat, lon):
    """Coastal region name for a landfall point: Indian state (TN incl. Puducherry) or country."""
    p = Point(lon, lat)
    best, bd = None, 0.7
    for n, c, g in admin1():
        if g.bounds[0] - 1 < lon < g.bounds[2] + 1 and g.bounds[1] - 1 < lat < g.bounds[3] + 1:
            d = g.distance(p)
            if d < bd:
                best, bd = (n, c), d
    if best is None:
        return "Other coast"
    n, c = best
    if c != "India":
        return c
    if n in ("Tamil Nadu", "Puducherry"):
        return "Tamil Nadu & Puducherry"
    return n


_land = {}


def land_geom():
    """Natural Earth 10 m land without the small islands (as in landfall.py)."""
    if "geom" not in _land:
        shp = shpreader.natural_earth(resolution="10m", category="physical", name="land")
        polys = []
        for g in shpreader.Reader(shp).geometries():
            polys += [p for p in getattr(g, "geoms", [g]) if p.area >= 2.0]
        _land["geom"] = unary_union(polys)
        _land["prep"] = prep(_land["geom"])
    return _land["geom"], _land["prep"]


def tn_regions():
    """Tamil Nadu and four regions from the district polygons (boundaries/tn_districts.geojson)."""
    if "tn" in _geo:
        return _geo["tn"]
    with open(os.path.join(HERE, "boundaries", "tn_districts.geojson")) as f:
        gj = json.load(f)
    groups = {
        "North coastal TN": ["Chennai", "Thiruvallur", "Chengalpattu", "Kanchipuram", "Ranipet",
                             "Villupuram", "Cuddalore", "Kallakurichi"],
        "Delta TN": ["Thanjavur", "Thiruvarur", "Nagapattinam", "Mayiladuthurai", "Pudukkottai", "Ariyalur",
                     "Perambalur"],
        "South TN": ["Ramanathapuram", "Tuticorin", "Tirunelveli", "Tenkasi", "Kanniyakumari", "Virudhunagar",
                     "Madurai", "Sivaganga", "Theni"],
    }
    named = {d for v in groups.values() for d in v}
    geoms = {k: [] for k in groups}
    geoms["Interior & west TN"] = []
    allg = []
    for ft in gj["features"]:
        g = shape(ft["geometry"]).buffer(0)
        allg.append(g)
        d = ft["properties"]["district"]
        for k, v in groups.items():
            if d in v:
                geoms[k].append(g)
        if d not in named:
            geoms["Interior & west TN"].append(g)
    out = {"Tamil Nadu": unary_union(allg)}
    out.update({k: unary_union(v) for k, v in geoms.items()})
    _geo["tn"] = out
    return out


def state_polys():
    """Area polygons for the state-wise tables."""
    if "areas" in _geo:
        return _geo["areas"]
    out = dict(tn_regions())
    want = ["Kerala", "Karnataka", "Andhra Pradesh", "Telangana", "Odisha", "Puducherry"]
    for n, c, g in admin1():
        if c == "India" and n in want:
            out[n] = g
    _geo["areas"] = out
    return out


# Common 0.1° analysis grid for area averages
FINE_LAT = np.arange(5.0, 24.01, 0.1)
FINE_LON = np.arange(72.0, 90.01, 0.1)


def fine_masks():
    if "masks" not in _geo:
        X, Y = np.meshgrid(FINE_LON, FINE_LAT)
        pts = [Point(x, y) for x, y in zip(X.ravel(), Y.ravel())]
        masks = {}
        for name, g in state_polys().items():
            pg = prep(g)
            b = g.bounds
            m = np.array([b[0] <= p.x <= b[2] and b[1] <= p.y <= b[3] and pg.contains(p) for p in pts])
            masks[name] = m.reshape(X.shape)
        _geo["masks"] = masks
    return _geo["masks"]


def to_grid(lat, lon, field, lat2, lon2):
    """Bilinear regrid (last two axes) of field(lat, lon) onto lat2, lon2."""
    from scipy.interpolate import RegularGridInterpolator
    lat, lon = np.asarray(lat), np.asarray(lon)
    X, Y = np.meshgrid(lon2, lat2)
    pts = np.column_stack([Y.ravel(), X.ravel()])
    f = np.asarray(field)
    lead = f.shape[:-2]
    flat = f.reshape((-1,) + f.shape[-2:])
    out = np.empty((flat.shape[0], len(lat2), len(lon2)), np.float32)
    for i in range(flat.shape[0]):
        it = RegularGridInterpolator((lat, lon), flat[i], bounds_error=False, fill_value=np.nan)
        out[i] = it(pts).reshape(len(lat2), len(lon2))
    return out.reshape(lead + (len(lat2), len(lon2)))


def area_mean(fine_field, mask):
    v = fine_field[..., mask]
    return np.nanmean(v, axis=-1) if v.shape[-1] else np.full(fine_field.shape[:-2], np.nan)


def base_map(ax, extent, states=True, gridlabels=True):
    proj = ccrs.PlateCarree()
    ax.set_extent(extent, crs=proj)
    ax.set_facecolor("#0d1422")
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor=LAND, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor=COAST, linewidth=0.6, zorder=3)
    ax.add_feature(india_borders(), zorder=3)
    if states:
        ax.add_feature(state_feature(), zorder=3)
    gl = ax.gridlines(draw_labels=gridlabels, linewidth=0.3, color=GRID, x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"color": MUTED, "size": 7}
    return proj


def caption(fig, text, y=0.005):
    fig.text(0.01, y, text, color=MUTED, fontsize=7.5, ha="left", va="bottom")


# ============================================================
# A. GENESIS PROBABILITY
# ============================================================

GEN_LAT = np.arange(0.0, 27.01, 0.5)
GEN_LON = np.arange(45.0, 100.01, 0.5)


def genesis_events(tracks):
    """First point at which each member's system reaches each level.
    -> DataFrame(storm, member, level, step, lat, lon, existing) — existing = already at that level at T+0."""
    rows = []
    for (storm, m), t in tracks.groupby(["storm", "member"]):
        t = t.sort_values("step")
        kt = t.wind.to_numpy() * MS_TO_KT
        for lev, _, thr in GEN_LEVELS:
            ok = np.ones(len(t), bool) if thr is None else (kt >= thr)
            if ok.any():
                i = int(np.argmax(ok))
                r = t.iloc[i]
                rows.append((storm, int(m), lev, int(r.step), float(r.lat), float(r.lon), int(r.step) == 0))
    return pd.DataFrame(rows, columns=["storm", "member", "level", "step", "lat", "lon", "existing"])


def prob_grid(ev, n=N_ENS, radius=GEN_RADIUS_KM):
    """Share of members with at least one event within `radius` km of each grid point."""
    LO, LA = np.meshgrid(GEN_LON, GEN_LAT)
    hit = {}
    for m, g in ev.groupby("member"):
        h = np.zeros(LO.shape, bool)
        for la, lo in zip(g.lat, g.lon):
            h |= haversine_km(LA, LO, la, lo) <= radius
        hit[m] = h
    if not hit:
        return np.zeros(LO.shape)
    return np.sum(list(hit.values()), axis=0) / n


PROB_LEVELS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PROB_CMAP = mcolors.ListedColormap(["#1f4e8c", "#2a78c2", "#33a5d6", "#40c9b0", "#8edc5f", "#e8e04a",
                                    "#f5b43c", "#f07c32", "#e0412f", "#a8182e"])


def draw_prob_map(P, title, sub, path, ev=None, extent=(45, 100, 0, 27), cmap=PROB_CMAP,
                  levels=PROB_LEVELS, cbar_label="Probability (%)", hatch=None):
    fig = plt.figure(figsize=(9.6, 5.6))
    ax = fig.add_axes([0.04, 0.1, 0.84, 0.8], projection=ccrs.PlateCarree())
    proj = base_map(ax, extent)
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    data = np.ma.masked_less(P * 100 if P.max() <= 1.0001 and cbar_label.startswith("Prob") else P, levels[0])
    cf = ax.contourf(GEN_LON, GEN_LAT, data, levels=levels, cmap=cmap, norm=norm, transform=proj,
                     alpha=0.88, zorder=1, extend="neither")
    if hatch is not None:
        ax.contourf(GEN_LON, GEN_LAT, hatch, levels=[0.5, 1.5], colors="none", hatches=["..."],
                    transform=proj, zorder=2)
    if ev is not None and len(ev):
        ax.scatter(ev.lon, ev.lat, s=6, color="white", alpha=0.55, lw=0, transform=proj, zorder=4)
    cax = fig.add_axes([0.9, 0.16, 0.018, 0.68])
    cb = fig.colorbar(cf, cax=cax, ticks=levels)
    cb.set_label(cbar_label, color=TEXT)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    fig.text(0.04, 0.955, title, fontsize=12.5, fontweight="bold", color=TEXT)
    fig.text(0.04, 0.918, sub, fontsize=8.5, color=MUTED)
    caption(fig, "Share of the 51 ECMWF ENS members whose system first reaches this intensity within "
                 f"{GEN_RADIUS_KM} km of the point (dots: member genesis points). Not an official forecast.")
    return save(fig, path)


def section_genesis(run, tracks, outdir, n=N_ENS):
    ev = genesis_events(tracks)
    new = ev[~ev.existing]
    res = {"windows": [], "basin": [], "existing": []}
    for st, g in ev[ev.existing & (ev.level == "lpa")].groupby("storm"):
        res["existing"].append({"storm": st, "lat": r1(g.lat.median()), "lon": r1(g.lon.median())})
    for wk, wlabel, h0, h1 in GEN_WINDOWS:
        maps = {}
        for lev, llabel, _ in GEN_LEVELS:
            e = new[(new.level == lev) & (new.step > h0) & (new.step <= h1)]
            P = prob_grid(e, n)
            png = f"{outdir}/genesis_{lev}_{wk}.png"
            draw_prob_map(P, f"Genesis probability · {llabel}", f"ECMWF ENS · init {fmt_run(run)} · {wlabel}",
                          png, ev=e)
            maps[lev] = {"png": os.path.relpath(png, SITE_DIR), "max": r1(P.max() * 100, 0)}
        res["windows"].append({"id": wk, "label": wlabel, "maps": maps})
        # basin-wide probabilities: share of members with any new event in the window
        for basin, suf in (("Bay of Bengal", "B"), ("Arabian Sea", "A")):
            row = {"window": wk, "basin": basin}
            for lev, _, _ in GEN_LEVELS:
                e = new[(new.level == lev) & (new.step > h0) & (new.step <= h1) & new.storm.str.endswith(suf)]
                row[lev] = r1(100 * e.member.nunique() / n, 0)
            res["basin"].append(row)
    ev.assign(run=run).to_csv(os.path.join(STATE, "genesis", f"{run}.csv"), index=False)
    res["consistency"] = genesis_consistency(run, outdir)
    return res


def genesis_consistency(run, outdir, n_runs=4):
    """Where do the latest runs agree on development? For the valid period every run covers,
    count the runs with P(depression forms within 300 km) >= 20 %."""
    files = sorted(glob.glob(os.path.join(STATE, "genesis", "*.csv")))[-n_runs:]
    runs = [os.path.basename(f)[:-4] for f in files]
    if not runs:
        return None
    t_end = min(run_time(r) for r in runs) + timedelta(hours=360)
    t_start = run_time(run)
    if t_end - t_start < timedelta(days=3):
        return None
    count = np.zeros((len(GEN_LAT), len(GEN_LON)))
    for f, r in zip(files, runs):
        ev = pd.read_csv(f)
        vt = run_time(r) + pd.to_timedelta(ev.step, unit="h")
        e = ev[(ev.level == "dep") & (~ev.existing.astype(bool)) & (vt > t_start) & (vt <= t_end)]
        count += prob_grid(e) >= CONSIST_PROB
    png = f"{outdir}/genesis_consistency.png"
    levels = [0.5, 1.5, 2.5, 3.5, 4.5][:len(runs) + 1]
    cmap = mcolors.ListedColormap(["#33a5d6", "#8edc5f", "#f5b43c", "#e0412f"][:len(levels) - 1])
    draw_prob_map(count, "Run-to-run consistency · depression development",
                  f"Runs (of the last {len(runs)}) with P ≥ {int(CONSIST_PROB * 100)}% · valid "
                  f"{t_start:%d %b %HZ} – {t_end:%d %b %HZ}", png, cmap=cmap, levels=levels,
                  cbar_label="Number of runs")
    return {"png": os.path.relpath(png, SITE_DIR), "runs": runs, "valid_from": t_start.isoformat(),
            "valid_to": t_end.isoformat(), "max_runs": int(count.max())}


# ============================================================
# B. CYCLONE ANALYSIS (days 1-15)
# ============================================================

def mean_track(tr):
    g = tr.groupby("step").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("member", "nunique"),
                               wind=("wind", "median"), mslp=("mslp", "median")).reset_index()
    return g[g.n >= max(2, tr.member.nunique() / 2)]


def interp_track(t, dt=1.0):
    st = t.step.to_numpy(float)
    if len(st) < 2:
        return t.lat.to_numpy(), t.lon.to_numpy()
    s = np.arange(st[0], st[-1] + 1e-6, dt)
    return np.interp(s, st, t.lat), np.interp(s, st, t.lon)


def member_outcome(t):
    """Landfall region / skirts the coast / recurves / dissipates over sea, for one member track."""
    geom, pl = land_geom()
    t = t.sort_values("step")
    la, lo = interp_track(t, 1.0)
    if len(la) and pl.contains(Point(lo[0], la[0])):
        return "Starts over land", None
    for i in range(1, len(la)):
        if pl.contains(Point(lo[i], la[i])):
            step = float(t.step.iloc[0]) + i          # hourly interpolation from the first step
            prev = t[t.step <= step]
            w = prev.wind.iloc[-1] * MS_TO_KT if len(prev) else np.nan
            return "Landfall: " + region_of(la[i], lo[i]), {"lat": la[i], "lon": lo[i], "step": step, "kt": w}
    dmin = min(geom.distance(Point(x, y)) for x, y in zip(lo[::6], la[::6])) * 111
    if dmin < 100:
        return "Moves close to the coast, no landfall", None
    imin = int(np.argmin(lo))
    if lo[-1] - lo[imin] > 1.0 and la[-1] > la[imin]:
        return "Recurves north-east over sea", None
    return "Weakens over sea", None


def system_stats(storm, tr, hres, run, n=N_ENS):
    t0 = run_time(run)
    tracked = tr.member.nunique()
    peak = tr.groupby("member").wind.max() * MS_TO_KT
    reach = {c: r1(100 * (peak >= lo).sum() / n, 0) for c, lo, _ in CLASSES[1:6]}
    outcomes, lfs = {}, []
    for m, t in tr.groupby("member"):
        o, lf = member_outcome(t)
        outcomes[o] = outcomes.get(o, 0) + 1
        if lf:
            lfs.append(dict(lf, member=int(m), region=o[10:]))
    outcomes["Does not form / not tracked"] = n - tracked
    scen = sorted(({"scenario": k, "pct": r1(100 * v / n, 0), "members": v} for k, v in outcomes.items() if v),
                  key=lambda x: -x["members"])
    mt = mean_track(tr)
    days = []
    for d in range(0, 16):
        s = d * 24
        g = tr[tr.step == s]
        row = {"day": d, "valid": (t0 + timedelta(hours=s)).strftime("%d %b"), "members": int(g.member.nunique())}
        if len(g) >= 2:
            mla, mlo = g.lat.mean(), g.lon.mean()
            kt = g.wind * MS_TO_KT
            row.update(lat=r1(mla), lon=r1(mlo), wind_med=r1(kt.median(), 0), wind_p90=r1(kt.quantile(0.9), 0),
                       cls=CLASSES[wind_class(kt.median())][0],
                       p_dep=r1(100 * (kt >= 17).sum() / n, 0), p_cs=r1(100 * (kt >= 34).sum() / n, 0),
                       spread_km=r1(haversine_km(g.lat, g.lon, mla, mlo).mean(), 0),
                       mslp=r1(g.mslp.median(), 0))
        days.append(row)
    start = tr[tr.step == tr.step.min()]
    hres_t = None
    if hres is not None and len(hres):
        hs = hres[hres.storm == storm]
        if not len(hs):          # HRES ids can differ (genesis ids): nearest start within 300 km
            for s2, g2 in hres.groupby("storm"):
                f = g2[g2.step == g2.step.min()]
                if haversine_km(f.lat.mean(), f.lon.mean(), start.lat.mean(), start.lon.mean()).min() < 300:
                    hs = g2
                    break
        if len(hs):
            hres_t = hs.sort_values("step")
    return {
        "storm": storm, "basin": "Bay of Bengal" if storm.endswith("B") else "Arabian Sea",
        "tracked": int(tracked), "members": n, "p_form": r1(100 * tracked / n, 0),
        "start": {"lat": r1(start.lat.mean()), "lon": r1(start.lon.mean()), "step": int(tr.step.min())},
        "reach": reach, "peak_med": r1(peak.median(), 0), "peak_p10": r1(peak.quantile(0.1), 0),
        "peak_p90": r1(peak.quantile(0.9), 0),
        "peak_step_med": r1(tr.loc[tr.groupby("member").wind.idxmax()].step.median(), 0),
        "scenarios": scen, "days": days,
        "landfalls": lfs,
        "mean_track": [[int(s), r1(a, 2), r1(o, 2)] for s, a, o in zip(mt.step, mt.lat, mt.lon)],
        "hres": None if hres_t is None else {
            "peak": r1(hres_t.wind.max() * MS_TO_KT, 0),
            "track": [[int(s), r1(a, 2), r1(o, 2)] for s, a, o in zip(hres_t.step, hres_t.lat, hres_t.lon)],
            "wind": [[int(s), r1(w * MS_TO_KT, 0)] for s, w in zip(hres_t.step, hres_t.wind)]},
    }


def draw_system_track(run, storm, tr, info, path):
    lats = list(tr.lat)
    lons = list(tr.lon)
    lat0, lat1 = max(-2, min(lats) - 3), max(lats) + 3
    lon0, lon1 = min(lons) - 3, max(lons) + 3
    if lon1 - lon0 < 16:
        c = (lon0 + lon1) / 2; lon0, lon1 = c - 8, c + 8
    if lat1 - lat0 < 12:
        c = (lat0 + lat1) / 2; lat0, lat1 = c - 6, c + 6
    # strike probability (any point of the member track within 120 km, hourly interpolation)
    glat = np.arange(lat0, lat1 + 0.01, 0.25)
    glon = np.arange(lon0, lon1 + 0.01, 0.25)
    LO, LA = np.meshgrid(glon, glat)
    hit = np.zeros(LO.shape)
    for m, t in tr.groupby("member"):
        la, lo = interp_track(t.sort_values("step"), 1.0)
        h = np.zeros(LO.shape, bool)
        for a, o in zip(la[::2], lo[::2]):
            box = (np.abs(LA - a) < 1.3) & (np.abs(LO - o) < 1.4)
            h[box] |= haversine_km(LA[box], LO[box], a, o) <= STRIKE_RADIUS_KM
        hit += h
    P = 100 * hit / N_ENS

    fig = plt.figure(figsize=(9.6, 9.6 * (lat1 - lat0) / (lon1 - lon0) * 0.92 + 0.9))
    ax = fig.add_axes([0.04, 0.09, 0.84, 0.82], projection=ccrs.PlateCarree())
    proj = base_map(ax, (lon0, lon1, lat0, lat1))
    lv = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cf = ax.contourf(glon, glat, np.ma.masked_less(P, 5), levels=lv, cmap=PROB_CMAP,
                     norm=mcolors.BoundaryNorm(lv, PROB_CMAP.N), alpha=0.55, transform=proj, zorder=1)
    for m, t in tr.groupby("member"):
        t = t.sort_values("step")
        ax.plot(t.lon, t.lat, color="#9fb3d1", lw=0.45, alpha=0.5, transform=proj, zorder=2)
        cls = [wind_class(w * MS_TO_KT) for w in t.wind]
        ax.scatter(t.lon[::2], t.lat[::2], s=5, c=[CLASSES[c][2] for c in cls[::2]], lw=0, transform=proj,
                   zorder=3)
    mt = mean_track(tr)
    ax.plot(mt.lon, mt.lat, color="white", lw=3.2, transform=proj, zorder=5)
    ax.plot(mt.lon, mt.lat, color=BG, lw=1.2, transform=proj, zorder=6)
    t0 = run_time(run)
    for _, r in mt[mt.step % 24 == 0].iterrows():
        ax.text(r.lon + 0.2, r.lat + 0.2, (t0 + timedelta(hours=int(r.step))).strftime("%d"), color=TEXT,
                fontsize=7, transform=proj, zorder=7,
                bbox=dict(facecolor=BG, edgecolor="none", alpha=0.6, pad=0.5))
    if info.get("hres"):
        h = np.array(info["hres"]["track"], float)
        ax.plot(h[:, 2], h[:, 1], color="#ff4d4d", lw=1.8, ls="--", transform=proj, zorder=6)
    used = sorted({wind_class(w * MS_TO_KT) for w in tr.wind})
    handles = [Line2D([], [], ls="", marker="o", ms=5, color=CLASSES[k][2], label=CLASSES[k][0]) for k in used]
    handles += [Line2D([], [], color="white", lw=2.5, label="Ensemble mean (dates)")]
    if info.get("hres"):
        handles += [Line2D([], [], color="#ff4d4d", lw=1.8, ls="--", label="HRES deterministic")]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=7.5, ncol=2, title="IMD class (member points)",
                    title_fontsize=7.5)
    leg.get_title().set_color(TEXT)
    cax = fig.add_axes([0.9, 0.16, 0.018, 0.66])
    cb = fig.colorbar(cf, cax=cax, ticks=lv)
    cb.set_label(f"Strike probability within {STRIKE_RADIUS_KM} km (%)", color=TEXT)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    alias = {v: k for k, v in NAMES.items()}.get(storm)
    fig.text(0.04, 0.955, f"System {storm}{' (Invest)' if storm[:1] == '9' else ''} · ECMWF ENS tracks, days 1-15"
             + (f" · ECMWF id {alias}" if alias else ""), fontsize=12.5, fontweight="bold")
    fig.text(0.04, 0.925, f"Init {fmt_run(run)} · {info['tracked']}/{N_ENS} members track it · "
                          f"P(≥ depression) {info['reach']['D']:.0f}% · P(≥ cyclonic storm) {info['reach']['CS']:.0f}%",
             fontsize=8.5, color=MUTED)
    caption(fig, "Thin lines: member tracks · dots coloured by IMD class from the member's 10-m wind "
                 "(ENS winds run lower than observed) · Not an official forecast.")
    return save(fig, path)


def draw_system_plume(run, storm, tr, info, path):
    t0 = run_time(run)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.6, 6.8), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
    fig.subplots_adjust(left=0.08, right=0.84, top=0.9, bottom=0.1)
    bands = CLASSES[1:7]
    for i, (c, lo, col) in enumerate(bands):
        hi = bands[i + 1][1] if i + 1 < len(bands) else 140
        ax.axhspan(lo, hi, color=col, alpha=0.08, lw=0)
        ax.text(1.005, (lo + min(hi, 125)) / 2, c, transform=ax.get_yaxis_transform(), color=col, fontsize=7.5,
                va="center")
    for m, t in tr.groupby("member"):
        t = t.sort_values("step")
        x = [t0 + timedelta(hours=int(s)) for s in t.step]
        ax.plot(x, t.wind * MS_TO_KT, color="#9fb3d1", lw=0.5, alpha=0.45)
        ax2.plot(x, t.mslp, color="#9fb3d1", lw=0.5, alpha=0.45)
    g = tr.groupby("step")
    q = pd.DataFrame({"p10": g.wind.quantile(0.1), "p50": g.wind.median(), "p90": g.wind.quantile(0.9),
                      "n": g.member.nunique(), "pm": g.mslp.median()}).reset_index()
    q = q[q.n >= 5]
    x = [t0 + timedelta(hours=int(s)) for s in q.step]
    ax.fill_between(x, q.p10 * MS_TO_KT, q.p90 * MS_TO_KT, color=ACCENT, alpha=0.22, lw=0, label="10-90 %")
    ax.plot(x, q.p50 * MS_TO_KT, color="white", lw=2.2, label="Ensemble median")
    ax2.plot(x, q.pm, color="white", lw=2.0)
    if info.get("hres"):
        hw = np.array(info["hres"]["wind"], float)
        ax.plot([t0 + timedelta(hours=int(s)) for s in hw[:, 0]], hw[:, 1], color="#ff4d4d", lw=1.8,
                label="HRES deterministic")
    ax.set_ylim(0, max(70, float(np.nanmax(tr.wind * MS_TO_KT)) + 8))
    ax.set_ylabel("10-m wind (kt)")
    ax2.set_ylabel("Central MSLP (hPa)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(color=GRID, lw=0.4)
    ax2.grid(color=GRID, lw=0.4)
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.text(0.08, 0.955, f"System {storm} · ensemble intensity plume", fontsize=12.5, fontweight="bold")
    fig.text(0.08, 0.925, f"ECMWF ENS · init {fmt_run(run)} · median peak {info['peak_med']:.0f} kt "
                          f"(10-90 %: {info['peak_p10']:.0f}–{info['peak_p90']:.0f} kt)", fontsize=8.5, color=MUTED)
    caption(fig, "Grey: members · shading: 10th-90th percentile while ≥5 members track it · bands: IMD classes. "
                 "Not an official forecast.")
    return save(fig, path)


def section_cyclones(run, tracks, hres, outdir, n=N_ENS):
    systems = []
    for storm, tr in tracks.groupby("storm"):
        if tr.member.nunique() < max(3, 0.06 * n):
            continue
        info = system_stats(storm, tr, hres, run, n)
        info["track_png"] = os.path.relpath(draw_system_track(run, storm, tr, info,
                                                              f"{outdir}/sys_{storm}_track.png"), SITE_DIR)
        info["plume_png"] = os.path.relpath(draw_system_plume(run, storm, tr, info,
                                                              f"{outdir}/sys_{storm}_plume.png"), SITE_DIR)
        systems.append(info)
        print(f"  system {storm}: {info['tracked']}/{n} members, P(D) {info['reach']['D']}%, "
              f"P(CS) {info['reach']['CS']}%")
    systems.sort(key=lambda s: -s["tracked"])
    return systems


# ============================================================
# PDI
# ============================================================

def pdi_track(wind_kt, steps):
    """1e-6 Σ v³ Δt/6h over fixes with v ≥ 34 kt (kt³ per 6-hourly fix)."""
    w = np.asarray(wind_kt, float)
    s = np.asarray(steps, float)
    if len(w) < 2:
        return 0.0
    dt = np.diff(np.concatenate([s, [s[-1] + (s[-1] - s[-2])]])) / 6.0
    v = np.where(np.isfinite(w) & (w >= 34), w, 0)
    return float(np.sum(v ** 3 * dt) * 1e-6)


def section_pdi(run, tracks, outdir, n=N_ENS):
    t0 = run_time(run)
    per_member = np.zeros(n)
    members = sorted(tracks.member.unique())
    idx = {m: i for i, m in enumerate(members[:n])}
    for (storm, m), t in tracks.groupby(["storm", "member"]):
        if m in idx:
            t = t.sort_values("step")
            per_member[idx[m]] += pdi_track(t.wind * MS_TO_KT, t.step)
    fc = {"median": r1(np.median(per_member), 2), "p10": r1(np.percentile(per_member, 10), 2),
          "p90": r1(np.percentile(per_member, 90), 2), "mean": r1(per_member.mean(), 2),
          "p_any": r1(100 * (per_member > 0).mean(), 0)}
    ib = None
    try:
        ib = od.demo_ibtracs() if DEMO else od.ibtracs_ni(STATE)
    except Exception as e:
        print(f"  IBTrACS not available: {e}")
    res = {"forecast": fc, "window": f"{t0:%d %b} – {(t0 + timedelta(days=15)):%d %b}"}
    if ib is None or not len(ib):
        return res
    ib = ib.sort_values(["sid", "time"])
    win0, win1 = (t0.month, t0.day), (t0 + timedelta(days=15))
    years = range(1991, 2026)
    seas, wins, counts = [], [], []
    for y in years:
        s = ib[(ib.time >= f"{y}-10-01") & (ib.time < f"{y + 1}-01-01")]
        p = sum(pdi_track(g.wind_kt, np.arange(len(g)) * 6.0) for _, g in s.groupby("sid"))
        pk = s.groupby("sid").wind_kt.max()
        seas.append(p)
        counts.append({"year": y, "dep": int((pk >= 17).sum()), "cs": int((pk >= 34).sum()),
                       "scs": int((pk >= 48).sum()), "pdi": round(p, 2)})
        a = datetime(y, t0.month, t0.day)
        b = a + timedelta(days=15)
        w = ib[(ib.time >= a) & (ib.time < b)]
        wins.append(sum(pdi_track(g.wind_kt, np.arange(len(g)) * 6.0) for _, g in w.groupby("sid")))
    wins, seas = np.array(wins), np.array(seas)
    cur = ib[(ib.time >= f"{t0.year}-10-01") & (ib.time < t0.replace(tzinfo=None))]
    cur_pdi = sum(pdi_track(g.wind_kt, np.arange(len(g)) * 6.0) for _, g in cur.groupby("sid"))
    cur_pk = cur.groupby("sid").wind_kt.max()
    res.update({
        "clim_window_mean": r1(wins.mean(), 2), "clim_window_median": r1(np.median(wins), 2),
        "clim_window_p_zero": r1(100 * (wins == 0).mean(), 0),
        "p_exceed_clim": r1(100 * (per_member > wins.mean()).mean(), 0),
        "clim_season_mean": r1(seas.mean(), 2),
        "clim_counts": {"dep": r1(np.mean([c["dep"] for c in counts]), 1),
                        "cs": r1(np.mean([c["cs"] for c in counts]), 1),
                        "scs": r1(np.mean([c["scs"] for c in counts]), 1)},
        "season_to_date": {"pdi": r1(cur_pdi, 2), "dep": int((cur_pk >= 17).sum()), "cs": int((cur_pk >= 34).sum()),
                           "last_fix": None if not len(cur) else str(cur.time.max())[:16]},
        "seasons": counts,
    })
    # chart
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.25})
    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.06, right=0.98)
    yrs = [c["year"] for c in counts]
    a1.bar(yrs, seas, color="#2a78c2", width=0.75, label="Observed OND PDI (IBTrACS)")
    a1.axhline(seas.mean(), color="#f5b43c", lw=1.4, ls="--", label=f"1991-2025 mean {seas.mean():.1f}")
    a1.bar([t0.year], [cur_pdi], color="#40c9b0", width=0.75, label=f"{t0.year} so far (to {t0:%d %b})")
    a1.set_ylabel("PDI (10⁶ kt³)")
    a1.set_title("Season (Oct-Dec) PDI, North Indian Ocean", color=TEXT, fontsize=10)
    a1.legend(fontsize=7.5, loc="upper left")
    a1.grid(axis="y", color=GRID, lw=0.4)
    bp = a2.boxplot([wins, per_member], widths=0.55, patch_artist=True, whis=(10, 90), showfliers=False)
    for patch, col in zip(bp["boxes"], ["#2a78c2", "#e0412f"]):
        patch.set_facecolor(col); patch.set_alpha(0.7); patch.set_edgecolor(TEXT)
    for k in ("whiskers", "caps", "medians"):
        for l in bp[k]:
            l.set_color(TEXT)
    a2.set_xticks([1, 2], [f"Observed\n1991-2025", "ECMWF ENS\nforecast"])
    a2.set_title(f"Same 15 days ({res['window']})", color=TEXT, fontsize=10)
    a2.set_ylabel("PDI (10⁶ kt³)")
    a2.grid(axis="y", color=GRID, lw=0.4)
    fig.text(0.06, 0.95, "Power Dissipation Index · forecast period vs history", fontsize=12.5, fontweight="bold")
    fig.text(0.06, 0.905, "PDI = 10⁻⁶ Σ v³ over 6-hourly fixes ≥ 34 kt. A 15-day forecast PDI is NOT a seasonal "
                          "PDI. Observed winds: IMD (3-min) via IBTrACS; ENS winds run low.", fontsize=8.2, color=MUTED)
    res["png"] = os.path.relpath(save(fig, f"{outdir}/pdi.png"), SITE_DIR)
    return res


# ============================================================
# C. 30-DAY NORTHEAST MONSOON RAINFALL OUTLOOK
# ============================================================

DEMO = False
MAP_EXTENT = (72.5, 90, 5, 23.5)
_landmask = {}


def fine_land():
    """Land mask on the 0.1° grid (CPC normals are gauge-based, land only)."""
    if "m" not in _landmask:
        _, pl = land_geom()
        X, Y = np.meshgrid(FINE_LON, FINE_LAT)
        _landmask["m"] = np.array([pl.contains(Point(x, y)) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
    return _landmask["m"]


def fill_nan(field):
    """Nearest-neighbour fill of NaNs (ocean in the gauge climatology) so coastal points regrid cleanly."""
    from scipy.ndimage import distance_transform_edt
    out = field.copy()
    for k in range(out.shape[0]):
        m = ~np.isfinite(out[k])
        if m.any() and (~m).any():
            idx = distance_transform_edt(m, return_distances=False, return_indices=True)
            out[k] = out[k][tuple(idx)]
    return out


def climatology():
    if DEMO:
        return od.demo_climatology()
    return od.cpc_climatology(STATE)


def doy_index(d):
    i = d.timetuple().tm_yday - 1
    if d.month > 2 and (d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0)):
        i -= 1                                    # the climatology has 365 days
    return min(i, 364)


RAIN_LEVELS = [0, 1, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400]
RAIN_CMAP = mcolors.ListedColormap(["#1b2433", "#b4d7f0", "#7ab8e8", "#3f8fdc", "#2a6fb8", "#24a36b", "#7fcf3f",
                                    "#f2e33a", "#f5a52f", "#ef5b2b", "#c9233a", "#8a1a7a"])
ANOM_LEVELS = [-100, -60, -40, -19, 19, 40, 60, 100, 200]
ANOM_CMAP = mcolors.ListedColormap(["#8c510a", "#bf812d", "#dfc27d", "#e8e8e0", "#80cdc1", "#35978f",
                                    "#01665e", "#003c30"])


def category(dep):
    if not np.isfinite(dep):
        return "n/a"
    return "Above normal" if dep > CAT_EDGE else "Below normal" if dep < -CAT_EDGE else "Near normal"


def draw_week_map(run, wk, mean, dep, agree, src_label, path):
    fig = plt.figure(figsize=(11.2, 5.9))
    proj = ccrs.PlateCarree()
    a1 = fig.add_axes([0.03, 0.12, 0.44, 0.74], projection=proj)
    a2 = fig.add_axes([0.51, 0.12, 0.44, 0.74], projection=proj)
    base_map(a1, MAP_EXTENT)
    base_map(a2, MAP_EXTENT)
    n1 = mcolors.BoundaryNorm(RAIN_LEVELS, RAIN_CMAP.N)
    c1 = a1.pcolormesh(FINE_LON, FINE_LAT, mean, cmap=RAIN_CMAP, norm=n1, transform=proj, zorder=1, shading="auto")
    land = fine_land()
    n2 = mcolors.BoundaryNorm(ANOM_LEVELS, ANOM_CMAP.N)
    c2 = a2.pcolormesh(FINE_LON, FINE_LAT, np.ma.masked_where(~land | ~np.isfinite(dep), np.clip(dep, -100, 199)),
                       cmap=ANOM_CMAP, norm=n2, transform=proj, zorder=1, shading="auto")
    st = np.ma.masked_where(~land, agree)
    a2.contourf(FINE_LON, FINE_LAT, st, levels=[0.5, 1.5], colors="none", hatches=["...."], transform=proj, zorder=2)
    for ax, c, lab, ticks in ((a1, c1, "Total rain (mm)", RAIN_LEVELS),
                              (a2, c2, "Departure from 1991-2020 normal (%)", ANOM_LEVELS)):
        cb = fig.colorbar(c, ax=ax, orientation="horizontal", fraction=0.05, pad=0.06, ticks=ticks)
        cb.set_label(lab, color=TEXT)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
    t0 = run_time(run)
    _, name, days, h0, h1, _ = wk
    d0, d1 = (t0 + timedelta(hours=h0)), (t0 + timedelta(hours=h1))
    fig.text(0.03, 0.955, f"NEM rainfall outlook · {name} ({days}) · {d0:%d %b} – {d1:%d %b}",
             fontsize=12.5, fontweight="bold")
    fig.text(0.03, 0.915, f"{src_label} · ensemble mean · init {fmt_run(run)} · dots: ≥80% of members agree "
                          "on the sign of the departure", fontsize=8.5, color=MUTED)
    caption(fig, "Normal: NOAA CPC Unified gauge 1991-2020 (land only). Raw model output, no bias correction; "
                 "weeks 3+ have low skill. Not an official forecast.")
    return save(fig, path)


def area_table(members_fine, clim_fine, week_conf):
    """Per area: member area-means -> mean, spread, normal, departure, category probabilities."""
    rows = []
    for name, mask in fine_masks().items():
        if not mask.any():
            continue
        mem = area_mean(members_fine, mask)            # (members,)
        norm = float(area_mean(clim_fine[None], mask)[0])
        mean = float(np.nanmean(mem))
        dep = 100 * (mean - norm) / norm if norm > 0.5 else np.nan
        hi, lo = norm * (1 + CAT_EDGE / 100), norm * (1 - CAT_EDGE / 100)
        p_above, p_below = 100 * np.mean(mem > hi), 100 * np.mean(mem < lo)
        p_near = 100 - p_above - p_below
        cat = category(dep)
        p_cat = {"Above normal": p_above, "Below normal": p_below, "Near normal": p_near}.get(cat, np.nan)
        conf = "High" if p_cat >= 65 else "Moderate" if p_cat >= 45 else "Low"
        order = ["Low", "Moderate", "High"]
        conf = order[min(order.index(conf), order.index(week_conf))]
        rows.append({"area": name, "mean": r1(mean), "p10": r1(np.nanpercentile(mem, 10)),
                     "p90": r1(np.nanpercentile(mem, 90)), "p25": r1(np.nanpercentile(mem, 25)),
                     "p75": r1(np.nanpercentile(mem, 75)), "normal": r1(norm), "dep": r1(dep, 0),
                     "category": cat, "p_above": r1(p_above, 0), "p_near": r1(p_near, 0),
                     "p_below": r1(p_below, 0), "confidence": conf, "members": int(len(mem))})
    return rows


def section_nem(run, outdir):
    t0 = run_time(run)
    clat, clon, clim = climatology()
    clim = fill_nan(np.asarray(clim, np.float32))
    weeks_out, total_mean, total_norm = [], {}, {}
    ens = od.demo_ens_tp(run, [168, 336]) if DEMO else od.ens_tp(run, [168, 336])
    cfs = None
    need_cfs = any(w[5] == "cfs" for w in WEEKS)
    if need_cfs:
        vf = (t0 + timedelta(hours=336)).replace(hour=0)
        vt = (t0 + timedelta(hours=720)).replace(hour=0) + timedelta(days=1)
        try:
            cfs = od.demo_cfs(vf, vt) if DEMO else od.cfs_lagged(vf, vt)
        except Exception as e:
            print(f"  CFSv2 failed: {e!r}")
    for wk in WEEKS:
        key, name, days, h0, h1, src = wk
        d0 = (t0 + timedelta(hours=h0)).date()
        ndays = (h1 - h0) // 24
        dates = [d0 + timedelta(days=i) for i in range(ndays)]
        cw = clim[[doy_index(d) for d in dates]].sum(0)
        clim_f = to_grid(clat, clon, cw, FINE_LAT, FINE_LON)
        if src == "ens":
            if h1 not in ens or (h0 and h0 not in ens):
                continue
            la, lo, a1 = ens[h1]
            a0 = ens[h0][2] if h0 else 0
            mem = np.maximum(a1 - a0, 0)
            label, conf = f"ECMWF ENS ({mem.shape[0]} members)", "High" if key == "w1" else "Moderate"
        else:
            if cfs is None:
                continue
            la, lo, cdays, arr, labels = cfs
            sel = [i for i, d in enumerate(cdays) if d in dates]
            if len(sel) < max(1, ndays - 1):
                continue
            mem = arr[:, sel].sum(1) * ndays / len(sel)
            label, conf = f"NOAA CFSv2 lagged ensemble ({arr.shape[0]} members)", "Low"
        mem_f = to_grid(la, lo, mem, FINE_LAT, FINE_LON)
        mean_f = np.nanmean(mem_f, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            dep_f = np.where(clim_f > 0.5 * ndays / 7, 100 * (mean_f - clim_f) / clim_f, np.nan)
        above = np.mean(mem_f > clim_f[None], 0)
        agree = ((above >= 0.8) | (above <= 0.2)).astype(float)
        png = draw_week_map(run, wk, mean_f, dep_f, agree, label, f"{outdir}/nem_{key}.png")
        if key == "w5":
            conf = "Low"
        areas = area_table(mem_f, clim_f, conf)
        for a in areas:
            total_mean[a["area"]] = total_mean.get(a["area"], 0) + a["mean"]
            total_norm[a["area"]] = total_norm.get(a["area"], 0) + a["normal"]
        weeks_out.append({"id": key, "name": name, "days": days, "from": dates[0].isoformat(),
                          "to": dates[-1].isoformat(), "source": label, "lead_confidence": conf,
                          "png": os.path.relpath(png, SITE_DIR), "areas": areas})
        print(f"  NEM {name}: {label}")
    total = []
    for a in total_mean:
        n = total_norm[a]
        dep = 100 * (total_mean[a] - n) / n if n > 0.5 else np.nan
        total.append({"area": a, "mean": r1(total_mean[a]), "normal": r1(n), "dep": r1(dep, 0),
                      "category": category(dep)})
    return {"weeks": weeks_out, "total": total, "weeks_covered": [w["id"] for w in weeks_out],
            "climatology": "NOAA CPC Unified gauge daily normal 1991-2020 (land only)"}


# ============================================================
# D. ACTIVE / BREAK NEM DETECTOR + ENVIRONMENT MAPS
# ============================================================

def box_mean(f, box):
    la, lo, v = f
    a = (la >= box[0]) & (la <= box[1])
    b = (lo >= box[2]) & (lo <= box[3])
    return float(np.nanmean(v[np.ix_(a, b)]))


def indicators(d):
    out = {}
    if "vo850" in d:
        out["vort"] = box_mean(d["vo850"], AB_BOX_BOB) * 1e5
    if "u850" in d:
        out["east"] = -box_mean(d["u850"], AB_BOX_EAST)
    if "u850" in d and "tcwv" in d:
        la, lo, u = d["u850"]
        out["moist"] = box_mean((la, lo, d["tcwv"][2] * np.maximum(-u, 0)), AB_BOX_COAST)
    if "r700" in d and "r500" in d:
        out["rh"] = (box_mean(d["r700"], AB_BOX_TN) + box_mean(d["r500"], AB_BOX_TN)) / 2
    if "msl" in d:
        out["grad"] = box_mean(d["msl"], AB_BOX_NORTH) - box_mean(d["msl"], AB_BOX_BOB)
    return out


def mjo_now():
    rows = od.demo_rmm() if DEMO else od.rmm_recent()
    if not rows:
        return None
    last = rows[-1]
    fav = last["amp"] >= 1 and last["phase"] in MJO_GOOD
    unf = last["amp"] >= 1 and last["phase"] in MJO_BAD
    return dict(last, effect="favourable" if fav else "unfavourable" if unf else "neutral",
                modifier=8 if fav else -8 if unf else 0, recent=rows[-20:])


def section_active_break(run, hres, outdir):
    if not hres:
        raise RuntimeError("no ECMWF HRES fields for this run")
    t0 = run_time(run)
    steps = sorted(s for s in hres if s % 24 == 0)
    mjo = mjo_now()
    days = []
    tn_mask = fine_masks().get("Tamil Nadu")
    for s in steps:
        ind = indicators(hres[s])
        parts, wsum, sc = {}, 0, 0
        for k, lab, unit, q, a, w in AB_INDICATORS:
            if k in ind and np.isfinite(ind[k]):
                v = float(np.clip((ind[k] - q) / (a - q), 0, 1))
                parts[k] = {"value": r1(ind[k], 1), "score": r1(100 * v, 0)}
                sc += w * v
                wsum += w
        score = 100 * sc / wsum if wsum else np.nan
        mod = 0
        if mjo and np.isfinite(score):
            mod = mjo["modifier"] * max(0, 1 - s / 24 / 15)
            score = float(np.clip(score + mod, 0, 100))
        rain = None
        if s + 24 in hres and "tp" in hres[s + 24] and "tp" in hres[s] and tn_mask is not None:
            la, lo, b = hres[s + 24]["tp"]
            a = hres[s]["tp"][2]
            rain = float(area_mean(to_grid(la, lo, np.maximum(b - a, 0), FINE_LAT, FINE_LON), tn_mask))
        state = "Active" if score >= 60 else "Moderate" if score >= 40 else "Subdued / break-like"
        days.append({"step": int(s), "valid": (t0 + timedelta(hours=s)).strftime("%a %d %b %HZ"),
                     "date": (t0 + timedelta(hours=s)).date().isoformat(), "score": r1(score, 0),
                     "state": state, "mjo_adj": r1(mod, 1), "parts": parts, "tn_rain_hres": r1(rain)})
    # chart
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    fig.subplots_adjust(top=0.82, bottom=0.2, left=0.07, right=0.92)
    xs = np.arange(len(days))
    sc = [d["score"] if d["score"] is not None else 0 for d in days]
    cols = ["#3ddc97" if v >= 60 else "#f5b43c" if v >= 40 else "#5b6b88" for v in sc]
    ax.bar(xs, sc, color=cols, width=0.65, zorder=2)
    ax.axhline(60, color="#3ddc97", lw=0.9, ls="--")
    ax.axhline(40, color="#f5b43c", lw=0.9, ls="--")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Active-NEM score (0-100)")
    ax.set_xticks(xs, [d["valid"][:10] for d in days], rotation=0, fontsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax2 = ax.twinx()
    rr = [d["tn_rain_hres"] if d["tn_rain_hres"] is not None else np.nan for d in days]
    ax2.plot(xs, rr, color=ACCENT, marker="o", lw=1.6, label="HRES TN mean rain next 24 h")
    ax2.set_ylabel("TN area-mean rain (mm/24 h)", color=ACCENT)
    ax2.tick_params(axis="y", colors=ACCENT)
    top = np.nanmax(rr) if len(rr) and np.isfinite(rr).any() else 0
    ax2.set_ylim(0, max(10, top * 1.3))
    handles = [Patch(color="#3ddc97", label="Active (≥60)"), Patch(color="#f5b43c", label="Moderate (40-60)"),
               Patch(color="#5b6b88", label="Subdued / break-like (<40)"),
               Line2D([], [], color=ACCENT, marker="o", label="HRES TN mean rain")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=7.5)
    mj = f"MJO {mjo['date']}: phase {mjo['phase']}, amplitude {mjo['amp']:.1f} ({mjo['effect']})" if mjo else \
        "MJO index not available"
    fig.text(0.07, 0.94, "Active / break NEM detector · Tamil Nadu", fontsize=12.5, fontweight="bold")
    fig.text(0.07, 0.895, f"ECMWF HRES init {fmt_run(run)} · days 0-10 · {mj}. An indicator of conditions, "
                          "not a guarantee of rain on a date.", fontsize=8.2, color=MUTED)
    png = save(fig, f"{outdir}/active_break.png")
    # history for the verification of the detector
    hist = os.path.join(STATE, "verif", "activebreak.csv")
    new = pd.DataFrame([{"run": run, "date": d["date"], "lead_day": d["step"] // 24, "score": d["score"]}
                        for d in days])
    if os.path.exists(hist):
        new = pd.concat([pd.read_csv(hist), new]).drop_duplicates(["run", "date"], keep="last")
    new.to_csv(hist, index=False)
    ind_meta = [{"key": k, "label": lab, "units": u, "quiet": q, "active": a, "weight": w}
                for k, lab, u, q, a, w in AB_INDICATORS]
    return {"days": days, "mjo": mjo, "png": os.path.relpath(png, SITE_DIR), "indicators": ind_meta}


def draw_environment(run, hres, step, path):
    if step not in hres:
        return None
    d = hres[step]
    t0 = run_time(run)
    fig = plt.figure(figsize=(14, 4.9))
    proj = ccrs.PlateCarree()
    ext = (68, 100, 0, 25)
    axs = [fig.add_axes([0.01 + i * 0.33, 0.13, 0.31, 0.72], projection=proj) for i in range(3)]
    for ax in axs:
        base_map(ax, ext, gridlabels=False)
    # 1. MSLP + 850 wind
    la, lo, p = d["msl"]
    cf = axs[0].contourf(lo, la, p, levels=np.arange(996, 1016.1, 1), cmap="RdYlBu", transform=proj, zorder=1,
                         extend="both", alpha=0.85)
    axs[0].contour(lo, la, p, levels=np.arange(980, 1020, 2), colors="#1b1f2a", linewidths=0.5, transform=proj, zorder=2)
    ula, ulo, u = d["u850"]
    v = d["v850"][2]
    sk = max(1, int(round(1.0 / abs(ulo[1] - ulo[0]))))
    axs[0].streamplot(ulo[::sk], ula[::sk], u[::sk, ::sk], v[::sk, ::sk], color="white", linewidth=0.5,
                      density=1.4, arrowsize=0.6, transform=proj, zorder=3)
    fig.colorbar(cf, ax=axs[0], orientation="horizontal", fraction=0.05, pad=0.04).set_label("MSLP (hPa)")
    # 2. 850 vorticity
    la, lo, vo = d["vo850"]
    cf2 = axs[1].contourf(lo, la, vo * 1e5, levels=[-10, -5, -2, 2, 5, 10, 15, 20, 30, 40], cmap="RdBu_r",
                          transform=proj, zorder=1, extend="both")
    fig.colorbar(cf2, ax=axs[1], orientation="horizontal", fraction=0.05, pad=0.04).set_label("850 hPa vorticity (10⁻⁵ s⁻¹)")
    # 3. shear
    la, lo, u2 = d["u200"]
    sh = np.hypot(u2 - d["u850"][2], d["v200"][2] - d["v850"][2]) * MS_TO_KT
    cf3 = axs[2].contourf(lo, la, sh, levels=[0, 10, 15, 20, 25, 30, 40, 50, 60], cmap="YlOrRd", transform=proj,
                          zorder=1, extend="max", alpha=0.9)
    axs[2].contour(lo, la, sh, levels=[20], colors="white", linewidths=1.0, transform=proj, zorder=2)
    fig.colorbar(cf3, ax=axs[2], orientation="horizontal", fraction=0.05, pad=0.04).set_label("200-850 hPa shear (kt) · white: 20 kt")
    for ax, t in zip(axs, ["MSLP & 850 hPa streamlines", "850 hPa relative vorticity", "Vertical wind shear"]):
        ax.set_title(t, color=TEXT, fontsize=9.5)
    vt = t0 + timedelta(hours=step)
    fig.text(0.01, 0.95, f"Environment · day {step // 24} · valid {vt:%d %b %HZ}", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.915, f"ECMWF IFS HRES init {fmt_run(run)} · low shear (<20 kt) and positive vorticity "
                          "favour development", fontsize=8.5, color=MUTED)
    return save(fig, path)


# ============================================================
# E. TRACK CONSISTENCY MONITOR
# ============================================================

def load_history():
    p = os.path.join(STATE, "systems_history.json")
    return json.load(open(p)) if os.path.exists(p) else []


def save_history(hist):
    json.dump(hist[-HISTORY_RUNS:], open(os.path.join(STATE, "systems_history.json"), "w"))


def run_record(run, systems):
    t0 = run_time(run)
    out = []
    for s in systems:
        lf = {}
        for sc in s["scenarios"]:
            if sc["scenario"].startswith("Landfall: "):
                lf[sc["scenario"][10:]] = sc["pct"]
        d = {r["day"]: r for r in s["days"]}
        out.append({"storm": s["storm"], "p_form": s["p_form"], "p_dep": s["reach"]["D"], "p_cs": s["reach"]["CS"],
                    "peak_med": s["peak_med"], "peak_p10": s["peak_p10"], "peak_p90": s["peak_p90"],
                    "spread72": d.get(3, {}).get("spread_km"), "spread120": d.get(5, {}).get("spread_km"),
                    "landfall": lf,
                    "track": [[(t0 + timedelta(hours=st)).strftime("%Y-%m-%dT%H"), a, o]
                              for st, a, o in s["mean_track"]]})
    return {"run": run, "systems": out}


def match(sysA, sysB):
    """Mean distance (km) between two mean tracks at their common valid times (>=2), else None."""
    a = {t: (la, lo) for t, la, lo in sysA["track"]}
    common = [(a[t], (la, lo)) for t, la, lo in sysB["track"] if t in a]
    if len(common) < 2:
        return None
    return float(np.mean([haversine_km(p[0], p[1], q[0], q[1]) for p, q in common]))


def section_consistency(run, systems, outdir):
    hist = [h for h in load_history() if h["run"] != run]
    hist.append(run_record(run, systems))
    hist.sort(key=lambda h: h["run"])
    save_history(hist)
    latest = hist[-1]
    out = []
    for s in latest["systems"]:
        chain = [(latest["run"], s)]
        ref = s
        for h in reversed(hist[:-1]):
            best, bd = None, 450.0
            for o in h["systems"]:
                d = match(ref, o)
                if d is not None and d < bd:
                    best, bd = o, d
            if best is not None:
                chain.append((h["run"], best))
                ref = best
        chain = chain[::-1]
        runs_considered = [h["run"] for h in hist]
        present = len(chain)
        # verdicts
        v = {"appearances": f"Present in {present} of the last {len(runs_considered)} runs"}
        if present >= 2:
            (r_prev, p), (r_last, l) = chain[-2], chain[-1]
            pa = {t: (a, o) for t, a, o in p["track"]}
            common = [(t, a, o) for t, a, o in l["track"] if t in pa]
            tgt = (run_time(r_last) + timedelta(hours=72)).strftime("%Y-%m-%dT%H")
            pick = next((c for c in common if c[0] >= tgt), common[-1] if common else None)
            if pick:
                t, a, o = pick
                pa_, po_ = pa[t]
                dist = haversine_km(a, o, pa_, po_)
                dn, de = (a - pa_) * 111, (o - po_) * 111 * math.cos(math.radians(a))
                direc = ("north" if dn > 0 else "south") if abs(dn) > abs(de) else ("east" if de > 0 else "west")
                v["track_shift"] = (f"Mean position for {t[8:10]}/{t[5:7]} {t[11:13]}Z moved {dist:.0f} km {direc} "
                                    "since the previous run") if dist >= 50 else "Track position steady (<50 km change)"
            sp = [c[1].get("spread72") for c in chain if c[1].get("spread72") is not None]
            if len(sp) >= 2:
                v["spread"] = ("Spread decreasing" if sp[-1] < sp[0] * 0.9 else "Spread increasing"
                               if sp[-1] > sp[0] * 1.1 else "Spread about the same") + f" (day-3 spread {sp[0]:.0f} → {sp[-1]:.0f} km)"
            pk = [c[1].get("peak_med") for c in chain if c[1].get("peak_med") is not None]
            if len(pk) >= 2:
                v["intensity"] = ("Strengthening" if pk[-1] > pk[0] + 5 else "Weakening" if pk[-1] < pk[0] - 5
                                  else "Intensity steady") + f" (median peak {pk[0]:.0f} → {pk[-1]:.0f} kt)"
            tops = [max(c[1]["landfall"], key=c[1]["landfall"].get) if c[1]["landfall"] else "no landfall"
                    for c in chain]
            v["landfall"] = ("Most likely landfall region unchanged: " + tops[-1]) if len(set(tops[-3:])) == 1 \
                else "Most likely landfall region changed: " + " → ".join(dict.fromkeys(tops[-4:]))
        png = draw_consistency(s["storm"], chain, f"{outdir}/consistency_{s['storm']}.png")
        out.append({"storm": s["storm"], "chain": [{"run": r, "p_dep": c["p_dep"], "p_cs": c["p_cs"],
                                                   "peak_med": c["peak_med"], "peak_p10": c["peak_p10"],
                                                   "peak_p90": c["peak_p90"], "spread72": c.get("spread72"),
                                                   "spread120": c.get("spread120"), "landfall": c["landfall"],
                                                   "id": c["storm"]} for r, c in chain],
                    "verdict": v, "png": os.path.relpath(png, SITE_DIR)})
    return {"runs": [h["run"] for h in hist], "systems": out}


def draw_consistency(storm, chain, path):
    fig = plt.figure(figsize=(12, 6.2))
    proj = ccrs.PlateCarree()
    pts = [(a, o) for _, c in chain for _, a, o in c["track"]]
    la = [p[0] for p in pts]; lo = [p[1] for p in pts]
    ext = (min(lo) - 3, max(lo) + 3, max(-2, min(la) - 3), max(la) + 3)
    if ext[1] - ext[0] < 14:
        c = (ext[0] + ext[1]) / 2; ext = (c - 7, c + 7, ext[2], ext[3])
    ax = fig.add_axes([0.02, 0.1, 0.5, 0.78], projection=proj)
    base_map(ax, ext)
    cmap = plt.get_cmap("plasma")
    n = len(chain)
    for i, (r, c) in enumerate(chain):
        tr = np.array([[a, o] for _, a, o in c["track"]], float)
        col = cmap(0.15 + 0.75 * i / max(1, n - 1))
        ax.plot(tr[:, 1], tr[:, 0], color=col, lw=2.4 if i == n - 1 else 1.4, transform=proj,
                label=f"{run_time(r):%d %b %HZ} · {c['storm']}")
    ax.legend(fontsize=7, loc="lower left", title="Ensemble-mean track by run", title_fontsize=7.5)
    x = np.arange(n)
    labels = [f"{r[6:8]}/{r[4:6]}\n{r[-2:]}Z" for r, _ in chain]
    a1 = fig.add_axes([0.58, 0.58, 0.18, 0.3]); a2 = fig.add_axes([0.8, 0.58, 0.18, 0.3])
    a3 = fig.add_axes([0.58, 0.12, 0.18, 0.3]); a4 = fig.add_axes([0.8, 0.12, 0.18, 0.3])
    a1.plot(x, [c["p_dep"] for _, c in chain], marker="o", color="#4f8ff7", label="≥ depression")
    a1.plot(x, [c["p_cs"] for _, c in chain], marker="o", color="#22c38e", label="≥ cyclonic storm")
    a1.set_ylim(0, 100); a1.set_title("Development probability (%)", fontsize=8.5, color=TEXT)
    a1.legend(fontsize=6.5)
    med = [c["peak_med"] for _, c in chain]
    a2.fill_between(x, [c["peak_p10"] for _, c in chain], [c["peak_p90"] for _, c in chain], color=ACCENT, alpha=0.25)
    a2.plot(x, med, marker="o", color="white"); a2.set_title("Peak wind, median & 10-90% (kt)", fontsize=8.5, color=TEXT)
    a3.plot(x, [c.get("spread72") or np.nan for _, c in chain], marker="o", color="#f5b43c", label="day 3")
    a3.plot(x, [c.get("spread120") or np.nan for _, c in chain], marker="o", color="#ef5b2b", label="day 5")
    a3.set_title("Track spread (km)", fontsize=8.5, color=TEXT); a3.legend(fontsize=6.5)
    regions = sorted({k for _, c in chain for k in c["landfall"]})
    bottom = np.zeros(n)
    pal = ["#e0412f", "#f5b43c", "#40c9b0", "#2a78c2", "#c03bd6", "#8edc5f", "#9fb3c8"]
    for k, reg in enumerate(regions):
        vals = np.array([c["landfall"].get(reg, 0) for _, c in chain], float)
        a4.bar(x, vals, bottom=bottom, color=pal[k % len(pal)], label=reg[:18], width=0.7)
        bottom += vals
    a4.set_ylim(0, 100); a4.set_title("Landfall share by region (%)", fontsize=8.5, color=TEXT)
    if regions:
        a4.legend(fontsize=5.8, loc="upper left")
    for a in (a1, a2, a3, a4):
        a.set_xticks(x, labels, fontsize=6.5)
        a.grid(color=GRID, lw=0.4)
        a.tick_params(labelsize=7)
    fig.text(0.02, 0.95, f"Track consistency monitor · {chain[-1][1]['storm']}", fontsize=12.5, fontweight="bold")
    fig.text(0.02, 0.915, f"Successive ECMWF ENS runs matched by their mean tracks · {n} run(s)", fontsize=8.5,
             color=MUTED)
    return save(fig, path)


# ============================================================
# F. VERIFICATION
# ============================================================

def rain_steps(run):
    """Steps bounding the first three 24-h rain-days (08:30-08:30 IST = 03-03 UTC) after the run."""
    a = (27 - run_time(run).hour) % 24 or 24
    return [a, a + 24, a + 48, a + 72]


def rain_day_of(run, k):
    a = rain_steps(run)[k + 1]
    return (run_time(run) + timedelta(hours=a)).date()


def archive_run(run, tracks, hres_tracks, ens_rain=None, hres_rain=None, n=N_ENS):
    """Save what is needed to verify this run later (small files in the state)."""
    t0 = run_time(run)
    rows = []
    for src, tr in (("ens", tracks), ("hres", hres_tracks)):
        if tr is None or not len(tr):
            continue
        for storm, t in tr.groupby("storm"):
            if src == "ens":
                mt = mean_track(t)
                for _, r in mt.iterrows():
                    rows.append((src, storm, int(r.step), round(r.lat, 2), round(r.lon, 2),
                                 round(r.wind * MS_TO_KT, 1), int(r.n)))
            else:
                for _, r in t.sort_values("step").iterrows():
                    rows.append((src, storm, int(r.step), round(r.lat, 2), round(r.lon, 2),
                                 round(r.wind * MS_TO_KT, 1), 1))
    pd.DataFrame(rows, columns=["src", "storm", "step", "lat", "lon", "wind_kt", "n"]).to_csv(
        os.path.join(STATE, "verif", "tracks", f"{run}.csv"), index=False)
    # ECMWF analysis positions (T+0 median of members)
    a = tracks[tracks.step == 0].groupby("storm").agg(lat=("lat", "median"), lon=("lon", "median"),
                                                     wind=("wind", "median"), n=("member", "nunique"))
    a = a[a.n >= 10].reset_index()
    ap = os.path.join(STATE, "verif", "analysis.csv")
    new = pd.DataFrame({"time": t0.strftime("%Y-%m-%dT%H"), "storm": a.storm, "lat": a.lat.round(2),
                        "lon": a.lon.round(2), "wind_kt": (a.wind * MS_TO_KT).round(1)})
    if os.path.exists(ap):
        new = pd.concat([pd.read_csv(ap), new]).drop_duplicates(["time", "storm"], keep="last")
    new.to_csv(ap, index=False)
    # predicted geneses (depression) per system
    ev = genesis_events(tracks)
    ev = ev[(ev.level == "dep") & ~ev.existing]
    gp = os.path.join(STATE, "verif", "genesis_pred.csv")
    pr = []
    for storm, g in ev.groupby("storm"):
        p = g.member.nunique() / n
        if p >= 0.1:
            pr.append({"run": run, "storm": storm, "p": round(p, 3),
                       "t_gen": (t0 + timedelta(hours=float(g.step.median()))).strftime("%Y-%m-%dT%H"),
                       "lat": round(g.lat.median(), 2), "lon": round(g.lon.median(), 2)})
    old = pd.read_csv(gp) if os.path.exists(gp) else pd.DataFrame(columns=["run", "storm", "p", "t_gen", "lat", "lon"])
    old = old[old.run != run]
    pd.concat([old, pd.DataFrame(pr)]).to_csv(gp, index=False)
    # rain fields over Tamil Nadu (first 3 rain-days)
    if ens_rain or hres_rain:
        st = rain_steps(run)
        out = {"days": np.array([rain_day_of(run, k).isoformat() for k in range(3)])}
        def cut(la, lo, f):
            a = (la >= VERIF_BOX[0] - 0.5) & (la <= VERIF_BOX[1] + 0.5)
            b = (lo >= VERIF_BOX[2] - 0.5) & (lo <= VERIF_BOX[3] + 0.5)
            return la[a], lo[b], f[..., a, :][..., b]
        if ens_rain and all(s in ens_rain for s in st):
            la, lo, _ = ens_rain[st[0]]
            daily = np.stack([np.maximum(ens_rain[st[k + 1]][2] - ens_rain[st[k]][2], 0) for k in range(3)], 1)
            la2, lo2, daily = cut(la, lo, daily)               # (member, day, lat, lon)
            out.update(lat=la2, lon=lo2, ens_mean=daily.mean(0).astype(np.float32),
                       prob=np.stack([(daily >= t).mean(0) for t in RAIN_THRESH]).astype(np.float32))
        if hres_rain and all(s in hres_rain and "tp" in hres_rain[s] for s in st):
            la, lo, _ = hres_rain[st[0]]["tp"]
            daily = np.stack([np.maximum(hres_rain[st[k + 1]]["tp"][2] - hres_rain[st[k]]["tp"][2], 0)
                              for k in range(3)])
            la2, lo2, daily = cut(la, lo, daily)
            out.update(hlat=la2, hlon=lo2, hres=daily.astype(np.float32))
        if len(out) > 1:
            np.savez_compressed(os.path.join(STATE, "verif", "rain", f"{run}.npz"), **out)


def prune_archive(now):
    cut = (now - timedelta(days=ARCHIVE_DAYS)).strftime("%Y%m%d_%H")
    for d in ("tracks", "rain"):
        for f in glob.glob(os.path.join(STATE, "verif", d, "*")):
            if os.path.basename(f)[:11] < cut:
                os.remove(f)
    for f in glob.glob(os.path.join(STATE, "obs", "*.json")):
        if os.path.basename(f)[8:18] < (now - timedelta(days=ARCHIVE_DAYS)).date().isoformat():
            os.remove(f)


def obs_day(d):
    if DEMO:
        return od.demo_tnsmart(d)
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    if ist < datetime(d.year, d.month, d.day, 9, 15, tzinfo=ist.tzinfo):
        return []
    return od.tnsmart_day(d, os.path.join(STATE, "obs"))


def sample(la, lo, field, pts):
    from scipy.interpolate import RegularGridInterpolator
    it = RegularGridInterpolator((la, lo), field, bounds_error=False, fill_value=np.nan)
    return it(pts)


def observed_geneses(ib, ana, since):
    """Observed systems reaching depression (≥17 kt): IBTrACS first, ECMWF analyses where IBTrACS has none yet."""
    out = []
    if ib is not None and len(ib):
        s = ib[ib.time >= since.replace(tzinfo=None)]
        for sid, g in s.groupby("sid"):
            g = g.sort_values("time")
            ok = g[g.wind_kt >= 17]
            if len(ok) >= 2:
                r = ok.iloc[0]
                out.append({"src": "IBTrACS", "id": str(g.name.iloc[0]) if isinstance(g.name.iloc[0], str) else sid,
                            "time": r.time.to_pydatetime().replace(tzinfo=timezone.utc), "lat": r.lat, "lon": r.lon,
                            "peak": float(g.wind_kt.max())})
    if ana is not None and len(ana):
        for storm, g in ana.groupby("storm"):
            g = g.sort_values("time")
            ok = g[g.wind_kt >= 17]
            if len(ok) >= 2:
                r = ok.iloc[0]
                t = datetime.strptime(r.time, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
                if t < since:
                    continue
                if any(abs((t - o["time"]).total_seconds()) < 4 * 86400 and
                       haversine_km(r.lat, r.lon, o["lat"], o["lon"]) < 800 for o in out):
                    continue
                out.append({"src": "ECMWF analysis", "id": storm, "time": t, "lat": r.lat, "lon": r.lon,
                            "peak": float(g.wind_kt.max())})
    return out


def section_verification(now, outdir):
    since = now - timedelta(days=ARCHIVE_DAYS)
    ib = None
    try:
        ib = None if DEMO else od.ibtracs_ni(STATE)
    except Exception as e:
        print(f"  IBTrACS not available: {e}")
    ap = os.path.join(STATE, "verif", "analysis.csv")
    ana = pd.read_csv(ap) if os.path.exists(ap) else None
    leads = [24, 48, 72, 96, 120]
    terr = {src: {L: [] for L in leads} for src in ("ens", "hres")}
    ierr = {src: {L: [] for L in leads} for src in ("ens", "hres")}
    track_files = sorted(glob.glob(os.path.join(STATE, "verif", "tracks", "*.csv")))
    ana_t = None
    if ana is not None and len(ana):
        ana_t = ana.assign(t=pd.to_datetime(ana.time, format="%Y-%m-%dT%H"))
    ib_t = ib[ib.time >= since.replace(tzinfo=None)] if ib is not None and len(ib) else None
    for f in track_files:
        run = os.path.basename(f)[:-4]
        t0 = run_time(run)
        try:
            fc = pd.read_csv(f)
        except Exception:
            continue
        for (src, storm), g in fc.groupby(["src", "storm"]):
            for L in leads:
                r = g[g.step == L]
                if not len(r):
                    continue
                r = r.iloc[0]
                vt = (t0 + timedelta(hours=L)).replace(tzinfo=None)
                if vt > now.replace(tzinfo=None):
                    continue
                obs = None
                if ib_t is not None:
                    c = ib_t[ib_t.time == vt]
                    if len(c):
                        d = haversine_km(c.lat.values, c.lon.values, r.lat, r.lon)
                        if d.min() < 500:
                            o = c.iloc[int(d.argmin())]
                            obs = (o.lat, o.lon, o.wind_kt)
                if obs is None and ana_t is not None:
                    c = ana_t[ana_t.t == vt]
                    if len(c):
                        same = c[c.storm == storm]
                        if len(same):
                            o = same.iloc[0]
                            obs = (o.lat, o.lon, o.wind_kt)
                        else:
                            d = haversine_km(c.lat.values, c.lon.values, r.lat, r.lon)
                            if d.min() < 500:
                                o = c.iloc[int(d.argmin())]
                                obs = (o.lat, o.lon, o.wind_kt)
                if obs is None:
                    continue
                terr[src][L].append(float(haversine_km(r.lat, r.lon, obs[0], obs[1])))
                if np.isfinite(obs[2]):
                    ierr[src][L].append(float(r.wind_kt - obs[2]))
    track = [{"lead_h": L, "ens_km": r1(np.mean(terr["ens"][L]), 0) if terr["ens"][L] else None,
              "ens_n": len(terr["ens"][L]),
              "hres_km": r1(np.mean(terr["hres"][L]), 0) if terr["hres"][L] else None,
              "hres_n": len(terr["hres"][L]),
              "ens_int_mae": r1(np.mean(np.abs(ierr["ens"][L])), 1) if ierr["ens"][L] else None,
              "ens_int_bias": r1(np.mean(ierr["ens"][L]), 1) if ierr["ens"][L] else None,
              "hres_int_mae": r1(np.mean(np.abs(ierr["hres"][L])), 1) if ierr["hres"][L] else None,
              "hres_int_bias": r1(np.mean(ierr["hres"][L]), 1) if ierr["hres"][L] else None} for L in leads]

    # detection / false alarms
    gp = os.path.join(STATE, "verif", "genesis_pred.csv")
    preds = pd.read_csv(gp) if os.path.exists(gp) else pd.DataFrame(columns=["run", "storm", "p", "t_gen", "lat", "lon"])
    preds = preds[preds.p >= 0.3]
    runs = [os.path.basename(f)[:-4] for f in track_files]
    obs_g = observed_geneses(ib, ana, since)
    bins = [("1-3 days", 24, 72), ("3-7 days", 72, 168), ("7-15 days", 168, 360)]
    det = {b[0]: [0, 0] for b in bins}
    for o in obs_g:
        o["detected_by"] = []
        for run in runs:
            lead = (o["time"] - run_time(run)).total_seconds() / 3600
            b = next((b for b in bins if b[1] <= lead < b[2]), None)
            if b is None:
                continue
            det[b[0]][1] += 1
            p = preds[preds.run == run]
            hit = False
            for _, q in p.iterrows():
                tg = datetime.strptime(q.t_gen, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
                if abs((tg - o["time"]).total_seconds()) <= 72 * 3600 and \
                        haversine_km(q.lat, q.lon, o["lat"], o["lon"]) <= 500:
                    hit = True
                    break
            det[b[0]][0] += hit
            if hit:
                o["detected_by"].append(run)
    fa, nver = 0, 0
    false_list = []
    for _, q in preds.iterrows():
        tg = datetime.strptime(q.t_gen, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
        if tg > now - timedelta(days=3) or tg < since:
            continue
        nver += 1
        if not any(abs((tg - o["time"]).total_seconds()) <= 72 * 3600 and
                   haversine_km(q.lat, q.lon, o["lat"], o["lon"]) <= 500 for o in obs_g):
            fa += 1
            false_list.append({"run": q.run, "storm": q.storm, "p": r1(100 * q.p, 0), "t_gen": q.t_gen})
    detection = {"bins": [{"lead": k, "hits": v[0], "opportunities": v[1],
                           "pod": r1(100 * v[0] / v[1], 0) if v[1] else None} for k, v in det.items()],
                 "far": r1(100 * fa / nver, 0) if nver else None, "predictions_verified": nver,
                 "false_alarms": fa, "false_list": false_list[-8:],
                 "observed": [{"src": o["src"], "id": o["id"], "time": o["time"].strftime("%Y-%m-%d %HZ"),
                               "lat": r1(o["lat"]), "lon": r1(o["lon"]), "peak_kt": r1(o["peak"], 0),
                               "detected_runs": len(o["detected_by"])} for o in obs_g]}

    # rainfall at TN-SMART stations
    stats = {k: {"ens": [], "hres": [], "obs": [], "probs": []} for k in range(3)}
    obs_cache = {}
    for f in sorted(glob.glob(os.path.join(STATE, "verif", "rain", "*.npz"))):
        z = np.load(f)
        days = [date.fromisoformat(d) for d in z["days"]]
        for k, d in enumerate(days):
            if d not in obs_cache:
                obs_cache[d] = obs_day(d)
            st = obs_cache[d]
            if not st:
                continue
            pts = np.array([[s["lat"], s["lon"]] for s in st])
            ob = np.array([s["rain"] for s in st])
            e = sample(z["lat"], z["lon"], z["ens_mean"][k], pts) if "ens_mean" in z else np.full(len(ob), np.nan)
            h = sample(z["hlat"], z["hlon"], z["hres"][k], pts) if "hres" in z else np.full(len(ob), np.nan)
            pr = np.stack([sample(z["lat"], z["lon"], z["prob"][i, k], pts) for i in range(len(RAIN_THRESH))], 1) \
                if "prob" in z else np.full((len(ob), len(RAIN_THRESH)), np.nan)
            stats[k]["ens"].append(e); stats[k]["hres"].append(h); stats[k]["obs"].append(ob)
            stats[k]["probs"].append(pr)
    rain = []
    for k in range(3):
        if not stats[k]["obs"]:
            rain.append({"lead_day": k + 1, "n": 0})
            continue
        ob = np.concatenate(stats[k]["obs"]); e = np.concatenate(stats[k]["ens"]); h = np.concatenate(stats[k]["hres"])
        pr = np.concatenate(stats[k]["probs"])
        row = {"lead_day": k + 1, "n": int(len(ob)), "days": len(stats[k]["obs"])}
        for nm, fc in (("ens", e), ("hres", h)):
            m = np.isfinite(fc)
            if m.any():
                row[f"{nm}_mae"] = r1(np.mean(np.abs(fc[m] - ob[m])))
                row[f"{nm}_rmse"] = r1(np.sqrt(np.mean((fc[m] - ob[m]) ** 2)))
                row[f"{nm}_bias"] = r1(np.mean(fc[m] - ob[m]))
        for i, t in enumerate(RAIN_THRESH):
            p = pr[:, i]
            m = np.isfinite(p)
            if m.sum() >= 20:
                o = (ob[m] >= t).astype(float)
                bs = np.mean((p[m] - o) ** 2)
                base = o.mean()
                ref = base * (1 - base)
                row[f"brier_{t}"] = r1(bs, 3)
                row[f"bss_{t}"] = r1(1 - bs / ref, 2) if ref > 0 else None
                row[f"freq_{t}"] = r1(100 * base, 0)
        rain.append(row)

    # active/break detector against observed TN rain (next rain-day)
    ab = []
    hist = os.path.join(STATE, "verif", "activebreak.csv")
    if os.path.exists(hist):
        h = pd.read_csv(hist)
        h = h[h.lead_day == 1].drop_duplicates("date", keep="last")
        for _, r in h.iterrows():
            d = date.fromisoformat(r.date) + timedelta(days=1)
            st = obs_cache.get(d)
            if st is None:
                st = obs_day(d) if d >= (now - timedelta(days=ARCHIVE_DAYS)).date() else []
                obs_cache[d] = st
            if st:
                ab.append({"date": r.date, "score": r.score, "tn_mean": r1(np.mean([s["rain"] for s in st]))})
    ab_sum = None
    if len(ab) >= 5:
        sc = np.array([a["score"] for a in ab], float); rn = np.array([a["tn_mean"] for a in ab], float)
        ab_sum = {"n": len(ab), "corr": r1(np.corrcoef(sc, rn)[0, 1], 2) if sc.std() > 0 and rn.std() > 0 else None,
                  "rain_active": r1(rn[sc >= 60].mean()) if (sc >= 60).any() else None,
                  "rain_moderate": r1(rn[(sc >= 40) & (sc < 60)].mean()) if ((sc >= 40) & (sc < 60)).any() else None,
                  "rain_subdued": r1(rn[sc < 40].mean()) if (sc < 40).any() else None}

    # chart
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.4))
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.98, hspace=0.42, wspace=0.22)
    x = np.arange(len(leads))
    for i, (src, col) in enumerate((("ens", ACCENT), ("hres", "#ff4d4d"))):
        vals = [t[f"{src}_km"] or 0 for t in track]
        axs[0, 0].bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=col, label="ENS mean" if src == "ens" else "HRES")
        vals = [t[f"{src}_int_bias"] or 0 for t in track]
        axs[0, 1].bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=col)
    for t_, xi in zip(track, x):
        axs[0, 0].text(xi, 5, f"n={t_['ens_n']}", ha="center", fontsize=6.5, color=TEXT)
    axs[0, 0].set_xticks(x, [f"{L} h" for L in leads]); axs[0, 0].set_title("Track error (km)", color=TEXT)
    axs[0, 0].legend(fontsize=7.5)
    axs[0, 1].axhline(0, color=MUTED, lw=0.8)
    axs[0, 1].set_xticks(x, [f"{L} h" for L in leads]); axs[0, 1].set_title("Intensity bias, forecast − observed (kt)", color=TEXT)
    xl = np.arange(3)
    for i, (key, col, lab) in enumerate((("ens_mae", ACCENT, "ENS mean MAE"), ("ens_rmse", "#40c9b0", "ENS mean RMSE"),
                                         ("hres_mae", "#ff4d4d", "HRES MAE"), ("hres_rmse", "#f5b43c", "HRES RMSE"))):
        axs[1, 0].bar(xl + (i - 1.5) * 0.2, [r.get(key) or 0 for r in rain], width=0.19, color=col, label=lab)
    axs[1, 0].set_xticks(xl, [f"Day {k + 1}\n(n={r['n']})" for k, r in enumerate(rain)])
    axs[1, 0].set_title("Station rainfall error, TN-SMART (mm/24 h)", color=TEXT)
    axs[1, 0].legend(fontsize=7, ncol=2)
    for i, (t, col) in enumerate(zip(RAIN_THRESH, ["#7ab8e8", "#24a36b", "#ef5b2b"])):
        axs[1, 1].bar(xl + (i - 1) * 0.27, [r.get(f"bss_{t}") or 0 for r in rain], width=0.25, color=col,
                      label=f"≥{t} mm")
    axs[1, 1].axhline(0, color=MUTED, lw=0.8)
    axs[1, 1].set_xticks(xl, [f"Day {k + 1}" for k in range(3)])
    axs[1, 1].set_title("ENS probability skill (Brier skill score)", color=TEXT)
    axs[1, 1].legend(fontsize=7)
    for a in axs.ravel():
        a.grid(axis="y", color=GRID, lw=0.4)
    pods = ", ".join(f"{b['lead']}: {b['pod']}%" if b["pod"] is not None else f"{b['lead']}: n/a"
                     for b in detection["bins"])
    fig.text(0.07, 0.955, "Forecast verification scoreboard", fontsize=12.5, fontweight="bold")
    fig.text(0.07, 0.92, f"Last {ARCHIVE_DAYS} days · {len(track_files)} archived runs · detection rate {pods} · "
                         f"false-alarm ratio {detection['far'] if detection['far'] is not None else 'n/a'}%",
             fontsize=8.5, color=MUTED)
    png = save(fig, f"{outdir}/verification.png")
    return {"png": os.path.relpath(png, SITE_DIR), "track": track, "detection": detection, "rain": rain,
            "active_break": {"summary": ab_sum, "recent": ab[-15:]}, "runs_archived": len(track_files),
            "period": f"{since:%d %b %Y} – {now:%d %b %Y}",
            "obs_sources": "Tracks/intensity: IBTrACS (IMD) where published, else ECMWF analysis (T+0). "
                           "Rain: TN-SMART station 24-h totals (08:30 IST)."}


# ============================================================
# KEY TAKEAWAYS
# ============================================================

def takeaways(s):
    out = []
    syss = (s.get("cyclones") or [])
    for c in syss[:2]:
        top = next((x for x in c["scenarios"] if x["scenario"].startswith("Landfall")), None)
        line = (f"{c['storm']} ({c['basin']}): {c['p_form']:.0f}% of members track it, "
                f"{c['reach']['D']:.0f}% reach depression, {c['reach']['CS']:.0f}% cyclonic storm; median peak "
                f"{c['peak_med']:.0f} kt.")
        if top:
            line += f" Most common outcome: {top['scenario']} ({top['pct']:.0f}%)."
        out.append(line)
    g = s.get("genesis") or {}
    for b in g.get("basin", []):
        if b["window"] in ("d6_10", "d11_15") and (b.get("dep") or 0) >= 20:
            out.append(f"New development signal over the {b['basin']} in {dict((w[0], w[1]) for w in GEN_WINDOWS)[b['window']].lower()}: "
                       f"{b['dep']:.0f}% of members form a depression.")
    n = s.get("nem") or {}
    later = []
    for w in n.get("weeks", []):
        tn = next((a for a in w["areas"] if a["area"] == "Tamil Nadu"), None)
        if not tn:
            continue
        if w["id"] in ("w1", "w2"):
            out.append(f"Tamil Nadu {w['name'].lower()} ({w['from'][8:10]}/{w['from'][5:7]}–{w['to'][8:10]}/{w['to'][5:7]}): "
                       f"{tn['mean']:.0f} mm vs normal {tn['normal']:.0f} mm → {tn['category'].lower()} "
                       f"({tn['confidence'].lower()} confidence).")
        else:
            later.append(f"{w['name']} {tn['category'].lower()}")
    if later:
        out.append("Extended range for Tamil Nadu (low confidence): " + ", ".join(later) + ".")
    ab = s.get("active_break") or {}
    if ab.get("days"):
        act = [d for d in ab["days"] if (d["score"] or 0) >= 60]
        if act:
            out.append(f"Active-NEM conditions indicated on {', '.join(d['valid'][4:10] for d in act[:5])}.")
        else:
            out.append("No strongly active NEM spell indicated in days 0-10.")
    for c in (s.get("consistency") or {}).get("systems", [])[:1]:
        out.append(f"{c['storm']}: {c['verdict'].get('appearances', '')}." + (" " + c["verdict"]["track_shift"] + "."
                                                                              if c["verdict"].get("track_shift") else ""))
    out.append("Model guidance only — follow IMD for official forecasts and warnings.")
    return out


# ============================================================
# RUN
# ============================================================

def drop_duplicates(tracks):
    """Genesis ids (70B, 71B …) that start on a named storm are the same system: keep the named one."""
    if not len(tracks):
        return tracks
    first = tracks[tracks.step == tracks.groupby("storm").step.transform("min")]
    starts = first.groupby("storm")[["lat", "lon"]].median()
    named = starts[~starts.index.str.startswith("7")]
    for storm, (la, lo) in starts.iterrows():
        if storm.startswith("7") and len(named) and haversine_km(la, lo, named.lat.values, named.lon.values).min() < 250:
            tracks = tracks[tracks.storm != storm]
    return tracks


NAMES = {}


def get_tracks(run, rename=True):
    if DEMO:
        k = int(run[-2:]) / 12 + (int(run[6:8]) % 3)
        return od.demo_ens_tracks(run, shift=0.3 * k), od.demo_hres_tracks(run)
    ens = od.ens_tracks(run)
    if ens is None:
        return None, None
    try:
        hres = od.hres_tracks(run)
    except Exception:
        hres = None
    ens = drop_duplicates(ens)
    hres = drop_duplicates(hres) if hres is not None else None
    names = od.official_names(ens, run_time(run)) if rename else {}
    NAMES.update(names)
    if names:                     # show JTWC ids (e.g. Invest 91B) instead of ECMWF's genesis ids (70B …)
        ens = ens.assign(storm=ens.storm.replace(names))
        if hres is not None:
            hres = hres.assign(storm=hres.storm.replace(names))
    return ens, hres


def process(run):
    t0 = run_time(run)
    outdir = os.path.join(OUT_ROOT, run)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    print(f"\n=== Outlook for ECMWF {run} ===")
    NAMES.clear()
    tracks, hres_tracks = get_tracks(run)
    if tracks is None:
        print("  no ENS tracks for this run")
        return False
    print(f"  ENS: {tracks.storm.nunique()} NIO systems; HRES track: {'yes' if hres_tracks is not None else 'no'}")
    s = {"run": run, "init": t0.isoformat(), "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
         "demo": DEMO, "errors": {}}

    def step(name, fn):
        try:
            s[name] = fn()
        except Exception as e:
            traceback.print_exc()
            s[name] = None
            s["errors"][name] = f"{type(e).__name__}: {e}"

    step("genesis", lambda: section_genesis(run, tracks, outdir))
    step("cyclones", lambda: section_cyclones(run, tracks, hres_tracks, outdir))
    step("pdi", lambda: section_pdi(run, tracks, outdir))
    step("consistency", lambda: section_consistency(run, s.get("cyclones") or [], outdir))
    hres = {}
    try:
        steps = list(range(0, 241, 24))
        hres = od.demo_hres(run, steps + [264]) if DEMO else od.hres_fields(run, steps)
        if DEMO:
            for k, v in hres.items():
                v["tp"] = (v["msl"][0], v["msl"][1], np.full_like(v["msl"][2], 3.0 * k / 24))
    except Exception as e:
        s["errors"]["hres"] = repr(e)
    step("active_break", lambda: section_active_break(run, hres, outdir))
    step("environment", lambda: [{"day": st // 24, "png": os.path.relpath(p, SITE_DIR)} for st in (24, 72, 120)
                                 for p in [draw_environment(run, hres, st, f"{outdir}/environment_d{st // 24}.png")] if p])
    step("nem", lambda: section_nem(run, outdir))
    # verification archive (rain fields for the first 3 rain-days) and scoreboard
    try:
        rs = rain_steps(run)
        er = od.demo_ens_tp(run, rs) if DEMO else od.ens_tp(run, rs)
        hr = ({k: {"tp": (v[0], v[1], v[2][0] * 1.1)} for k, v in od.demo_ens_tp(run, rs, n=1).items()} if DEMO
              else od.hres_fields(run, rs, only={"tp"}))
        archive_run(run, tracks, hres_tracks, er, hr)
    except Exception as e:
        traceback.print_exc()
        s["errors"]["archive"] = repr(e)
    step("verification", lambda: section_verification(datetime.now(timezone.utc), outdir))
    s["names"] = {v: k for k, v in NAMES.items()}       # shown id -> ECMWF id
    s["takeaways"] = takeaways(s)
    json.dump(s, open(os.path.join(outdir, "summary.json"), "w"), indent=1, default=str)
    update_manifest(run)
    print(f"  done ({len(s['errors'])} section errors: {', '.join(s['errors']) or 'none'})")
    return True


def update_manifest(run):
    path = os.path.join(SITE_DIR, "data", "manifest_outlook.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    man = json.load(open(path)) if os.path.exists(path) else {"runs": []}
    runs = [r for r in man["runs"] if r["run"] != run]
    runs.append({"run": run, "summary": f"outlook/{run}/summary.json"})
    runs.sort(key=lambda r: r["run"], reverse=True)
    for old in runs[KEEP_RUNS:]:
        d = os.path.join(OUT_ROOT, old["run"])
        if os.path.isdir(d):
            shutil.rmtree(d)
    man["runs"] = runs[:KEEP_RUNS]
    man["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    json.dump(man, open(path, "w"), indent=1)


def done_runs():
    path = os.path.join(SITE_DIR, "data", "manifest_outlook.json")
    return {r["run"] for r in json.load(open(path))["runs"]} if os.path.exists(path) else set()


def candidate_runs(now, n=6):
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 12)
    out = []
    for _ in range(n):
        out.append(t.strftime("%Y%m%d_%H"))
        t -= timedelta(hours=12)
    return out


def backfill(days, now):
    """Archive ENS tracks (and HRES rain) of past 00Z/12Z runs for verification (ECMWF S3 archive)."""
    have = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(STATE, "verif", "tracks", "*.csv"))}
    for run in reversed(candidate_runs(now - timedelta(hours=12), n=days * 2)):
        if run in have:
            continue
        tr, hr = get_tracks(run, rename=False)
        if tr is None:
            print(f"  backfill {run}: no tracks")
            continue
        rain = None
        try:
            rain = od.hres_fields(run, rain_steps(run), only={"tp"}) if not DEMO else None
        except Exception:
            pass
        archive_run(run, tr, hr, None, rain)
        print(f"  backfill {run}: archived")


def main():
    global DEMO
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="YYYYMMDD_HH (00 or 12)")
    ap.add_argument("--demo", action="store_true", help="synthetic data, no downloads")
    ap.add_argument("--backfill-days", type=int, default=int(os.environ.get("OUTLOOK_BACKFILL_DAYS", "0")))
    ap.add_argument("--force", action="store_true", help="redo a run that is already on the site")
    a = ap.parse_args()
    if not a.run and os.environ.get("OUTLOOK_RUN"):
        a.run, a.force = os.environ["OUTLOOK_RUN"], True
    DEMO = a.demo
    for d in ("genesis", "verif/tracks", "verif/rain", "obs"):
        os.makedirs(os.path.join(STATE, d), exist_ok=True)
    now = datetime.now(timezone.utc)
    if a.backfill_days or not glob.glob(os.path.join(STATE, "verif", "tracks", "*.csv")):
        backfill(a.backfill_days or (3 if DEMO else 14), now)
    if a.run:
        runs = [a.run]
    else:
        runs = []
        for r in candidate_runs(now):
            if DEMO or (od.tf_bufr(r) and od.ens_run_complete(r)):
                runs = [r]
                break
        if not runs:
            print("No complete ECMWF ENS 00Z/12Z run found.")
            return 0
        if runs[0] in done_runs() and not a.force:
            print(f"Outlook for {runs[0]} already on the site.")
            return 0
    changed = False
    for r in runs:
        changed |= process(r)
    prune_archive(now)
    if changed:
        shutil.copy(os.path.join(HERE, "web", "outlook.html"), SITE_DIR) if os.path.exists(
            os.path.join(HERE, "web", "outlook.html")) else None
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("new=true\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
