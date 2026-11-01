# ============================================================
# LOW DETECTION, TRACKING AND MULTI-MODEL CONSENSUS
# ============================================================
#
# 1. Detect lows in every member at every step (MSLP minima with a
#    pressure deficit; wind circulation check where winds exist —
#    same idea as the V3 detector in the weathernext3 project).
# 2. Link each member's lows in time into tracks.
# 3. Group tracks from all members/models into systems (L1, L2 ...).
# 4. Weighted consensus track per system, spread, support, landfall.

import numpy as np
from scipy.ndimage import minimum_filter, gaussian_filter
from shapely.geometry import Point
from shapely.prepared import prep
import cartopy.feature as cfeature
from shapely.ops import unary_union

from . import config as C
from .grib import TRACK

STEPS = C.TRACK_STEPS
DT = STEPS[1] - STEPS[0]

MIN_DEFICIT_HPA = 1.0          # ring median minus centre pressure
CIRCULATION_FLOOR = 0.55       # share of ring points with cyclonic flow (winds only)
MAX_MOVE_KM = 40.0 * DT        # 12 h -> 480 km
MIN_TRACK_STEPS = 3            # at least 24 h of life
CLUSTER_KM = 400.0
MIN_SUPPORT = 0.25             # share of group weight needed to show a system
SEA_DEFICIT = 2.0              # peak consensus deficit (hPa) for a low that is over sea at some point
LAND_DEFICIT = 3.0             # ... for a low that stays over land (monsoon lows / depressions)

LAND = prep(unary_union(list(cfeature.LAND.with_scale("50m").geometries())))

CAT = [("LPA", 0, "#9e9e9e"), ("D", 17, "#4daf4a"), ("DD", 28, "#e6c300"), ("CS", 34, "#ff7f00"),
       ("SCS", 48, "#e41a1c"), ("VSCS", 64, "#984ea3"), ("ESCS", 90, "#5b1a72"), ("SuCS", 120, "#2b0a3d")]


def category(kt):
    if kt is None or not np.isfinite(kt):
        return CAT[0]
    out = CAT[0]
    for row in CAT:
        if kt >= row[1]:
            out = row
    return out


def km(lat1, lon1, lat2, lon2):
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ------------------------------------------------------------
# 1. DETECTION
# ------------------------------------------------------------

def detect(msl, u=None, v=None):
    if not np.isfinite(msl).any():
        return []
    p = gaussian_filter(np.nan_to_num(msl, nan=np.nanmean(msl)), 1.0)
    size = int(round(3.0 / TRACK.step)) | 1
    w, e, s, n = C.DETECT_AREA
    cands = []
    for iy, ix in np.argwhere(p == minimum_filter(p, size=size, mode="nearest")):
        lat, lon = float(TRACK.lat[iy]), float(TRACK.lon[ix])
        if not (w <= lon <= e and s <= lat <= n):
            continue
        d = km(lat, lon, TRACK.LAT2, TRACK.LON2)
        ring = (d >= 2 * 111) & (d <= 4 * 111)
        deficit = float(np.median(p[ring]) - p[iy, ix])
        if deficit < MIN_DEFICIT_HPA:
            continue
        wind = None
        if u is not None and np.isfinite(u).any():
            inner = (d >= 0.8 * 111) & (d <= 2.0 * 111)
            dx = np.radians(TRACK.LON2 - lon) * np.cos(np.radians(lat))
            dy = np.radians(TRACK.LAT2 - lat)
            r = np.hypot(dx, dy)
            r[r == 0] = np.nan
            tang = (u * (-dy / r) + v * (dx / r))[inner]
            if np.mean(tang > 0) < CIRCULATION_FLOOR:
                continue
            core = d <= 2.5 * 111
            wind = float(np.nanmax(np.hypot(u, v)[core]) * 1.94384)
        cands.append({"lat": lat, "lon": lon, "msl": float(msl[iy, ix]), "deficit": deficit, "wind": wind})
    return cands


# ------------------------------------------------------------
# 2. TRACKING (per member)
# ------------------------------------------------------------

def track_member(f):
    by_step = [detect(f["msl"][k], f.get("u10", [None] * len(STEPS))[k], f.get("v10", [None] * len(STEPS))[k])
               for k in range(len(STEPS))]
    active, done = [], []
    for k, cands in enumerate(by_step):
        pairs = []
        for ti, t in enumerate(active):
            last = t[-1]
            for ci, c in enumerate(cands):
                d = km(last["lat"], last["lon"], c["lat"], c["lon"])
                if d <= MAX_MOVE_KM:
                    pairs.append((d + 30 * abs(c["msl"] - last["msl"]), ti, ci))
        pairs.sort()
        used_t, used_c, nxt = set(), set(), []
        for _, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            nxt.append(active[ti] + [dict(cands[ci], k=k)])
        done += [t for ti, t in enumerate(active) if ti not in used_t]
        nxt += [[dict(c, k=k)] for ci, c in enumerate(cands) if ci not in used_c]
        active = nxt
    done += active
    return [t for t in done if len(t) >= MIN_TRACK_STEPS]


# ------------------------------------------------------------
# 3 + 4. SYSTEMS AND CONSENSUS
# ------------------------------------------------------------

_PLAINS = []


def _plains():
    """India (GoI outline) south of 26.5N, west of the north-east hills."""
    if not _PLAINS:
        from .maps import boundaries
        from shapely.geometry import box
        _PLAINS.append(prep(boundaries()["india"].intersection(box(68.0, 8.0, 89.5, 26.5))))
    return _PLAINS[0]


def _mean_track(tracks):
    """Weighted mean position per step. tracks: list of (weight, track)."""
    out = {}
    wtot = sum(w for w, _ in tracks)
    for k in range(len(STEPS)):
        pts = [(w, p) for w, t in tracks for p in t if p["k"] == k]
        ws = sum(w for w, _ in pts)
        if not pts or ws < 0.25 * wtot:
            continue
        lat = sum(w * p["lat"] for w, p in pts) / ws
        lon = sum(w * p["lon"] for w, p in pts) / ws
        winds = [(w, p["wind"]) for w, p in pts if p["wind"] is not None]
        out[k] = {
            "lat": lat, "lon": lon,
            "msl": sum(w * p["msl"] for w, p in pts) / ws,
            "deficit": sum(w * p["deficit"] for w, p in pts) / ws,
            "wind": sum(w * x for w, x in winds) / sum(w for w, _ in winds) if winds else None,
            "spread": sum(w * km(lat, lon, p["lat"], p["lon"]) for w, p in pts) / ws,
            "share": ws / wtot,
        }
    return out


def _match(mean, track):
    d = [km(mean[p["k"]]["lat"], mean[p["k"]]["lon"], p["lat"], p["lon"]) for p in track if p["k"] in mean]
    return np.mean(d) if len(d) >= 2 else np.inf


def systems(group_fields, group):
    """group_fields: {model: {member: fields}} -> list of systems, strongest support first."""
    weights = C.GROUPS[group]["models"]
    present = [m for m in group_fields if m in weights]
    wsum = sum(weights[m] for m in present)
    tracks = []  # (weight, model, member, track)
    for m in present:
        n = len(group_fields[m])
        for member, f in group_fields[m].items():
            for t in track_member(f):
                tracks.append((weights[m] / wsum / n, m, member, t))

    clusters = []
    for w, m, member, t in sorted(tracks, key=lambda x: -x[0] * len(x[3])):
        best, bd = None, CLUSTER_KM
        for c in clusters:
            if (m, member) in c["who"]:
                continue
            d = _match(c["mean"], t)
            if d < bd:
                best, bd = c, d
        if best is None:
            best = {"tracks": [], "who": set()}
            clusters.append(best)
        best["tracks"].append((w, t))
        best["who"].add((m, member))
        best["members"] = best.get("members", []) + [(w, m, member, t)]
        best["mean"] = _mean_track(best["tracks"])

    out = []
    for c in clusters:
        support = sum(w for w, _ in c["tracks"])
        if support < (MIN_SUPPORT if group == "ens" else 0.3) or len(c["mean"]) < 2:
            continue
        # a real low: a clear pressure deficit, over sea at some point (or deep enough over land)
        peak_def = max(p["deficit"] for p in c["mean"].values())
        any_sea = any(not LAND.contains(Point(p["lon"], p["lat"])) for p in c["mean"].values())
        if not ((any_sea and peak_def >= SEA_DEFICIT) or peak_def >= LAND_DEFICIT):
            continue
        # must form over the sea or over the Indian plains (where monsoon lows form);
        # this drops false lows over high ground and desert heat lows
        g = c["mean"][min(c["mean"])]
        gp = Point(g["lon"], g["lat"])
        if LAND.contains(gp) and not _plains().contains(gp):
            continue
        models = sorted({m for _, m, _, _ in c["members"]})
        out.append({"support": support, "mean": c["mean"], "members": c["members"], "models": models})
    out.sort(key=lambda s: -s["support"])
    return out


BASINS = [("Bay of Bengal", lambda la, lo: lo >= 78 and not LAND.contains(Point(lo, la))),
          ("Arabian Sea", lambda la, lo: lo < 78 and not LAND.contains(Point(lo, la)))]


def basin(lat, lon):
    for name, test in BASINS:
        if test(lat, lon):
            return name
    return "Over land"


SECTORS = [  # name, (W, E, S, N) of the stretch of coast
    ("Sri Lanka", (79.5, 82.0, 5.8, 9.9)),
    ("South Tamil Nadu coast", (77.0, 80.5, 8.0, 10.5)),
    ("North Tamil Nadu & Puducherry coast", (79.0, 80.6, 10.5, 13.6)),
    ("South Andhra coast", (79.5, 81.5, 13.6, 16.0)),
    ("North Andhra coast", (80.5, 84.8, 16.0, 19.2)),
    ("Odisha coast", (84.5, 87.5, 19.2, 21.6)),
    ("West Bengal & Bangladesh coast", (86.5, 92.0, 21.4, 23.5)),
    ("Myanmar coast", (92.0, 99.0, 9.0, 23.0)),
    ("Kerala coast", (74.5, 77.3, 8.0, 12.8)),
    ("Karnataka & Goa coast", (73.5, 75.2, 12.8, 15.9)),
    ("Maharashtra coast", (72.5, 73.8, 15.9, 20.2)),
    ("Gujarat coast", (68.0, 73.0, 20.2, 24.0)),
    ("Pakistan coast", (61.5, 68.5, 24.0, 26.0)),
    ("Oman & Yemen coast", (45.0, 60.5, 12.0, 23.0)),
    ("Somalia & Gulf of Aden coast", (41.0, 51.5, 0.0, 12.0)),
    ("Andaman & Nicobar Islands", (92.0, 94.5, 6.5, 14.0)),
    ("Bangladesh coast", (89.0, 92.5, 20.5, 23.0)),
]


def landfall(track):
    """(sector, step index) of the first sea -> land crossing, or None."""
    for a, b in zip(track, track[1:]):
        if not LAND.contains(Point(a["lon"], a["lat"])) and LAND.contains(Point(b["lon"], b["lat"])):
            la, lo = (a["lat"] + b["lat"]) / 2, (a["lon"] + b["lon"]) / 2
            for name, (w, e, s, n) in SECTORS:
                if w <= lo <= e and s <= la <= n:
                    return name, b["k"]
            return "Other coast", b["k"]
    return None


def landfall_probs(system):
    tot = sum(w for w, *_ in system["members"])
    probs, when = {}, {}
    for w, _, _, t in system["members"]:
        lf = landfall(t)
        key = lf[0] if lf else "No landfall by T+168 h"
        probs[key] = probs.get(key, 0) + w / tot
        if lf:
            when.setdefault(key, []).append((w, lf[1]))
    timing = {k: STEPS[int(round(sum(w * s for w, s in v) / sum(w for w, _ in v)))] for k, v in when.items()}
    return sorted(probs.items(), key=lambda x: -x[1]), timing


def link_det(ens_sys, det_systems):
    """The deterministic system closest to an ensemble system (shares >= 2 steps within 450 km)."""
    best, bd = None, 450.0
    for d in det_systems:
        common = [k for k in ens_sys["mean"] if k in d["mean"]]
        if len(common) < 2:
            continue
        dist = np.mean([km(ens_sys["mean"][k]["lat"], ens_sys["mean"][k]["lon"],
                           d["mean"][k]["lat"], d["mean"][k]["lon"]) for k in common])
        if dist < bd:
            best, bd = d, dist
    return best
