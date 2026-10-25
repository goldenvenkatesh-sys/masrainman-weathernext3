# ============================================================
# MASRAINMAN — ENSEMBLE 24-HOUR RAIN CHARTS PER DISTRICT
#   ECMWF IFS ENS (51 members, open data)  +  WeatherNext 3 (percentiles)
# ============================================================
#
# For every district headquarters in ens/districts.csv (TN, Kerala,
# Karnataka, AP, Telangana) it draws a chart like the WeatherBell
# "ECMWF Ens 24-Hour QPF" plume:
#   top    : a grid of 24-hour rain totals, one row per ensemble member
#            (WeatherNext 3: one row per percentile + the mean), one
#            column every 6 h, coloured and numbered in mm
#   bottom : control / mean lines with the spread of the ensemble
#
# WeatherNext 3: the store we can read has only the ensemble mean and the
# 10/25/50/75/90th percentiles of hourly rain, not the members. The 24-h
# percentile rows are sums of hourly percentiles: the mean row is exact,
# the percentile rows are only a guide (they over-spread a little).
#
# Output (published on the site under /ens/):
#   site/ens/manifest.json
#   site/ens/<model>/<run>/<district-id>.png
#
# Usage:
#   python ens_meteogram.py                     # all models, newest runs not on the site
#   python ens_meteogram.py --check             # only say if there is a new run (no heavy imports)
#   python ens_meteogram.py --models ecmwf_ens  # one model
#   python ens_meteogram.py --model wn3 --run 20261025_00hr_01_preds   # force one run
#   python ens_meteogram.py --demo              # synthetic data, no downloads
# ============================================================

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SITE_DIR = os.environ.get("SITE_DIR", os.path.join(HERE, "site"))
ENS_DIR = os.path.join(SITE_DIR, "ens")
DISTRICTS_CSV = os.path.join(HERE, "ens", "districts.csv")

KEEP_RUNS = int(os.environ.get("ENS_KEEP_RUNS", "2"))      # runs kept on the site, per model
STEP_H = 6                                                 # one column every 6 h
PERIOD_H = 24                                              # 24-hour totals
LAST_H = 360
IST = timezone(timedelta(hours=5, minutes=30))

# ---- ECMWF open data (ENS = stream enfo; 00/12Z run to T+360)
ECMWF_ROOTS = [r.strip() for r in os.environ.get(
    "ECMWF_ENS_ROOTS",
    "https://data.ecmwf.int/forecasts,https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
).split(",") if r.strip()]
ENS_RESOLS = [r.strip() for r in os.environ.get("ENS_RESOLS", "0p25").split(",") if r.strip()]
HTTP_TIMEOUT = (10, 120)

# ---- WeatherNext 3 statistics store (same as render.py)
STATS_BUCKET = "weathernext3_statistics_spatial"
RUN_ROOT_PREFIX = "weathernext_3_0_0_statistics/zarr/2026_to_present/"
WN3_RUN_PATTERN = r"^(\d{8})_(00|06|12|18)hr_01_preds$"
WN3_VAR = os.environ.get("WN3_ENS_VAR", "total_precipitation_1hr")
WN3_STATS = ["p90", "p75", "p50", "p25", "p10", "mean"]          # top -> bottom rows
WN3_STAT_LABEL = {"p90": "p90", "p75": "p75", "p50": "median", "p25": "p25", "p10": "p10",
                  "mean": "mean"}

MODELS = {
    "ecmwf_ens": {"label": "ECMWF ENS (members)", "short": "ECMWF ENS",
                  "credit": "Source: ECMWF open data (CC BY 4.0)"},
    "wn3": {"label": "WeatherNext 3 (percentiles)", "short": "WeatherNext 3",
            "credit": "Source: Google DeepMind WeatherNext 3 (experimental)"},
}

# Rain colour scale (mm) — same as the site's rain maps (render.py)
RAIN_LEVELS = [0.2, 0.6, 1.2, 1.8, 2.5, 3.75, 5, 6.25, 7.5, 10, 12.5, 15, 17.5, 20, 22.5,
               26, 30, 35, 40, 45, 50, 62.5, 75, 87.5, 100, 125, 150, 175, 200, 225, 250,
               275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
RAIN_TICKS = [0.2, 1.2, 2.5, 5, 7.5, 12.5, 17.5, 22.5, 30, 40, 50, 75, 100, 150, 200,
              250, 300, 350, 400, 450, 500]
RAIN_COLOURS = [
    "#d0d0d0", "#bdbdbd", "#a3a3a3", "#8c8c8c", "#6f6f6f",
    "#c4f7b8", "#a2ef8f", "#7fe36a", "#5ad146", "#33b52b", "#12951c",
    "#0b3fb8", "#2266dd", "#4a95ee", "#8fc8f7",
    "#fff3a3", "#ffde5c", "#ffc430", "#ffa51f", "#ff8a16", "#ff6a10",
    "#f2450c", "#df1a0a", "#c40606", "#a10000", "#7c0b05",
    "#5e2a1e", "#7d4a3c", "#a07264", "#c49d91", "#e6cfc8",
    "#ddd0ee", "#c4b3e6", "#a991d8", "#8e6fc8", "#7650b5", "#6630a0",
    "#8020a8", "#a01ab8", "#c626cf",
]
RAIN_OVER = "#e83ce8"

STATE_CODE = {"Tamil Nadu": "tn", "Kerala": "kl", "Karnataka": "ka",
              "Andhra Pradesh": "ap", "Telangana": "tg"}


# ============================================================
# DISTRICTS
# ============================================================

def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_districts():
    out = []
    with open(DISTRICTS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append({
                "id": f"{STATE_CODE[row['state']]}-{slug(row['district'])}",
                "state": row["state"], "district": row["district"], "hq": row["hq"],
                "lat": float(row["lat"]), "lon": float(row["lon"]),
            })
    ids = [d["id"] for d in out]
    assert len(ids) == len(set(ids)), "duplicate district ids"
    return out


def columns():
    """Valid hours of the 24-h totals: T+24, T+30 … T+360."""
    return list(range(PERIOD_H, LAST_H + 1, STEP_H))


# ============================================================
# MANIFEST
# ============================================================

def manifest_path():
    return os.path.join(ENS_DIR, "manifest.json")


def read_manifest():
    try:
        with open(manifest_path(), encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        m = {}
    m.setdefault("models", {})
    for key, cfg in MODELS.items():
        m["models"].setdefault(key, {"label": cfg["label"], "runs": []})
        m["models"][key]["label"] = cfg["label"]
    return m


def write_manifest(m, districts):
    m["districts"] = [{k: d[k] for k in ("id", "state", "district", "hq", "lat", "lon")}
                      for d in districts]
    m["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    os.makedirs(ENS_DIR, exist_ok=True)
    with open(manifest_path(), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)


def add_run(m, model, entry):
    runs = [r for r in m["models"][model]["runs"] if r["run"] != entry["run"]]
    runs.append(entry)
    runs.sort(key=lambda r: r["init"], reverse=True)
    keep, drop = runs[:KEEP_RUNS], runs[KEEP_RUNS:]
    for r in drop:
        shutil.rmtree(os.path.join(ENS_DIR, model, r["run"]), ignore_errors=True)
    m["models"][model]["runs"] = keep


def known_runs(m, model):
    return {r["run"] for r in m["models"][model]["runs"]}


def newest_known_init(m, model):
    runs = m["models"][model]["runs"]
    return max((r["init"] for r in runs), default="")


# ============================================================
# ECMWF ENS (open data)
# ============================================================

_tls = threading.local()


def http():
    s = getattr(_tls, "s", None)
    if s is None:
        import requests
        s = requests.Session()
        s.headers["User-Agent"] = "MasRainman-ens-meteogram"
        _tls.s = s
    return s


def ens_base(root, run_time, resol):
    ymd, hh = run_time.strftime("%Y%m%d"), run_time.hour
    return f"{root}/{ymd}/{hh:02d}z/ifs/{resol}/enfo", f"{ymd}{hh:02d}0000"


def ens_find_source(run_time):
    """First (root, resol) that has the T+360 index of this run, else None."""
    for resol in ENS_RESOLS:
        for root in ECMWF_ROOTS:
            base, stamp = ens_base(root, run_time, resol)
            try:
                r = http().head(f"{base}/{stamp}-{LAST_H}h-enfo-ef.index",
                                timeout=(5, 20), allow_redirects=True)
            except Exception as exc:
                print(f"  (ENS check {root} {resol}: {type(exc).__name__})")
                continue
            if r.status_code == 200:
                return {"root": root, "resol": resol}
    return None


def ens_latest_runs(n=3):
    """Newest complete 00/12Z ENS runs: [(run_name, source)], newest first."""
    t = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 12)
    found = []
    for _ in range(8):
        src = ens_find_source(t)
        if src:
            found.append((t.strftime("%Y%m%d_%H"), src))
            if len(found) >= n:
                break
        t -= timedelta(hours=12)
    return found


def _merge_ranges(parts):
    """[(off, len)] -> [(start, end_inclusive, [(off, len)…])] joining touching ranges."""
    out = []
    for off, ln in sorted(parts):
        if out and off <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], off + ln - 1)
            out[-1][2].append((off, ln))
        else:
            out.append([off, off + ln - 1, [(off, ln)]])
    return out


def _get(url, headers=None):
    last = None
    for _ in range(4):
        try:
            r = http().get(url, timeout=HTTP_TIMEOUT, headers=headers or {})
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
    raise last


def ens_step_messages(base, stamp, step):
    """Raw GRIB messages of tp (all 51 members) for one step."""
    idx = _get(f"{base}/{stamp}-{step}h-enfo-ef.index").text
    parts, types = [], set()
    for line in idx.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("param") == "tp":
            parts.append((int(rec["_offset"]), int(rec["_length"])))
            types.add(rec.get("type"))
    if step == STEP_H:
        print(f"  step {step}: {len(parts)} tp messages (types {sorted(map(str, types))})")
    if len(parts) < 50:     # 50 perturbed members (+ control when it is published)
        raise RuntimeError(f"step {step}: only {len(parts)} tp messages in the index")
    msgs = []
    url = f"{base}/{stamp}-{step}h-enfo-ef.grib2"
    for start, end, pieces in _merge_ranges(parts):
        blob = _get(url, headers={"Range": f"bytes={start}-{end}"}).content
        if len(blob) != end - start + 1:
            raise RuntimeError(f"step {step}: short read {len(blob)}/{end - start + 1}")
        for off, ln in pieces:
            msgs.append(blob[off - start: off - start + ln])
    return msgs


def grid_points(gid, districts):
    """Index into a regular lat/lon GRIB field of the grid point nearest to each district."""
    import eccodes as ec
    ni = ec.codes_get(gid, "Ni")
    nj = ec.codes_get(gid, "Nj")
    lat0 = ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
    lon0 = ec.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
    di = ec.codes_get(gid, "iDirectionIncrementInDegrees")
    dj = ec.codes_get(gid, "jDirectionIncrementInDegrees")
    jpos = ec.codes_get(gid, "jScansPositively")
    out = []
    for d in districts:
        j = int(round((d["lat"] - lat0) / dj)) if jpos else int(round((lat0 - d["lat"]) / dj))
        i = int(round(((d["lon"] - lon0) % 360.0) / di)) % ni
        j = min(max(j, 0), nj - 1)
        glat = lat0 + j * dj if jpos else lat0 - j * dj
        glon = (lon0 + i * di + 180.0) % 360.0 - 180.0
        out.append((j * ni + i, round(glat, 3), round(glon, 3)))
    return out, di


def load_ecmwf_ens(run_name, source, districts):
    """-> dict(values [district, member(51), column] in mm, grid points, grid_deg)."""
    import eccodes as ec

    run_time = datetime.strptime(run_name, "%Y%m%d_%H").replace(tzinfo=timezone.utc)
    base, stamp = ens_base(source["root"], run_time, source["resol"])
    steps = list(range(STEP_H, LAST_H + 1, STEP_H))
    print(f"\nECMWF ENS {run_name} ({source['resol']}, {source['root']}): "
          f"{len(steps)} steps x 51 members of tp ...")

    state = {}
    lock = threading.Lock()

    def fetch(step):
        msgs = ens_step_messages(base, stamp, step)
        acc = np.full((len(districts), 51), np.nan, dtype=np.float32)
        for raw in msgs:
            gid = ec.codes_new_from_message(raw)
            try:
                with lock:
                    if "pts" not in state:
                        state["pts"], state["grid"] = grid_points(gid, districts)
                        state["units"] = ec.codes_get(gid, "units")
                num = ec.codes_get(gid, "number") if ec.codes_is_defined(gid, "number") else 0
                vals = ec.codes_get_values(gid)
                acc[:, int(num)] = vals[[p[0] for p in state["pts"]]]
            finally:
                ec.codes_release(gid)
        if np.isnan(acc[:, 1:]).any():
            raise RuntimeError(f"step {step}: some perturbed members missing")
        return step, acc

    with ThreadPoolExecutor(max_workers=8) as pool:
        res = {}
        for i, (step, acc) in enumerate(pool.map(fetch, steps), 1):
            res[step] = acc
            if i % 10 == 0:
                print(f"  {i}/{len(steps)} steps")

    scale = 1000.0 if str(state.get("units", "m")).strip() == "m" else 1.0
    # running total since T+0, [district, member, hour]; T+0 = 0
    hours = [0] + steps
    tot = np.stack([np.zeros_like(res[steps[0]])] + [res[s] for s in steps], axis=-1) * scale
    has_ctrl = not np.isnan(tot[:, 0, 1:]).any()
    if not has_ctrl:
        print("  (control run has no rain in the open data - grid shows the 50 members)")
        tot[:, 0, :] = 0
    tot = np.maximum.accumulate(np.maximum(tot, 0), axis=-1)       # packing noise can dip it
    hidx = {h: k for k, h in enumerate(hours)}
    cols = columns()
    q = np.stack([tot[..., hidx[h]] - tot[..., hidx[h - PERIOD_H]] for h in cols], axis=-1)
    q = np.maximum(q, 0).astype(np.float32)
    if not has_ctrl:
        q = q[:, 1:]                                                # [district, 50, column]
    print(f"  done: max 24-h member total {np.nanmax(q):.0f} mm")
    return {"values": q, "points": [(p[1], p[2]) for p in state["pts"]],
            "grid_deg": float(state["grid"]), "resol": source["resol"], "control": has_ctrl}


# ============================================================
# WEATHERNEXT 3 (statistics store)
# ============================================================

def wn3_session():
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(creds)


def wn3_list_runs(session):
    runs, params = [], {"prefix": RUN_ROOT_PREFIX, "delimiter": "/", "maxResults": 1000}
    url = f"https://storage.googleapis.com/storage/v1/b/{STATS_BUCKET}/o"
    while True:
        r = session.get(url, params=params)
        r.raise_for_status()
        d = r.json()
        for p in d.get("prefixes", []):
            name = p.rstrip("/").split("/")[-1]
            if re.match(WN3_RUN_PATTERN, name):
                runs.append(name)
        if not d.get("nextPageToken"):
            break
        params["pageToken"] = d["nextPageToken"]
    return sorted(set(runs), reverse=True)


def wn3_open(run_name):
    import obstore
    import zarr
    prefix = f"{RUN_ROOT_PREFIX}{run_name}/predictions.zarr"
    store = None
    try:
        from obstore.auth.google import GoogleCredentialProvider
        store = obstore.store.GCSStore(bucket=STATS_BUCKET, prefix=prefix,
                                       credential_provider=GoogleCredentialProvider())
    except Exception as exc:
        print(f"(credential provider not used: {exc})")
    if store is None:
        key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        store = obstore.store.GCSStore(bucket=STATS_BUCKET, prefix=prefix,
                                       config={"google_service_account_path": key} if key else {})
    return zarr.open_group(zarr.storage.ObjectStore(store, read_only=True), mode="r", zarr_format=3)


def wn3_complete(root):
    arr = root[f"{WN3_VAR}_mean"]
    if int(arr.shape[0]) < LAST_H:
        return False
    v = float(np.asarray(arr[LAST_H - 1, arr.shape[1] // 2, arr.shape[2] // 2]))
    return np.isfinite(v)


def load_wn3(run_name, districts, root=None):
    """-> dict(values [district, stat(6), column] in mm, grid points)."""
    root = root or wn3_open(run_name)
    lat = np.asarray(root["lat_0p1"][:], dtype=np.float64)
    lon = np.asarray(root["lon_0p1"][:], dtype=np.float64)
    lon180 = (lon + 180.0) % 360.0 - 180.0
    jj = np.array([int(np.argmin(np.abs(lat - d["lat"]))) for d in districts])
    ii = np.array([int(np.argmin(np.abs(lon180 - d["lon"]))) for d in districts])
    j0, j1, i0, i1 = jj.min(), jj.max() + 1, ii.min(), ii.max() + 1
    print(f"\nWeatherNext 3 {run_name}: {len(WN3_STATS)} fields, box {j1 - j0}x{i1 - i0} x {LAST_H} h ...")

    cols = columns()
    out = np.zeros((len(districts), len(WN3_STATS), len(cols)), dtype=np.float32)

    def one(k_stat):
        k, stat = k_stat
        arr = root[f"{WN3_VAR}_{stat}"]
        box = np.asarray(arr[0:LAST_H, j0:j1, i0:i1], dtype=np.float32)       # m, hourly
        pts = np.nan_to_num(box[:, jj - j0, ii - i0]) * 1000.0                 # [hour, district] mm
        c = np.concatenate([np.zeros((1, pts.shape[1]), np.float32),
                            np.cumsum(np.maximum(pts, 0), axis=0)], axis=0)    # c[h] = total to T+h
        for n, h in enumerate(cols):
            out[:, k, n] = c[h] - c[h - PERIOD_H]
        print(f"  {stat} loaded")

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one, enumerate(WN3_STATS)))

    pts = [(round(float(lat[j]), 3), round(float(lon180[i]), 3)) for j, i in zip(jj, ii)]
    print(f"  done: max 24-h p90 {out[:, 0].max():.0f} mm, mean {out[:, -1].max():.0f} mm")
    return {"values": out, "points": pts, "grid_deg": 0.1, "resol": "0p1"}


# ============================================================
# DRAWING
# ============================================================

def compass(dlat, dlon):
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ang = (math.degrees(math.atan2(dlon, dlat)) + 360) % 360
    return names[int((ang + 11.25) // 22.5) % 16]


def km_between(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def text_colour(hex_colour):
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"


def draw_chart(out_png, model, run_name, init, district, grid_pt, grid_deg, values):
    """values: [rows, columns] in mm (ECMWF: 51 members c00..50; WN3: WN3_STATS order)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cols = columns()
    ncol = len(cols)
    valid = [init + timedelta(hours=h) for h in cols]
    is_ens = model == "ecmwf_ens"

    has_ctrl = is_ens and values.shape[0] == 51
    if is_ens:
        # rows bottom -> top: c00, 01 … 50 (like WeatherBell); no c00 if control is missing
        row_labels = (["c00"] if has_ctrl else []) + [f"{m:02d}" for m in range(1, 51)]
        grid_vals = values                      # row 0 = control at the bottom
        fs_cell, fs_row = 5.0, 5.5
        fig_h, top_ratio = 13.0, 2.2
    else:
        order = WN3_STATS[::-1]                  # bottom -> top: mean, p10 … p90
        row_labels = [WN3_STAT_LABEL[s] for s in order]
        grid_vals = values[[WN3_STATS.index(s) for s in order]]
        fs_cell, fs_row = 7.5, 8.5
        fig_h, top_ratio = 9.5, 0.9

    cmap = ListedColormap(RAIN_COLOURS)
    cmap.set_under("white")
    cmap.set_over(RAIN_OVER)
    norm = BoundaryNorm(RAIN_LEVELS, cmap.N)

    fig = plt.figure(figsize=(16, fig_h), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[top_ratio, 1], width_ratios=[60, 1],
                          left=0.045, right=0.955, top=0.905, bottom=0.09 if is_ens else 0.125,
                          hspace=0.08, wspace=0.015)
    ax = fig.add_subplot(gs[0, 0])
    axl = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 1])

    nrow = grid_vals.shape[0]
    mesh = ax.pcolormesh(np.arange(ncol + 1), np.arange(nrow + 1), grid_vals,
                         cmap=cmap, norm=norm, edgecolors="white", linewidth=0.3)
    for r in range(nrow):
        for c in range(ncol):
            v = float(grid_vals[r, c])
            if v < 0.2:
                continue
            k = min(int(np.searchsorted(RAIN_LEVELS, v, side="right")) - 1, len(RAIN_COLOURS) - 1)
            colour = RAIN_OVER if v >= RAIN_LEVELS[-1] else RAIN_COLOURS[k]
            ax.text(c + 0.5, r + 0.5, f"{v:.0f}", ha="center", va="center",
                    fontsize=fs_cell, color=text_colour(colour))
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.set_xticks([])
    if is_ens:
        ticks = [k for k in range(nrow) if (k % 2 == 0) == has_ctrl]
        ax.set_yticks([k + 0.5 for k in ticks])
        ax.set_yticklabels([row_labels[k] for k in ticks], fontsize=fs_row)
    else:
        ax.set_yticks([k + 0.5 for k in range(nrow)])
        ax.set_yticklabels(row_labels, fontsize=fs_row)
        ax.axhline(1, color="black", lw=1.2)          # line between mean and percentiles
    for c in range(ncol):                              # mark each day (00z)
        if valid[c].hour == 0:
            ax.axvline(c, color="#444", lw=0.6)

    # ---- lower panel
    x = np.arange(ncol) + 0.5
    if is_ens:
        ctrl, mean = (values[0] if has_ctrl else None), values.mean(axis=0)
        lo, hi = values.min(axis=0), values.max(axis=0)
        p25, p75 = np.percentile(values, 25, axis=0), np.percentile(values, 75, axis=0)
        axl.vlines(x, lo, hi, color="black", lw=0.6)
        axl.hlines(hi, x - 0.18, x + 0.18, color="black", lw=0.6)
        axl.bar(x, p75 - p25, bottom=p25, width=0.45, color="#bbbbbb", edgecolor="#777", lw=0.4,
                label="25–75%")
        if ctrl is not None:
            axl.plot(x, ctrl, color="#1f3fbf", marker="o", ms=2.2, lw=1.1, label="Control")
        axl.plot(x, mean, color="#1c8a2e", marker="o", ms=2.2, lw=1.1, label="Mean")
        top = float(hi.max())
    else:
        g = {s: values[WN3_STATS.index(s)] for s in WN3_STATS}
        axl.fill_between(x, g["p10"], g["p90"], color="#cfe3f7", label="p10–p90")
        axl.fill_between(x, g["p25"], g["p75"], color="#8fbde8", label="p25–p75")
        axl.plot(x, g["p50"], color="#1f3fbf", lw=1.0, ls="--", label="Median")
        axl.plot(x, g["mean"], color="#1c8a2e", marker="o", ms=2.2, lw=1.3, label="Mean")
        top = float(g["p90"].max())
    axl.set_xlim(0, ncol)
    axl.set_ylim(0, max(10.0, top * 1.08))
    axl.grid(axis="y", color="#ddd", lw=0.6)
    axl.set_ylabel("mm / 24 h", fontsize=9)
    axl.tick_params(axis="y", labelsize=8)
    axl.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axl.set_xticks(x)
    axl.set_xticklabels([f"{t:%d}-{t:%H}z" for t in valid], rotation=90, fontsize=6.5)
    for c in range(ncol):
        if valid[c].hour == 0:
            axl.axvline(c, color="#999", lw=0.5)
        if valid[c].hour == 0 and c + 2 < ncol:
            axl.annotate(f"{valid[c]:%a %d %b}", xy=(c + 2, 0), xycoords=axl.get_xaxis_transform(),
                         xytext=(0, -40), textcoords="offset points",
                         ha="center", va="top", fontsize=7.5)

    # ---- colour bar
    cb = fig.colorbar(mesh, cax=cax, ticks=RAIN_TICKS, extend="max")
    cb.ax.tick_params(labelsize=7)
    cb.ax.set_yticklabels([f"{t:g}" for t in RAIN_TICKS])

    # ---- titles
    cfg = MODELS[model]
    glat, glon = grid_pt
    dist = km_between(glat, glon, district["lat"], district["lon"])
    where = compass(district["lat"] - glat, district["lon"] - glon)
    init_ist = init.astimezone(IST)
    head = (f"{cfg['short']} {grid_deg:g}°  •  Init {init:%Hz %d %b %Y} ({init_ist:%-I:%M %p} IST)"
            f"  •  24-Hour Rain (mm)")
    if not is_ens:
        head += "  •  percentiles of hourly rain, summed"
    elif not has_ctrl:
        head += "  •  50 members"
    fig.text(0.045, 0.975, head, fontsize=13, weight="bold", va="top")
    fig.text(0.045, 0.948,
             f"{district['hq']} — {district['district']} district, {district['state']}  "
             f"[{district['lat']:.2f}°N, {district['lon']:.2f}°E]", fontsize=11.5, va="top")
    fig.text(0.045, 0.925,
             f"Grid point: [{glat:.2f}°N, {glon:.2f}°E], {dist:.1f} km to the {where} of it. "
             f"Each column = rain in the 24 h ending at that time (UTC).",
             fontsize=9.5, va="top", color="#333")
    fig.text(0.955, 0.012, f"{cfg['credit']}  •  #MasRainman  •  Not an official forecast",
             fontsize=8, ha="right", color="#444")

    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    try:   # 256-colour PNG: about a third of the size, looks the same
        from PIL import Image
        with Image.open(out_png) as im:
            q = im.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        q.save(out_png, optimize=True)
    except Exception as exc:
        print(f"  (PNG not compressed: {exc})")


def _draw_job(args):
    draw_chart(*args)
    return args[0]


def render_run(model, run_name, init, districts, data, entry_extra=None):
    out_dir = os.path.join(ENS_DIR, model, run_name)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for k, d in enumerate(districts):
        jobs.append((os.path.join(out_dir, f"{d['id']}.png"), model, run_name, init, d,
                     data["points"][k], data["grid_deg"], data["values"][k]))
    print(f"Drawing {len(jobs)} {MODELS[model]['short']} charts ...")
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 2)) as pool:
        for n, _ in enumerate(pool.map(_draw_job, jobs, chunksize=4), 1):
            if n % 25 == 0:
                print(f"  {n}/{len(jobs)}")

    # small JSON of the numbers too (handy for checks / later use)
    with open(os.path.join(out_dir, "values.json"), "w") as f:
        json.dump({"columns_utc": [(init + timedelta(hours=h)).strftime("%Y-%m-%dT%HZ")
                                   for h in columns()],
                   "rows": ((["c00"] if data["values"].shape[1] == 51 else [])
                            + [f"{m:02d}" for m in range(1, 51)]) if model == "ecmwf_ens"
                   else WN3_STATS,
                   "districts": {d["id"]: np.round(data["values"][k], 1).tolist()
                                 for k, d in enumerate(districts)}}, f, separators=(",", ":"))

    entry = {"run": run_name, "init": init.strftime("%Y-%m-%dT%H:%MZ"),
             "grid_deg": data["grid_deg"],
             "made": datetime.now(IST).strftime("%Y-%m-%dT%H:%M+05:30")}
    entry.update(entry_extra or {})
    return entry


# ============================================================
# MAIN
# ============================================================

def find_new(m, models, force_model=None, force_run=None):
    """-> list of (model, run_name, source/None)."""
    todo = []
    if force_run:
        if force_model == "ecmwf_ens":
            t = datetime.strptime(force_run, "%Y%m%d_%H").replace(tzinfo=timezone.utc)
            src = ens_find_source(t)
            if not src:
                raise SystemExit(f"ECMWF ENS {force_run} is not (fully) published")
            todo.append(("ecmwf_ens", force_run, src))
        else:
            todo.append(("wn3", force_run, None))
        return todo

    if "ecmwf_ens" in models:
        try:
            runs = ens_latest_runs(1)
            if runs and runs[0][0] not in known_runs(m, "ecmwf_ens"):
                todo.append(("ecmwf_ens", runs[0][0], runs[0][1]))
            print(f"ECMWF ENS newest complete run: {runs[0][0] if runs else 'none'}")
        except Exception as exc:
            print(f"ECMWF ENS check failed: {type(exc).__name__}: {exc}")

    if "wn3" in models:
        try:
            s = wn3_session()
            known = known_runs(m, "wn3")
            newest = newest_known_init(m, "wn3")
            for name in wn3_list_runs(s)[:6]:
                mm = re.match(WN3_RUN_PATTERN, name)
                init = datetime.strptime(mm.group(1) + mm.group(2), "%Y%m%d%H")
                if name in known or init.strftime("%Y-%m-%dT%H:%MZ") <= newest:
                    break
                # completeness: is the last hour of the mean rain written?
                try:
                    ok = wn3_complete(wn3_open(name))
                except Exception as exc:
                    print(f"  WN3 {name}: {type(exc).__name__}: {exc}")
                    ok = False
                if ok:
                    todo.append(("wn3", name, None))
                    break
                print(f"  WN3 {name}: not complete yet")
            print(f"WeatherNext 3 new run: {[t[1] for t in todo if t[0] == 'wn3'] or 'none'}")
        except Exception as exc:
            print(f"WeatherNext 3 check failed: {type(exc).__name__}: {exc}")
    return todo


def demo_data(model, districts):
    rng = np.random.default_rng(1)
    ncol = len(columns())
    if model == "ecmwf_ens":
        base = np.clip(rng.gamma(0.6, 8, (len(districts), 1, ncol)), 0, None)
        vals = base * rng.gamma(1.2, 1.0, (len(districts), 51, ncol))
        vals[:, :, 20:30] *= 6
    else:
        mean = np.clip(rng.gamma(0.6, 8, (len(districts), ncol)), 0, None)
        vals = np.stack([mean * f for f in (3.0, 1.8, 0.8, 0.3, 0.05, 1.0)], axis=1)
    pts = [(round(d["lat"] * 4) / 4, round(d["lon"] * 4) / 4) for d in districts]
    return {"values": vals.astype(np.float32), "points": pts,
            "grid_deg": 0.25 if model == "ecmwf_ens" else 0.1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ecmwf_ens,wn3")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--run", default="")
    ap.add_argument("--check", action="store_true", help="only report whether a new run exists")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--only", default="", help="comma list of district ids (testing)")
    a = ap.parse_args()

    districts = load_districts()
    if a.only:
        keep = set(a.only.split(","))
        districts = [d for d in districts if d["id"] in keep]
    models = [x.strip() for x in a.models.split(",") if x.strip()]
    m = read_manifest()

    if a.demo:
        init = datetime(2026, 10, 25, 0, tzinfo=timezone.utc)
        for model in models:
            run = "20261025_00" if model == "ecmwf_ens" else "20261025_00hr_01_preds"
            add_run(m, model, render_run(model, run, init, districts, demo_data(model, districts)))
        write_manifest(m, districts)
        return

    todo = find_new(m, models, a.model, a.run)
    if a.check:
        new = "true" if todo else "false"
        print(f"new={new}")
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"new={new}\n")
        return

    failed, done = [], 0
    for model, run, src in todo:
        try:
            if model == "ecmwf_ens":
                init = datetime.strptime(run, "%Y%m%d_%H").replace(tzinfo=timezone.utc)
                data = load_ecmwf_ens(run, src, districts)
                extra = {"resol": data["resol"]}
            else:
                mm = re.match(WN3_RUN_PATTERN, run)
                init = datetime.strptime(mm.group(1) + mm.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)
                root = wn3_open(run)
                if not wn3_complete(root):
                    print(f"WN3 {run} not complete yet - skipped")
                    continue
                data = load_wn3(run, districts, root)
                extra = {}
            add_run(m, model, render_run(model, run, init, districts, data, extra))
            write_manifest(m, districts)
            done += 1
            print(f"{MODELS[model]['short']} {run}: done")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failed.append(f"{model} {run}: {type(exc).__name__}: {exc}")
    if not todo:
        print("No new runs.")
    if failed:
        print("FAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
