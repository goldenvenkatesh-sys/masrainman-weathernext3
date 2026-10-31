"""
Cyclone tracks from gridded model fields, for models that publish no track files of their own:
  GFS   : GEFS 31 members, 0.5° GRIB2 on the NOAA open-data bucket (noaa-gefs-pds, AWS)
  UKMET : Met Office global deterministic 10 km NetCDF on the Met Office AWS bucket
          (the MOGREPS-G ensemble is not published openly, so this is one track)
  ICON  : ICON-EPS global 40 members (icosahedral GRIB2 on opendata.dwd.de, regridded to 0.25°)
  GEM   : GEPS 21 members, 0.5° GRIB2 on dd.weather.gc.ca (MSC Datamart)
The ICON and GEM files are found from the servers' directory listings, so small changes in
file naming don't break them.

Each system is seeded from ECMWF ENS's starting position for the same run (same ids as the ECMWF maps,
e.g. 71B). Every 6 h (12 h where the model only has 12-hourly output) the tracker takes the lowest MSLP
within 2° per 6 h of the expected position (last position + last motion) and keeps it if it is a closed low: the mean MSLP on the
2.5–3.5° ring around it is at least 0.5 hPa higher (weak pre-genesis lows included). Two misses in a row end the track; so does
reaching land (only the first landfall is needed). Wind = max 10-m wind within 2° of the centre.

tracks(model, run, seeds) returns the same table as landfall.read_ecmwf_tf:
storm, member, step, lat, lon, mslp (hPa), wind (m/s).
"""
import io
import re
import bz2
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# region read from the global fields (North Indian Ocean with some margin)
LAT0, LAT1, LON0, LON1 = -5.0, 32.0, 45.0, 110.0
SEARCH_DEG = 2.0
SEED_SEARCH_DEG = 3.0
RING = (2.5, 3.5)
MIN_DEPTH_HPA = 0.5
WIND_RADIUS_DEG = 2.0
STEP_H = 6

GEFS = "https://noaa-gefs-pds.s3.amazonaws.com/gefs.{d}/{h}/atmos/pgrb2ap5/{m}.t{h}z.pgrb2a.0p50.f{s:03d}"
GEFS_MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
GEFS_LAST = 384

UKMO = ("https://met-office-atmospheric-model-data.s3.eu-west-2.amazonaws.com/global-deterministic-10km/"
        "{r}/{v}-PT{s:04d}H00M-{f}.nc")
UKMO_LAST = 168          # 00Z and 12Z runs (06Z / 18Z stop at ~66 h)

ICON_EPS = "https://opendata.dwd.de/weather/nwp/icon-eps/grib/{h}/"
ICON_GRID = "https://opendata.dwd.de/weather/lib/cdo/icon_grid_0024_R02B06_G.nc.bz2"   # ICON-EPS global grid
ICON_MEMBERS = 40

GEPS_DIRS = ["https://dd.weather.gc.ca/{d}/WXO-DD/ensemble/geps/grib2/raw/{h}/{s:03d}/",
             "https://dd.weather.gc.ca/ensemble/geps/grib2/raw/{h}/{s:03d}/"]
GEPS_MEMBERS = 21
GEPS_LAST = 384

_session = requests.Session()


_down = {}       # host -> failed requests in a row; a host that keeps failing is skipped for this run


def _get(url, rng=None, tries=4):
    host = url.split("/")[2]
    for i in range(tries):
        if _down.get(host, 0) >= 6:
            return None
        try:
            r = _session.get(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}, timeout=120)
            if r.status_code in (200, 206):
                _down[host] = 0
                return r.content
            if r.status_code == 404:
                _down[host] = 0
                return None
        except requests.RequestException:
            pass
        _down[host] = _down.get(host, 0) + 1
        time.sleep(2 ** i)      # busy bucket or dropped connection: back off and retry
    return None


# ------------------------------------------------------------ GEFS

def _grib_fields(data):
    import eccodes
    out = {}
    pos = 0
    while pos < len(data):
        h = eccodes.codes_new_from_message(data[pos:])
        if h is None:
            break
        try:
            n = eccodes.codes_get(h, "totalLength")
            name = eccodes.codes_get(h, "shortName")
            ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
            vals = eccodes.codes_get_values(h).reshape(nj, ni)
            la0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
            la1 = eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees")
            lo0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
            dlo = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        finally:
            eccodes.codes_release(h)
        out[name] = (vals, np.linspace(la0, la1, nj), lo0 + dlo * np.arange(ni))
        pos += n
    return out


def gefs_field(run, member, step):
    """(lats, lons, mslp hPa, wind m/s) over the region, or None if not online."""
    d, h = run.split("_")
    url = GEFS.format(d=d, h=h, m=member, s=step)
    idx = _get(url + ".idx")
    if not idx:
        return None
    lines = idx.decode().strip().splitlines()
    starts = [int(l.split(":")[1]) for l in lines]
    want = {}
    for i, l in enumerate(lines):
        for key in (":PRMSL:mean sea level:", ":UGRD:10 m above ground:", ":VGRD:10 m above ground:"):
            if key in l:
                want[key] = (starts[i], starts[i + 1] - 1 if i + 1 < len(starts) else "")
    if len(want) < 3:
        return None
    f = {}
    for rng in want.values():
        data = _get(url, rng)
        if not data:
            return None
        f.update(_grib_fields(data))
    p, lats, lons = f["prmsl"]
    u, v = f["10u"][0], f["10v"][0]
    return _cut(lats, lons, p / 100.0, np.hypot(u, v))


# ------------------------------------------------------------ UKMET

def _nc_var(data, names):
    import h5py
    with h5py.File(io.BytesIO(data), "r") as ds:
        var = next(k for k in names if k in ds)
        a = ds[var][()].astype(float)
        if "scale_factor" in ds[var].attrs:
            a = a * ds[var].attrs["scale_factor"] + ds[var].attrs.get("add_offset", 0)
        lat = ds["latitude"][()]
        lon = ds["longitude"][()]
    return a, lat, lon


def ukmo_field(run, member, step):
    d, h = run.split("_")
    r = f"{d}T{h}00Z"
    v = (datetime.strptime(run, "%Y%m%d_%H") + timedelta(hours=step)).strftime("%Y%m%dT%H00Z")
    pm = _get(UKMO.format(r=r, v=v, s=step, f="pressure_at_mean_sea_level"))
    wm = _get(UKMO.format(r=r, v=v, s=step, f="wind_speed_at_10m"))
    if not pm or not wm:
        return None
    p, lats, lons = _nc_var(pm, ["air_pressure_at_sea_level"])
    w, _, _ = _nc_var(wm, ["wind_speed"])
    return _cut(lats, lons, p / 100.0, w)


# ------------------------------------------------------------ directory listings (DWD, MSC)

def _listing(url):
    """File / directory names linked from an HTML directory listing ([] if not there)."""
    data = _get(url)
    if not data:
        return []
    names = re.findall(r'href="([^"?#]+)"', data.decode("utf-8", "replace"))
    names = [n.rstrip("/").split("/")[-1] for n in names if not n.startswith("..")]
    return [n for n in names if n]


def _pick(names, *patterns):
    """First name matching all the regex patterns (case-insensitive)."""
    for n in names:
        if all(re.search(p, n, re.I) for p in patterns):
            return n
    return None


def _grib_members(data, n_members):
    """All messages of a GRIB file -> {member index: (values 2D or 1D, lats, lons)} (lats/lons None
    for unstructured grids). Control = 0, perturbed members by their perturbationNumber."""
    import eccodes
    out = {}
    pos = 0
    while pos < len(data):
        h = eccodes.codes_new_from_message(data[pos:])
        if h is None:
            break
        try:
            n = eccodes.codes_get(h, "totalLength")
            try:
                m = int(eccodes.codes_get(h, "perturbationNumber"))
            except Exception:
                m = 0
            vals = eccodes.codes_get_values(h)
            lats = lons = None
            if eccodes.codes_get(h, "gridType") == "regular_ll":
                ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
                vals = vals.reshape(nj, ni)
                lats = np.linspace(eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"),
                                   eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees"), nj)
                lons = (eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
                        + eccodes.codes_get(h, "iDirectionIncrementInDegrees") * np.arange(ni))
        finally:
            eccodes.codes_release(h)
        if m < n_members:
            out[m] = (vals, lats, lons)
        pos += n
    return out


def _bz2(data):
    return bz2.decompress(data) if data and data[:3] == b"BZh" else data


# ------------------------------------------------------------ ICON-EPS

_icon_grid = None
_icon_lock = threading.Lock()


def icon_grid():
    """Cell-centre lat/lon (degrees) of the ICON-EPS global grid and a regridder to 0.25° over the region."""
    global _icon_grid
    with _icon_lock:
        if _icon_grid is None:
            import h5py
            data = _bz2(_get(ICON_GRID))
            if not data:
                raise RuntimeError("ICON-EPS grid file not found: " + ICON_GRID)
            with h5py.File(io.BytesIO(data), "r") as ds:
                clat = np.degrees(ds["clat"][()])
                clon = np.degrees(ds["clon"][()])
            _icon_grid = _icosa_regridder(clat, clon)
    return _icon_grid


def _icosa_regridder(clat, clon, res=0.25):
    """Nearest-cell lookup from an icosahedral grid to a regular grid over the region."""
    from scipy.spatial import cKDTree
    clon = np.where(clon > 180, clon - 360, clon)
    sel = np.where((clat >= LAT0 - 1) & (clat <= LAT1 + 1) & (clon >= LON0 - 1) & (clon <= LON1 + 1))[0]
    lats = np.arange(LAT0, LAT1 + res / 2, res)
    lons = np.arange(LON0, LON1 + res / 2, res)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    tree = cKDTree(np.c_[clat[sel], clon[sel] * np.cos(np.radians(clat[sel]))])
    _, k = tree.query(np.c_[LA.ravel(), LO.ravel() * np.cos(np.radians(LA.ravel()))])
    return lats, lons, sel[k].reshape(LA.shape)


def _icon_file(run, var, step):
    d, h = run.split("_")
    folder = ICON_EPS.format(h=h)
    sub = _pick(_listing(folder), rf"^{var}$")
    if not sub:
        return None
    name = _pick(_listing(folder + sub + "/"), rf"_{d}{h}_", rf"_{step:03d}_", r"single-level")
    return _bz2(_get(folder + sub + "/" + name)) if name else None


def icon_step(run, step):
    """{member: (lats, lons, mslp hPa, wind m/s)} for one ICON-EPS step, or {} if not online."""
    files = {v: _icon_file(run, v, step) for v in ("pmsl", "u_10m", "v_10m")}
    if not all(files.values()):
        return {}
    lats, lons, idx = icon_grid()
    p, u, v = (_grib_members(files[k], ICON_MEMBERS + 1) for k in ("pmsl", "u_10m", "v_10m"))
    out = {}
    for m in p:
        if m in u and m in v:
            out[m] = (lats, lons, p[m][0][idx] / 100.0, np.hypot(u[m][0][idx], v[m][0][idx]))
    return out


def icon_steps(run):
    h = int(run[-2:])
    last = 180 if h in (0, 12) else 120
    return [s for s in list(range(0, 121, 6)) + list(range(132, 181, 12)) if s <= last]


def icon_complete(run):
    d, h = run.split("_")
    folder = ICON_EPS.format(h=h)
    sub = _pick(_listing(folder), r"^pmsl$")
    names = _listing(folder + sub + "/") if sub else []
    return bool(_pick(names, rf"_{d}{h}_", rf"_{icon_steps(run)[-1]:03d}_"))


# ------------------------------------------------------------ GEM GEPS

def _geps_folder(run, step):
    d, h = run.split("_")
    for tpl in GEPS_DIRS:
        folder = tpl.format(d=d, h=h, s=step)
        names = _listing(folder)
        if _pick(names, r"PRMSL", rf"_{d}{h}_"):
            return folder, names
    return None, []


def geps_step(run, step):
    d, h = run.split("_")
    folder, names = _geps_folder(run, step)
    if not folder:
        return {}
    want = {"p": _pick(names, r"_PRMSL_", rf"_{d}{h}_", r"allmbrs"),
            "u": _pick(names, r"_UGRD_", r"_TGL_10", rf"_{d}{h}_", r"allmbrs"),
            "v": _pick(names, r"_VGRD_", r"_TGL_10", rf"_{d}{h}_", r"allmbrs")}
    if not all(want.values()):
        return {}
    f = {k: _grib_members(_get(folder + n) or b"", GEPS_MEMBERS) for k, n in want.items()}
    out = {}
    for m, (p, lats, lons) in f["p"].items():
        if m in f["u"] and m in f["v"]:
            out[m] = _cut(lats, lons, p / 100.0, np.hypot(f["u"][m][0], f["v"][m][0]))
    return out


def geps_complete(run):
    return _geps_folder(run, GEPS_LAST)[0] is not None


# ------------------------------------------------------------ tracking

def _cut(lats, lons, p, w):
    lons = np.where(lons > 180, lons - 360, lons)
    if lats[0] > lats[-1]:
        lats, p, w = lats[::-1], p[::-1], w[::-1]
    iy = (lats >= LAT0) & (lats <= LAT1)
    ix = (lons >= LON0) & (lons <= LON1)
    return lats[iy], lons[ix], p[np.ix_(iy, ix)], w[np.ix_(iy, ix)]


def find_centre(field, guess, radius):
    lats, lons, p, w = field
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    d = np.hypot(LA - guess[0], (LO - guess[1]) * np.cos(np.radians(guess[0])))
    inside = d <= radius
    if not inside.any():
        return None
    k = np.nanargmin(np.where(inside, p, np.inf))
    j, i = np.unravel_index(k, p.shape)
    la, lo, pmin = lats[j], lons[i], p[j, i]
    dc = np.hypot(LA - la, (LO - lo) * np.cos(np.radians(la)))
    ring = (dc >= RING[0]) & (dc <= RING[1])
    if not ring.any() or np.nanmean(p[ring]) - pmin < MIN_DEPTH_HPA:
        return None
    wind = np.nanmax(w[dc <= WIND_RADIUS_DEG])
    return float(la), float(lo), float(pmin), float(wind)


def _per_member(field_fn, members, workers=24):
    """Turn a one-member reader into a one-step reader for all members."""
    def step_fn(run, step):
        with ThreadPoolExecutor(workers) as ex:
            got = ex.map(lambda m: field_fn(run, m, step), members)
        return {i: f for i, f in enumerate(got) if f is not None}
    return step_fn


def model_setup(model, run):
    """(read one step -> {member index: field}, steps, number of members)"""
    if model == "gfs":
        return _per_member(gefs_field, GEFS_MEMBERS), list(range(0, GEFS_LAST + 1, STEP_H)), len(GEFS_MEMBERS)
    if model == "ukmet":
        return _per_member(ukmo_field, ["det"], 1), list(range(0, UKMO_LAST + 1, STEP_H)), 1
    if model == "icon":
        return icon_step, icon_steps(run), ICON_MEMBERS + 1
    if model == "gem":
        return geps_step, list(range(0, GEPS_LAST + 1, STEP_H)), GEPS_MEMBERS
    raise ValueError(model)


def tracks(model, run, seeds, on_land):
    """seeds: {storm: (lat, lon)} at T+0. on_land(lat, lon) -> bool."""
    step_fn, steps, n_members = model_setup(model, run)
    # state per (storm, member): [pos, motion per 6 h, misses, done]
    state = {(s, m): [seed, (0.0, 0.0), 0, False] for s, seed in seeds.items() for m in range(n_members)}
    rows = []
    prev_step = 0
    for step in steps:
        if all(st[3] for st in state.values()):
            break
        fields = step_fn(run, step)
        if not fields:
            if step == 0:
                break
            continue          # a missing step: carry on with the next one
        k = (step - prev_step) / STEP_H if step else 0.0
        for (s, m), st in state.items():
            if st[3] or fields.get(m) is None:
                continue
            pos, mot, misses, _ = st
            guess = (pos[0] + mot[0] * k, pos[1] + mot[1] * k)
            c = find_centre(fields[m], guess, SEED_SEARCH_DEG if step == 0 else SEARCH_DEG * max(1.0, k))
            if c is None:
                st[2] += 1
                st[0] = guess
                st[3] = st[2] >= 2 or step == 0
                continue
            la, lo, pm, wd = c
            rows.append((s, m, step, la, lo, pm, wd))
            st[1] = ((la - pos[0]) / k, (lo - pos[1]) / k) if step else (0.0, 0.0)
            st[0], st[2] = (la, lo), 0
            st[3] = on_land(la, lo) and step > 0
        prev_step = step
        print(f"    {model} {run} T+{step}: {sum(not st[3] for st in state.values())} tracks running")
    return pd.DataFrame(rows, columns=["storm", "member", "step", "lat", "lon", "mslp", "wind"])


def _exists(url):
    host = url.split("/")[2]
    for i in range(3):
        if _down.get(host, 0) >= 6:
            return False
        try:
            ok = _session.head(url, timeout=30).status_code == 200
            _down[host] = 0
            return ok
        except requests.RequestException:
            _down[host] = _down.get(host, 0) + 1
            time.sleep(2 ** i)
    return False


def gefs_complete(run):
    d, h = run.split("_")
    return _exists(GEFS.format(d=d, h=h, m="gep30", s=GEFS_LAST) + ".idx")


def ukmo_complete(run):
    d, h = run.split("_")
    v = (datetime.strptime(run, "%Y%m%d_%H") + timedelta(hours=UKMO_LAST)).strftime("%Y%m%dT%H00Z")
    url = UKMO.format(r=f"{d}T{h}00Z", v=v, s=UKMO_LAST, f="wind_speed_at_10m")
    return _exists(url)
