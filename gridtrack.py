"""
Cyclone tracks from gridded model fields, for models that publish no track files of their own:
  GFS   : GEFS 31 members, 0.5° GRIB2 on the NOAA open-data bucket (noaa-gefs-pds, AWS)
  UKMET : Met Office global deterministic 10 km NetCDF on the Met Office AWS bucket
          (the MOGREPS-G ensemble is not published openly, so this is one track)

Each system is seeded from ECMWF ENS's starting position for the same run (same ids as the ECMWF maps,
e.g. 71B). Every 6 h the tracker takes the lowest MSLP within 2° of the expected position
(last position + last 6-h motion) and keeps it if it is a closed low: the mean MSLP on the
2.5–3.5° ring around it is at least 0.5 hPa higher (weak pre-genesis lows included). Two misses in a row end the track; so does
reaching land (only the first landfall is needed). Wind = max 10-m wind within 2° of the centre.

tracks(model, run, seeds) returns the same table as landfall.read_ecmwf_tf:
storm, member, step, lat, lon, mslp (hPa), wind (m/s).
"""
import io
import time
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

_session = requests.Session()


def _get(url, rng=None, tries=5):
    for i in range(tries):
        try:
            r = _session.get(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}, timeout=120)
            if r.status_code in (200, 206):
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
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


def tracks(model, run, seeds, on_land, workers=24):
    """seeds: {storm: (lat, lon)} at T+0. on_land(lat, lon) -> bool."""
    field_fn, members, last = {
        "gfs": (gefs_field, GEFS_MEMBERS, GEFS_LAST),
        "ukmet": (ukmo_field, ["det"], UKMO_LAST),
    }[model]
    # state per (storm, member): [pos, motion, misses, done]
    state = {(s, m): [seed, (0.0, 0.0), 0, False] for s, seed in seeds.items() for m in members}
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for step in range(0, last + 1, STEP_H):
            need = sorted({m for (s, m), st in state.items() if not st[3]})
            if not need:
                break
            fields = dict(zip(need, ex.map(lambda m: field_fn(run, m, step), need)))
            if all(f is None for f in fields.values()):
                if step == 0:
                    return pd.DataFrame(columns=["storm", "member", "step", "lat", "lon", "mslp", "wind"])
                break
            for (s, m), st in state.items():
                if st[3] or fields.get(m) is None:
                    continue
                pos, mot, misses, _ = st
                guess = (pos[0] + mot[0], pos[1] + mot[1])
                c = find_centre(fields[m], guess, SEED_SEARCH_DEG if step == 0 else SEARCH_DEG)
                if c is None:
                    st[2] += 1
                    st[0] = guess
                    st[3] = st[2] >= 2 or step == 0
                    continue
                la, lo, pm, wd = c
                rows.append((s, members.index(m), step, la, lo, pm, wd))
                st[1] = (la - pos[0], lo - pos[1]) if step else (0.0, 0.0)
                st[0], st[2] = (la, lo), 0
                st[3] = on_land(la, lo) and step > 0
            print(f"    {model} {run} T+{step}: {sum(not st[3] for st in state.values())} tracks running")
    return pd.DataFrame(rows, columns=["storm", "member", "step", "lat", "lon", "mslp", "wind"])


def _exists(url):
    for i in range(3):
        try:
            return _session.head(url, timeout=30).status_code == 200
        except requests.RequestException:
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
