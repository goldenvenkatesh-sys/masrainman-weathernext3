# ============================================================
# FIELDS FOR TRACKING — MSLP (+10-m wind for deterministic models)
# ============================================================
#
# Each fetcher returns {"members": {member: {"msl": [n_steps, ny, nx] hPa,
#                                            "u10"/"v10": m/s (deterministic only)}}}
# on the 0.25 deg tracking grid for C.TRACK_STEPS. Missing steps are NaN.

import os
import time
import logging

import numpy as np

from . import config as C
from .grib import messages, regrid, TRACK
from .fetch import NotAvailable, _cache, _download, _icon_file, icon_regridder, icon_prefetch

LOG = logging.getLogger("mme")
STEPS = C.TRACK_STEPS
NAN = lambda: np.full((len(STEPS),) + TRACK.LAT2.shape, np.nan, np.float32)

ECMWF_PARAM = {"msl": "msl", "10u": "u10", "10v": "v10"}


def _put(store, member, var, step, field):
    if step not in STEPS:
        return
    d = store.setdefault(member, {})
    if var not in d:
        d[var] = NAN()
    d[var][STEPS.index(step)] = field


# ------------------------------------------------------------
# ECMWF: IFS HRES, AIFS Single (msl + winds), IFS ENS, AIFS ENS (msl)
# ------------------------------------------------------------

def track_ecmwf(model, init):
    from ecmwf.opendata import Client
    eng = {"ecmwf_hres": "ifs", "ecmwf_ens": "ifs", "aifs_single": "aifs-single", "aifs_ens": "aifs-ens"}[model]
    ens = model in ("ecmwf_ens", "aifs_ens")
    req = dict(date=init.strftime("%Y%m%d"), time=init.hour, step=STEPS,
               stream="enfo" if ens else "oper", type="pf" if ens else "fc",
               param=["msl"] if ens else ["msl", "10u", "10v"])
    if ens:
        req["number"] = list(range(1, C.TRACK_MEMBERS[model] + 1))
    path = _cache(init, model, f"track_{len(req.get('number', [0]))}.grib2")
    if not os.path.exists(path):
        try:
            Client(source=os.environ.get("ECMWF_SOURCE", "ecmwf"), model=eng, resol="0p25") \
                .retrieve(target=path + ".part", **req)
        except Exception as exc:
            raise NotAvailable(f"{model}: {exc}")
        os.replace(path + ".part", path)
    store = {}
    for m in messages(path):
        var = ECMWF_PARAM.get(m["shortName"])
        if var:
            v = regrid(m, TRACK)
            _put(store, m["number"], var, m["end"], v / 100.0 if var == "msl" else v)
    return store


def ecmwf_sst(init):
    """Skin temperature over the sea (deg C) from IFS HRES at T+0."""
    from ecmwf.opendata import Client
    path = _cache(init, "ecmwf_hres", "sst.grib2")
    if not os.path.exists(path):
        Client(source=os.environ.get("ECMWF_SOURCE", "ecmwf"), model="ifs", resol="0p25").retrieve(
            date=init.strftime("%Y%m%d"), time=init.hour, stream="oper", type="fc", step=0,
            param=["skt", "lsm"], target=path + ".part")
        os.replace(path + ".part", path)
    f = {m["shortName"]: regrid(m, TRACK) for m in messages(path)}
    sst = f["skt"] - 273.15
    sst[f["lsm"] > 0.3] = np.nan
    return sst


# ------------------------------------------------------------
# NCEP GFS (msl + winds) and GEFS members (msl) via NOMADS filter
# ------------------------------------------------------------

NOMADS = "https://nomads.ncep.noaa.gov/cgi-bin/"
SUB = (f"&subregion=&toplat={C.TRACK_LAT[1] + 1:.0f}&bottomlat={C.TRACK_LAT[0] - 1:.0f}"
       f"&leftlon={C.TRACK_LON[0] - 1:.0f}&rightlon={C.TRACK_LON[1] + 1:.0f}")
NCEP_VAR = {"prmsl": "msl", "10u": "u10", "10v": "v10"}


def _nomads(url, path):
    for attempt in range(4):
        try:
            return _download(url, path)
        except NotAvailable:
            raise
        except Exception:
            time.sleep(5 * (attempt + 1))  # NOMADS throttles bursts
    return _download(url, path)


def track_gfs(init):
    ymd, hh = init.strftime("%Y%m%d"), init.strftime("%H")
    store = {}
    for s in STEPS:
        url = (f"{NOMADS}filter_gfs_0p25.pl?dir=%2Fgfs.{ymd}%2F{hh}%2Fatmos&file=gfs.t{hh}z.pgrb2.0p25.f{s:03d}"
               f"&var_PRMSL=on&var_UGRD=on&var_VGRD=on&lev_mean_sea_level=on&lev_10_m_above_ground=on{SUB}")
        for m in messages(_nomads(url, _cache(init, "gfs", f"trk_f{s:03d}.grib2"))):
            var = NCEP_VAR.get(m["shortName"])
            if var:
                v = regrid(m, TRACK)
                _put(store, 0, var, s, v / 100.0 if var == "msl" else v)
    return store


def track_gefs(init):
    ymd, hh = init.strftime("%Y%m%d"), init.strftime("%H")
    store = {}
    for n in range(1, C.TRACK_MEMBERS["gefs"] + 1):
        for s in STEPS:
            url = (f"{NOMADS}filter_gefs_atmos_0p50a.pl?dir=%2Fgefs.{ymd}%2F{hh}%2Fatmos%2Fpgrb2ap5"
                   f"&file=gep{n:02d}.t{hh}z.pgrb2a.0p50.f{s:03d}&var_PRMSL=on&lev_mean_sea_level=on{SUB}")
            try:
                path = _nomads(url, _cache(init, "gefs", f"trk_p{n:02d}_f{s:03d}.grib2"))
            except NotAvailable:
                continue
            for m in messages(path):
                _put(store, n, "msl", s, regrid(m, TRACK) / 100.0)
            time.sleep(0.4)
    if not store:
        raise NotAvailable("GEFS members")
    return store


# ------------------------------------------------------------
# CMC GEPS (all members in one file per step)
# ------------------------------------------------------------

def track_geps(init):
    ymd, hh, ymdh = init.strftime("%Y%m%d"), init.strftime("%H"), init.strftime("%Y%m%d%H")
    store = {}
    for s in STEPS:
        url = (f"https://dd.weather.gc.ca/{ymd}/WXO-DD/ensemble/geps/grib2/raw/{hh}/{s:03d}/"
               f"CMC_geps-raw_PRMSL_MSL_0_latlon0p5x0p5_{ymdh}_P{s:03d}_allmbrs.grib2")
        for m in messages(_download(url, _cache(init, "geps", f"PRMSL_P{s:03d}.grib2"))):
            if m["number"] <= C.TRACK_MEMBERS["geps"]:
                _put(store, m["number"], "msl", s, regrid(m, TRACK) / 100.0)
    return store


# ------------------------------------------------------------
# DWD ICON (msl + winds, icosahedral)
# ------------------------------------------------------------

def track_icon(init):
    last = 180 if init.hour in (0, 12) else 120
    rg = icon_regridder(init, TRACK)
    icon_prefetch(init, [(name, s) for s in STEPS if s <= last for name in ("pmsl", "u_10m", "v_10m")])
    store = {}
    for s in STEPS:
        if s > last:
            continue
        for var, name in (("msl", "pmsl"), ("u10", "u_10m"), ("v10", "v_10m")):
            v = rg(next(messages(_icon_file(init, name, s)))["values"])
            _put(store, 0, var, s, v / 100.0 if var == "msl" else v)
    return store


def track_wn3(init):
    if not (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("WN3_ENABLED")):
        raise NotAvailable("WeatherNext 3: no Google credentials in this environment")
    from .wn3 import wn3_track_fields
    return wn3_track_fields(init)


TRACK_FETCHERS = {
    "ecmwf_hres": lambda i: track_ecmwf("ecmwf_hres", i),
    "aifs_single": lambda i: track_ecmwf("aifs_single", i),
    "ecmwf_ens": lambda i: track_ecmwf("ecmwf_ens", i),
    "aifs_ens": lambda i: track_ecmwf("aifs_ens", i),
    "gfs": track_gfs, "gefs": track_gefs, "geps": track_geps, "icon": track_icon, "wn3": track_wn3,
}


def load_track_group(group, init):
    """{model: {member: fields}} for the models of a group that are available."""
    out = {}
    for model in C.GROUPS[group]["models"]:
        try:
            out[model] = TRACK_FETCHERS[model](init)
            LOG.info("%s: %d member(s) for tracking", model, len(out[model]))
        except NotAvailable as exc:
            LOG.warning("%s not available for tracking (%s)", model, exc)
        except Exception as exc:
            LOG.exception("%s tracking fetch failed: %s", model, exc)
    return out
