# ============================================================
# RAINFALL FETCHERS — one per model
# ============================================================
#
# Every fetcher answers the same question:
#     accumulated precipitation (mm) since `init`, on the common grid,
#     at each requested lead hour.
# Leads that fall between two output steps are linearly interpolated
# (accumulations grow smoothly, so this is a safe approximation).
#
# Daily totals are then differences between consecutive leads.

import bz2
import os
import logging
from datetime import timedelta

import numpy as np
import requests

from . import config as C
from .grib import messages, regrid, IconRegridder, RAIN

LOG = logging.getLogger("mme")
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "MASRainman-MME/1.0"


class NotAvailable(Exception):
    pass


def _cache(init, model, name):
    d = os.path.join(C.CACHE_DIR, init.strftime("%Y%m%d%H"), model)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def _download(url, path, timeout=180):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    r = HTTP.get(url, timeout=timeout)
    if r.status_code == 404:
        raise NotAvailable(url)
    r.raise_for_status()
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, path)
    return path


def _brackets(leads, steps):
    """For each lead: (lower step, upper step, weight of upper)."""
    steps = sorted(steps)
    out = {}
    for L in leads:
        if L > steps[-1]:
            continue
        if L in steps:
            out[L] = (L, L, 0.0)
            continue
        lo = max(s for s in steps if s < L)
        hi = min(s for s in steps if s > L)
        out[L] = (lo, hi, (L - lo) / (hi - lo))
    return out


def _interp(leads, steps, acc_at):
    """acc_at(list_of_steps) -> {step: field}. Returns {lead: field} for leads within range."""
    br = _brackets(leads, steps)
    need = sorted({s for lo, hi, _ in br.values() for s in (lo, hi)})
    fields = acc_at(need)
    out = {}
    for L, (lo, hi, w) in br.items():
        if lo in fields and hi in fields:
            out[L] = (1 - w) * fields[lo] + w * fields[hi]
    return out


# ------------------------------------------------------------
# ECMWF open data: IFS HRES, IFS ENS, AIFS Single, AIFS ENS
# ------------------------------------------------------------

def _ecmwf_steps(model, init):
    if model in ("aifs_single", "aifs_ens"):
        return list(range(0, 361, 6))
    last = 360 if init.hour in (0, 12) else 144
    if model == "ecmwf_hres":
        last = 240 if init.hour in (0, 12) else 90
    return list(range(0, min(144, last) + 1, 3)) + list(range(150, last + 1, 6))


def _ecmwf_retrieve(model, init, steps, kind, members=None):
    from ecmwf.opendata import Client
    eng = "aifs-single" if model == "aifs_single" else "aifs-ens" if model == "aifs_ens" else "ifs"
    stream = "enfo" if model in ("ecmwf_ens", "aifs_ens") else "oper"
    path = _cache(init, model, f"tp_{kind}{len(members) if members else ''}_s{len(steps)}.grib2")
    if os.path.exists(path):
        return path
    client = Client(source=os.environ.get("ECMWF_SOURCE", "ecmwf"), model=eng, resol="0p25")
    req = dict(date=init.strftime("%Y%m%d"), time=init.hour, stream=stream, type=kind,
               step=steps, param="tp", target=path + ".part")
    if members:
        req["number"] = members
    try:
        client.retrieve(**req)
    except Exception as exc:
        raise NotAvailable(f"{model} {kind}: {exc}")
    os.replace(path + ".part", path)
    return path


def fetch_ecmwf(model, init, leads):
    steps = _ecmwf_steps(model, init)

    def acc_at(need):
        need = [s for s in need if s > 0]
        sums, counts = {}, {}
        kinds = [("fc", None)] if model in ("ecmwf_hres", "aifs_single") else \
                [("cf", None), ("pf", list(range(1, C.ENS_MEMBERS[model] + 1)))]
        for kind, members in kinds:
            try:
                path = _ecmwf_retrieve(model, init, need, kind, members)
            except NotAvailable:
                if kind == "cf":  # IFS ENS open data has no control run for tp: members only
                    LOG.info("%s: no control forecast, using perturbed members only", model)
                    continue
                raise
            for m in messages(path):
                v = regrid(m)
                if model in ("ecmwf_hres", "ecmwf_ens"):
                    v = v * 1000.0  # IFS tp is in metres
                sums[m["end"]] = sums.get(m["end"], 0) + v
                counts[m["end"]] = counts.get(m["end"], 0) + 1
        out = {s: sums[s] / counts[s] for s in sums}
        out[0] = np.zeros_like(next(iter(out.values())))
        return out

    return _interp(leads, steps, acc_at)


# ------------------------------------------------------------
# NCEP GFS (0.25) and GEFS ensemble mean (0.5) via NOMADS filter
# ------------------------------------------------------------

NOMADS = "https://nomads.ncep.noaa.gov/cgi-bin/"
SUB = f"&subregion=&toplat={C.GRID_LAT[1] + 1:.0f}&bottomlat={C.GRID_LAT[0] - 1:.0f}" \
      f"&leftlon={C.GRID_LON[0] - 1:.0f}&rightlon={C.GRID_LON[1] + 1:.0f}"


def fetch_gfs(init, leads):
    ymd, hh = init.strftime("%Y%m%d"), init.strftime("%H")
    steps = list(range(0, 121, 1)) + list(range(123, 385, 3))

    def acc_at(need):
        out = {0: None}
        for s in need:
            if s == 0:
                continue
            url = (f"{NOMADS}filter_gfs_0p25.pl?dir=%2Fgfs.{ymd}%2F{hh}%2Fatmos"
                   f"&file=gfs.t{hh}z.pgrb2.0p25.f{s:03d}&var_APCP=on{SUB}")
            path = _download(url, _cache(init, "gfs", f"f{s:03d}.grib2"))
            for m in messages(path):
                if m["start"] == 0:  # running total since the start
                    out[s] = regrid(m)
        out[0] = np.zeros_like(next(v for v in out.values() if v is not None))
        return out

    return _interp(leads, steps, acc_at)


def fetch_gefs(init, leads):
    """GEFS mean only has 6-h buckets, so add them up."""
    ymd, hh = init.strftime("%Y%m%d"), init.strftime("%H")
    steps = list(range(0, 385, 6))

    def acc_at(need):
        total, out = None, {}
        for s in range(6, max(need) + 1, 6):
            url = (f"{NOMADS}filter_gefs_atmos_0p50a.pl?dir=%2Fgefs.{ymd}%2F{hh}%2Fatmos%2Fpgrb2ap5"
                   f"&file=geavg.t{hh}z.pgrb2a.0p50.f{s:03d}&var_APCP=on{SUB}")
            path = _download(url, _cache(init, "gefs", f"f{s:03d}.grib2"))
            m = next(x for x in messages(path) if x["end"] - x["start"] == 6)
            v = regrid(m)
            total = v if total is None else total + v
            out[s] = total
        out[0] = np.zeros_like(total)
        return out

    return _interp(leads, steps, acc_at)


# ------------------------------------------------------------
# CMC GEPS (0.5, all members in one file per step)
# ------------------------------------------------------------

def fetch_geps(init, leads):
    ymd, hh, ymdh = init.strftime("%Y%m%d"), init.strftime("%H"), init.strftime("%Y%m%d%H")
    steps = list(range(0, 193, 3)) + list(range(198, 385, 6))

    def acc_at(need):
        out = {}
        for s in need:
            if s == 0:
                continue
            url = (f"https://dd.weather.gc.ca/{ymd}/WXO-DD/ensemble/geps/grib2/raw/{hh}/{s:03d}/"
                   f"CMC_geps-raw_APCP_SFC_0_latlon0p5x0p5_{ymdh}_P{s:03d}_allmbrs.grib2")
            path = _download(url, _cache(init, "geps", f"P{s:03d}.grib2"))
            fields = [regrid(m) for m in messages(path)]
            out[s] = np.mean(fields, axis=0)
        out[0] = np.zeros_like(next(iter(out.values())))
        return out

    return _interp(leads, steps, acc_at)


# ------------------------------------------------------------
# DWD ICON global (icosahedral, regridded here)
# ------------------------------------------------------------

_ICON_RG = {}
ICON_URL = "https://opendata.dwd.de/weather/nwp/icon/grib/"


def icon_prefetch(init, items, workers=8):
    """Download many ICON files at once — DWD is slow per file but fine in parallel.
    items: [(var, step)]. Missing files are skipped here and reported when read."""
    from concurrent.futures import ThreadPoolExecutor

    def one(item):
        try:
            _icon_file(init, *item)
        except Exception as exc:
            LOG.warning("ICON %s %s: %s", item[0], item[1], exc)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, items))


def icon_regridder(init, grid):
    if id(grid) not in _ICON_RG:
        clat = next(messages(_icon_file(init, "clat")))["values"]
        clon = next(messages(_icon_file(init, "clon")))["values"]
        _ICON_RG[id(grid)] = IconRegridder(clat, clon, grid)
    return _ICON_RG[id(grid)]


def _icon_file(init, var, step=None):
    hh, ymdh = init.strftime("%H"), init.strftime("%Y%m%d%H")
    if step is None:
        name = f"icon_global_icosahedral_time-invariant_{ymdh}_{var.upper()}.grib2.bz2"
    else:
        name = f"icon_global_icosahedral_single-level_{ymdh}_{step:03d}_{var.upper()}.grib2.bz2"
    path = _download(f"{ICON_URL}{hh}/{var}/{name}", _cache(init, "icon", name))
    out = path[:-4]
    if not os.path.exists(out):
        with open(out, "wb") as f:
            f.write(bz2.decompress(open(path, "rb").read()))
    return out


def fetch_icon(init, leads):
    last = 180 if init.hour in (0, 12) else 120
    steps = list(range(0, 79)) + list(range(81, last + 1, 3))
    rg = icon_regridder(init, RAIN)

    def acc_at(need):
        out = {}
        icon_prefetch(init, [("tot_prec", s) for s in need if s > 0])
        for s in need:
            if s == 0:
                continue
            out[s] = rg(next(messages(_icon_file(init, "tot_prec", s)))["values"])
        out[0] = np.zeros_like(next(iter(out.values())))
        return out

    return _interp(leads, steps, acc_at)


# ------------------------------------------------------------
# WeatherNext 3 (Google Cloud, same access as the weathernext3 project)
# ------------------------------------------------------------

def fetch_wn3(init, leads):
    if not (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("WN3_ENABLED")):
        raise NotAvailable("WeatherNext 3: no Google credentials in this environment")
    from .wn3 import wn3_accum
    return wn3_accum(init, leads)


FETCHERS = {
    "ecmwf_hres": lambda i, l: fetch_ecmwf("ecmwf_hres", i, l),
    "aifs_single": lambda i, l: fetch_ecmwf("aifs_single", i, l),
    "ecmwf_ens": lambda i, l: fetch_ecmwf("ecmwf_ens", i, l),
    "aifs_ens": lambda i, l: fetch_ecmwf("aifs_ens", i, l),
    "gfs": fetch_gfs,
    "gefs": fetch_gefs,
    "geps": fetch_geps,
    "icon": fetch_icon,
    "wn3": fetch_wn3,
}


def day_leads(init):
    """Lead hours of the 03 UTC day boundaries: [b0, b1, ... b15]."""
    first = (C.DAY_END_UTC_HOUR - init.hour) % 24
    if first == 0 and init.hour != C.DAY_END_UTC_HOUR:
        first = 24
    return [first + 24 * k for k in range(C.N_DAYS + 1)]


def daily_totals(model, init):
    """(n_days, ny, nx) array of 24-h totals; NaN where the model does not reach.

    Falls back to the model's previous cycle (12 h older) when the requested one is missing.
    """
    for back in (0, 12):
        run = init - timedelta(hours=back)
        leads = [L + back for L in day_leads(init)]
        try:
            acc = FETCHERS[model](run, leads)
        except NotAvailable as exc:
            LOG.warning("%s %s not available (%s)", model, run.strftime("%Y%m%d%HZ"), exc)
            continue
        if len(acc) < 2:
            continue
        shape = next(iter(acc.values())).shape
        days = np.full((C.N_DAYS,) + shape, np.nan, np.float32)
        for k in range(C.N_DAYS):
            a, b = leads[k], leads[k + 1]
            if a in acc and b in acc:
                days[k] = np.clip(acc[b] - acc[a], 0, None)
        LOG.info("%s: run %s, %d days", model, run.strftime("%Y%m%d%HZ"),
                 int(np.isfinite(days[:, 0, 0]).sum()))
        return days, run
    return None, None
