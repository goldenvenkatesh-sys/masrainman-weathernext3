"""
MasRainman NEM & Cyclone Outlook — data sources.

Everything the outlook (nem_outlook.py) computes comes from these open data sets:

  ECMWF ENS (51 members)   cyclone tracks (…-enfo-tf.bufr) and total precipitation (…-enfo-ef.grib2)
                           data.ecmwf.int, mirror/archive on AWS S3 (ecmwf-forecasts)      CC BY 4.0
  ECMWF IFS HRES           cyclone track (…-oper-tf.bufr, when published), 850/700/500/200 hPa wind,
                           humidity, vorticity, MSLP, TCWV (…-oper-fc*.grib2)                CC BY 4.0
  NOAA CFSv2               6-hourly precipitation rate time series of the 4 members of each cycle
                           (noaa-cfs-pds on AWS), lagged ensemble for days 15-30
  NOAA CPC Unified gauge   daily precipitation climatology 1991-2020 (land only), PSL
  IBTrACS v04 (NI basin)   observed tracks and intensity (IMD New Delhi wind first), NCEI
  BoM RMM index            observed MJO phase and amplitude
  TN-SMART (TNSDMA)        observed 24-h station rainfall (08:30-08:30 IST) for verification

Only byte ranges of the wanted GRIB messages are downloaded (from the .index / .idx files).
`demo_*` functions make synthetic data of the same shape for testing without downloads.
"""
import io
import os
import re
import json
import time
import gzip
from datetime import datetime, timedelta, timezone, date
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Analysis domain (North Indian Ocean + peninsula)
LAT0, LAT1, LON0, LON1 = 0.0, 30.0, 60.0, 100.0

ECMWF_ROOTS = [os.environ.get("ECMWF_ROOT", "https://data.ecmwf.int/forecasts"),
               "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"]
CFS_ROOT = "https://noaa-cfs-pds.s3.amazonaws.com"
CPC_CLIM_URLS = [
    "https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_global_precip/precip.day.ltm.1991-2020.nc",
    "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.day.ltm.1991-2020.nc",
]
IBTRACS_URLS = [
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NI.list.v04r01.csv",
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r00/access/csv/ibtracs.NI.list.v04r00.csv",
]
RMM_URLS = [
    "http://www.bom.gov.au/clim_data/IDCKGEM000/rmm.74toRealtime.txt",
    "https://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
]
TNSMART_URL = "https://beta-tnsmart.rimes.int/Rainfall_Python_IDW/Hourly_json/hourly_rainfall_{d}.json"
TNSMART_HEADERS = {"User-Agent": "Mozilla/5.0 (MasRainman outlook verification)",
                   "Referer": "https://beta-tnsmart.rimes.int/index.php/RTDAS/Hourly_data_TNDRRA_c/public"}

CACHE = os.environ.get("OUTLOOK_CACHE") or os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "outlook_cache")
TIMEOUT = (10, 120)


def _http():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "MasRainman-NEM-outlook"
    return s


_session = None


def http():
    global _session
    if _session is None:
        _session = _http()
    return _session


def _get(url, retries=3, **kw):
    """GET with a few retries; returns the response or None (404 / failures)."""
    last = None
    for a in range(retries):
        try:
            r = http().get(url, timeout=kw.pop("timeout", TIMEOUT), **kw)
            if r.status_code in (200, 206):
                return r
            if r.status_code in (403, 404):
                return None
            last = r.status_code
        except Exception as e:      # network hiccup
            last = e
        time.sleep(3 * (a + 1))
    print(f"    ({url.split('/')[-1]}: {last})")
    return None


# ============================================================
# GRIB helpers
# ============================================================

def grib_fields(blob):
    """Decode GRIB message bytes -> list of (keys, lats, lons, values cut to the domain).
    Regular lat/lon and regular Gaussian grids; lons returned in -180..180, ascending."""
    import eccodes
    import tempfile
    out = []
    tmp = tempfile.NamedTemporaryFile(suffix=".grib", delete=False)   # eccodes needs a real file
    tmp.write(blob)
    tmp.close()
    f = open(tmp.name, "rb")
    while True:
        h = eccodes.codes_new_from_file(f, eccodes.CODES_PRODUCT_GRIB)
        if h is None:
            break
        try:
            keys = {}
            for k in ("shortName", "level", "number", "startStep", "endStep", "dataDate", "dataTime",
                      "units", "typeOfLevel", "dataType"):
                try:
                    keys[k] = eccodes.codes_get(h, k)
                except Exception:
                    keys[k] = None
            ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
            vals = eccodes.codes_get_values(h)
            vals = np.where(vals > 9e20, np.nan, vals).reshape(nj, ni)
            if eccodes.codes_get(h, "gridType") == "regular_gg":
                lats = np.array(eccodes.codes_get_array(h, "distinctLatitudes"), float)
                lons = np.array(eccodes.codes_get_array(h, "distinctLongitudes"), float)
                if eccodes.codes_get(h, "jScansPositively") == 0 and lats[0] < lats[-1]:
                    lats = lats[::-1]
            else:
                la0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
                la1 = eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees")
                lo0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
                lats = np.linspace(la0, la1, nj)
                lons = lo0 + np.arange(ni) * 360.0 / ni
        finally:
            eccodes.codes_release(h)
        lons = (lons + 180) % 360 - 180
        o = np.argsort(lons)
        lons, vals = lons[o], vals[:, o]
        if lats[0] > lats[-1]:
            lats, vals = lats[::-1], vals[::-1]
        la = (lats >= LAT0 - 1) & (lats <= LAT1 + 1)
        lo = (lons >= LON0 - 1) & (lons <= LON1 + 1)
        out.append((keys, lats[la], lons[lo], vals[np.ix_(la, lo)].astype(np.float32)))
    f.close()
    os.remove(tmp.name)
    return out


def _ranges_from_index(text, want):
    """ECMWF .index (JSON lines) -> [(offset, length, record)] for records where want(rec)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            if want(rec):
                out.append((int(rec["_offset"]), int(rec["_length"]), rec))
    return out


def _fetch_ranges(url, ranges, workers=8):
    def one(rg):
        off, ln, _ = rg
        r = _get(url, headers={"Range": f"bytes={off}-{off + ln - 1}"})
        if r is None or len(r.content) != ln:
            raise RuntimeError(f"range {off}+{ln} failed for {url.split('/')[-1]}")
        return r.content
    with ThreadPoolExecutor(workers) as ex:
        return b"".join(ex.map(one, sorted(ranges, key=lambda x: x[0])))


def ecmwf_messages(run, step, stream, kinds, want, model="ifs", resol="0p25"):
    """Fetch the wanted GRIB messages of one step of an ECMWF open-data file.
    run: 'YYYYMMDD_HH'. kinds: file endings to try, e.g. ['enfo-ef'] or ['oper-fc', 'oper-fc-pl', 'oper-fc-sfc'].
    Several kinds are combined (new-style separate pl/sfc files). Returns decoded grib_fields()."""
    d, h = run.split("_")
    stamp = f"{d}{h}0000"
    os.makedirs(CACHE, exist_ok=True)
    fields = []
    for root in ECMWF_ROOTS:
        base = f"{root}/{d}/{h}z/{model}/{resol}/{stream}/{stamp}-{step}h"
        got_any = False
        for kind in kinds:
            r = _get(f"{base}-{kind}.index", retries=2)
            if r is None:
                continue
            rg = _ranges_from_index(r.text, want)
            if not rg:
                continue
            fields += grib_fields(_fetch_ranges(f"{base}-{kind}.grib2", rg))
            got_any = True
        if got_any:
            return fields
    return fields


# ============================================================
# ECMWF ENS / HRES
# ============================================================

def ens_tp(run, steps):
    """ECMWF ENS total precipitation (mm since T+0) -> {step: (lat, lon, array[member, lat, lon])}."""
    out = {}
    for s in steps:
        fl = ecmwf_messages(run, s, "enfo", ["enfo-ef"],
                            lambda r: r.get("param") == "tp" and r.get("type") in ("cf", "pf"))
        if not fl:
            print(f"  ENS tp T+{s}: not found")
            continue
        fl.sort(key=lambda x: int(x[0].get("number") or 0))
        scale = 1000.0 if str(fl[0][0].get("units")).strip() == "m" else 1.0
        out[s] = (fl[0][1], fl[0][2], np.stack([f[3] for f in fl]) * scale)
        print(f"  ENS tp T+{s}: {len(fl)} members")
    return out


HRES_PL = {("u", 850), ("v", 850), ("u", 200), ("v", 200), ("r", 700), ("r", 500), ("vo", 850)}
HRES_SFC = {"msl", "tcwv", "tp"}


def hres_fields(run, steps, only=None):
    """ECMWF IFS HRES fields for the active/break detector and the environment maps.
    only: e.g. {"tp"} to fetch just those surface params. -> {step: {'u850':(lat,lon,arr), 'msl':…, …}}"""
    def want(r):
        p = r.get("param")
        if only is not None:
            return r.get("levtype", "sfc") == "sfc" and p in only
        if r.get("levtype") == "pl":
            try:
                return (p, int(r.get("levelist", -1))) in HRES_PL
            except ValueError:
                return False
        return r.get("levtype", "sfc") == "sfc" and p in HRES_SFC
    out = {}
    for s in steps:
        fl = ecmwf_messages(run, s, "oper", ["oper-fc", "oper-fc-pl", "oper-fc-sfc"], want)
        if not fl:
            print(f"  HRES T+{s}: not found")
            continue
        d = {}
        for k, la, lo, v in fl:
            name = k["shortName"]
            if k.get("typeOfLevel") == "isobaricInhPa":
                name = f"{name}{k['level']}"
            if name == "tp":
                v = v * (1000.0 if str(k.get("units")).strip() == "m" else 1.0)
            if name == "msl":
                v = v / 100.0
            d[name] = (la, lo, v)
        out[s] = d
    return out


def _arr(eccodes, h, key, n):
    try:
        a = np.array(eccodes.codes_get_array(h, key), dtype=float)
    except Exception:
        return np.full(n, np.nan)
    if a.size == 1:
        a = np.repeat(a, n)
    a[np.abs(a) > 1e90] = np.nan
    return a


def read_tf_bufr(path, default_member=0):
    """ECMWF tropical-cyclone track BUFR (ENS or HRES) -> DataFrame(storm, member, step, lat, lon, mslp, wind m/s)
    for the North Indian Ocean (ids ending B or A)."""
    import eccodes
    rows = []
    with open(path, "rb") as f:
        while True:
            h = eccodes.codes_bufr_new_from_file(f)
            if h is None:
                break
            try:
                eccodes.codes_set(h, "unpack", 1)
                sid = eccodes.codes_get(h, "stormIdentifier").strip()
                if not sid.endswith(("B", "A")):
                    continue
                n = eccodes.codes_get(h, "numberOfSubsets")
                mem = _arr(eccodes, h, "ensembleMemberNumber", n)
                mem = np.where(np.isfinite(mem), mem, default_member)
                steps = [0] + [int(x) for x in eccodes.codes_get_array(h, "timePeriod")]
                for k, step in enumerate(steps):
                    la = _arr(eccodes, h, f"#{2 * k + 2}#latitude", n)
                    lo = _arr(eccodes, h, f"#{2 * k + 2}#longitude", n)
                    p = _arr(eccodes, h, f"#{k + 1}#pressureReducedToMeanSeaLevel", n)
                    w = _arr(eccodes, h, f"#{k + 1}#windSpeedAt10M", n)
                    for i in range(n):
                        if np.isfinite(la[i]):
                            rows.append((sid, int(mem[i]), step, la[i], lo[i], p[i] / 100.0, w[i]))
            finally:
                eccodes.codes_release(h)
    return pd.DataFrame(rows, columns=["storm", "member", "step", "lat", "lon", "mslp", "wind"])


def tf_bufr(run, stream="enfo"):
    """Download the ENS ('enfo') or HRES ('oper') track BUFR of a run. Path or None."""
    d, h = run.split("_")
    s = 360 if h in ("00", "12") else 144
    if stream == "oper":
        s = 240 if h in ("00", "12") else 90
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"tf_{stream}_{run}.bufr")
    if os.path.exists(path):
        return path
    for root in ECMWF_ROOTS:
        for step in ([s] if stream == "enfo" else [s, 360, 144]):
            r = _get(f"{root}/{d}/{h}z/ifs/0p25/{stream}/{d}{h}0000-{step}h-{stream}-tf.bufr", retries=2)
            if r is not None and r.content[:4] == b"BUFR":
                open(path, "wb").write(r.content)
                return path
    return None


def ens_tracks(run):
    p = tf_bufr(run, "enfo")
    return read_tf_bufr(p) if p else None


def hres_tracks(run):
    p = tf_bufr(run, "oper")
    return read_tf_bufr(p) if p else None


def ens_run_complete(run):
    """True once the ENS track BUFR and the last tp step needed by the outlook are online."""
    d, h = run.split("_")
    for root in ECMWF_ROOTS:
        try:
            r = http().head(f"{root}/{d}/{h}z/ifs/0p25/enfo/{d}{h}0000-360h-enfo-ef.index",
                            timeout=(5, 20), allow_redirects=True)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False


# ============================================================
# JTWC names (invests 90-99 and numbered systems) for ECMWF's own ids
# ============================================================
# ECMWF's track files call a disturbance it is following 70B, 71B … (its own genesis ids) until the
# system gets an official number (e.g. 02B). JTWC's best-track files (b-decks, mirrored by NOAA/SSD)
# carry the names forecasters use, such as Invest 91B.

JTWC_DIRS = ["https://www.ssd.noaa.gov/PS/TROP/DATA/ATCF/JTWC/"]
JTWC_ABIO = ["https://www.metoc.navy.mil/jtwc/products/abioweb.txt"]


def jtwc_invests_abio(now):
    """Invests named in JTWC's Indian Ocean outlook (ABIO): 'INVEST 91B ... NEAR 9.6N 90.9E'."""
    import re as _re
    out = []
    for url in JTWC_ABIO:
        r = _get(url, retries=2, timeout=(10, 30))
        if r is None:
            continue
        txt = " ".join(r.text.split())
        for m in _re.finditer(r"INVEST\s+(9\d[AB])\)?.{0,200}?(\d{1,2}\.\d)([NS])\s+(\d{2,3}\.\d)([EW])", txt):
            lat = float(m.group(2)) * (1 if m.group(3) == "N" else -1)
            lon = float(m.group(4)) * (1 if m.group(5) == "E" else -1)
            sid = m.group(1)
            if sid not in [o["id"] for o in out]:
                out.append({"id": sid, "name": f"Invest {sid}", "time": now, "lat": lat, "lon": lon, "wind_kt": np.nan})
    return out


def _ll(v):
    v = v.strip()
    x = float(v[:-1]) / 10.0
    return -x if v[-1] in "SW" else x


def jtwc_systems(now, max_age_h=36):
    """Active North Indian Ocean systems in JTWC b-decks: [{id, name, time, lat, lon, wind_kt}]."""
    import re as _re
    out = []
    for d in JTWC_DIRS:
        r = _get(d, retries=2, timeout=(10, 30))
        if r is None:
            continue
        files = sorted(set(_re.findall(r"bio(\d\d)(\d{4})\.dat", r.text)))
        for cy, yr in files:
            if int(yr) != now.year:
                continue
            b = _get(f"{d}bio{cy}{yr}.dat", retries=2, timeout=(10, 30))
            if b is None or not b.text.strip():
                continue
            last = [p.strip() for p in b.text.strip().splitlines()[-1].split(",")]
            try:
                t = datetime.strptime(last[2], "%Y%m%d%H").replace(tzinfo=timezone.utc)
                lat, lon, w = _ll(last[6]), _ll(last[7]), float(last[8])
            except (IndexError, ValueError):
                continue
            if (now - t).total_seconds() > max_age_h * 3600:
                continue
            suf = "A" if lon < 77.5 else "B"
            sid = f"{int(cy):02d}{suf}"
            name = last[27] if len(last) > 27 and last[27] and last[27].upper() not in ("INVEST", "") else ""
            out.append({"id": sid, "name": (f"Invest {sid}" if int(cy) >= 90 else (name.title() or sid)),
                        "time": t, "lat": lat, "lon": lon, "wind_kt": w})
        if out:
            break
    print(f"  JTWC b-decks: {[o['id'] for o in out]}")
    if not any(o["id"][:1] == "9" for o in out):
        try:
            ab = jtwc_invests_abio(now)
            print(f"  JTWC ABIO invests: {[o['id'] for o in ab]}")
            out += [a for a in ab if a["id"] not in [o["id"] for o in out]]
        except Exception as e:
            print(f"  JTWC ABIO failed: {e}")
    return out


def official_names(tracks, run_dt, now=None, radius_km=350):
    """Map ECMWF storm ids to JTWC ids for systems that exist at the start of the run:
    {'70B': '91B', …}. Empty when JTWC data is not reachable."""
    if tracks is None or not len(tracks):
        return {}
    try:
        js = jtwc_systems(now or datetime.now(timezone.utc))
    except Exception as e:
        print(f"  JTWC names not available: {e}")
        return {}
    if not js:
        return {}
    first = tracks[tracks.step == tracks.groupby("storm").step.transform("min")]
    start = first.groupby("storm").agg(lat=("lat", "median"), lon=("lon", "median"), step=("step", "min"))
    out = {}
    for storm, r in start.iterrows():
        if r.step > 12 or not str(storm)[:1] == "7":
            continue            # only systems present at T+0 that ECMWF knows by a genesis id
        best, bd = None, radius_km
        for j in js:
            if abs((j["time"] - run_dt).total_seconds()) > 36 * 3600:
                continue
            p = np.pi / 180
            a = (np.sin((j["lat"] - r.lat) * p / 2) ** 2 + np.cos(r.lat * p) * np.cos(j["lat"] * p)
                 * np.sin((j["lon"] - r.lon) * p / 2) ** 2)
            d = 12742 * np.arcsin(np.sqrt(a))
            if d < bd:
                best, bd = j, d
        if best and best["id"] not in out.values():
            out[storm] = best["id"]
    if out:
        print("  JTWC names: " + ", ".join(f"ECMWF {k} = {v}" for k, v in out.items()))
    return out


# ============================================================
# NOAA CFSv2 (days 15-30)
# ============================================================

def _cfs_idx_ranges(text, need_h):
    """wgrib2 .idx -> one byte range (start, end) covering messages whose forecast end hour <= need_h."""
    lines = [l.split(":") for l in text.strip().splitlines()]
    offs = [int(l[1]) for l in lines]
    last = None
    for i, l in enumerate(lines):
        t = ":".join(l)
        m = re.search(r"(\d+)-(\d+) hour", t) or re.search(r":(\d+) hour (?:fcst|ave)", t)
        if m and int(m.groups()[-1]) <= need_h:
            last = i
    if last is None:
        # unknown inventory wording: messages are 6-hourly in time order, so take enough of them
        last = min(len(lines) - 1, need_h // 6 + 4)
    end = offs[last + 1] - 1 if last + 1 < len(offs) else None
    return 0, end


def cfs_member_rain(init, member, valid_from, valid_to):
    """One CFSv2 member's daily rain (mm/day) for UTC days in [valid_from, valid_to).
    -> (lat, lon, days list, array[day, lat, lon]) on the native Gaussian grid cut to the domain, or None."""
    ymd, hh = init.strftime("%Y%m%d"), init.strftime("%H")
    url = f"{CFS_ROOT}/cfs.{ymd}/{hh}/time_grib_{member:02d}/prate.{member:02d}.{ymd}{hh}.daily.grb2"
    need_h = int((valid_to - init).total_seconds() // 3600) + 6
    idx = _get(url + ".idx", retries=2)
    if idx is not None and not getattr(cfs_member_rain, "shown", False):
        cfs_member_rain.shown = True
        print("    CFS idx sample: " + " | ".join(idx.text.splitlines()[:2] + idx.text.splitlines()[-1:]))
    if idx is not None:
        rg = _cfs_idx_ranges(idx.text, need_h)
        if rg is None:
            return None
        hdr = {"Range": f"bytes={rg[0]}-{'' if rg[1] is None else rg[1]}"}
        r = _get(url, headers=hdr, timeout=(10, 600))
        print(f"    CFS {url.split('/')[-1]}: {'%.0f MB' % (len(r.content) / 1e6) if r is not None else 'download failed'}")
    else:
        try:
            hd = http().head(url, timeout=(5, 20))
            size = int(hd.headers.get("Content-Length", 0)) if hd.status_code == 200 else 0
        except Exception:
            size = 0
        r = _get(url, timeout=(10, 900)) if 0 < size < 350e6 else None
    if r is None:
        return None
    days = {}
    for k, la, lo, v in grib_fields(r.content):
        s0, s1 = int(k["startStep"] or 0), int(k["endStep"] or 0)
        if s1 <= s0:
            s0, s1 = s1 - 6, s1
        t0 = init + timedelta(hours=s0)
        if not (valid_from <= t0 < valid_to):
            continue
        day = t0.date()
        mm = np.maximum(v, 0) * (s1 - s0) * 3600.0          # kg m-2 s-1 (rate) -> mm in the block
        days.setdefault(day, [la, lo, np.zeros_like(mm), 0])
        days[day][2] += mm
        days[day][3] += (s1 - s0)
    full = sorted(d for d, x in days.items() if x[3] >= 24)
    if not full:
        print(f"    CFS {url.split('/')[-1]}: no full days in {valid_from:%d %b}-{valid_to:%d %b} "
              f"({len(days)} partial days, {len(r.content) // 1e6:.0f} MB)")
        return None
    la, lo = days[full[0]][0], days[full[0]][1]
    return la, lo, full, np.stack([days[d][2] for d in full])


def cfs_lagged(valid_from, valid_to, n_cycles=4, now=None):
    """Lagged CFSv2 ensemble: the 4 members of the newest n_cycles cycles that are online.
    -> (lat, lon, days, array[member, day, lat, lon], [member labels]) or None."""
    t = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 6 + 12)                    # CFS appears ~8-12 h after the cycle
    members, labels, grid = [], [], None
    tried = 0
    while len({l.split(" m")[0] for l in labels}) < n_cycles and tried < 12:
        tried += 1
        got = 0
        for m in (1, 2, 3, 4):
            res = cfs_member_rain(t, m, valid_from, valid_to)
            if res is None:
                continue
            la, lo, days, arr = res
            if grid is None:
                grid = (la, lo, days)
            if days != grid[2]:
                common = [d for d in grid[2] if d in days]
                if len(common) < len(grid[2]):
                    continue
                arr = arr[[days.index(d) for d in common]]
            members.append(arr)
            labels.append(f"{t:%d%HZ} m{m}")
            got += 1
        print(f"  CFSv2 {t:%Y%m%d %HZ}: {got} members")
        t -= timedelta(hours=6)
    if not members:
        return None
    return grid[0], grid[1], grid[2], np.stack(members), labels


# ============================================================
# Climatology, IBTrACS, MJO, TN-SMART
# ============================================================

def cpc_climatology(state_dir):
    """CPC Unified gauge daily climatology 1991-2020 (mm/day, land only) for the domain.
    -> (lat, lon, clim[365, lat, lon]). Saved as npz in the outlook state so it downloads once."""
    path = os.path.join(state_dir, "clim_cpc_1991_2020.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["lat"], z["lon"], z["clim"].astype(np.float32)
    import xarray as xr
    ds = None
    try:
        ds = xr.open_dataset(CPC_CLIM_URLS[0])          # OPeNDAP subset (needs netCDF4 with DAP)
    except Exception as e:
        print(f"  CPC OPeNDAP failed ({e}); downloading the file")
        os.makedirs(CACHE, exist_ok=True)
        local = os.path.join(CACHE, "cpc_ltm.nc")
        if not os.path.exists(local):
            r = _get(CPC_CLIM_URLS[1], timeout=(10, 1200))
            if r is None:
                raise RuntimeError("CPC climatology not reachable")
            open(local, "wb").write(r.content)
        ds = xr.open_dataset(local)
    da = ds["precip"]
    lat = da["lat"].values
    lon = da["lon"].values
    la = (lat >= LAT0 - 1) & (lat <= LAT1 + 1)
    lo = (lon >= LON0 - 1) & (lon <= LON1 + 1)
    clim = da.isel(lat=np.where(la)[0], lon=np.where(lo)[0]).values.astype(np.float32)
    lat, lon = lat[la], lon[lo]
    if lat[0] > lat[-1]:
        lat, clim = lat[::-1], clim[:, ::-1]
    clim = clim[:365]
    os.makedirs(state_dir, exist_ok=True)
    np.savez_compressed(path, lat=lat, lon=lon, clim=clim.astype(np.float16))
    return lat, lon, clim


def ibtracs_ni(state_dir, max_age_days=7):
    """IBTrACS North Indian basin 6-hourly fixes since 1980: DataFrame
    (sid, name, season, time, lat, lon, wind_kt, basin) — wind = IMD (New Delhi) 3-min wind when
    present, else WMO, else 0.88 x JTWC 1-min. Cached in the state as csv.gz, refreshed weekly."""
    path = os.path.join(state_dir, "ibtracs_ni.csv.gz")
    fresh = os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_days * 86400
    if not fresh:
        for url in IBTRACS_URLS:
            r = _get(url, timeout=(10, 600))
            if r is None:
                continue
            cols = ["SID", "SEASON", "SUBBASIN", "NAME", "ISO_TIME", "LAT", "LON", "WMO_WIND",
                    "USA_WIND", "NEWDELHI_WIND"]
            df = pd.read_csv(io.BytesIO(r.content), skiprows=[1], usecols=lambda c: c in cols,
                             low_memory=False, na_values=[" ", ""], keep_default_na=True)
            df = df[pd.to_numeric(df.SEASON, errors="coerce") >= 1980]
            num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
            wind = num("NEWDELHI_WIND")
            wind = wind.fillna(num("WMO_WIND")).fillna(num("USA_WIND") * 0.88)
            out = pd.DataFrame({"sid": df.SID, "name": df.NAME, "season": num("SEASON").astype(int),
                                "time": pd.to_datetime(df.ISO_TIME), "lat": num("LAT"), "lon": num("LON"),
                                "wind_kt": wind, "basin": df.SUBBASIN})
            out = out[out.time.dt.hour % 6 == 0]
            os.makedirs(state_dir, exist_ok=True)
            out.to_csv(path, index=False, compression="gzip")
            break
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    return df


def rmm_recent(days=40):
    """BoM RMM MJO index: list of {date, rmm1, rmm2, phase, amp} (newest last), or []."""
    for url in RMM_URLS:
        try:
            r = http().get(url, timeout=(10, 60), headers={"User-Agent": "Mozilla/5.0"})
        except Exception:
            continue
        if r.status_code != 200:
            continue
        rows = []
        for line in r.text.splitlines():
            p = line.split()
            if len(p) >= 7 and p[0].isdigit() and len(p[0]) == 4:
                try:
                    y, m, d = int(p[0]), int(p[1]), int(p[2])
                    r1, r2, ph, amp = float(p[3]), float(p[4]), int(p[5]), float(p[6])
                except ValueError:
                    continue
                if abs(r1) < 900 and amp < 900:
                    rows.append({"date": f"{y:04d}-{m:02d}-{d:02d}", "rmm1": r1, "rmm2": r2,
                                 "phase": ph, "amp": amp})
        if rows:
            return rows[-days:]
    return []


def tnsmart_day(d, cache_dir):
    """TN-SMART 24-h station totals for rain-day d (08:30 IST on d-1 to 08:30 IST on d):
    list of {district, station, lat, lon, rain}. Only reported stations. [] if unavailable."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"tnsmart_{d.isoformat()}.json")
    if os.path.exists(path):
        return json.load(open(path))
    import requests
    try:
        r = requests.get(TNSMART_URL.format(d=d.isoformat()), headers=TNSMART_HEADERS, timeout=60)
        if r.status_code != 200:
            return []
        recs = r.json()
    except Exception:
        return []
    out = []
    for s in recs:
        try:
            lat, lon = float(s.get("latitude")), float(s.get("longitude"))
        except (TypeError, ValueError):
            continue
        if lat > 70 and lon < 20:
            lat, lon = lon, lat
        tot = s.get("total")
        if not (7.5 <= lat <= 14.0 and 75.8 <= lon <= 80.6) or not isinstance(tot, (int, float)):
            continue
        if 0 <= tot <= 1500:
            out.append({"district": (s.get("district_name") or "").strip(),
                        "station": (s.get("station_name") or "").strip(),
                        "lat": round(lat, 4), "lon": round(lon, 4), "rain": float(tot)})
    if out:
        json.dump(out, open(path, "w"))
    return out


# ============================================================
# DEMO DATA (synthetic, for testing the outlook without downloads)
# ============================================================

def _demo_rng(run):
    return np.random.default_rng(int(run.replace("_", "")) % 2 ** 31)


def demo_ens_tracks(run, n=51, shift=0.0):
    """A synthetic Bay of Bengal system (70B) moving WNW towards north Tamil Nadu / south Andhra,
    plus a weaker Arabian Sea genesis signal (72A) in a few members."""
    rng = _demo_rng(run)
    rows = []
    for m in range(n):
        if rng.random() < 0.88:
            lat, lon = 9.0 + rng.normal(0, 0.4), 89.0 + rng.normal(0, 0.5) + shift
            dlat, dlon = 0.22 + rng.normal(0, 0.08), -0.55 + rng.normal(0, 0.12)
            peak = max(10, rng.normal(26, 8))
            t_peak = rng.uniform(84, 132)
            for step in range(0, 241, 6):
                w = peak * np.exp(-((step - t_peak) / 70) ** 2) + 6
                rows.append(("70B", m, step, lat, lon, 1008 - (w - 6) * 1.1, w))
                lat += dlat + rng.normal(0, 0.03)
                lon += dlon + rng.normal(0, 0.03)
                if lon < 77.5:
                    break
        if rng.random() < 0.25:
            lat, lon = 11 + rng.normal(0, 1), 66 + rng.normal(0, 1)
            for step in range(144, 361, 6):
                w = 8 + (step - 144) / 20 + rng.normal(0, 1)
                rows.append(("72A", m, step, lat, lon, 1006 - (w - 8), w))
                lat += 0.1
                lon -= 0.15
    return pd.DataFrame(rows, columns=["storm", "member", "step", "lat", "lon", "mslp", "wind"])


def demo_hres_tracks(run):
    t = demo_ens_tracks(run, n=1)
    return t.assign(member=0, wind=t.wind * 1.25)


def _grid(res):
    lat = np.arange(LAT0, LAT1 + 1e-6, res)
    lon = np.arange(LON0, LON1 + 1e-6, res)
    return lat, lon, *np.meshgrid(lon, lat)


def demo_ens_tp(run, steps, n=51):
    rng = _demo_rng(run)
    lat, lon, X, Y = _grid(0.25)
    out = {}
    base = None
    for s in sorted(steps):
        days = s / 24
        field = []
        for m in range(n):
            amp = rng.lognormal(0, 0.35)
            cx, cy = 80.0 + rng.normal(0, 0.6), 12.5 + rng.normal(0, 0.8)
            core = 160 * np.exp(-(((X - cx) / 1.6) ** 2 + ((Y - cy) / 2.0) ** 2)) * min(days, 7) / 7
            bg = 4.5 * days * np.exp(-((Y - 11) / 8) ** 2) * (1 + 0.4 * np.sin(X / 3))
            field.append((core * amp + bg).astype(np.float32))
        out[s] = (lat, lon, np.stack(field))
    return out


def demo_hres(run, steps):
    lat, lon, X, Y = _grid(0.25)
    out = {}
    for s in steps:
        cx, cy = 89 - s / 24 * 2.2, 9 + s / 24 * 0.9
        r2 = ((X - cx) ** 2 + (Y - cy) ** 2)
        vort = 25 * np.exp(-r2 / 4)
        u = -6 - 6 * np.exp(-r2 / 10) * (Y - cy) / 2
        v = 6 * np.exp(-r2 / 10) * (X - cx) / 2
        out[s] = {"u850": (lat, lon, u.astype(np.float32)), "v850": (lat, lon, v.astype(np.float32)),
                  "u200": (lat, lon, (-12 + Y * 0.5).astype(np.float32)),
                  "v200": (lat, lon, np.zeros_like(u, np.float32)),
                  "r700": (lat, lon, (55 + 30 * np.exp(-r2 / 30)).astype(np.float32)),
                  "r500": (lat, lon, (45 + 30 * np.exp(-r2 / 30)).astype(np.float32)),
                  "vo850": (lat, lon, (vort * 1e-5).astype(np.float32)),
                  "msl": (lat, lon, (1010 - 8 * np.exp(-r2 / 6) + (Y - 10) * 0.15).astype(np.float32)),
                  "tcwv": (lat, lon, (45 + 15 * np.exp(-r2 / 20)).astype(np.float32))}
    return out


def demo_cfs(valid_from, valid_to, n=16):
    rng = np.random.default_rng(7)
    lat, lon, X, Y = _grid(0.9375)
    days = [(valid_from + timedelta(days=i)).date() for i in range((valid_to - valid_from).days)]
    arr = np.stack([np.stack([np.maximum(0, (6 + 3 * np.sin(i / 3)) * np.exp(-((Y - 11) / 5) ** 2)
                                        * rng.lognormal(0, 0.6) + rng.normal(0, 0.5, X.shape))
                              for i in range(len(days))]) for _ in range(n)]).astype(np.float32)
    return lat, lon, days, arr, [f"demo m{i}" for i in range(n)]


def demo_climatology():
    lat, lon, X, Y = _grid(0.5)
    doy = np.arange(365)[:, None, None]
    nem = np.exp(-((doy - 315) / 30.0) ** 2)                              # NEM peak ~ mid-November
    tn = np.exp(-(((X - 79.2) / 2.2) ** 2 + ((Y - 11) / 2.5) ** 2))
    clim = (1.5 + 8 * nem * tn + 2 * np.exp(-((Y - 20) / 6) ** 2) * np.exp(-((doy - 200) / 40.0) ** 2))
    return lat, lon, clim.astype(np.float32)


def demo_ibtracs():
    rng = np.random.default_rng(3)
    rows = []
    for y in range(1991, 2027):
        for k in range(rng.poisson(2.2)):
            t0 = datetime(y, 10, 1) + timedelta(days=float(rng.uniform(0, 85)))
            if y == 2026 and t0 > datetime(2026, 10, 25):
                continue
            lat, lon = rng.uniform(8, 15), rng.uniform(84, 90)
            peak, life = rng.uniform(20, 110), int(rng.uniform(8, 24))
            for i in range(life):
                w = peak * np.sin(np.pi * (i + 1) / (life + 1))
                rows.append((f"{y}DEMO{k}", f"DEMO{k}", y, t0 + timedelta(hours=6 * i), lat, lon, w, "BB"))
                lat += 0.2
                lon -= 0.4
    return pd.DataFrame(rows, columns=["sid", "name", "season", "time", "lat", "lon", "wind_kt", "basin"])


def demo_rmm():
    out = []
    for i in range(40):
        a = np.radians(200 + i * 6)
        out.append({"date": (date(2026, 9, 22) + timedelta(days=i)).isoformat(),
                    "rmm1": 1.4 * np.cos(a), "rmm2": 1.4 * np.sin(a),
                    "phase": int(((np.degrees(np.arctan2(np.sin(a), np.cos(a))) + 180) // 45 + 4) % 8 + 1),
                    "amp": 1.4})
    return out


def demo_tnsmart(d):
    rng = np.random.default_rng(d.toordinal())
    out = []
    for i in range(300):
        la, lo = rng.uniform(8.2, 13.4), rng.uniform(76.8, 80.2)
        out.append({"district": "Demo", "station": f"S{i}", "lat": la, "lon": lo,
                    "rain": float(max(0, rng.gamma(0.6, 12) * (1 + (lo - 77) / 3)))})
    return out
