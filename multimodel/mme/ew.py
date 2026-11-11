# ============================================================
# EW INDEX FOR EVERY MULTI-MODEL MODEL — South India + North India, Day 1-N
# ============================================================
#
# Same index and chart as the WeatherNext 3 maps page (../ew_index.py), for all models of the
# multi-model job, on IMD rain days (08:30 IST -> 08:30 IST):
#   ensembles with members (ECMWF ENS, AIFS ENS, CMC GEPS): true ensemble EFI from the members
#   everything else (IFS HRES, AIFS Single, ICON, GFS, AIGFS, WeatherNext 3 mean, GEFS mean):
#   the run as a single member
# Output: <out_dir>/ew/<model>/ew_south_dNN.png, ew_north_dNN.png
# ============================================================

import os
import sys
import logging
from datetime import timedelta, timezone

import numpy as np

from . import config as C
from .fetch import MEMBER_ACC, day_leads, daily_totals
from .grib import RAIN

ROOT = os.path.dirname(C.HERE)      # C.HERE is multimodel/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import ew_index  # noqa: E402

LOG = logging.getLogger("mme")
EW_DAYS = int(os.environ.get("EW_DAYS", "10"))
EW_MODELS = ["ecmwf_hres", "ecmwf_ens", "aifs_single", "aifs_ens", "wn3", "gfs", "gefs", "aigfs",
             "icon", "geps"]
SOURCES = {
    "ecmwf_hres": "ECMWF open data (CC BY 4.0)", "ecmwf_ens": "ECMWF open data (CC BY 4.0)",
    "aifs_single": "ECMWF open data (CC BY 4.0)", "aifs_ens": "ECMWF open data (CC BY 4.0)",
    "wn3": "WeatherNext 3 © 2026 DeepMind Technologies Limited, experimental",
    "gfs": "NOAA/NCEP open data", "gefs": "NOAA/NCEP open data", "aigfs": "NOAA open data",
    "icon": "DWD open data (CC BY 4.0)", "geps": "ECCC / MSC open data",
}
_BORDERS = None


def borders():
    global _BORDERS
    if _BORDERS is None:
        import json
        import cartopy.crs as ccrs
        from cartopy.feature import ShapelyFeature
        from shapely.geometry import shape
        with open(os.path.join(ROOT, "boundaries", "borders_india_view.geojson")) as f:
            gj = json.load(f)
        _BORDERS = ShapelyFeature([shape(ft["geometry"]) for ft in gj["features"]], ccrs.PlateCarree())
    return _BORDERS


def ew_maps(init, out_dir, have=None):
    """Draw EW maps for every model. have = {model: (days, run)} already fetched by the blends."""
    if not ew_index.available():
        LOG.warning("EW index skipped: no rain climate (download the 'efi-climate' release into climate/)")
        return {}
    have = dict(have or {})
    out = {}
    for model in EW_MODELS:
        try:
            if model not in have:
                days, run = daily_totals(model, init)
                if days is None:
                    continue
                have[model] = (days, run)
            days, run = have[model]
            back = int(round((init - run).total_seconds() / 3600))
            leads = [L + back for L in day_leads(init)]
            mem = MEMBER_ACC.get((model, run), {})
            name = C.MODEL_NAMES[model]
            kind = "members"
            label = f"{name} ({len(next(iter(mem.values())))} members)" if mem else name
            if not mem and model in ("gefs", "wn3"):
                label += " mean"
                kind = "mean"
            elif not mem:
                kind = "single"
            m_info = {"name": name, "label": label, "kind": kind,
                      "run": run.strftime("%Y-%m-%dT%H:%MZ"), "days": []}
            for k in range(min(EW_DAYS, C.N_DAYS)):
                a, b = leads[k], leads[k + 1]
                if mem and a in mem and b in mem:
                    fc = np.clip(mem[b] - mem[a], 0, None)
                elif np.isfinite(days[k, 0, 0]):
                    fc = days[k]
                else:
                    continue
                mid = run + timedelta(hours=(a + b) / 2)
                field = ew_index.ew_field(fc, RAIN.lat, RAIN.lon, mid)
                if field is None:
                    continue
                files = {}
                for layer in ew_index.LAYERS:
                    fn = f"{layer}_d{k + 1:02d}.png"
                    os.makedirs(os.path.join(out_dir, "ew", model), exist_ok=True)
                    ew_index.draw_ew_map(os.path.join(out_dir, "ew", model, fn), layer,
                                         label,
                                         [SOURCES[model]], borders(), run, a, b, field)
                    files[layer] = fn
                end = (run + timedelta(hours=b)).astimezone(timezone(timedelta(hours=5, minutes=30)))
                m_info["days"].append({"day": k + 1, "end_ist": end.strftime("%Y-%m-%dT%H:%M+05:30"),
                                       **files})
            if m_info["days"]:
                out[model] = m_info
                LOG.info("EW index: %s, %d days", label, len(m_info["days"]))
        except Exception as exc:  # one model must not stop the others
            LOG.exception("EW index %s failed: %s", model, exc)
    return out
