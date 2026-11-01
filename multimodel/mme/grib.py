# ============================================================
# GRIB READING + REGRIDDING TO THE COMMON GRID
# ============================================================

import numpy as np
import eccodes
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from . import config as C

class Grid:
    def __init__(self, lat, lon, step):
        self.lat = np.round(np.arange(lat[0], lat[1] + 1e-6, step), 3)
        self.lon = np.round(np.arange(lon[0], lon[1] + 1e-6, step), 3)
        self.LON2, self.LAT2 = np.meshgrid(self.lon, self.lat)
        self.step = step


RAIN = Grid(C.GRID_LAT, C.GRID_LON, C.GRID_STEP)
TRACK = Grid(C.TRACK_LAT, C.TRACK_LON, C.TRACK_STEP)
TLAT, TLON, TLAT2, TLON2 = RAIN.lat, RAIN.lon, RAIN.LAT2, RAIN.LON2


def _step_range(s):
    """'18-24' -> (18, 24) hours; also handles minute units such as '0m'."""
    def hours(x):
        x = x.strip()
        return int(x[:-1]) // 60 if x.endswith("m") else int(x.rstrip("h"))
    parts = s.split("-")
    return hours(parts[0]), hours(parts[-1])


def messages(path):
    """Yield one dict per GRIB message: header keys + 'values' (+ lat/lon for regular grids)."""
    with open(path, "rb") as fh:
        while True:
            g = eccodes.codes_grib_new_from_file(fh)
            if g is None:
                break
            try:
                m = {k: eccodes.codes_get(g, k) for k in ("shortName", "stepRange", "dataType", "gridType")}
                m["number"] = eccodes.codes_get(g, "number") if eccodes.codes_is_defined(g, "number") else 0
                vals = eccodes.codes_get_double_array(g, "values")
                miss = eccodes.codes_get(g, "missingValue")
                vals[vals == miss] = np.nan
                if m["gridType"] == "regular_ll":
                    ni, nj = eccodes.codes_get(g, "Ni"), eccodes.codes_get(g, "Nj")
                    la0 = eccodes.codes_get(g, "latitudeOfFirstGridPointInDegrees")
                    la1 = eccodes.codes_get(g, "latitudeOfLastGridPointInDegrees")
                    lo0 = eccodes.codes_get(g, "longitudeOfFirstGridPointInDegrees")
                    lo1 = eccodes.codes_get(g, "longitudeOfLastGridPointInDegrees")
                    if lo1 < lo0:
                        lo1 += 360.0
                    m["lat"] = np.linspace(la0, la1, nj)
                    m["lon"] = np.linspace(lo0, lo1, ni)
                    m["values"] = vals.reshape(nj, ni)
                else:
                    m["values"] = vals
                m["start"], m["end"] = _step_range(str(m["stepRange"]))
                yield m
            finally:
                eccodes.codes_release(g)


def regrid(m, grid=RAIN):
    """Regular lat/lon message -> target grid (bilinear)."""
    lat, lon, v = m["lat"], m["lon"], m["values"]
    lon = np.where(lon > 180, lon - 360, lon) if lon.max() > 180 and grid.lon[-1] <= 180 else lon
    order = np.argsort(lon)
    lon, v = lon[order], v[:, order]
    if lat[0] > lat[-1]:
        lat, v = lat[::-1], v[::-1]
    # crop to the domain (+1 deg) before interpolating, so global fields stay fast
    j = np.where((lat >= grid.lat[0] - 1) & (lat <= grid.lat[-1] + 1))[0]
    i = np.where((lon >= grid.lon[0] - 1) & (lon <= grid.lon[-1] + 1))[0]
    lat, lon, v = lat[j], lon[i], v[np.ix_(j, i)]
    f = RegularGridInterpolator((lat, lon), v, bounds_error=False, fill_value=np.nan)
    return f((grid.LAT2, grid.LON2)).astype(np.float32)


class IconRegridder:
    """ICON icosahedral cells -> target grid (inverse-distance, 4 nearest cells)."""

    def __init__(self, clat, clon, grid=RAIN):
        pad = 1.0
        sel = ((clat >= grid.lat[0] - pad) & (clat <= grid.lat[-1] + pad)
               & (clon >= grid.lon[0] - pad) & (clon <= grid.lon[-1] + pad))
        self.sel = np.where(sel)[0]
        tree = cKDTree(np.c_[clon[sel] * np.cos(np.radians(clat[sel])), clat[sel]])
        q = np.c_[grid.LON2.ravel() * np.cos(np.radians(grid.LAT2.ravel())), grid.LAT2.ravel()]
        d, i = tree.query(q, k=4)
        w = 1.0 / np.maximum(d, 1e-6) ** 2
        self.idx, self.w = i, w / w.sum(1, keepdims=True)
        self.shape = grid.LAT2.shape

    def __call__(self, values):
        v = values[self.sel][self.idx]
        return (v * self.w).sum(1).reshape(self.shape).astype(np.float32)
