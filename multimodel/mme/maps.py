# ============================================================
# RAINFALL MAPS (approved MASRainman design)
# ============================================================
#
# Borders: India outline + state lines as per the Government of India
# (DataMeet, Survey of India based); neighbours from the India-view
# border file used in the weathernext3 project.

import os
import json

import numpy as np
import shapefile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.path import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import shape, box
from shapely.ops import unary_union

from . import config as C
from .grib import TLAT2, TLON2

PC = ccrs.PlateCarree()
HALO = [pe.withStroke(linewidth=2.6, foreground="white")]

_B = {}


DATAMEET = "https://raw.githubusercontent.com/datameet/maps/master/"
BOUNDARY_FILES = {
    "india-composite.geojson": "Country/india-composite.geojson",
    "Admin2.shp": "States/Admin2.shp", "Admin2.shx": "States/Admin2.shx", "Admin2.dbf": "States/Admin2.dbf",
    "2011_Dist.shp": "Districts/Census_2011/2011_Dist.shp", "2011_Dist.shx": "Districts/Census_2011/2011_Dist.shx",
    "2011_Dist.dbf": "Districts/Census_2011/2011_Dist.dbf",
}


def _ensure_boundaries():
    """Download the GoI-compliant boundaries (DataMeet) the first time; reuse the India-view
    neighbour borders from the weathernext3 repo when this runs inside it."""
    import requests
    os.makedirs(C.BOUNDARY_DIR, exist_ok=True)
    for name, path in BOUNDARY_FILES.items():
        dst = os.path.join(C.BOUNDARY_DIR, name)
        if not os.path.exists(dst):
            r = requests.get(DATAMEET + path, timeout=300)
            r.raise_for_status()
            with open(dst, "wb") as f:
                f.write(r.content)
    nb = os.path.join(C.BOUNDARY_DIR, "borders_india_view.geojson")
    if not os.path.exists(nb):
        import shutil
        shutil.copy(os.path.join(os.path.dirname(C.HERE), "boundaries", "borders_india_view.geojson"), nb)


def boundaries():
    if _B:
        return _B
    _ensure_boundaries()

    def shp(name, key):
        r = shapefile.Reader(os.path.join(C.BOUNDARY_DIR, name))
        return [(rec.as_dict()[key], shape(s.__geo_interface__).buffer(0).simplify(0.004))
                for s, rec in zip(r.shapes(), r.records())]

    with open(os.path.join(C.BOUNDARY_DIR, "india-composite.geojson")) as f:
        _B["india"] = unary_union([shape(ft["geometry"]) for ft in json.load(f)["features"]]).simplify(0.004)
    with open(os.path.join(C.BOUNDARY_DIR, "borders_india_view.geojson")) as f:
        _B["neighbours"] = [shape(ft["geometry"]) for ft in json.load(f)["features"]]
    _B["states"] = dict(shp("Admin2", "ST_NM"))
    _B["districts"] = shp("2011_Dist", "DISTRICT")
    return _B


def state_shape(state):
    st = boundaries()["states"]
    g = st[state]
    if state == "Tamil Nadu":  # Tamil Nadu & Puducherry (Puducherry + Karaikal enclaves)
        g = g.union(st["Puducherry"].intersection(box(78.5, 10.5, 80.5, 12.5)))
    return g


def state_districts(state):
    shp_ = state_shape(state)
    return [(n, g) for n, g in boundaries()["districts"] if shp_.contains(g.representative_point())]


class Placer:
    """Greedy label placement: nudge a label until it no longer overlaps one already placed."""

    def __init__(self, ax, extent):
        fig = ax.figure
        bb = ax.get_position()
        self.dx = (extent[1] - extent[0]) / (bb.width * fig.get_figwidth() * 72)
        self.dy = (extent[3] - extent[2]) / (bb.height * fig.get_figheight() * 72)
        self.boxes = []

    def place(self, x, y, text, fs, lines=1, ha="left"):
        w = 0.62 * fs * max(len(s) for s in text.split("\n")) * self.dx
        h = 1.25 * fs * lines * self.dy
        for k in [0, 1, -1, 2, -2, 3, -3, 4, -4]:
            yy = y + k * h * 0.55
            x0 = x - w / 2 if ha == "center" else x
            b = (x0, yy - h / 2, x0 + w, yy + h / 2)
            if not any(b[0] < o[2] and b[2] > o[0] and b[1] < o[3] and b[3] > o[1] for o in self.boxes):
                self.boxes.append(b)
                return yy
        self.boxes.append(b)
        return y


def label_cities(ax, cities, extent, fs=9):
    placer = Placer(ax, extent)
    inside = [(n, a, b) for n, a, b in cities if extent[0] < b < extent[1] and extent[2] < a < extent[3]]
    for n, a, b in inside:
        placer.boxes.append((b - 4 * placer.dx, a - 4 * placer.dy, b + 4 * placer.dx, a + 4 * placer.dy))
    for n, a, b in inside:
        ax.plot(b, a, "o", ms=5, mfc="#111", mec="white", mew=1, transform=PC, zorder=25)
        y = placer.place(b + 6 * placer.dx, a, n, fs)
        if abs(y - a) > 1e-6:
            ax.plot([b, b + 6 * placer.dx], [a, y], "-", color="#111", lw=.6, transform=PC, zorder=25)
        ax.text(b + 6 * placer.dx, y, n, fontsize=fs, fontweight="bold", color="#111", va="center",
                transform=PC, zorder=26, path_effects=HALO)


def district_means(field, dists):
    pts = np.c_[TLON2.ravel(), TLAT2.ravel()]
    vals = field.ravel()
    out = {}
    for n, g in dists:
        polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
        x0, y0, x1, y1 = g.bounds
        sel = np.where((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))[0]
        m = np.zeros(len(sel), bool)
        for p in polys:
            m |= Path(np.asarray(p.exterior.coords)).contains_points(pts[sel])
        if not m.any():  # very small district: nearest grid point to its centre
            c = g.representative_point()
            out[n] = float(field[np.abs(TLAT2[:, 0] - c.y).argmin(), np.abs(TLON2[0] - c.x).argmin()])
        else:
            out[n] = float(np.nanmean(vals[sel[m]]))
    return out


def rain_map(field, region, period, out_png, title, when, models_line):
    """Draw one map. Returns {district: mean mm} for state maps, else {}."""
    B = boundaries()
    label, ext, state = C.REGIONS[region]
    w, h = ext[1] - ext[0], ext[3] - ext[2]
    figw = 13
    figh = max(6.5, min(14, figw * h / w / np.cos(np.radians((ext[2] + ext[3]) / 2)) + 0.5))

    fig = plt.figure(figsize=(figw, figh + 1.6), facecolor="white")
    ax = fig.add_axes([0.06, 0.17, 0.88, 0.72], projection=PC)
    ax.set_extent(ext, crs=PC)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e3eef8", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#efede7", zorder=0)
    ax.add_geometries([B["india"]], PC, facecolor="#fdfcf8", edgecolor="none", zorder=0.5)

    lv, ticks = C.RAIN_LV[period]
    cmap = ListedColormap(C.RAIN_CL)
    cmap.set_under("#ffffff")
    cmap.set_over("#d9d9d9")
    cf = ax.contourf(TLON2, TLAT2, field, levels=lv, cmap=cmap, norm=BoundaryNorm(lv, len(C.RAIN_CL)),
                     extend="both", transform=PC, zorder=2, alpha=.95)

    means = {}
    if state:  # only the selected state
        shp_ = state_shape(state)
        dists = state_districts(state)
        outside = box(ext[0] - 3, ext[2] - 3, ext[1] + 3, ext[3] + 3).difference(shp_)
        ax.add_geometries([outside], PC, facecolor="white", edgecolor="none", zorder=10)
        ax.add_geometries([g for _, g in dists], PC, facecolor="none", edgecolor="#555",
                          linewidth=0.5, linestyle=(0, (3, 1.5)), zorder=15)
        ax.add_geometries([shp_], PC, facecolor="none", edgecolor="black", linewidth=1.4, zorder=17)
        means = district_means(field, dists)
        fs = 9.5 if len(dists) <= 15 else 8
        placer = Placer(ax, ext)
        for n, g in sorted(dists, key=lambda d: -d[1].area):
            p = g.representative_point()
            y = placer.place(p.x, p.y, f"{n}\n{means[n]:.0f} mm", fs, lines=2, ha="center")
            ax.text(p.x, y, n, fontsize=fs, fontweight="bold", ha="center", va="bottom",
                    color="#111", transform=PC, zorder=26, path_effects=HALO)
            ax.text(p.x, y, f"{means[n]:.0f} mm", fontsize=fs - .5, ha="center", va="top",
                    color="#08306b", fontweight="bold", transform=PC, zorder=26, path_effects=HALO)
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        ax.spines["geo"].set_visible(False)
    else:
        ax.coastlines(resolution="50m", linewidth=0.6, color="#333", zorder=15)
        ax.add_geometries(B["neighbours"], PC, facecolor="none", edgecolor="#333", linewidth=0.6, zorder=15)
        ax.add_geometries(list(B["states"].values()), PC, facecolor="none", edgecolor="#3a3a3a",
                          linewidth=0.55, zorder=16)
        ax.add_geometries([B["india"]], PC, facecolor="none", edgecolor="black", linewidth=1.3, zorder=17)
        label_cities(ax, C.SOUTH_CITIES if region == "south_india" else C.NORTH_CITIES, ext)
        gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="gray", alpha=0.35, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 9}

    cax = fig.add_axes([0.16, 0.085, 0.68, 0.022])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=ticks, extendfrac=0.045)
    cb.ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=10)
    cb.ax.tick_params(length=0)
    cb.set_label({"24h": "24-Hour Rainfall (mm)", "7day": "7-Day Accumulated Rainfall (mm)",
                  "15day": "15-Day Accumulated Rainfall (mm)"}[period], fontsize=11, fontweight="bold")

    top = dict(xy=(0.5, 1.0), xycoords="axes fraction", textcoords="offset points", ha="center", va="bottom")
    ax.annotate(title, xytext=(0, 50), fontsize=17, fontweight="bold", **top)
    ax.annotate(f"{label} · {when}", xytext=(0, 27), fontsize=12.5, fontweight="bold", **top)
    ax.annotate(models_line, xytext=(0, 7), fontsize=10.5, color="#333", **top)

    fig.text(0.5, 0.032, "Data: ECMWF Open Data (CC BY 4.0), NOAA GFS/GEFS, DWD ICON, ECCC GEPS, WeatherNext 3 | "
             "Borders as per Government of India | © MASRainman Experimental MME",
             ha="center", fontsize=8, color="dimgray")
    fig.text(0.5, 0.014, "Not an official forecast — follow IMD for weather warnings.",
             ha="center", fontsize=7.5, color="dimgray")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    return means
