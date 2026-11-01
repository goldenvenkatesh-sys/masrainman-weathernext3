# ============================================================
# LPA DESK + CYCLONE TRACKER MAPS
# ============================================================

import os
from datetime import timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from shapely.ops import unary_union

from . import config as C
from .grib import TRACK
from .maps import boundaries, add_logo_left
from .tracker import STEPS, category, CAT

PC = ccrs.PlateCarree()
SST_LV = list(range(24, 34))
SST_CL = ["#1b3f86", "#0a84b8", "#20b2aa", "#8fd14f", "#ffe600", "#ff9900", "#ff4500", "#d40000", "#6a006a"]
LPA_EXTENT = [55, 100, 2, 28]          # full view when there is no active system
ZOOM_LIMITS = (45.0, 100.0, -2.0, 30.0)
ZOOM_MIN_W = 24.0                      # degrees of longitude at least, when zoomed on a system

CITIES = [("Mumbai", 19.08, 72.88), ("Goa", 15.50, 73.83), ("Mangaluru", 12.91, 74.86),
          ("Kochi", 9.93, 76.27), ("Thiruvananthapuram", 8.52, 76.94), ("Chennai", 13.08, 80.27),
          ("Puducherry", 11.93, 79.83), ("Nellore", 14.44, 79.99), ("Machilipatnam", 16.19, 81.14),
          ("Visakhapatnam", 17.69, 83.22), ("Puri", 19.81, 85.83), ("Bhubaneswar", 20.30, 85.82),
          ("Kolkata", 22.57, 88.36), ("Port Blair", 11.62, 92.73), ("Colombo", 6.93, 79.85),
          ("Trincomalee", 8.57, 81.23), ("Dhaka", 23.81, 90.41), ("Yangon", 16.84, 96.17),
          ("Bengaluru", 12.97, 77.59), ("Hyderabad", 17.39, 78.49), ("Ahmedabad", 23.02, 72.57),
          ("Karachi", 24.86, 67.01), ("Muscat", 23.59, 58.41), ("Kavaratti", 10.57, 72.64),
          ("Nagapattinam", 10.77, 79.84), ("Kakinada", 16.99, 82.25), ("Chittagong", 22.36, 91.78),
          ("Tirupati", 13.63, 79.42), ("Vijayawada", 16.51, 80.65), ("Madurai", 9.93, 78.12)]

BOX = dict(boxstyle="round,pad=.25", fc="#fff3d6", ec="#d62728", lw=1)
RED = dict(boxstyle="round,pad=.35", fc="white", ec="#c00000", lw=1.2)
CITY_BOX = dict(boxstyle="round,pad=.15", fc="#fff3d6", ec="none", alpha=.92)
WHITE_EDGE = [pe.Stroke(linewidth=6, foreground="white"), pe.Normal()]
FOOT_L = ("Data: ECMWF Open Data (CC BY 4.0), NOAA GFS/GEFS, DWD ICON, ECCC GEPS, WeatherNext 3\n"
          "© MASRainman Experimental Channel")
FOOT_R = "Not an Official Forecast Track\nFollow IMD for Official Warnings"


def _base(ax, extent, sst=None):
    B = boundaries()
    ax.set_extent(extent, crs=PC)
    cf = None
    if sst is not None:
        cmap = ListedColormap(SST_CL)
        cmap.set_under("#0b1f5c")
        cmap.set_over("#3a0040")
        cf = ax.contourf(TRACK.LON2, TRACK.LAT2, sst, levels=SST_LV, cmap=cmap,
                         norm=BoundaryNorm(SST_LV, len(SST_CL)), extend="both", transform=PC, zorder=1)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="white", zorder=2)
    else:
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e3eef8", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#efede7", zorder=0)
        ax.add_geometries([B["india"]], PC, facecolor="white", edgecolor="none", zorder=0.5)
    ax.coastlines(resolution="50m", linewidth=0.9, color="black", zorder=15)
    ax.add_geometries(B["neighbours"], PC, facecolor="none", edgecolor="black", linewidth=0.6, zorder=15)
    ax.add_geometries(list(B["states"].values()), PC, facecolor="none", edgecolor="#777", linewidth=0.45,
                      linestyle=":", zorder=15)
    ax.add_geometries([B["india"]], PC, facecolor="none", edgecolor="black", linewidth=1.2, zorder=16)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 11}
    return cf


def zoom_extent(points, aspect=1.45, pad=(5.0, 4.0)):
    """Map extent around the given (lon, lat) points, at least ZOOM_MIN_W wide, inside ZOOM_LIMITS."""
    if not points:
        return LPA_EXTENT
    lo = [p[0] for p in points]
    la = [p[1] for p in points]
    w = max(max(lo) - min(lo) + 2 * pad[0], ZOOM_MIN_W, (max(la) - min(la) + 2 * pad[1]) * aspect)
    h = w / aspect
    cx, cy = (max(lo) + min(lo)) / 2, (max(la) + min(la)) / 2
    W, E, S, N = ZOOM_LIMITS
    w, h = min(w, E - W), min(h, N - S)
    cx = min(max(cx, W + w / 2), E - w / 2)
    cy = min(max(cy, S + h / 2), N - h / 2)
    return [cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2]


# ------------------------------------------------------------
# LABEL PLACEMENT — every label tries several spots and takes the first free one
# ------------------------------------------------------------

class Labels:
    def __init__(self, ax):
        ax.apply_aspect()
        fig = ax.figure
        bb = ax.get_position()
        x0, x1, y0, y1 = ax.get_extent(PC)
        self.ax, self.ext = ax, (x0, x1, y0, y1)
        self.dx = (x1 - x0) / (bb.width * fig.get_figwidth() * 72)    # degrees per point
        self.dy = (y1 - y0) / (bb.height * fig.get_figheight() * 72)
        self.boxes = []

    def reserve_axes(self, fx0, fy0, fx1, fy1):
        """Keep a part of the map (axes fractions) free, e.g. under the legend or footers."""
        x0, x1, y0, y1 = self.ext
        self.boxes.append((x0 + fx0 * (x1 - x0), y0 + fy0 * (y1 - y0), x0 + fx1 * (x1 - x0), y0 + fy1 * (y1 - y0)))

    def reserve_point(self, x, y, size_pt=7):
        self.boxes.append((x - size_pt * self.dx, y - size_pt * self.dy, x + size_pt * self.dx, y + size_pt * self.dy))

    def _free(self, b):
        x0, x1, y0, y1 = self.ext
        if b[0] < x0 or b[2] > x1 or b[1] < y0 or b[3] > y1:
            return False
        return not any(b[0] < o[2] and b[2] > o[0] and b[1] < o[3] and b[3] > o[1] for o in self.boxes)

    def add(self, x, y, text, fs, gap_pt=8, skip=False, leader=True, near_only=False, **kw):
        n_lines = text.count("\n") + 1
        w = (0.6 * fs * max(len(s) for s in text.split("\n")) + 8) * self.dx
        h = (1.25 * fs * n_lines + 6) * self.dy
        g, gx, gy = gap_pt, gap_pt * self.dx, gap_pt * self.dy
        if near_only:  # city names: right next to the dot, right side first
            spots = [(gx, -h / 2), (gx, gy * 0.3), (gx, -h - gy * 0.3), (-gx - w, -h / 2), (-w / 2, gy), (-w / 2, -gy - h)]
        else:
            spots = [(gx, gy), (gx, -gy - h), (-gx - w, gy), (-gx - w, -gy - h),
                     (gx, -h / 2), (-gx - w, -h / 2), (-w / 2, gy), (-w / 2, -gy - h)]
        for mult in ((1,) if near_only else (1, 2.2, 3.6)):
            for ox, oy in spots:
                ox2 = ox + (np.sign(ox) if ox else 0) * (mult - 1) * gx * 2
                oy2 = oy + (np.sign(oy + h / 2) if abs(oy + h / 2) > 1e-9 else 0) * (mult - 1) * gy * 2
                b = (x + ox2, y + oy2, x + ox2 + w, y + oy2 + h)
                if self._free(b):
                    self.boxes.append(b)
                    tx, ty = b[0] + 4 * self.dx, b[1] + 3 * self.dy
                    if leader and mult > 1:
                        cxp, cyp = min(max(x, b[0]), b[2]), min(max(y, b[1]), b[3])
                        self.ax.plot([x, cxp], [y, cyp], "-", color="#555", lw=.7, transform=PC, zorder=28)
                    self.ax.text(tx, ty, text, fontsize=fs, ha="left", va="bottom", transform=PC,
                                 zorder=kw.pop("zorder", 30), clip_on=True, **kw)
                    return True
        if skip:
            return False
        b = (x + gx, y + gy, x + gx + w, y + gy + h)  # last resort for must-show labels
        self.boxes.append(b)
        self.ax.text(b[0] + 4 * self.dx, b[1] + 3 * self.dy, text, fontsize=fs, ha="left", va="bottom",
                     transform=PC, zorder=kw.pop("zorder", 30), clip_on=True, **kw)
        return True


def _cities(ax, labels, extent, fs=10.5):
    """City dots + names; a name that would cover a track label or another name is left out."""
    for n, a, b in CITIES:
        if extent[0] < b < extent[1] and extent[2] < a < extent[3]:
            ax.plot(b, a, "o", ms=5.5, color="black", transform=PC, zorder=25)
            labels.reserve_point(b, a, 4)
    for n, a, b in CITIES:
        if extent[0] < b < extent[1] and extent[2] < a < extent[3]:
            labels.add(b, a, n, fs, gap_pt=5, skip=True, leader=False, near_only=True, fontweight="bold",
                       bbox=CITY_BOX, zorder=26)


def _titles(ax, t1, t2, t3):
    top = dict(xy=(0.5, 1.0), xycoords="axes fraction", textcoords="offset points", ha="center", va="bottom")
    ax.annotate(t1, xytext=(0, 52), fontsize=20, fontweight="bold", **top)
    ax.annotate(t2, xytext=(0, 28), fontsize=14, fontweight="bold", color="#9b0000", **top)
    ax.annotate(t3, xytext=(0, 8), fontsize=12, color="#333", **top)


def _footers(ax):
    ax.text(0.008, 0.012, FOOT_L, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color="#9b0000",
            va="bottom", zorder=40, bbox=RED)
    ax.text(0.992, 0.012, FOOT_R, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color="#9b0000",
            va="bottom", ha="right", zorder=40, bbox=RED)


def _save(fig, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    main_ax = next(a for a in fig.axes if hasattr(a, "projection"))
    add_logo_left(fig, main_ax, title_mid_pt=36)
    plt.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------
# LPA DESK FRAME
# ------------------------------------------------------------

MAX_LPA_SYSTEMS = 3


def lpa_extent(ens_sys, links):
    """One view for all frames of a cycle: zoomed on the main system (L1); others show if they fall inside."""
    pts = []
    for i, s in enumerate(ens_sys[:1]):
        pts += [(p["lon"], p["lat"]) for p in s["mean"].values()]
        d = links.get(i)
        if d is not None:
            pts += [(p["lon"], p["lat"]) for p in d["mean"].values()]
    return zoom_extent(pts)


def lpa_frame(out, init, k, sst, det_fields, ens_msl, ens_sys, det_sys, links, show_sst=True, extent=None):
    lead = STEPS[k]
    ens_sys = ens_sys[:MAX_LPA_SYSTEMS]
    extent = extent or LPA_EXTENT
    valid = init + timedelta(hours=lead)
    fig = plt.figure(figsize=(16, 12), facecolor="white")
    ax = fig.add_axes([0.05, 0.10, 0.82, 0.76], projection=PC)
    cf = _base(ax, extent, sst if show_sst else None)
    labels = Labels(ax)
    labels.reserve_axes(0, 0, 1, 0.075)          # footers
    labels.reserve_axes(0, 0.075, 0.27, 0.30)    # legend

    lv = np.arange(990, 1022, 2)
    if det_fields is not None and np.isfinite(det_fields["msl"][k]).any():
        cs = ax.contour(TRACK.LON2, TRACK.LAT2, det_fields["msl"][k], levels=lv, colors="black",
                        linewidths=1.4, transform=PC, zorder=17)
        ax.clabel(cs, fontsize=9.5, fmt="%d", inline_spacing=4)
    if ens_msl is not None and np.isfinite(ens_msl[k]).any():
        cs2 = ax.contour(TRACK.LON2, TRACK.LAT2, ens_msl[k], levels=lv, colors="#d62728", linewidths=1.0,
                         linestyles="--", transform=PC, zorder=17)
        ax.clabel(cs2, fontsize=8.5, fmt="%d", inline_spacing=4)
    if det_fields is not None and np.isfinite(det_fields["u10"][k]).any():
        s = max(1, int((extent[1] - extent[0]) / 30 / TRACK.step))   # about 30 barbs across
        kt = 1.94384
        ax.barbs(TRACK.LON2[::s, ::s], TRACK.LAT2[::s, ::s], det_fields["u10"][k][::s, ::s] * kt,
                 det_fields["v10"][k][::s, ::s] * kt, length=5.5, linewidth=0.55, color="#333", alpha=.7,
                 transform=PC, zorder=18, barb_increments={"half": 5, "full": 10, "flag": 50})

    # a system is drawn only until its consensus track ends (crossed the coast and weakened)
    alive = [(n, s) for n, s in enumerate(ens_sys, 1) if k <= max(s["mean"])]

    # member tracks: faint, so the consensus tracks stay visible
    for _, sysm in alive:
        for _, _, _, t in sorted(sysm["members"], key=lambda x: -x[0])[:40]:
            ax.plot([p["lon"] for p in t], [p["lat"] for p in t], "-", color="#7a7a7a", lw=.6, alpha=.35,
                    transform=PC, zorder=19)

    pending = []   # labels, placed after every marker is down so none of them gets covered
    for n, sysm in alive:
        mean = sysm["mean"]
        ks = sorted(mean)
        lo, la = [mean[j]["lon"] for j in ks], [mean[j]["lat"] for j in ks]
        d = links.get(n - 1)
        if d is not None:
            dk = sorted(d["mean"])
            dlo, dla = [d["mean"][j]["lon"] for j in dk], [d["mean"][j]["lat"] for j in dk]
            ax.plot(dlo, dla, "--", color="#1f4fb5", lw=2.4, transform=PC, zorder=21, path_effects=WHITE_EDGE)
            ax.plot(dlo, dla, "s", ms=6, mfc="white", mec="#1f4fb5", mew=1.5, transform=PC, zorder=21)
        ax.plot(lo, la, "-", color="#d62728", lw=3.4, transform=PC, zorder=22, path_effects=WHITE_EDGE)
        ax.plot(lo, la, "o", ms=7.5, mfc="white", mec="#d62728", mew=1.8, transform=PC, zorder=23)
        for j in ks:
            labels.reserve_point(mean[j]["lon"], mean[j]["lat"], 6)
        if k in mean:
            c = mean[k]
            ax.plot(c["lon"], c["lat"], "o", ms=24, mfc="#d62728", mec="white", mew=2.5, transform=PC, zorder=31)
            ax.text(c["lon"], c["lat"], "L", fontsize=14, fontweight="bold", color="white", ha="center",
                    va="center", transform=PC, zorder=32, clip_on=True)
            labels.reserve_point(c["lon"], c["lat"], 14)
            pending.append((0, c["lon"], c["lat"], f"L{n} ({c['msl']:.0f} hPa)", 12, dict(BOX, lw=1.4)))
        for j in ks:
            if STEPS[j] % 24 == 0 and STEPS[j] > 0 and j != k:
                pending.append((1, mean[j]["lon"], mean[j]["lat"], f"+{STEPS[j]}h ({mean[j]['msl']:.0f})", 10, BOX))

    for _, x, y, text, fs, box in sorted(pending, key=lambda p: p[0]):
        labels.add(x, y, text, fs, gap_pt=9, fontweight="bold", color="#9b0000", bbox=box)
    _cities(ax, labels, extent)

    handles = [plt.Line2D([], [], color="#d62728", lw=3.4, marker="o", mfc="white", label="Ensemble weighted mean track"),
               plt.Line2D([], [], color="#1f4fb5", lw=2.4, ls="--", marker="s", mfc="white", label="Deterministic weighted track"),
               plt.Line2D([], [], color="#7a7a7a", lw=1, label="Ensemble member tracks"),
               plt.Line2D([], [], color="black", lw=1.4, label="MSLP — deterministic blend (hPa)"),
               plt.Line2D([], [], color="#d62728", lw=1.0, ls="--", label="MSLP — ensemble mean (hPa)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.085), fontsize=9.5,
              framealpha=.94).set_zorder(40)
    if not alive:
        msg = (f"No active low at +{lead}h — earlier systems have weakened" if ens_sys
               else "No persistent low in the ensemble consensus")
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center",
                fontsize=15, fontweight="bold", color="#9b0000", bbox=RED, zorder=40)
    _footers(ax)
    _titles(ax, "MASRainman LPA Desk — Multi-model Ensemble & Deterministic Tracker",
            f"Weighted Multi-Lead Tracking & Synoptic MSLP (+{lead}h) — North Indian Ocean",
            f"Init: {init:%Y-%m-%d %H:%M}Z | Valid: {valid:%Y-%m-%d %H:%M}Z (+{lead}h) | "
            f"SST background: {'ON' if show_sst else 'OFF'}")
    if cf is not None:
        bb = ax.get_position()
        cax = fig.add_axes([bb.x1 + 0.02, bb.y0 + 0.03, 0.022, bb.height - 0.06])
        cb = fig.colorbar(cf, cax=cax, ticks=SST_LV, extendfrac=0.05)
        cb.set_label("Sea Surface Temperature (°C)", fontsize=13, fontweight="bold")
    _save(fig, out)


# ------------------------------------------------------------
# CYCLONE TRACKER
# ------------------------------------------------------------

def cyclone_map(out, init, n, sysm, det, probs, timing, sst=None):
    mean = sysm["mean"]
    ks = sorted(mean)
    pts = [(mean[j]["lon"], mean[j]["lat"]) for j in ks]
    ext = zoom_extent(pts, aspect=1.3, pad=(6.0, 5.0))

    fig = plt.figure(figsize=(16, 13), facecolor="white")
    ax = fig.add_axes([0.05, 0.10, 0.82, 0.76], projection=PC)
    _base(ax, ext, None)
    labels = Labels(ax)
    labels.reserve_axes(0, 0, 1, 0.075)          # footers
    labels.reserve_axes(0, 0.78, 0.42, 1)        # landfall box
    labels.reserve_axes(0.62, 0.075, 1, 0.27)    # legend

    for _, _, _, t in sysm["members"]:
        ax.plot([p["lon"] for p in t], [p["lat"] for p in t], "-", color="#6d7b8a", lw=.55, alpha=.35,
                transform=PC, zorder=4)
    cone = unary_union([Point(mean[j]["lon"], mean[j]["lat"]).buffer(max(mean[j]["spread"], 40) / 111)
                        for j in ks])
    ax.add_geometries([cone], PC, facecolor="#ffd8a8", edgecolor="#e07b00", alpha=.5, lw=1.4, zorder=3)

    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="black", lw=2.8, transform=PC, zorder=20,
            path_effects=WHITE_EDGE)
    pending = []
    for j in ks:
        wind = det["mean"][j]["wind"] if det is not None and j in det["mean"] else mean[j]["wind"]
        name, _, col = category(wind)
        ax.plot(mean[j]["lon"], mean[j]["lat"], "o", ms=13, mfc=col, mec="black", mew=1.3, transform=PC, zorder=21)
        labels.reserve_point(mean[j]["lon"], mean[j]["lat"], 8)
        if STEPS[j] % 24 == 0:
            wtxt = f"{wind:.0f} kt" if wind is not None else "—"
            pending.append((mean[j]["lon"], mean[j]["lat"], f"T+{STEPS[j]} h · {name}\n{wtxt} · {mean[j]['msl']:.0f} hPa",
                            dict(boxstyle="round,pad=.25", fc="white", ec=col, lw=1.2, alpha=.95)))
    for x, y, text, box in pending:
        labels.add(x, y, text, 9.5, gap_pt=10, fontweight="bold", bbox=box)
    _cities(ax, labels, ext, fs=10)

    lines = [f"System L{n} · support {sysm['support'] * 100:.0f}% of ensemble weight", "Landfall probability:"]
    lines += [f"  {k:<36} {v * 100:4.0f}%" + (f"  ~T+{timing[k]} h" if k in timing else "") for k, v in probs[:5]]
    ax.text(0.012, 0.985, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#888", alpha=.94), zorder=40)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, mfc=c, mec="black", label=nm) for nm, _, c in CAT[1:7]]
    handles += [plt.Line2D([], [], color="#6d7b8a", lw=1, label="Ensemble member tracks"),
                plt.Rectangle((0, 0), 1, 1, fc="#ffd8a8", ec="#e07b00", label="Uncertainty cone (weighted spread)")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.995, 0.085), fontsize=9.5, ncol=2,
              framealpha=.94, title="IMD category (deterministic-blend 10-m wind)", title_fontsize=9).set_zorder(40)
    _footers(ax)
    _titles(ax, "MASRainman Cyclone Desk — Multi-model Ensemble Tracker",
            f"System L{n}: Weighted Consensus Track & Intensity T+0 → T+{STEPS[ks[-1]]}h",
            f"Init: {init:%Y-%m-%d %H:%M}Z | Models: {', '.join(C.MODEL_NAMES[m] for m in sysm['models'])}")
    _save(fig, out)
