# ============================================================
# EW INDEX — "Precipitation, 24h (EW index)" maps for South India and North India
# ============================================================
#
# Like the meteologix / ECMWF "EW index" chart: how extreme the model's 24-h rain is
# compared with the normal rain climate of that place and time of year.
#
#   0      the forecast is an ordinary day for this place and season
#   0.5    clearly unusual rain
#   0.8+   very unusual / extreme for this place and season
#   1      wetter than anything in the 1991-2020 climate
#
# It is the ECMWF Extreme Forecast Index formula (efi_core.py) applied to one forecast
# (a deterministic run counts as a single member):
#     EFI = 2/pi * integral_0^1 (p - F_f(q_c(p))) / sqrt(p (1 - p)) dp
# Climate: ERA5 1991-2020 daily 24-h rain, 0.25°, ±15 days around the date
# (efi_climate.py, kept as assets of the GitHub release "efi-climate").
# The model's 24-h rain is averaged onto the same 0.25° grid first, so a 0.1° IFS
# field is not compared with a smoother 0.25° climate.
# Where the model gives less than MIN_RAIN_MM in the 24 h the index is not shown
# (in a dry season a drizzle can be "unusual" without mattering).
#
# render.py calls draw_ew_maps() for every 24-h frame of every model with rain:
#   ew_south_T048.png, ew_north_T048.png  (+ GIF loops)
# ============================================================

import os
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import masrainman_logo  # noqa: F401  (MasRainman logo on every chart, also from the multi-model job)
import efi_core
import efi_climate

IST = timedelta(hours=5, minutes=30)
MIN_RAIN_MM = float(os.environ.get("EW_MIN_RAIN_MM", "1.0"))

REGIONS = {
    "ew_south": {
        "name": "South India",
        "extent": (68.0, 94.0, 5.0, 21.0),     # lon0, lon1, lat0, lat1
        "figsize": (11.5, 8.6),
        "cities": [
            ("Chennai", 13.08, 80.27), ("Bengaluru", 12.97, 77.59), ("Hyderabad", 17.39, 78.49),
            ("Thiruvananthapuram", 8.52, 76.94), ("Kochi", 9.93, 76.27), ("Kozhikode", 11.26, 75.78),
            ("Mangaluru", 12.91, 74.86), ("Madurai", 9.93, 78.12), ("Coimbatore", 11.02, 76.96),
            ("Thanjavur", 10.79, 79.14), ("Puducherry", 11.94, 79.81), ("Nellore", 14.44, 79.99),
            ("Tirupati", 13.63, 79.42), ("Vijayawada", 16.51, 80.65), ("Visakhapatnam", 17.69, 83.22),
            ("Kurnool", 15.83, 78.04), ("Ballari", 15.14, 76.92), ("Belagavi", 15.85, 74.50),
            ("Mysuru", 12.30, 76.64), ("Kanyakumari", 8.08, 77.54), ("Panaji", 15.49, 73.83),
            ("Solapur", 17.66, 75.91), ("Bhubaneswar", 20.30, 85.82), ("Colombo", 6.93, 79.86),
            ("Jaffna", 9.66, 80.01), ("Trincomalee", 8.59, 81.21), ("Port Blair", 11.62, 92.73),
        ],
    },
    "ew_north": {
        "name": "North India",
        "extent": (66.0, 98.0, 20.0, 37.5),
        "figsize": (12.5, 8.2),
        "cities": [
            ("New Delhi", 28.61, 77.21), ("Jaipur", 26.91, 75.79), ("Lucknow", 26.85, 80.95),
            ("Patna", 25.59, 85.14), ("Kolkata", 22.57, 88.36), ("Guwahati", 26.14, 91.74),
            ("Bhopal", 23.26, 77.41), ("Ahmedabad", 23.02, 72.57), ("Chandigarh", 30.73, 76.78),
            ("Srinagar", 34.08, 74.80), ("Leh", 34.15, 77.58), ("Shimla", 31.10, 77.17),
            ("Dehradun", 30.32, 78.03), ("Amritsar", 31.63, 74.87), ("Jodhpur", 26.24, 73.02),
            ("Bikaner", 28.02, 73.31), ("Varanasi", 25.32, 82.97), ("Ranchi", 23.34, 85.31),
            ("Raipur", 21.25, 81.63), ("Jabalpur", 23.18, 79.99), ("Indore", 22.72, 75.86),
            ("Agra", 27.18, 78.01), ("Gangtok", 27.33, 88.61), ("Shillong", 25.58, 91.89),
            ("Imphal", 24.82, 93.94), ("Udaipur", 24.59, 73.71), ("Bhuj", 23.24, 69.67),
            ("Kathmandu", 27.72, 85.32), ("Dhaka", 23.81, 90.41),
        ],
    },
}
LAYERS = list(REGIONS)

# Colour bands like the meteologix EW index chart: white -> light blue -> blue -> navy -> purple -> magenta
EW_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0001]
EW_COLOURS = ["#e3eefb", "#c6dcf6", "#a9cbf2", "#8cb9ee", "#6aa6ea", "#3f8fe6", "#1f6fd1",
              "#1553ad", "#0f3a8a", "#141f6b", "#3b0f7a", "#6a0d8f", "#a015b0"]
EW_TICKS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

_CLIM = {}


def climate_for(valid_mid):
    """(q (101, ny, nx), lat, lon) of the ±15-day climate window nearest the date, or None."""
    w = efi_climate.nearest_window(valid_mid.date())
    if w not in _CLIM:
        _CLIM[w] = efi_climate.load_clim("rain24", w)
    return _CLIM[w]


def available():
    """True when the rain climate files are present (downloaded from the 'efi-climate' release)."""
    d = efi_climate.CLIM_DIR
    return os.path.isdir(d) and any(f.startswith("rain24_") for f in os.listdir(d))


def to_climate_grid(rain, rlat, rlon, clat, clon):
    """Average a model rain field onto the climate's 0.25° cells (box mean; NaN where no model point)."""
    rlat = np.asarray(rlat, float)
    rlon = np.asarray(rlon, float)
    dy = float(clat[1] - clat[0])
    dx = float(clon[1] - clon[0])
    iy = np.rint((rlat - clat[0]) / dy).astype(int)
    ix = np.rint((rlon - clon[0]) / dx).astype(int)
    oky = (iy >= 0) & (iy < len(clat))
    okx = (ix >= 0) & (ix < len(clon))
    sub = np.nan_to_num(np.asarray(rain, float)[np.ix_(oky, okx)], nan=0.0)
    YY, XX = np.meshgrid(iy[oky], ix[okx], indexing="ij")
    flat = (YY * len(clon) + XX).ravel()
    n = len(clat) * len(clon)
    tot = np.bincount(flat, weights=sub.ravel(), minlength=n)
    cnt = np.bincount(flat, minlength=n)
    out = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return out.reshape(len(clat), len(clon)).astype(np.float32)


def ew_field(rain24, rlat, rlon, valid_mid):
    """EW index on the climate grid. rain24 is one 24-h rain field (ny, nx), or ensemble members
    (N, ny, nx) for a true ensemble EFI. Returns (ew, rain_on_grid, lat, lon) or None;
    rain_on_grid is the (ensemble-mean) rain, used for the 'shown where >= MIN_RAIN_MM' mask."""
    clim = climate_for(valid_mid)
    if clim is None:
        return None
    q, clat, clon = clim
    fields = np.asarray(rain24)
    if fields.ndim == 2:
        fields = fields[None]
    members = np.stack([to_climate_grid(f, rlat, rlon, clat, clon) for f in fields])
    r = members.mean(axis=0)
    ew = efi_core.efi_from_members(q, np.nan_to_num(members))
    ew = np.where(np.isfinite(r) & (r >= MIN_RAIN_MM), ew, np.nan)
    return ew, r, clat, clon


def _cmap():
    cmap = ListedColormap(EW_COLOURS)
    cmap.set_under((1, 1, 1, 0))
    return cmap, BoundaryNorm(EW_LEVELS, cmap.N)


def _smooth(lon, lat, f, factor=4):
    from scipy.ndimage import zoom
    g = np.nan_to_num(f, nan=-0.2)
    s = zoom(g, factor, order=1)
    return (np.linspace(lon[0], lon[-1], s.shape[1]), np.linspace(lat[0], lat[-1], s.shape[0]), s)


def draw_ew_map(out_png, layer, model_label, credits, borders, init_time, start_h, end_h, field):
    reg = REGIONS[layer]
    lon0, lon1, lat0, lat1 = reg["extent"]
    ew, rain, lat, lon = field
    jj = (lat >= lat0 - 1) & (lat <= lat1 + 1)
    ii = (lon >= lon0 - 1) & (lon <= lon1 + 1)
    ew_r, lat_r, lon_r = ew[np.ix_(jj, ii)], lat[jj], lon[ii]
    rain_r = rain[np.ix_(jj, ii)]
    cmap, norm = _cmap()

    fig = plt.figure(figsize=reg["figsize"], facecolor="white")
    ax = fig.add_axes([0.01, 0.19, 0.98, 0.78], projection=ccrs.PlateCarree())
    ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#ffffff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ffffff", zorder=0)

    slon, slat, sew = _smooth(lon_r, lat_r, ew_r)
    ax.contourf(slon, slat, sew, levels=EW_LEVELS, cmap=cmap, norm=norm,
                antialiased=True, transform=ccrs.PlateCarree(), zorder=1)
    cs = ax.contour(slon, slat, sew, levels=[0.5, 0.7, 0.9], colors=["#7a7a7a", "#555555", "#ffffff"],
                    linewidths=[0.5, 0.6, 0.6], transform=ccrs.PlateCarree(), zorder=2)
    ax.clabel(cs, fmt="%.1f", fontsize=7.5, inline=True)

    ax.coastlines(resolution="50m", linewidth=0.9, color="black", zorder=5)
    ax.add_feature(borders, facecolor="none", edgecolor="black", linewidth=0.8, zorder=6)
    ax.add_feature(cfeature.STATES.with_scale("50m"), facecolor="none", edgecolor="#9a9a9a",
                   linewidth=0.35, zorder=4)

    halo = [pe.withStroke(linewidth=2.2, foreground="black")]
    for name, la, lo in reg["cities"]:
        if lon0 < lo < lon1 and lat0 < la < lat1:
            ax.plot(lo, la, "s", ms=4.2, mfc="white", mec="black", mew=0.8,
                    transform=ccrs.PlateCarree(), zorder=8)
            ax.text(lo, la + 0.16, name, ha="center", va="bottom", fontsize=8.2, color="white",
                    path_effects=halo, transform=ccrs.PlateCarree(), zorder=9)

    # Strongest signal in the region (index and the model's own 24-h rain there)
    inside = ((lat_r[:, None] >= lat0) & (lat_r[:, None] <= lat1) &
              (lon_r[None, :] >= lon0) & (lon_r[None, :] <= lon1))
    ok = np.isfinite(ew_r) & inside
    peak = ""
    if ok.any():
        k = np.nanargmax(np.where(ok, ew_r, -9))
        py, px = np.unravel_index(k, ew_r.shape)
        if ew_r[py, px] >= 0.1:
            peak = (f"Max EW {ew_r[py, px]:.2f} at {lat_r[py]:.1f}N {lon_r[px]:.1f}E "
                    f"({rain_r[py, px]:.0f} mm in 24 h)")
            ax.plot(lon_r[px], lat_r[py], marker="x", ms=9, mew=2.2, color="#ff2fd0",
                    transform=ccrs.PlateCarree(), zorder=10)

    t0 = init_time + timedelta(hours=start_h)
    t1 = init_time + timedelta(hours=end_h)
    fmt = lambda t: (t + IST).strftime("%a %d %b %Y, %I:%M %p IST")
    fig.text(0.012, 0.137, "Precipitation, 24h (EW index)", ha="left", va="center",
             fontsize=17, fontweight="bold")
    fig.text(0.988, 0.152, f"From {fmt(t0)}", ha="right", va="center", fontsize=9.5)
    fig.text(0.988, 0.124, f"to {fmt(t1)}", ha="right", va="center", fontsize=9.5)
    fig.text(0.012, 0.985, f"{reg['name']} • {model_label} • Init {init_time.strftime('%Hz %d %b %Y')} "
             f"• T+{start_h} to T+{end_h}", ha="left", va="center", fontsize=11, fontweight="bold")

    cax = fig.add_axes([0.04, 0.055, 0.92, 0.022])
    n = len(EW_COLOURS)
    cax.imshow(np.arange(n)[None, :], cmap=cmap, norm=plt.Normalize(-0.5, n - 0.5),
               aspect="auto", extent=(0, n, 0, 1))
    cax.set_yticks([])
    cax.set_xticks(range(n))
    cax.set_xticklabels([f"{t:g}" for t in EW_TICKS], fontsize=8.5)
    cax.xaxis.set_ticks_position("top")
    cax.tick_params(length=0, pad=2)
    fig.text(0.04, 0.035, peak or "No unusual rain (EW index below 0.1 everywhere)",
             ha="left", va="center", fontsize=8.6, fontweight="bold", color="#3b0f7a")
    fig.text(0.5, 0.018, f"{credits[0]} | Climate: ERA5 1991–2020 (Copernicus/ECMWF, via WeatherBench 2) "
             f"| EW index (ECMWF EFI formula) & chart: MasRainman", ha="center", fontsize=6.6, color="dimgray")
    fig.text(0.5, 0.004, f"0 = normal day for the place and season, 1 = wetter than any day in 1991–2020. "
             f"Shown where the model gives ≥{MIN_RAIN_MM:g} mm. India's border as per the Government of India.",
             ha="center", fontsize=6.6, color="dimgray")

    plt.savefig(out_png, dpi=170, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def draw_ew_maps(out_dir, data, init_time, h, model_label, credits, borders, period_h=24):
    """Both regions for the 24 h ending at T+h. Returns {layer: filename} (empty if no climate)."""
    hours = data["hours"]
    start = h - period_h
    if start < 0 or (start > 0 and start not in hours) or h not in hours:
        return {}
    rain = data["rain"][hours.index(h)] - (data["rain"][hours.index(start)] if start > 0 else 0)
    rain = np.maximum(rain, 0)
    field = ew_field(rain, data["rlat"], data["rlon"], init_time + timedelta(hours=h - period_h / 2))
    if field is None:
        return {}
    out = {}
    for layer in LAYERS:
        fname = f"{layer}_T{h:03d}.png"
        draw_ew_map(os.path.join(out_dir, fname), layer, model_label, credits, borders,
                    init_time, start, h, field)
        out[layer] = fname
    return out
