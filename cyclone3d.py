"""
3D cyclone / heavy-rain view from the GPM satellite radar (NASA/JAXA), for the WeatherNext 3 site.

The GPM radar (Ku band, product 2A.Ku) measures rain at every height, from the sea up to about 18 km,
in a 245 km wide strip under the satellite. Once a day this script looks at the radar passes of the
last 24 hours over the Bay of Bengal, the Arabian Sea and South India, keeps the (up to three) passes
with the most heavy rain, and for each makes:

  <id>.mp4   the rain in 3D, the camera turning around it: coloured by rain rate, then by
             rain / mixed rain and snow / snow and ice
  <id>.png   one still picture
  <id>.txt   a short caption
  where <id> = gpm3d_YYYY-MM-DD_HHMM (time of the pass, IST)

and adds them to index.json in the output folder (newest first, the last 30 passes kept).
The site's "3D cyclone view" page (web/cyclone3d.html) reads that folder.

Data: NASA GES DISC, found with NASA's CMR search. Needs a free NASA Earthdata login in the
environment: EARTHDATA_USER and EARTHDATA_PASS.

  python cyclone3d.py --out cyclone3d          # real data
  python cyclone3d.py --out /tmp/c3d --demo    # made-up storm, to check the pictures
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw", "gpm")
IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
RADAR = "GPM_2AKu"                       # GPM radar, Ku band, 3D rain profile
AREA = (70.0, 4.0, 95.0, 24.0)           # W, S, E, N: Arabian Sea, South India, Bay of Bengal
HALF = 4.5                               # each picture: ± this many degrees along the track around the storm
MAX_PASSES = 3
KEEP = 30

W, H = 1080, 1350
FPS = 20
BG = "#0b0e14"
TEXT = "#eceff4"
SOFT = "#c6ccd6"
MUTED = "#8a92a0"
GOLD = "#e8a33d"

RATE_B = [0.1, 0.5, 1, 2, 3, 5, 10, 15, 20, 25, 1000]       # mm/h, like JAXA's scale
RATE_C = ["#1a1aa8", "#1f5cff", "#29a3ff", "#26d6c8", "#4fd94f", "#e6e633", "#ffb31a", "#ff7a1a", "#f03c1e", "#a30f14"]
RATE_CMAP = ListedColormap(RATE_C)
RATE_NORM = BoundaryNorm(RATE_B, RATE_CMAP.N)
PHASE_C = {0: "#f2f4f8", 1: "#5fd0ff", 2: "#2035ff"}          # 0 snow/ice, 1 mixed, 2 rain
PHASE_NAME = {2: "Rain", 1: "Mixed rain and snow", 0: "Snow and ice"}
STATE_FILES = ["tn_districts.geojson", "ka_districts.geojson", "ap_districts.geojson", "kl_districts.geojson"]


# ----------------------------------------------------------------- fonts, header, helpers
def _fonts():
    fam = "DejaVu Sans"
    for fn in ("MuktaMalar-Regular.ttf", "MuktaMalar-SemiBold.ttf", "MuktaMalar-Bold.ttf"):
        p = os.path.join(HERE, "fonts", fn)
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            fam = font_manager.FontProperties(fname=p).get_name()
    plt.rcParams["font.family"] = [fam, "DejaVu Sans"]


_fonts()


def header(fig, kicker, title, sub, logo_h=104, gap=26):
    """Kicker, title and time lines with the logo just to their right; the whole group centred."""
    rows = [(0.955, kicker, GOLD, 20), (0.918, title, TEXT, 34), (0.888, sub, SOFT, 19)]
    texts = [fig.text(0.5, y, s, color=c, fontsize=fs, weight="bold" if fs > 19 else "normal", ha="center")
             for y, s, c, fs in rows]
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in texts]
    wt = max(b.width for b in boxes)
    im = None
    p = os.path.join(HERE, "assets", "masrainman_logo_dark.png")
    if os.path.exists(p):
        from PIL import Image
        im = Image.open(p).convert("RGBA")
        im = im.resize((round(im.width * logo_h / im.height), logo_h), Image.LANCZOS)
    lw = im.width + gap if im is not None else 0
    left = (W - (wt + lw)) / 2
    for t in texts:
        t.set_x((left + wt / 2) / W)
    if im is not None:
        top, bot = max(b.y1 for b in boxes), min(b.y0 for b in boxes)
        fig.figimage(np.asarray(im), xo=int(left + wt + gap), yo=int((top + bot) / 2 - logo_h / 2), zorder=10)


def hex2rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def t_ist(t):
    return t.astimezone(IST).strftime("%I:%M %p, %d %b %Y").lstrip("0").replace(" 0", " ")


def note(msg, level="notice"):
    """Print, and on GitHub Actions also show it on the run page."""
    print(("::%s::" % level if os.environ.get("GITHUB_ACTIONS") else "") + msg)


def load_geo(fn):
    p = os.path.join(HERE, "boundaries", "states", fn)
    if not os.path.exists(p):
        return []
    from shapely.geometry import shape
    out = []
    for f in json.load(open(p, encoding="utf-8"))["features"]:
        pr = f.get("properties") or {}
        name = pr.get("district") or pr.get("DISTRICT") or pr.get("name") or "?"
        out.append((str(name).title(), shape(f["geometry"])))
    return out


def lines_of(geom):
    """Every outline / line of a geometry as Nx2 arrays."""
    t = geom.geom_type
    if t == "Polygon":
        yield np.asarray(geom.exterior.coords)
    elif t == "LineString":
        yield np.asarray(geom.coords)
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            yield from lines_of(g)


def base_lines():
    """(lines, colour, width): coastline, India's border, South Indian districts and state edges."""
    out = []
    try:
        import cartopy.io.shapereader as shpreader
        rd = shpreader.Reader(shpreader.natural_earth("50m", "physical", "coastline"))
        from shapely.geometry import box as sbox
        b = sbox(AREA[0] - 3, AREA[1] - 3, AREA[2] + 3, AREA[3] + 3)
        out.append(([l for g in rd.geometries() if g.intersects(b) for l in lines_of(g.intersection(b))],
                    "#b8c2d4", 1.0))
    except Exception as e:  # noqa: BLE001
        print("  (no coastline:", e, ")")
    p = os.path.join(HERE, "boundaries", "borders_india_view.geojson")
    if os.path.exists(p):
        from shapely.geometry import shape
        out.append(([l for f in json.load(open(p, encoding="utf-8"))["features"]
                     for l in lines_of(shape(f["geometry"]))], "#8f9bb0", 0.8))
    from shapely.ops import unary_union
    for fn in STATE_FILES:
        geo = load_geo(fn)
        if geo:
            out.append(([l for _, g in geo for l in lines_of(g)], "#4f5869", 0.5))
            out.append((list(lines_of(unary_union([g for _, g in geo]))), "#dfe4ec", 1.2))
    return out


def where_is(la, lo, districts):
    """'Srikakulam district', 'Bay of Bengal, off Srikakulam', 'Arabian Sea' …"""
    from shapely.geometry import Point
    p = Point(lo, la)
    best = None
    for n, g in districts:
        if g.contains(p):
            return f"over {n} district"
        d = g.distance(p)
        if best is None or d < best[0]:
            best = (d, n)
    sea = "Bay of Bengal" if lo >= 79.5 or (lo >= 78 and la < 9) else "Arabian Sea" if lo <= 77.5 else "Indian Ocean"
    if la < 6 and 78 <= lo <= 82:
        sea = "Indian Ocean, south of Sri Lanka"
    if best and best[0] < 1.6:
        return f"{sea}, off {best[1]}"
    return sea


class Video:
    """Raw RGB frames → H.264 MP4 through ffmpeg."""

    def __init__(self, path, w=W, h=H, fps=FPS):
        exe = shutil.which("ffmpeg")
        if not exe:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        self.p = subprocess.Popen([exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                   "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "veryfast",
                                   "-movflags", "+faststart", path], stdin=subprocess.PIPE)

    def add(self, fig):
        fig.canvas.draw()
        self.p.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")


# ----------------------------------------------------------------- NASA access
class EarthdataSession:
    """requests session that keeps the NASA login across GES DISC's redirect to urs.earthdata.nasa.gov."""

    def __init__(self):
        import requests
        user, pw = os.environ.get("EARTHDATA_USER"), os.environ.get("EARTHDATA_PASS")
        if not user or not pw:
            raise SystemExit("NASA Earthdata login missing: set the EARTHDATA_USER and EARTHDATA_PASS secrets.")

        class S(requests.Session):
            AUTH_HOST = "urs.earthdata.nasa.gov"

            def rebuild_auth(self, prepared, response):
                if "Authorization" in prepared.headers:
                    a, b = urlparse(response.request.url).hostname, urlparse(prepared.url).hostname
                    if a != b and self.AUTH_HOST not in (a, b):
                        del prepared.headers["Authorization"]

        self.s = S()
        self.s.auth = (user, pw)

    def get(self, url, dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        for attempt in range(3):
            try:
                with self.s.get(url, stream=True, timeout=300) as r:
                    if r.status_code == 401:
                        raise SystemExit("NASA Earthdata login was refused (HTTP 401): check EARTHDATA_USER / "
                                         "EARTHDATA_PASS, and that 'NASA GESDISC DATA ARCHIVE' is approved in the "
                                         "Earthdata profile (Applications → Authorized Apps).")
                    r.raise_for_status()
                    if "text/html" in r.headers.get("content-type", ""):
                        raise RuntimeError("got a web page instead of data (login not accepted?)")
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                os.replace(tmp, dest)
                return dest
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"    download failed ({e}); try {attempt + 1}/3")
        raise RuntimeError("could not download " + url)


def cmr_search(short_name, t0, t1, bbox):
    """Granules (title, start, end, https data link), oldest first; newest version per start time."""
    import requests
    params = {"short_name": short_name, "page_size": 200, "sort_key": "start_date",
              "temporal": f"{t0:%Y-%m-%dT%H:%M:%SZ},{t1:%Y-%m-%dT%H:%M:%SZ}",
              "bounding_box": ",".join(f"{v:.2f}" for v in bbox)}
    r = requests.get(CMR, params=params, timeout=60)
    r.raise_for_status()
    out = {}
    for e in r.json()["feed"]["entry"]:
        link = next((l["href"] for l in e.get("links", [])
                     if l.get("rel", "").endswith("/data#") and l["href"].startswith("https")
                     and not l["href"].endswith((".xml", ".html"))), None)
        if not link:
            continue
        st = datetime.fromisoformat(e["time_start"].replace("Z", "+00:00"))
        en = datetime.fromisoformat(e["time_end"].replace("Z", "+00:00"))
        if st not in out or e["title"] > out[st][0]:
            out[st] = (e["title"], st, en, link)
    return [out[k] for k in sorted(out)]


# ----------------------------------------------------------------- radar passes
def radar_read(path):
    import h5py
    with h5py.File(path, "r") as f:
        g = f["FS" if "FS" in f else "NS"]
        lat, lon = g["Latitude"][:], g["Longitude"][:]
        rate = g["SLV/precipRate"][:].astype("f4")
        surf = g["SLV/precipRateNearSurface"][:].astype("f4")
        phase = g["DSD/phase"][:] if "DSD/phase" in g else None
        st = g["ScanTime"]
        times = [datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), tzinfo=UTC)
                 for y, mo, d, h, mi, s in zip(st["Year"][:], st["Month"][:], st["DayOfMonth"][:],
                                               st["Hour"][:], st["Minute"][:], st["Second"][:])]
        nb = rate.shape[2]
        if "PRE/height" in g:
            hgt = g["PRE/height"][:].astype("f4") / 1000.0
        else:  # bins are 125 m apart; the last bin is at the ellipsoid
            zen = g["PRE/localZenithAngle"][:] if "PRE/localZenithAngle" in g else np.zeros(lat.shape)
            hgt = ((nb - 1 - np.arange(nb))[None, None, :] * 0.125 * np.cos(np.radians(zen))[..., None]).astype("f4")
    rate[rate < 0] = 0
    surf[surf < 0] = 0
    return dict(lat=lat, lon=lon, rate=rate, surf=surf, phase=phase, hgt=hgt, times=times)


def storm_cut(d):
    """The ±HALF° stretch of the pass around its heaviest rain inside AREA; None if little rain."""
    lat, lon, surf = d["lat"], d["lon"], d["surf"]
    inside = (lon >= AREA[0]) & (lon <= AREA[2]) & (lat >= AREA[1]) & (lat <= AREA[3])
    if inside.sum() < 50:
        return None
    s = np.where(inside, surf, 0).sum(axis=1)
    if s.max() <= 0:
        return None
    k = np.convolve(s, np.ones(25), mode="same")              # ~ 120 km along the track
    i = int(np.argmax(k))
    clat = float(np.nanmean(lat[i]))
    mid = np.nanmean(lat, axis=1)
    near = np.where((np.abs(mid - clat) <= HALF) & inside.any(axis=1))[0]
    near = near[(near >= i - 900) & (near <= i + 900)]        # same stretch of the orbit
    if len(near) < 10:
        return None
    sl = slice(near[0], near[-1] + 1)
    c = {k_: (v[sl] if isinstance(v, np.ndarray) else v) for k_, v in d.items() if k_ != "times"}
    c["times"] = d["times"][sl]
    c["heavy"] = int((c["rate"] >= 5).sum())
    c["rainy"] = int((c["rate"] >= 0.5).sum())
    return c


def find_passes(sess):
    now = datetime.now(UTC)
    grans = cmr_search(RADAR, now - timedelta(hours=72), now, AREA)
    if not grans:
        note("Radar: no GPM passes found in the last 72 hours (NASA's radar files can be a day or two behind)")
        return []
    newest = grans[-1][2]
    use = [g for g in grans if g[2] >= min(now, newest) - timedelta(hours=24)]
    note(f"Radar: {len(grans)} pass(es) in 72 h, {len(use)} to check; newest file ends {newest:%d %b %H:%M} UTC")
    found = []
    for g in use:
        p = sess.get(g[3], os.path.join(RAW, os.path.basename(urlparse(g[3]).path)))
        try:
            c = storm_cut(radar_read(p))
        finally:
            os.remove(p)                 # big files: don't keep them
        if c is None:
            note(f"Radar pass {g[1]:%d %b %H:%M} UTC: no rain over the area")
            continue
        note(f"Radar pass {c['times'][0]:%d %b %H:%M} UTC: {c['rainy']} rain points, {c['heavy']} heavy")
        if c["rainy"] >= 300:
            found.append(c)
    found.sort(key=lambda c: -c["heavy"])
    return found[:MAX_PASSES]


# ----------------------------------------------------------------- the 3D picture
def make(c, out_dir, lines, districts):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
    tmid = c["times"][len(c["times"]) // 2]
    pid = f"gpm3d_{tmid.astimezone(IST):%Y-%m-%d_%H%M}"

    lat, lon, rate, hgt, surf = c["lat"], c["lon"], c["rate"], c["hgt"], c["surf"]
    m = (rate >= 0.2) & (hgt > 0) & (hgt < 19)
    LAT = np.broadcast_to(lat[..., None], rate.shape)[m]
    LON = np.broadcast_to(lon[..., None], rate.shape)[m]
    Z, R = hgt[m], rate[m]
    PH = (c["phase"][m] // 100).astype(int) if c["phase"] is not None else None
    if PH is not None:
        ok = PH <= 2
        LAT, LON, Z, R, PH = LAT[ok], LON[ok], Z[ok], R[ok], PH[ok]
    n, cap_n = len(R), 70000
    if n > cap_n:            # keep the heavier rain, thin the light rain
        rng = np.random.default_rng(1)
        keep = (R >= 5) | (rng.random(n) < (cap_n - (R >= 5).sum()) / max((R < 5).sum(), 1))
        LAT, LON, Z, R = LAT[keep], LON[keep], Z[keep], R[keep]
        PH = PH[keep] if PH is not None else None
    order = np.argsort(R)    # heavy rain drawn last
    LAT, LON, Z, R = LAT[order], LON[order], Z[order], R[order]
    PH = PH[order] if PH is not None else None

    x0, x1 = float(np.nanmin(lon)), float(np.nanmax(lon))
    y0, y1 = float(np.nanmin(lat)), float(np.nanmax(lat))
    kx = math.cos(math.radians((y0 + y1) / 2))
    sm = surf >= 0.2
    j, i = np.unravel_index(np.argmax(surf), surf.shape)
    where = where_is(float(lat[j, i]), float(lon[j, i]), districts)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    header(fig, "SATELLITE RADAR · 3D VIEW", where[0].upper() + where[1:] if len(where) < 34 else where.split(",")[0],
           f"GPM pass at {t_ist(tmid)} IST")
    ax = fig.add_axes([-0.06, 0.13, 1.12, 0.78], projection="3d", facecolor=BG)
    ax.set_box_aspect(((x1 - x0) * kx, (y1 - y0), 0.42 * max((x1 - x0) * kx, y1 - y0)), zoom=1.05)

    def setup():
        ax.cla()
        ax.set_facecolor(BG)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(0, 18)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0.06, 0.08, 0.12, 1.0))
            a._axinfo["grid"]["color"] = (0.25, 0.28, 0.34, 0.5)
        ax.tick_params(colors=MUTED, labelsize=11)
        ax.set_zlabel("km", color=MUTED, fontsize=12)
        for group, col, lw in lines:
            for ln in group:
                ok = (ln[:, 0] > x0 - 0.3) & (ln[:, 0] < x1 + 0.3) & (ln[:, 1] > y0 - 0.3) & (ln[:, 1] < y1 + 0.3)
                if ok.sum() > 1:
                    q = ln.astype(float).copy()
                    q[~ok] = np.nan
                    ax.plot(q[:, 0], q[:, 1], 0, color=col, lw=lw)
        for e in (0, -1):    # edges of the radar strip
            ax.plot(lon[:, e], lat[:, e], 0, color="#6f86b8", lw=1.0, ls="--")

    leg = fig.add_axes([0.08, 0.075, 0.84, 0.022])
    foot = fig.text(0.5, 0.032, "", ha="center", color=MUTED, fontsize=14)

    def legend(cols, labels, ticks, text):
        leg.cla()
        leg.imshow(np.arange(len(cols))[None, :], cmap=ListedColormap(cols), aspect="auto", extent=[0, len(cols), 0, 1])
        leg.set_yticks([])
        leg.set_xticks(ticks)
        leg.set_xticklabels(labels, color=SOFT, fontsize=14)
        leg.tick_params(length=0, pad=6)
        for s in leg.spines.values():
            s.set_visible(False)
        foot.set_text(text)

    def scene(by_phase):
        setup()
        ax.scatter(lon[sm], lat[sm], np.zeros(sm.sum()), c=surf[sm], cmap=RATE_CMAP, norm=RATE_NORM,
                   s=10, marker="s", alpha=0.55, depthshade=False, linewidths=0)
        if by_phase and PH is not None:
            col = np.array([hex2rgb(PHASE_C[p]) for p in (0, 1, 2)])[PH]
            ax.scatter(LON, LAT, Z, c=col, s=2.2, alpha=0.45, depthshade=False, linewidths=0)
        else:
            ax.scatter(LON, LAT, Z, c=R, cmap=RATE_CMAP, norm=RATE_NORM, s=2.2, alpha=0.5,
                       depthshade=False, linewidths=0)

    src = "Source: NASA/JAXA GPM radar (Ku band, 2A.Ku)"
    vid = Video(os.path.join(out_dir, pid + ".mp4"))
    legend(RATE_C, [f"{v:g}" for v in RATE_B[:-1]], range(len(RATE_C)), "Rain rate (mm/h)  ·  " + src)
    scene(False)
    n1 = FPS * 6
    for k in range(n1):
        ax.view_init(elev=26 + 6 * math.sin(k / n1 * math.pi), azim=-75 + 95 * k / n1)
        vid.add(fig)
        if k == n1 // 2:
            fig.savefig(os.path.join(out_dir, pid + ".png"), facecolor=BG)
    if PH is not None:
        legend([PHASE_C[2], PHASE_C[1], PHASE_C[0]], [PHASE_NAME[2], PHASE_NAME[1], PHASE_NAME[0]],
               [0.5, 1.5, 2.5], "What is falling  ·  " + src)
        scene(True)
        n2 = FPS * 5
        for k in range(n2):
            ax.view_init(elev=18, azim=20 - 70 * k / n2)
            vid.add(fig)
    vid.close()
    plt.close(fig)

    top_rate = float(np.nanmax(surf))
    top_h = float(np.nanmax(Z[R >= 0.5])) if (R >= 0.5).any() else 0.0
    cap = [f"🌀 Satellite Radar 3D View · {where}", f"GPM pass at {t_ist(tmid)} IST", "",
           f"Heaviest rain under the radar: {top_rate:.1f} mm/h.",
           f"Rain clouds reached about {top_h:.0f} km high.", "",
           "One snapshot as the satellite flew over, not a total.",
           "Source: NASA/JAXA GPM Dual-frequency Precipitation Radar (Ku band)",
           "#MasRainman · masrainman.blogspot.com"]
    open(os.path.join(out_dir, pid + ".txt"), "w", encoding="utf-8").write("\n".join(cap) + "\n")
    print("  made", pid)
    return {"id": pid, "time_utc": tmid.strftime("%Y-%m-%dT%H:%MZ"), "label": t_ist(tmid) + " IST",
            "where": where, "max_rate": round(top_rate, 1), "top_km": round(top_h, 1),
            "files": {"mp4": pid + ".mp4", "png": pid + ".png", "txt": pid + ".txt"},
            "caption": "\n".join(cap)}


def update_index(out_dir, new):
    p = os.path.join(out_dir, "index.json")
    try:
        items = {i["id"]: i for i in json.load(open(p, encoding="utf-8"))["passes"]}
    except (OSError, ValueError, KeyError):
        items = {}
    for it in new:
        items[it["id"]] = it
    keep = sorted(items.values(), key=lambda i: i["time_utc"], reverse=True)[:KEEP]
    wanted = {f for it in keep for f in it["files"].values()} | {"index.json"}
    for fn in os.listdir(out_dir):
        if fn not in wanted and os.path.isfile(os.path.join(out_dir, fn)):
            os.remove(os.path.join(out_dir, fn))
    json.dump({"updated": datetime.now(IST).isoformat(timespec="minutes"), "passes": keep},
              open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Index: {len(new)} new, {len(keep)} kept.")


# ----------------------------------------------------------------- made-up storm for checking
def demo_pass():
    ns, nr, nb = 300, 49, 176
    cx, cy = 85.5, 15.5
    s = np.linspace(-1, 1, ns)[:, None]
    r = np.linspace(-1, 1, nr)[None, :]
    lat = cy + s * 5.5 + r * 0.4
    lon = cx + s * 1.6 - r * 1.1
    hgt = np.broadcast_to((nb - 1 - np.arange(nb)) * 0.125, (ns, nr, nb)).astype("f4")
    rr = np.hypot((lon - cx) / 1.0, (lat - cy) / 1.0)
    rng = np.random.default_rng(5)
    band = np.exp(-((rr - 0.9) / 0.25) ** 2) * (0.6 + 0.4 * np.sin(6 * np.arctan2(lat - cy, lon - cx))) \
        + 1.4 * np.exp(-((rr - 0.25) / 0.12) ** 2)
    surf = (28 * band + rng.gamma(0.4, 1.0, (ns, nr))).astype("f4")
    top = 5 + 10 * np.clip(band, 0, 1) + rng.random((ns, nr)) * 2
    rate = np.where(hgt < top[..., None], surf[..., None] * np.exp(-hgt / 6), 0).astype("f4")
    phase = np.where(hgt < 4.6, 200, np.where(hgt < 5.1, 150, 50)).astype("u1")
    t = datetime.now(UTC) - timedelta(hours=7)
    return storm_cut(dict(lat=lat, lon=lon, rate=rate, surf=surf, phase=phase, hgt=hgt,
                          times=[t + timedelta(seconds=0.6 * i) for i in range(ns)]))


def main():
    ap = argparse.ArgumentParser(description="3D cyclone view from the GPM radar")
    ap.add_argument("--out", default=os.path.join(HERE, "cyclone3d_out"))
    ap.add_argument("--demo", action="store_true", help="made-up storm (no NASA login needed)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    passes = [demo_pass()] if a.demo else find_passes(EarthdataSession())
    if not passes:
        note("Radar: no pass with enough rain over the area in the last 24 hours")
        update_index(a.out, [])
        return
    lines = base_lines()
    districts = [d for fn in STATE_FILES for d in load_geo(fn)]
    made = []
    for c in passes:
        try:
            made.append(make(c, a.out, lines, districts))
        except Exception as e:  # noqa: BLE001
            note(f"Could not draw a pass: {e}", "warning")
    update_index(a.out, made)
    if passes and not made:
        sys.exit(1)


if __name__ == "__main__":
    main()
