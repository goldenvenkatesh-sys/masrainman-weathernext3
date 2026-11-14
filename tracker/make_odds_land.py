# Builds tracker/odds_land.json for the Cyclone Odds page (tracker/odds.html), once.
#   python tracker/make_odds_land.py ne_50m_land.geojson
# Natural Earth 50 m land (public domain) clipped to the North Indian Ocean view:
#   "poly": simplified land outlines for drawing (lon/lat rings, small islands kept)
#   "mask": 0.1-degree land/sea bitmap (base64, row 0 = north) used to find each member's landfall.
#           Islands under 2 deg2 (Andaman & Nicobar, Lakshadweep, Maldives) are sea in the mask, like landfall.py.
# Borders on the page come from data.json (Government of India view), not from this file.
import base64
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.prepared import prep
import shapely

W, E, S, N, RES = 38.0, 118.0, -8.0, 40.0, 0.1
MIN_ISLAND_DRAW, MIN_ISLAND_MASK = 0.01, 2.0


def main(src):
    feats = json.load(open(src))["features"]
    clip = box(W, S, E, N)
    polys = []
    for f in feats:
        g = shape(f["geometry"]).intersection(clip)
        if g.is_empty:
            continue
        polys += list(getattr(g, "geoms", [g]))
    polys = [p for p in polys if p.geom_type == "Polygon"]
    draw = []
    for p in polys:
        if p.area < MIN_ISLAND_DRAW:
            continue
        ring = np.asarray(p.simplify(0.03, preserve_topology=True).exterior.coords)
        if len(ring) >= 4:
            draw.append([np.round(ring[:, 0], 2).tolist(), np.round(ring[:, 1], 2).tolist()])
    big = prep(unary_union([p for p in polys if p.area >= MIN_ISLAND_MASK]))
    nx, ny = int(round((E - W) / RES)), int(round((N - S) / RES))
    lon = W + (np.arange(nx) + 0.5) * RES
    lat = N - (np.arange(ny) + 0.5) * RES
    LO, LA = np.meshgrid(lon, lat)
    m = shapely.contains_xy(big.context, LO, LA)
    bits = np.packbits(m.astype(np.uint8).ravel())
    out = {"w": W, "e": E, "s": S, "n": N, "res": RES, "nx": nx, "ny": ny,
           "mask": base64.b64encode(bits.tobytes()).decode(), "poly": draw,
           "source": "Natural Earth 50 m land (public domain)"}
    dst = Path(__file__).with_name("odds_land.json")
    dst.write_text(json.dumps(out, separators=(",", ":")))
    print(dst, dst.stat().st_size, "bytes,", len(draw), "outlines,", int(m.sum()), "land cells")


if __name__ == "__main__":
    main(sys.argv[1])
