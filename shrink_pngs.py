#!/usr/bin/env python3
"""Shrink the site's map PNGs to 256-colour palette PNGs (about 60-70 % smaller, maps look the same).

    python shrink_pngs.py site

The site-data branch is pushed as one commit, and GitHub refuses pushes over ~2 GB, so the
images must stay small. Files already in palette mode are skipped, so after the first run only
new maps are converted. Runs on all CPU cores."""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from PIL import Image


def shrink(path):
    try:
        with Image.open(path) as im:
            if im.mode == "P":
                return 0
            before = os.path.getsize(path)
            q = im.convert("RGB").quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        tmp = path + ".tmp.png"
        q.save(tmp, optimize=True)
        if os.path.getsize(tmp) < before:
            os.replace(tmp, path)
            return before - os.path.getsize(path)
        os.remove(tmp)
    except Exception as exc:   # one odd file must not stop the job
        print(f"  skipped {path}: {exc}")
    return 0


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "site"
    files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(".png")]
    with ProcessPoolExecutor() as pool:
        saved = sum(pool.map(shrink, files, chunksize=16))
    print(f"shrink_pngs: {len(files)} PNGs checked, {saved / 1e6:.0f} MB saved")


if __name__ == "__main__":
    main()
