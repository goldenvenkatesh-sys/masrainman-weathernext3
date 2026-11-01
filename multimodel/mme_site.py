# ============================================================
# Writes <SITE_DIR>/manifest.json + index.html for the MME page
# and keeps only the newest KEEP_CYCLES cycles of each product.
# ============================================================

import os
import re
import json
import shutil
from datetime import datetime, timezone

from mme import config as C

KEEP = int(os.environ.get("KEEP_CYCLES", "2"))
HERE = os.path.dirname(os.path.abspath(__file__))


def collect(kind):
    root = os.path.join(C.SITE_DIR, kind)
    if not os.path.isdir(root):
        return []
    cycles = sorted((c for c in os.listdir(root)
                     if re.fullmatch(r"\d{10}", c) and os.path.exists(os.path.join(root, c, "run.json"))),
                    reverse=True)
    for old in cycles[KEEP:]:
        print("Removing old", kind, old)
        shutil.rmtree(os.path.join(root, old), ignore_errors=True)
    out = []
    for c in cycles[:KEEP]:
        with open(os.path.join(root, c, "run.json")) as f:
            info = json.load(f)
        info["id"] = c
        out.append(info)
    return out


def main():
    os.makedirs(C.SITE_DIR, exist_ok=True)
    manifest = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                "rain": collect("rain"), "tracks": collect("tracks")}
    with open(os.path.join(C.SITE_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    shutil.copy(os.path.join(HERE, "web", "index.html"), os.path.join(C.SITE_DIR, "index.html"))
    print("manifest:", len(manifest["rain"]), "rain cycle(s),", len(manifest["tracks"]), "tracker cycle(s)")


if __name__ == "__main__":
    main()
