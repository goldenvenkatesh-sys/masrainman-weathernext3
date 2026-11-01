# ============================================================
# MASRAINMAN MULTI-MODEL WEIGHTED RAINFALL — RUNNER
# ============================================================
#
# For one cycle (00 or 12 UTC) it:
#   1. Fetches each model's rainfall (ensemble group and deterministic group)
#   2. Converts it to 24-h totals for 08:30 IST -> 08:30 IST days 1-15
#   3. Blends each group with renormalised weights
#   4. Draws Day 1-15, 7-day and 15-day maps for all 7 regions
#   5. Writes district CSVs and site/rain/<cycle>/run.json
#
# Usage:
#   python run_rain.py                         # latest suitable cycle, everything
#   python run_rain.py --cycle 2026103112      # a given cycle
#   python run_rain.py --groups det --regions tamilnadu --days 3   # quick test
# ============================================================

import os
import csv
import json
import argparse
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from mme import config as C
from mme.blend import load_group, blend, totals
from mme.fetch import day_leads
from mme.maps import rain_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("mme")
IST = timezone(timedelta(hours=5, minutes=30))


def latest_cycle(min_age_h=9):
    now = datetime.now(timezone.utc) - timedelta(hours=min_age_h)
    return now.replace(hour=0 if now.hour < 12 else 12, minute=0, second=0, microsecond=0)


def day_label(init, k):
    end = init + timedelta(hours=day_leads(init)[k + 1])
    return end.astimezone(IST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", help="YYYYMMDDHH (00 or 12 UTC)")
    ap.add_argument("--groups", default="ens,det")
    ap.add_argument("--regions", default=",".join(C.REGIONS))
    ap.add_argument("--days", type=int, default=C.N_DAYS, help="draw only the first N daily maps")
    args = ap.parse_args()

    init = (datetime.strptime(args.cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            if args.cycle else latest_cycle())
    cyc = init.strftime("%Y%m%d%H")
    out_dir = os.path.join(C.SITE_DIR, "rain", cyc)
    LOG.info("Cycle %s -> %s", cyc, out_dir)

    info = {"cycle": init.strftime("%Y-%m-%dT%H:%MZ"), "groups": {}, "days": [],
            "rendered": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")}
    for k in range(C.N_DAYS):
        info["days"].append({"day": k + 1, "end_ist": day_label(init, k).strftime("%Y-%m-%dT%H:%M+05:30")})

    for group in args.groups.split(","):
        members = load_group(group, init)
        if len(members) < C.MIN_MODELS_PER_DAY:
            LOG.error("Group %s: only %d model(s) available, skipping", group, len(members))
            continue
        days, used = blend(group, members)
        title = f"MASRainman Weighted Multi-Model {C.GROUPS[group]['title']}"
        g_info = {"title": C.GROUPS[group]["title"],
                  "models": {m: {"name": C.MODEL_NAMES[m], "run": r.strftime("%Y-%m-%dT%H:%MZ")}
                             for m, (_, r) in members.items()},
                  "used": used, "maps": {}}
        np.savez_compressed(os.path.join(out_dir if os.path.isdir(out_dir) else _mk(out_dir), f"{group}_days.npz"),
                            days=days.astype(np.float16))

        products = [(f"d{k + 1:02d}", "24h", days[k], k) for k in range(min(args.days, C.N_DAYS))
                    if np.isfinite(days[k, 0, 0])]
        for n, key in ((7, "7day"), (15, "15day")):
            t = totals(days, n)
            if t is not None:
                products.append((key, key, t, n - 1))

        for region in args.regions.split(","):
            table = {}
            for key, period, field, k in products:
                end = day_label(init, k)
                if period == "24h":
                    when = f"24-Hour Rainfall Forecast Ending 08:30 IST - {end:%d %b %Y} (Day {k + 1})"
                    names = [C.MODEL_NAMES[m] for m, _ in used[k]]
                else:
                    start = day_label(init, -1)
                    when = f"{period[:-3]}-Day Rainfall Total · 08:30 IST {start:%d %b} → 08:30 IST {end:%d %b %Y}"
                    names = sorted({C.MODEL_NAMES[m] for kk in range(k + 1) for m, _ in used[kk]})
                models_line = " | ".join(names) + f" | Cycle: {init:%Y-%m-%d %H}Z"
                png = os.path.join(out_dir, group, f"{region}_{key}.png")
                LOG.info("Drawing %s", png)
                means = rain_map(field, region, period, png, title, when, models_line)
                g_info["maps"].setdefault(region, []).append(key)
                for d, v in means.items():
                    table.setdefault(d, {})[key] = round(v, 1)
            if table:
                keys = [p[0] for p in products]
                with open(os.path.join(out_dir, group, f"{region}_districts.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["district"] + keys)
                    for d in sorted(table):
                        w.writerow([d] + [table[d].get(k, "") for k in keys])
        info["groups"][group] = g_info

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(info, f, indent=1)
    LOG.info("Done: %s", out_dir)


def _mk(d):
    os.makedirs(d, exist_ok=True)
    return d


if __name__ == "__main__":
    main()
