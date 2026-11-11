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
from mme.fetch import day_leads, FETCHERS, NotAvailable
from mme.maps import rain_map
from mme.ew import ew_maps

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
    ap.add_argument("--totals", default=",".join(C.TOTAL_MODELS),
                    help="models for the single-model rain totals (24 h, 7 days, 15 days); '' = none")
    ap.add_argument("--no-ew", action="store_true", help="skip the EW index maps")
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

    fetched = {}                      # every model's daily rain, reused by the EW index maps
    for group in args.groups.split(","):
        members = load_group(group, init)
        fetched.update(members)
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
    with open(os.path.join(out_dir, "run.json"), "w") as f:   # the blends are safe even if the totals stall
        json.dump(info, f, indent=1)

    info["models"] = model_totals(init, out_dir, [m for m in args.totals.split(",") if m])
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(info, f, indent=1)

    # EW index (extreme 24-h rain) for every model, South + North India, IMD days
    if not args.no_ew:
        info["ew"] = ew_maps(init, out_dir, fetched)
        with open(os.path.join(out_dir, "run.json"), "w") as f:
            json.dump(info, f, indent=1)
    LOG.info("Done: %s", out_dir)


def model_totals(init, out_dir, models):
    """Rain totals of single models from the start of their run: next 24 h, 7 days and 15 days.

    These are the maps on the public "Rain totals" page. A model that is missing (or whose run
    is shorter) is simply left out; the previous cycle (12 h older) is used if this one is not out."""
    out = {}
    for model in models:
        acc, run = None, None
        for back in (0, 12):
            r = init - timedelta(hours=back)
            leads = [0] + [h for _, h, _ in C.TOTAL_PERIODS]
            try:
                if model == "ecmwf_hres":   # T+360 if this run has it, else the usual T+240
                    try:
                        a = FETCHERS["ecmwf_hres_360"](r, leads)
                    except NotAvailable:
                        a = FETCHERS[model](r, leads)
                else:
                    a = FETCHERS[model](r, leads)
            except NotAvailable as exc:
                LOG.warning("totals: %s %s not available (%s)", model, r.strftime("%Y%m%d%HZ"), exc)
                continue
            except Exception as exc:  # one broken source must not stop the others
                LOG.exception("totals: %s failed: %s", model, exc)
                break
            if 24 in a:
                acc, run = a, r
                break
        if acc is None:
            continue
        name = C.MODEL_NAMES[model]
        start = run.astimezone(IST)
        m_info = {"name": name, "run": run.strftime("%Y-%m-%dT%H:%MZ"), "periods": [], "maps": {}}
        for key, hours, label in C.TOTAL_PERIODS:
            if hours not in acc:
                continue
            field = np.clip(acc[hours] - acc.get(0, 0), 0, None)
            end = (run + timedelta(hours=hours)).astimezone(IST)
            m_info["periods"].append({"key": key, "hours": hours, "label": label,
                                      "end_ist": end.strftime("%Y-%m-%dT%H:%M+05:30")})
            when = f"Rainfall total, next {label} · {start:%d %b %H:%M} IST → {end:%d %b %Y %H:%M} IST"
            line = f"{name} | Run {run:%Y-%m-%d %H}Z"
            for region in C.TOTAL_REGIONS:
                png = os.path.join(out_dir, "models", model, f"{region}_{key}.png")
                LOG.info("Drawing %s", png)
                rain_map(field, region, key, png, f"{name} · Rainfall Total ({label})", when, line)
                m_info["maps"].setdefault(region, []).append(key)
        if m_info["periods"]:
            out[model] = m_info
    return out


def _mk(d):
    os.makedirs(d, exist_ok=True)
    return d


if __name__ == "__main__":
    main()
