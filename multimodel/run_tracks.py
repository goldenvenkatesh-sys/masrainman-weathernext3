# ============================================================
# MASRAINMAN LPA + CYCLONE TRACKER — RUNNER
# ============================================================
#
# For one cycle it:
#   1. Fetches MSLP for every ensemble member and MSLP + 10-m wind for
#      the deterministic models (T+0 ... T+168 h, every 12 h)
#   2. Detects and tracks lows in each member, groups them into systems
#   3. Builds weighted consensus tracks (ensemble and deterministic)
#   4. Draws LPA Desk frames (SST on and off) and a cyclone map for
#      systems that reach Deep Depression strength in the deterministic blend
#   5. Writes site/tracks/<cycle>/run.json and member-track CSVs
#
# Usage:
#   python run_tracks.py                      # latest suitable cycle
#   python run_tracks.py --cycle 2026103112
# ============================================================

import os
import csv
import json
import argparse
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from mme import config as C
from mme.track_fetch import load_track_group, ecmwf_sst
from mme.tracker import STEPS, systems, link_det, landfall_probs, basin, category
from mme.track_maps import lpa_frame, lpa_extent, cyclone_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("mme")

CYCLONE_MIN_KT = 28  # draw a cyclone map once the deterministic blend reaches Deep Depression


def latest_cycle(min_age_h=9):
    now = datetime.now(timezone.utc) - timedelta(hours=min_age_h)
    return now.replace(hour=0 if now.hour < 12 else 12, minute=0, second=0, microsecond=0)


def weighted_fields(fields, group, var, members_mean=True):
    """Weighted mean of a variable across the models of a group (NaN-aware, per step)."""
    weights = C.GROUPS[group]["models"]
    num = den = None
    for m, mem in fields.items():
        arrs = [f[var] for f in mem.values() if var in f]
        if not arrs:
            continue
        a = np.nanmean(arrs, axis=0) if members_mean else arrs[0]
        ok = np.isfinite(a)
        w = weights[m] * ok
        if num is None:
            num, den = np.zeros_like(a), np.zeros_like(a)
        num += np.where(ok, a, 0) * w
        den += w
    if num is None:
        return None
    with np.errstate(invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def table(mean, det=None):
    rows = []
    for k in sorted(mean):
        p = mean[k]
        wind = det["mean"][k]["wind"] if det is not None and k in det["mean"] else p["wind"]
        rows.append({"lead": STEPS[k], "lat": round(p["lat"], 2), "lon": round(p["lon"], 2),
                     "mslp": round(p["msl"], 1), "deficit": round(p["deficit"], 2),
                     "wind_kt": None if wind is None else round(wind, 1),
                     "cat": category(wind)[0], "spread_km": round(p["spread"]), "share": round(p["share"], 2)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", help="YYYYMMDDHH (00 or 12 UTC)")
    args = ap.parse_args()
    init = (datetime.strptime(args.cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            if args.cycle else latest_cycle())
    cyc = init.strftime("%Y%m%d%H")
    out_dir = os.path.join(C.SITE_DIR, "tracks", cyc)
    os.makedirs(out_dir, exist_ok=True)
    LOG.info("Tracking cycle %s -> %s", cyc, out_dir)

    ens_f = load_track_group("ens", init)
    det_f = load_track_group("det", init)

    ens_sys = systems(ens_f, "ens") if ens_f else []
    det_sys = systems(det_f, "det") if det_f else []
    links = {i: link_det(s, det_sys) for i, s in enumerate(ens_sys)}
    LOG.info("Systems: %d ensemble, %d deterministic", len(ens_sys), len(det_sys))

    det_fields = None
    if det_f:
        det_fields = {v: weighted_fields(det_f, "det", v, members_mean=False) for v in ("msl", "u10", "v10")}
        if det_fields["u10"] is None:
            det_fields["u10"] = det_fields["v10"] = np.full_like(det_fields["msl"], np.nan)
    ens_msl = weighted_fields(ens_f, "ens", "msl") if ens_f else None
    try:
        sst = ecmwf_sst(init)
    except Exception as exc:
        LOG.warning("SST not available: %s", exc)
        sst = None

    view = lpa_extent(ens_sys, links)   # zoomed on the systems, same for every frame
    frames = []
    for k, lead in enumerate(STEPS):
        names = {}
        for tag, show in (("sst", True), ("nosst", False)):
            if show and sst is None:
                continue
            name = f"lpa_{tag}_T{lead:03d}.png"
            LOG.info("Drawing %s", name)
            lpa_frame(os.path.join(out_dir, name), init, k, sst, det_fields, ens_msl, ens_sys, det_sys, links, show,
                      extent=view)
            names[tag] = name
        frames.append({"lead": lead, "valid": (init + timedelta(hours=lead)).strftime("%Y-%m-%dT%H:%MZ"), **names})

    info = {"cycle": init.strftime("%Y-%m-%dT%H:%MZ"), "frames": frames, "systems": [],
            "models": {"ens": {m: len(v) for m, v in ens_f.items()}, "det": {m: len(v) for m, v in det_f.items()}},
            "rendered": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")}

    with open(os.path.join(out_dir, "member_tracks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "model", "member", "weight", "lead", "lat", "lon", "mslp", "deficit", "wind_kt"])
        for n, s in enumerate(ens_sys, 1):
            for wt, m, member, t in s["members"]:
                for p in t:
                    w.writerow([f"L{n}", m, member, round(wt, 5), STEPS[p["k"]], round(p["lat"], 2),
                                round(p["lon"], 2), round(p["msl"], 1), round(p["deficit"], 2),
                                "" if p["wind"] is None else round(p["wind"], 1)])

    for n, s in enumerate(ens_sys, 1):
        det = links[n - 1]
        probs, timing = landfall_probs(s)
        k0 = min(s["mean"])
        rows = table(s["mean"], det)
        winds = [r["wind_kt"] for r in rows if r["wind_kt"] is not None]
        peak = max(winds) if winds else None
        entry = {"id": f"L{n}", "basin": basin(s["mean"][k0]["lat"], s["mean"][k0]["lon"]),
                 "support": round(s["support"], 3), "models": s["models"],
                 "members": len(s["members"]), "track": rows,
                 "det_track": table(det["mean"]) if det else None,
                 "min_mslp": round(min(r["mslp"] for r in rows), 1),
                 "peak_wind_kt": peak, "peak_cat": category(peak)[0],
                 "landfall": [{"where": k, "prob": round(v, 3), "lead": timing.get(k)} for k, v in probs]}
        if peak is not None and peak >= CYCLONE_MIN_KT:
            name = f"cyclone_L{n}.png"
            LOG.info("Drawing %s", name)
            cyclone_map(os.path.join(out_dir, name), init, n, s, det, probs, timing)
            entry["cyclone_map"] = name
        info["systems"].append(entry)

    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(info, f, indent=1)
    LOG.info("Done: %s", out_dir)


if __name__ == "__main__":
    main()
