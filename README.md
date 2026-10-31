# MasRainman · Model Low Centres (auto maps)

Automatic **MSLP + V3 low-centre** maps for 55–100E, 2–28N from three models:

| Model | Source | Runs | Length |
|---|---|---|---|
| WeatherNext 3 (statistics mean) | Google Cloud `weathernext3_statistics_spatial` | 00/06/12/18 UTC | T+360 |
| ECMWF IFS HRES | data.ecmwf.int open data (0.1° ~9 km when published, else 0.25°) | 00/06/12/18 UTC | 00/12Z T+360, 06/18Z T+144 |
| ECMWF AIFS Single | data.ecmwf.int open data (0.25°) | 00/06/12/18 UTC | 00/12Z T+360, 06/18Z T+144 |

**Rain charts (ECMWF IFS + AIFS):** besides the MSLP maps, every frame also has
*Rain (total)* — rain since the start of the run — and *Rain (24 h)* — rain in the
24 h before the map, on a wider 50–110°E, 0–42°N view with a 0.2–500 mm colour scale.
WeatherNext 3 gets the same two rain charts from `total_precipitation_1hr_mean` (hourly ensemble-mean rain, summed). Set `WN3_RAIN_VAR` to use another field, e.g. `total_precipitation_1hr_p90` or `imerg_tp_1hr_mean`. Rain charts appear for runs drawn after this change.

For every complete run it draws **T+24, T+48 …** to the end of the forecast,
with and without 10-m wind barbs, builds a GIF loop, writes CSVs of the V3 candidates
and persistent tracks, and publishes a viewer page with model and run selectors,
frame buttons, slider, play/speed, and PNG / GIF / CSV downloads.
The latest 4 runs of each model stay selectable. India's border is drawn as per the Government of India.

**Landfall maps:** `landfall.py` draws, for each Bay of Bengal / Arabian Sea system, every member's
first landfall coloured by intensity (TD / TS / Cat 1–5), the numbered top-5 landfall areas with the
share of members, and the ensemble mean track. They are on `landfall.html`.

| Model | Tracks from | Runs |
|---|---|---|
| ECMWF ENS (51) | ECMWF's own ensemble cyclone tracks (open data BUFR), genesis systems 70B, 71B … included | 00/06/12/18 UTC |
| GFS ensemble GEFS (31) | tracked here (`gridtrack.py`) in GEFS 0.5° MSLP + 10-m wind, NOAA open data on AWS | 00/06/12/18 UTC |
| UKMET global (1 track) | tracked here in Met Office global 10 km MSLP + 10-m wind, Met Office open data on AWS | 00/12 UTC (T+168) |

The GEFS and UKMET tracks start from ECMWF's position of each system in the same run, so each system has
the same id (e.g. 71B) on every model. Landfall means reaching the mainland or Sri Lanka; small islands
(Andaman & Nicobar, Lakshadweep) are not counted.
DeepMind FNV3, ICON-EPS and GEM GEPS have buttons on the page but stay greyed out: their member tracks/fields
are on deepmind.google.com, opendata.dwd.de and dd.weather.gc.ca only, which this repo's builder cannot use yet.
The UKMET ensemble (MOGREPS-G) is not openly published, hence one deterministic track.

**NEM & cyclone outlook:** `nem_outlook.py` builds `outlook.html` after every ECMWF 00Z and 12Z run:

| Part | What | Data |
|---|---|---|
| A · Genesis | Probability that an LPA / depression (≥17 kt) / cyclonic storm (≥34 kt) forms within 300 km, days 1-5, 6-10, 11-15, 1-15; basin-wide probabilities; run-to-run consistency map (last 4 runs) | ECMWF ENS track BUFR, 51 members |
| B · Cyclones days 1-15 | Per system: strike probability + member tracks by IMD class, ensemble mean + HRES track, intensity plume, day-by-day table, landfall / coast / recurve / dissipate scenarios, Day 1/3/5 environment (MSLP + 850 hPa flow, vorticity, shear) | ECMWF ENS + HRES |
| C · 30-day NEM rain | Weekly totals and departures (weeks 1-2 ENS, weeks 3-4 + days 29-30 CFSv2 lagged ensemble), Tamil Nadu + 4 TN regions + states, below/near/above probabilities, confidence | ECMWF ENS tp, NOAA CFSv2, CPC 1991-2020 gauge normal |
| D · Active / break | Daily 0-100 score for days 0-10 from Bay vorticity, easterlies, moisture flux, mid-level humidity, pressure gradient, MJO | ECMWF HRES, BoM RMM |
| E · Track consistency | Systems matched across the last 10 runs: appearance count, track shift, spread, intensity and landfall-region trends | stored run history |
| F · PDI | 15-day forecast PDI vs the same window and the OND seasons 1991-2025 | IBTrACS (IMD winds) |
| G · Verification | Track and intensity error, detection rate, false-alarm ratio, station rain MAE/RMSE and Brier skill | IBTrACS / ECMWF analyses, TN-SMART |

Every probability is a share of ensemble members. The history and verification archive live in
`site/outlook/state/` (kept on the `site-data` branch). To fill the verification archive straight away,
run the workflow with *outlook_backfill* = 30. Test without downloads: `python nem_outlook.py --demo`.

## Files

| File | What it does |
|---|---|
| `check_new.py` | Quick check (no heavy installs): is there a new complete run the site doesn't have yet? |
| `render.py` | Loads new runs, runs V3 detection + tracking, draws maps, GIFs, CSVs and manifests |
| `web/index.html` | The viewer page (copied into the site on every run) |
| `landfall.py` | Ensemble landfall maps (`site/landfall/<model>/<run>/`, `site/data/manifest_landfall.json`) |
| `gridtrack.py` | Cyclone tracker for models without track files (GEFS, UKMET) |
| `web/landfall.html` | The landfall maps page |
| `nem_outlook.py` | NEM & cyclone outlook (`site/outlook/<run>/`, `site/data/manifest_outlook.json`) |
| `outlook_data.py` | Downloads for the outlook (ECMWF, CFSv2, CPC, IBTrACS, RMM, TN-SMART) and demo data |
| `web/outlook.html` | The outlook dashboard |
| `boundaries/tn_districts.geojson` | Tamil Nadu districts (for the TN regions) |
| `boundaries/borders_india_view.geojson` | Land borders, India as per GoI (Natural Earth India view) |
| `.github/workflows/weathernext3.yml` | Runs every 3 hours, renders only new complete runs, saves and publishes the site |

## How a run works

1. Every 3 hours at :20 UTC (05:50, 08:50, 11:50, 14:50, 17:50, 20:50, 23:50, 02:50 IST)
   the workflow restores the current site from the `site-data` branch.
2. `check_new.py` looks for new complete runs. If there are none, the job stops here.
3. Otherwise `render.py` draws the new runs (max 2 per model per job) and drops runs
   beyond the latest 4.
4. The site is saved to the `site-data` branch (force-pushed, so old images don't pile up
   in git history) and deployed to **Cloudflare Pages** (project `masrainman-wn3`).

## One-time setup

1. **Google service account key** for WeatherNext 3 (GitHub cannot use the Colab login):
   - Google Cloud Console → the project that has your WeatherNext access →
     IAM & Admin → Service Accounts → *Create service account*.
   - Give it **Storage Object Viewer**.
   - Make sure this service account is allowed to read `weathernext3_statistics_spatial`
     (if your WeatherNext approval was for your Gmail account only, add the service account's
     email to the same WeatherNext access, or ask Google to add it).
   - Keys → *Add key* → JSON → download.

2. **Repository secrets**: repo → Settings → Secrets and variables → Actions → *New repository secret*:

   | Secret | Value |
   |---|---|
   | `GCP_SA_KEY` | The whole service-account JSON. Never upload the JSON file itself. |
   | `CLOUDFLARE_API_TOKEN` | Cloudflare API token with *Cloudflare Pages: Edit* |
   | `CLOUDFLARE_ACCOUNT_ID` | Your Cloudflare account ID |

   ECMWF open data needs no key.

3. **First run**: Actions tab → *MasRainman model maps* → *Run workflow*.
   It creates the `site-data` branch and the Cloudflare Pages project.

4. **Protect the page** (optional): in Cloudflare → Zero Trust → Access, add an application
   for the Pages domain so only people you allow can log in.

After that it runs by itself every 3 hours and only renders when a new run is complete.

## Options

- Force a specific run: *Run workflow* → pick the model and enter the run,
  e.g. WeatherNext 3 `20261019_06hr_01_preds`, IFS/AIFS `20261019_12`.
- Runs kept per model: `KEEP_RUNS` in the workflow (default 4).
- Save GitHub minutes by drawing only the 15-day ECMWF runs: uncomment
  `ECMWF_CYCLES: "00,12"` in the workflow.
- Only some models: `python render.py --models ifs,aifs`.
- Test without any downloads: `python render.py --demo` (synthetic cyclone, all models), then open
  the site through a local server: `cd site && python -m http.server`.

## Site layout

```
site/runs/<YYYYMMDD_HHhr_01_preds>/   WeatherNext 3     site/data/manifest.json
site/ecmwf/ifs/<YYYYMMDD_HH>/         ECMWF IFS         site/data/manifest_ifs.json
site/ecmwf/aifs/<YYYYMMDD_HH>/        ECMWF AIFS        site/data/manifest_aifs.json
site/data/models.json                 list of models for the page
```

## Notes

- Real-time WeatherNext 3 output is under Google DeepMind's experimental-data terms; check they allow
  sharing before giving the page to others.
- ECMWF open data is CC BY 4.0.
- These maps are not official forecasts. Please follow IMD for official forecasts and warnings.
