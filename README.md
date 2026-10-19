# MasRainman · WeatherNext 3 Low Centres (auto maps)

Automatic WeatherNext 3 **Mean MSLP + V3 low-centre** maps for 55–100E, 2–28N.

For every 00, 06, 12 and 18 UTC run it draws **T+24, T+48 … T+360** (15 frames),
with and without mean 10-m wind barbs, builds a GIF loop, and publishes a viewer page
with run selector, frame buttons, slider, play/speed, and PNG / GIF / CSV downloads.
The latest 4 runs stay selectable. India's border is drawn as per the Government of India.

## Files

| File | What it does |
|---|---|
| `render.py` | Finds new complete runs, runs V3 detection + tracking, draws maps, GIFs, CSVs, manifest |
| `web/index.html` | The viewer page (copied into the website on every run) |
| `boundaries/countries_india_view.geojson` | Country borders, India as per GoI (Natural Earth India view) |
| `.github/workflows/weathernext3.yml` | Runs every hour, renders only new complete runs, publishes to `gh-pages` |

## One-time setup

1. **Create the repo** (e.g. `masrainman-weathernext3`) and upload all files keeping the folders
   (`.github/workflows/…` must keep that exact path).

2. **Google service account key** (GitHub cannot use the Colab login):
   - Google Cloud Console → the project that has your WeatherNext access →
     IAM & Admin → Service Accounts → *Create service account*.
   - Give it **Storage Object Viewer**.
   - Make sure this service account is allowed to read `weathernext3_statistics_spatial`
     (if your WeatherNext approval was for your Gmail account only, add the service account's
     email to the same WeatherNext access, or ask Google to add it).
   - Keys → *Add key* → JSON → download.

3. **Add the key as a secret**: repo → Settings → Secrets and variables → Actions →
   *New repository secret* → name `GCP_SA_KEY`, paste the whole JSON. Never upload the JSON file itself.

4. **First run**: Actions tab → *WeatherNext 3 auto maps* → *Run workflow*.
   It creates the `gh-pages` branch (~20–30 min per run).

5. **Turn on the website**: Settings → Pages → Source *Deploy from a branch* →
   `gh-pages` / `(root)`. Page: `https://<your-user>.github.io/<repo>/`

After that it runs by itself every hour and only renders when a new run is complete.

## Options

- Force a specific run: *Run workflow* → enter e.g. `20261019_06hr_01_preds`.
- Runs kept on the site: `KEEP_RUNS` in the workflow (default 4).
- Test without Google Cloud: `python render.py --demo` (synthetic cyclone), then open `site/index.html`
  through a local server: `cd site && python -m http.server`.

## Notes

- `gh-pages` is force-pushed each time, so old images don't pile up in git history.
- GitHub Pages on a free account needs a **public** repo, and anyone with the link can see the page.
  If you want it private, use GitHub Pro or host the `site/` folder elsewhere.
- Real-time WeatherNext 3 output is under Google DeepMind's experimental-data terms; check they allow
  public posting before sharing the page widely.
