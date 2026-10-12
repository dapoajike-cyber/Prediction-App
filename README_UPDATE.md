# Keeping the climate data current

The app predicts from county-level PRISM climate stored in `county_monthly_climate.json`.
`update_prism_data.py` refreshes that file (and the climate parts of `county_details_*.json`).

## How the data is organised

| Kind of year | Meaning | What the app does |
|---|---|---|
| **Complete** (1968 … 2025) | all 12 months × 7 variables for every county | real PRISM values; the 5-year limit counts from the last complete year |
| **Partial** (the current year) | months PRISM has published so far | uses the real months, fills the rest from the county's own trend, and says which is which |
| **Future** | beyond the partial year | whole year from the county trend, up to 5 years past the last complete year |

When the last month of a partial year is published, the next run promotes it to a complete year
automatically (and the 5-year limit moves forward by one).

## Option A — from a PRISM county CSV (same layout as the existing file)

```
python update_prism_data.py --csv PRISM_Texas_County_Monthly_Climate_2026.csv
```
Months not yet published may be blank. History (years already complete) is left untouched
unless you add `--rebuild`.

## Option B — straight from PRISM (no CSV needed)

Same method as the original Colab pipeline (monthly 4 km rasters → county-polygon mean, Census
TIGER 2023 boundaries).

```
pip install -r requirements-update.txt
python update_prism_data.py --probe     # which months has PRISM published?
python update_prism_data.py --fetch     # fetch + rebuild
```
`--years 2026` re-fetches a specific year (PRISM revises its recent provisional months, so
re-fetching the current year each run is intended). `--dry-run` shows what would change without writing.

## Automatic schedule (GitHub Actions)

`.github/workflows/update-prism-data.yml` runs Option B on the 20th of every month, commits the
changed JSON files, and Render redeploys the backend by itself. If PRISM has nothing new, nothing is
committed. It can also be run by hand from the repository's **Actions** tab.
To refresh only once a year, change the cron line to `'0 14 20 1 *'` — but then the app will not
see newly published months of the current year until the following January.

## Safety checks built in

The script stops without writing anything if it finds a county the app does not know, a gap in the
complete years, or an out-of-range value (e.g. negative rainfall, a 90 °C temperature). It also
warns if a new year's statewide average is far outside the historical spread. Past years are never
modified unless you explicitly ask (`--rebuild`).

## What this does NOT change

* **Models are not retrained.** They keep predicting from whatever climate the files contain.
  Retraining is only needed when new *yield* data (USDA/NASS) is added.
* **CO₂** is taken from the last year in the training data (2025); CO₂ moves only ~2–3 ppm a year, so
  the effect is negligible.
* A prediction for the current year is a forecast until the growing window is complete; the app
  shows which months are recorded and which are estimated.
