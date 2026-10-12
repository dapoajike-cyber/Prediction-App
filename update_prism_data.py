#!/usr/bin/env python3
"""
update_prism_data.py -- keeps Texas Yield Explorer's county climate current.

WHAT IT DOES
------------
Refreshes the climate files the web app reads:

  county_monthly_climate.json   per-county monthly PRISM climate (the file the
                                backend uses for every prediction)
  county_details_Cotton.json    the climate parts of the per-county charts
  county_details_Corn.json      (annual precipitation / temperature)

It understands two kinds of year:

  * COMPLETE year  -- all 12 months x 7 variables exist for every county.
                      Used as ordinary history (and as the base for the
                      5-year extrapolation limit).
  * PARTIAL year   -- the current, still-in-progress year. Months PRISM has
                      published so far are real data; the app fills the rest
                      from each county's own trend and says so on screen.

When the last month of a partial year is published, the next run promotes it
to a complete year automatically -- nothing to edit by hand.

TWO WAYS TO GET NEW DATA
------------------------
1) From a county-level PRISM CSV (same layout as
   PRISM_Texas_County_Monthly_Climate_*.csv: County, Year, ppt_Jan ... vpdmax_Dec;
   months not yet published may be blank):

       python update_prism_data.py --csv PRISM_Texas_County_Monthly_Climate_2026.csv

2) Straight from PRISM (same method as the original Colab pipeline: download
   the monthly 4 km rasters, then take the county-polygon mean). Needs
   `pip install -r requirements-update.txt` and internet access:

       python update_prism_data.py --fetch

Useful options:
   --dry-run     report what would change; write nothing
   --probe       (with --fetch) only check which months PRISM has published
   --years 2026  (with --fetch) re-fetch specific years (PRISM revises its
                 provisional months, so re-fetching a recent year is normal)
   --data-dir D  folder holding the JSON files (default: next to this script)

Safe to run as often as you like: if nothing new is available, nothing changes.
"""
import argparse
import datetime
import json
import math
import os
import sys
import tempfile
import time
import warnings
import zipfile

import numpy as np

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
VARS = ['ppt', 'tmax', 'tmin', 'tmean', 'tdmean', 'vpdmin', 'vpdmax']
COLS = [f'{v}_{m}' for v in VARS for m in MONTHS]      # same order as the app's data file
MIN_YEAR = 1968                                          # first year the models were trained on
PRISM_BASE = 'https://data.prism.oregonstate.edu/time_series/us/an/4km'
TIGER_URL = 'https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip'

# Plausible monthly ranges, per variable. A value outside these means a bad
# download or a unit problem, so we stop instead of publishing it.
VALID_RANGE = {
    'ppt': (0, 1500),         # mm / month
    'tmax': (-40, 55),        # deg C
    'tmin': (-45, 45),
    'tmean': (-40, 50),
    'tdmean': (-50, 40),
    'vpdmin': (0, 80),        # hPa
    'vpdmax': (0, 250),
}


def norm(name):
    return ''.join(ch for ch in str(name).lower() if ch.isalnum())


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Loading / saving the app's compact climate file
# --------------------------------------------------------------------------
def load_store(path):
    """-> (store, last_complete_year) where store[county][year] = [84 values, None allowed]."""
    with open(path) as f:
        raw = json.load(f)
    if raw['cols'] != COLS:
        sys.exit('ERROR: column layout of county_monthly_climate.json is not the expected '
                 '84 columns (ppt, tmax, tmin, tmean, tdmean, vpdmin, vpdmax x Jan-Dec).')
    store = {}
    for county, rec in raw['counties'].items():
        store[county] = {y: list(row) for y, row in zip(rec['years'], rec['data'])}
        part = rec.get('partial')
        if part:
            store[county][part['year']] = list(part['data'])
    meta = raw.get('meta', {})
    last_complete = meta.get('last_complete_year')
    if last_complete is None:
        last_complete = max(max(y for y in yrs) for yrs in store.values())
    return store, last_complete


def is_complete_row(row):
    return row is not None and all(v is not None for v in row)


def year_status(store):
    """Classify years. Returns (complete_years, partial_year or None)."""
    all_years = sorted({y for yrs in store.values() for y in yrs if y >= MIN_YEAR})
    complete, incomplete = [], []
    for y in all_years:
        rows = [store[c].get(y) for c in store]
        if all(is_complete_row(r) for r in rows):
            complete.append(y)
        elif any(r is not None and any(v is not None for v in r) for r in rows):
            incomplete.append(y)
    if complete and complete != list(range(complete[0], complete[-1] + 1)):
        gaps = sorted(set(range(complete[0], complete[-1] + 1)) - set(complete))
        sys.exit(f'ERROR: complete years are not continuous (missing/incomplete: {gaps}). '
                 'Refusing to write a file with a hole in the history.')
    if incomplete:
        if len(incomplete) > 1 or incomplete[0] != complete[-1] + 1:
            sys.exit(f'ERROR: unexpected incomplete years {incomplete}. Only the single year after '
                     f'{complete[-1]} may be partial.')
        return complete, incomplete[0]
    return complete, None


def observed_through(store, partial_year):
    """Last month (name) such that every month up to it is fully published for every county."""
    last = None
    for m in MONTHS:
        idxs = [i for i, c in enumerate(COLS) if c.endswith('_' + m)]
        ok = all(all(store[c].get(partial_year, [None] * 84)[i] is not None for i in idxs) for c in store)
        if not ok:
            break
        last = m
    return last


def validate(store, complete_years, partial_year, base_complete):
    problems = []
    for county, yrs in store.items():
        for y, row in yrs.items():
            if y < MIN_YEAR:
                continue
            for i, v in enumerate(row):
                if v is None:
                    continue
                var = COLS[i].split('_')[0]
                lo, hi = VALID_RANGE[var]
                if not (isinstance(v, (int, float)) and math.isfinite(v)) or v < lo or v > hi:
                    problems.append(f'{county} {y} {COLS[i]} = {v} (expected {lo}..{hi})')
    if problems:
        sys.exit('ERROR: implausible values, nothing written:\n  ' + '\n  '.join(problems[:20]) +
                 (f'\n  ... and {len(problems) - 20} more' if len(problems) > 20 else ''))

    # Statewide sanity check on every new year: warn (not fail) if a column's
    # statewide mean is far outside its historical spread -- a hint that a
    # download or unit conversion went wrong.
    base = [y for y in complete_years if y in set(base_complete)]
    if len(base) < 6:
        return
    hist = np.array([np.mean([store[c][y] for c in store], axis=0) for y in base])   # years x 84
    mu, sd = hist.mean(axis=0), hist.std(axis=0)
    sd[sd == 0] = 1e-9
    new_years = [y for y in complete_years if y not in set(base_complete)] + ([partial_year] if partial_year else [])
    warn = []
    for y in new_years:
        arr = np.array([[np.nan if v is None else v for v in store[c][y]] for c in store if y in store[c]], dtype=float)
        if arr.size == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # all-blank columns (unpublished months)
            m = np.nanmean(arr, axis=0)
        z = np.abs(m - mu) / sd
        for i in np.where(np.nan_to_num(z) > 5)[0]:
            warn.append(f'{y} {COLS[i]}: statewide mean {m[i]:.2f} is {z[i]:.1f} std devs from history')
    for w in warn[:15]:
        log('  WARNING: ' + w)


def save_store(store, complete_years, partial_year, path):
    last_complete = complete_years[-1]
    counties = {}
    for county in sorted(store):
        yrs = [y for y in complete_years]
        rec = {'years': yrs, 'data': [store[county][y] for y in yrs]}
        if partial_year and store[county].get(partial_year) and any(v is not None for v in store[county][partial_year]):
            rec['partial'] = {'year': partial_year, 'data': store[county][partial_year]}
        counties[county] = rec
    meta = {
        'last_complete_year': last_complete,
        'partial_year': partial_year,
        'observed_through': observed_through(store, partial_year) if partial_year else None,
        'updated': datetime.date.today().isoformat(),
        'source': 'PRISM Climate Group, Oregon State University (county-polygon monthly means)',
    }
    out = {'cols': COLS, 'counties': counties, 'meta': meta}
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    os.replace(tmp, path)
    return meta


# --------------------------------------------------------------------------
# Source 1: a county-level PRISM CSV
# --------------------------------------------------------------------------
def merge_csv(store, csv_path, last_complete, rebuild):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df['County'] = df['County'].astype(str).str.strip()
    missing_cols = [c for c in COLS if c not in df.columns]
    if missing_cols:
        sys.exit(f'ERROR: CSV is missing climate columns, e.g. {missing_cols[:5]}')
    df = df.drop_duplicates(subset=['County', 'Year'], keep='last')
    by_norm = {norm(c): c for c in store}
    unknown = sorted({c for c in df['County'] if norm(c) not in by_norm})
    if unknown:
        sys.exit(f'ERROR: CSV has counties the app does not know: {unknown[:10]}')

    changed_years, skipped = set(), 0
    values = df[COLS].to_numpy(dtype=float)
    for county_raw, year_raw, vals in zip(df['County'].tolist(), df['Year'].tolist(), values):
        year = int(year_raw)
        if year < MIN_YEAR:
            continue
        if year <= last_complete and not rebuild:
            skipped += 1          # history is frozen unless --rebuild
            continue
        county = by_norm[norm(county_raw)]
        row = [None if not math.isfinite(v) else round(float(v), 2) for v in vals]
        if store[county].get(year) != row:
            changed_years.add(year)
        store[county][year] = row
    if skipped:
        log(f'  (ignored {skipped} CSV rows for years <= {last_complete}; use --rebuild to replace history)')
    return changed_years


# --------------------------------------------------------------------------
# Source 2: PRISM rasters -> county means (same method as the original pipeline)
# --------------------------------------------------------------------------
def prism_url(var, year, month):
    return f'{PRISM_BASE}/{var}/monthly/{year}/prism_{var}_us_25m_{year}{month:02d}.zip'


def download(url, retries=3):
    """-> bytes, or None if PRISM has not published this file (HTTP 404)."""
    import requests
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'download failed for {url}: {last}')


def raster_from_zip(blob, workdir):
    with zipfile.ZipFile(io_bytes(blob)) as z:
        z.extractall(workdir)
        names = z.namelist()
    for ext in ('.bil', '.tif'):
        for n in names:
            if n.lower().endswith(ext):
                return os.path.join(workdir, n)
    raise RuntimeError('no .bil/.tif raster found inside the PRISM zip')


def io_bytes(b):
    import io
    return io.BytesIO(b)


def load_counties(boundaries):
    import geopandas as gpd
    gdf = gpd.read_file(boundaries or TIGER_URL)
    if 'STATEFP' in gdf.columns:
        gdf = gdf[gdf['STATEFP'] == '48']
    gdf = gdf[['NAME', 'geometry']].reset_index(drop=True).to_crs(epsg=4269)
    return gdf


def county_means(raster_path, geoms):
    from rasterstats import zonal_stats
    stats = zonal_stats(geoms, raster_path, stats=['mean'], nodata=-9999, all_touched=True)
    return [None if s['mean'] is None or not math.isfinite(s['mean']) else round(float(s['mean']), 2) for s in stats]


def fetch_year(year, gdf, store, by_norm, probe=False):
    """Download every month PRISM has published for `year` and fill store.
    A month is all-or-nothing (all 7 variables) so a half-released month is never used."""
    names = list(gdf['NAME'])
    unknown = [n for n in names if norm(n) not in by_norm]
    if unknown:
        sys.exit(f'ERROR: boundary counties not in the app: {unknown[:10]}')
    geoms = list(gdf.geometry)
    got_months = 0
    for mi, m in enumerate(MONTHS, start=1):
        if probe:
            import requests
            ok = requests.head(prism_url('ppt', year, mi), timeout=60).status_code == 200
            log(f'  {year}-{mi:02d} ppt: {"published" if ok else "not published"}')
            if not ok:
                break
            got_months += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            month_vals = {}
            complete = True
            for var in VARS:
                blob = download(prism_url(var, year, mi))
                if blob is None:
                    complete = False
                    break
                sub = os.path.join(tmp, var)
                os.makedirs(sub, exist_ok=True)
                month_vals[var] = county_means(raster_from_zip(blob, sub), geoms)
            if not complete:
                log(f'  {year} {m}: not published yet -- stopping here')
                break
            for ci, name in enumerate(names):
                county = by_norm[norm(name)]
                row = store[county].setdefault(year, [None] * len(COLS))
                for var in VARS:
                    row[COLS.index(f'{var}_{m}')] = month_vals[var][ci]
            got_months += 1
            log(f'  {year} {m}: fetched')
    return got_months


# --------------------------------------------------------------------------
# Per-county chart files (annual precip / temperature for new complete years)
# --------------------------------------------------------------------------
def update_county_details(store, complete_years, changed_years, data_dir, dry_run):
    by_norm = {norm(c): c for c in store}
    for crop in ('Cotton', 'Corn'):
        path = os.path.join(data_dir, f'county_details_{crop}.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            details = json.load(f)
        added = refreshed = 0
        for county, d in details.items():
            sc = by_norm.get(norm(county))
            if sc is None:
                continue
            have = {r['year']: r for r in d['yearly']}
            have_t = {r['year']: r for r in d['annual_temp']}
            for y in complete_years:
                if y in have and y not in changed_years:
                    continue
                row = store[sc][y]
                ppt = round(float(sum(row[COLS.index(f'ppt_{m}')] for m in MONTHS)), 2)
                tmax = round(float(max(row[COLS.index(f'tmax_{m}')] for m in MONTHS)), 2)
                tmin = round(float(min(row[COLS.index(f'tmin_{m}')] for m in MONTHS)), 2)
                tmean = round(float(np.mean([row[COLS.index(f'tmean_{m}')] for m in MONTHS])), 2)
                if y in have:
                    have[y]['annual_precip_mm'] = ppt          # keep the yield that is already there
                    if y in have_t:
                        have_t[y].update({'tmax': tmax, 'tmin': tmin, 'tmean': tmean})
                    else:
                        d['annual_temp'].append({'year': y, 'tmax': tmax, 'tmin': tmin, 'tmean': tmean})
                    refreshed += 1
                else:
                    d['yearly'].append({'year': y, 'yield': None, 'annual_precip_mm': ppt})
                    d['annual_temp'].append({'year': y, 'tmax': tmax, 'tmin': tmin, 'tmean': tmean})
                    added += 1
            d['yearly'].sort(key=lambda r: r['year'])
            d['annual_temp'].sort(key=lambda r: r['year'])
        log(f'  county_details_{crop}.json: {added} year-entries added, {refreshed} refreshed')
        if (added or refreshed) and not dry_run:
            tmp = path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(details, f)
            os.replace(tmp, path)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--csv', help='county-level PRISM CSV with new rows (months not yet published may be blank)')
    src.add_argument('--fetch', action='store_true', help='download from PRISM and compute county means')
    ap.add_argument('--years', type=int, nargs='+', help='(--fetch) years to (re)fetch; default: every year after the last complete one, up to this year')
    ap.add_argument('--boundaries', help='(--fetch) county boundary file/URL; default: Census TIGER 2023')
    ap.add_argument('--rebuild', action='store_true', help='(--csv) replace ALL years from the CSV, including history')
    ap.add_argument('--probe', action='store_true', help='(--fetch) only report which months PRISM has published')
    ap.add_argument('--dry-run', action='store_true', help='show what would change; write nothing')
    ap.add_argument('--data-dir', default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    path = os.path.join(args.data_dir, 'county_monthly_climate.json')
    store, last_complete = load_store(path)
    base_complete, base_partial = year_status(store)
    log(f'Current data: complete through {base_complete[-1]}'
        + (f', partial {base_partial} ({observed_through(store, base_partial)})' if base_partial else '')
        + f', {len(store)} counties.')

    changed_years = set()
    if args.csv:
        log(f'Reading {args.csv} ...')
        changed_years = merge_csv(store, args.csv, base_complete[-1], args.rebuild)
    else:
        years = args.years or list(range(base_complete[-1] + 1, datetime.date.today().year + 1))
        gdf = load_counties(args.boundaries)
        by_norm = {norm(c): c for c in store}
        log(f'Fetching PRISM for {years} ({len(gdf)} county polygons) ...')
        for y in years:
            before = {c: list(store[c].get(y, [])) for c in store}
            fetch_year(y, gdf, store, by_norm, probe=args.probe)
            if not args.probe and any(store[c].get(y) != before[c] and store[c].get(y) for c in store):
                changed_years.add(y)
        if args.probe:
            return

    complete_years, partial_year = year_status(store)
    validate(store, complete_years, partial_year, base_complete)

    if not changed_years:
        log('Nothing new -- data files left unchanged.')
        return
    new_complete = [y for y in complete_years if y > base_complete[-1]]
    log(f'Changed years: {sorted(changed_years)}')
    log(f'Result: complete through {complete_years[-1]}'
        + (f', partial {partial_year} (real data through {observed_through(store, partial_year)})' if partial_year else ', no partial year')
        + (f'; newly COMPLETE: {new_complete}' if new_complete else ''))

    if args.dry_run:
        log('Dry run -- nothing written.')
        return
    meta = save_store(store, complete_years, partial_year, path)
    log(f'Wrote county_monthly_climate.json ({os.path.getsize(path):,} bytes), meta = {meta}')
    update_county_details(store, complete_years, changed_years & set(complete_years), args.data_dir, False)
    log('Done. Commit the changed JSON files; Render redeploys automatically.')


if __name__ == '__main__':
    main()
