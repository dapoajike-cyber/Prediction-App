"""
Texas Yield Explorer -- Flask backend (Render-ready).

This does NOT train anything. It loads model files produced by
TexasYieldExplorer_TrainAndExport.ipynb (run once in Colab) and serves
predictions from them.

IMPORTANT: models are loaded LAZILY, one at a time, only when actually
requested -- not all at once at startup. Loading every region/window
combination into memory simultaneously exceeded free-tier hosting's 512MB
RAM limit. A small cache keeps a handful of recently-used models in memory
and evicts the oldest when it gets full, so memory use stays bounded no
matter how many different combinations get requested over time.

Expected files in the same folder as this script:
  meta_Corn.json, meta_Cotton.json
  counties_Corn.json, counties_Cotton.json
  climate_trend_Corn.json, climate_trend_Cotton.json

Model files (model_<crop>_<region>_<window>_<yieldtype>.joblib) are NOT
expected to be present locally -- they are downloaded on demand from a
GitHub Release the first time each one is actually needed.
"""

import os
import re
import json
import urllib.request
from collections import OrderedDict

import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROPS = ['Corn', 'Cotton']

# Base URL where all model_*.joblib files live -- uploaded directly into the
# main branch of the repo (they're small enough now, ~1-11MB each, unlike the
# old single big files per crop). Using raw.githubusercontent.com serves the
# actual file content directly, same as a normal file download.
MODEL_BASE_URL = 'https://raw.githubusercontent.com/dapoajike-cyber/Prediction-App/main'

# How many models to keep cached in memory at once. Each is a handful of MB,
# so this stays well within free-tier memory limits.
MAX_CACHED_MODELS = 3

app = Flask(__name__)
CORS(app)

META = {}            # crop -> list of dicts (from meta_{crop}.json)
COUNTIES = {}        # crop -> list of dicts
CLIMATE_TREND = {}   # crop -> region -> list of yearly records
MODEL_CACHE = OrderedDict()  # filename -> loaded model dict (LRU)

# Full per-county, per-year, full-year (Jan-Dec) monthly climate record --
# shared across both crops since it's the same underlying PRISM data, not
# gated by whether a county had a reported yield that year. This is what
# lets predictions be scoped to one specific county's own real climate,
# instead of a regional average or an approximated ratio.
#
# Stored compactly as county -> {'years': [...], 'data': [[84 floats], ...]}
# with a single shared column-name list (COUNTY_CLIMATE_COLS), rather than a
# dict-of-84-keys per year -- repeating 84 string keys per year per county
# (254 counties x 58 years) was the single biggest driver of memory use on
# the free-tier host, so this trades a small amount of lookup code for a
# large reduction in loaded memory.
COUNTY_CLIMATE_COLS = []      # e.g. ['ppt_Jan', ..., 'vpdmax_Dec']
COUNTY_MONTHLY_CLIMATE = {}   # county -> {'years': [...], 'data': [[...], ...], optional 'partial': {...}}
COUNTY_CLIMATE_META = {}      # optional: last_complete_year, partial_year, observed_through
COUNTY_DEFAULT_REGION = {}    # county -> best-matching modeled region


def slugify(text):
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')


COUNTY_DETAILS = {}  # crop -> {county_name: {yield_stats, yearly, monthly_temp_climatology}}
                      # Optional, richer per-county data (stats + charts). Only
                      # present for crops where the raw source data was available
                      # to compute it -- currently Cotton only.

# ---- Load only the small metadata/lookup files at startup ----
for crop in CROPS:
    meta_path = os.path.join(BASE_DIR, f'meta_{crop}.json')
    counties_path = os.path.join(BASE_DIR, f'counties_{crop}.json')
    trend_path = os.path.join(BASE_DIR, f'climate_trend_{crop}.json')
    county_details_path = os.path.join(BASE_DIR, f'county_details_{crop}.json')

    if not os.path.exists(meta_path):
        print(f'WARNING: {meta_path} not found -- {crop} will be unavailable.')
        continue

    with open(meta_path) as f:
        META[crop] = json.load(f)
    with open(counties_path) as f:
        COUNTIES[crop] = json.load(f)
    with open(trend_path) as f:
        CLIMATE_TREND[crop] = json.load(f)

    if os.path.exists(county_details_path):
        with open(county_details_path) as f:
            COUNTY_DETAILS[crop] = json.load(f)
        print(f'Loaded county detail data for {crop}: {len(COUNTY_DETAILS[crop])} counties')
    else:
        print(f'No county detail data file for {crop} -- per-county charts unavailable for this crop.')

    print(f'Loaded metadata for {crop}: {len(META[crop])} model combinations available, '
          f'{len(COUNTIES[crop])} counties')

county_monthly_climate_path = os.path.join(BASE_DIR, 'county_monthly_climate.json')
if os.path.exists(county_monthly_climate_path):
    with open(county_monthly_climate_path) as f:
        raw = json.load(f)
    COUNTY_CLIMATE_COLS = raw['cols']
    COUNTY_MONTHLY_CLIMATE = raw['counties']
    # Optional: describes the in-progress current year, if the data file has
    # one (written by update_prism_data.py). Absent in older data files.
    COUNTY_CLIMATE_META = raw.get('meta', {})
    print(f'Loaded full monthly climate for {len(COUNTY_MONTHLY_CLIMATE)} counties '
          f'({len(COUNTY_CLIMATE_COLS)} columns/year). Meta: {COUNTY_CLIMATE_META}')
else:
    print('WARNING: county_monthly_climate.json not found -- per-county predictions unavailable.')

county_default_region_path = os.path.join(BASE_DIR, 'county_default_region.json')
if os.path.exists(county_default_region_path):
    with open(county_default_region_path) as f:
        COUNTY_DEFAULT_REGION = json.load(f)


def get_climate_trend(crop, region):
    """Returns the list of yearly climate records for a given crop+region.

    climate_trend_{crop}.json is keyed by region ('All Texas', 'High Plains',
    'North Central', 'South Central'), each a list of yearly records -- so
    that predictions for a division-specific model use that division's own
    historical climate, not the statewide average. Falls back to 'All Texas'
    if the requested region isn't found (e.g. an older data file), so this
    never hard-fails."""
    trend_by_region = CLIMATE_TREND.get(crop)
    if trend_by_region is None:
        return []
    if isinstance(trend_by_region, dict):
        return trend_by_region.get(region) or trend_by_region.get('All Texas') or []
    # Backward compatibility with the old flat-list format (pre region split).
    return trend_by_region


def find_meta_entry(crop, region, window_months_count, yield_type):
    for entry in META.get(crop, []):
        if (entry['region'] == region and entry['window_months_count'] == window_months_count
                and entry['yield_type'] == yield_type):
            return entry
    return None


def get_model_entry(crop, region, window_months_count, yield_type):
    """Loads a model on demand, downloading it first if necessary. Caches a
    small number of recently-used models in memory; evicts the oldest when full.
    Returns (entry, error_message) -- error_message is None on success, and
    distinguishes 'not in catalog' from 'found in catalog but download failed'."""
    meta_entry = find_meta_entry(crop, region, window_months_count, yield_type)
    if meta_entry is None:
        return None, (f'No trained model for crop={crop}, region={region}, '
                       f'window={window_months_count} months, yield_type={yield_type}.')

    filename = meta_entry.get('model_filename')
    if not filename:
        return None, f'Catalog entry found but has no model_filename recorded.'

    if filename in MODEL_CACHE:
        MODEL_CACHE.move_to_end(filename)
        return MODEL_CACHE[filename], None

    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath):
        url = f'{MODEL_BASE_URL}/{filename}'
        print(f'Downloading {filename} from {url} ...')
        try:
            # GitHub's raw content server can reject requests with no User-Agent
            # header -- urlretrieve()'s default request doesn't send one, so we
            # build the request explicitly instead.
            req = urllib.request.Request(url, headers={'User-Agent': 'texas-yield-explorer/1.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError(f'HTTP {response.status} from {url}')
                data = response.read()
            with open(filepath, 'wb') as f:
                f.write(data)
            print(f'  Downloaded {filename} ({len(data)/1024:.0f} KB)')
        except Exception as e:
            print(f'  ERROR downloading {filename}: {e}')
            return None, f'Model was found in the catalog but could not be downloaded ({e}).'

    try:
        entry = joblib.load(filepath)
    except Exception as e:
        print(f'  ERROR loading {filename}: {e}')
        if os.path.exists(filepath):
            os.remove(filepath)  # remove a possibly-corrupt partial download
        return None, f'Model file downloaded but could not be loaded ({e}).'

    while len(MODEL_CACHE) >= MAX_CACHED_MODELS:
        old_filename, _ = MODEL_CACHE.popitem(last=False)
        old_path = os.path.join(BASE_DIR, old_filename)
        if os.path.exists(old_path):
            os.remove(old_path)

    MODEL_CACHE[filename] = entry
    return entry, None


@app.route('/meta', methods=['GET'])
def meta():
    """Tells the frontend which crops/regions/windows/features are available."""
    all_models = []
    for crop in CROPS:
        all_models.extend(META.get(crop, []))
    # Describes how current the county climate file is, so the frontend can
    # label the in-progress year. Empty dict for older data files.
    last_complete = COUNTY_CLIMATE_META.get('last_complete_year')
    climate_data = dict(COUNTY_CLIMATE_META)
    if last_complete is not None:
        climate_data['max_prediction_year'] = last_complete + MAX_EXTRAPOLATION_YEARS
    return jsonify({'available_models': all_models, 'climate_data': climate_data})


@app.route('/predict', methods=['POST'])
def predict():
    """Predict yield from user-supplied climate values."""
    data = request.get_json(force=True)
    crop = data.get('crop')
    region = data.get('region')
    window_months_count = data.get('window_months_count')
    yield_type = data.get('yield_type', 'Overall_Yield')
    climate = data.get('climate', {})
    co2_ppm = data.get('co2_ppm')

    entry, error = get_model_entry(crop, region, window_months_count, yield_type)
    if entry is None:
        return jsonify({'detail': error}), 404

    feat_cols = entry['feat_cols']
    row = []
    missing = []
    for col in feat_cols:
        if col == 'CO2_ppm':
            if co2_ppm is not None:
                row.append(co2_ppm)
            else:
                trend = get_climate_trend(crop, region)
                row.append(trend[-1]['CO2_ppm'] if trend else 420.0)
        elif col in climate:
            row.append(climate[col])
        else:
            missing.append(col)

    if missing:
        return jsonify({'detail': f'Missing climate values: {missing}'}), 400

    X = np.array(row).reshape(1, -1)
    X_s = entry['scaler'].transform(X)
    pred = float(entry['model'].predict(X_s)[0])

    tree_preds = [float(t.predict(X_s)[0]) for t in entry['model'].estimators_]
    lo, hi = float(np.percentile(tree_preds, 10)), float(np.percentile(tree_preds, 90))

    return jsonify({
        'prediction': round(pred, 1),
        'range_low': round(lo, 1),
        'range_high': round(hi, 1),
        'unit': 'lb/acre' if crop == 'Cotton' else 'bu/acre',
        'n_training_samples': entry['n'],
    })


def _normalize_county_name(name):
    return ''.join(ch for ch in name.lower() if ch.isalnum())


def _find_county_climate_raw(county):
    """Looks up a county's {'years': [...], 'data': [[...], ...]} record,
    tolerating naming mismatches (e.g. 'DeWitt' vs 'De Witt') the same way
    county_detail does."""
    if county in COUNTY_MONTHLY_CLIMATE:
        return COUNTY_MONTHLY_CLIMATE[county]
    target = _normalize_county_name(county)
    for name, rec in COUNTY_MONTHLY_CLIMATE.items():
        if _normalize_county_name(name) == target:
            return rec
    return None


def _county_year_climate(county, year):
    """Returns {col: value} for one county+year (or None if that county/year
    isn't available), reconstructed on demand from the compact array storage."""
    rec = _find_county_climate_raw(county)
    if rec is None:
        return None
    years = rec['years']
    if year not in years:
        return None
    row = rec['data'][years.index(year)]
    return dict(zip(COUNTY_CLIMATE_COLS, row))


def _county_climate_column_series(county):
    """Returns (years, {col: [values aligned to years]}) for one county --
    used to fit an extrapolation trend per column."""
    rec = _find_county_climate_raw(county)
    if rec is None:
        return None, None
    years = rec['years']
    by_col = {col: [row[i] for row in rec['data']] for i, col in enumerate(COUNTY_CLIMATE_COLS)}
    return years, by_col


# Beyond this many years past a county's last recorded climate year, we
# refuse to extrapolate rather than silently returning a guess. Per Dr.
# Awal: a 58-year linear trend is a weak predictor of any single future
# year to begin with, and the further out you go, the more likely the
# extrapolated climate falls outside what the Random Forest ever saw in
# training -- at which point its prediction isn't really informed by
# anything, even though it still returns a confident-looking number.
MAX_EXTRAPOLATION_YEARS = 5


def _extrapolation_limit_error(county, year, max_known_year):
    limit_year = max_known_year + MAX_EXTRAPOLATION_YEARS
    return {
        'detail': (
            f'{year} is more than {MAX_EXTRAPOLATION_YEARS} years beyond {county} County\'s '
            f'last recorded climate year ({max_known_year}). To keep predictions reliable, this '
            f'app only extrapolates up to {MAX_EXTRAPOLATION_YEARS} years past a county\'s most '
            f'recent climate data -- try a year through {limit_year}.'
        )
    }


def _default_region_for_county(county):
    if county in COUNTY_DEFAULT_REGION:
        return COUNTY_DEFAULT_REGION[county]
    target = _normalize_county_name(county)
    for name, region in COUNTY_DEFAULT_REGION.items():
        if _normalize_county_name(name) == target:
            return region
    return 'All Texas'


MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _column_trend_value(years, series, year):
    """Linear trend (per county, per column) evaluated at `year`, fit on the
    complete historical years only. None if there is too little data."""
    xs = [yr for yr, v in zip(years, series) if v is not None]
    ys = [v for v in series if v is not None]
    if len(xs) < 2:
        return None
    coeffs = np.polyfit(np.array(xs), np.array(ys), deg=1)
    return float(np.polyval(coeffs, year))


def _resolve_county_climate(county, year, only_cols=None):
    """Single source of truth for 'what climate do we use for this county+year?'.

    Returns (info, None) on success or (None, (status, body)) on failure.
    info = {
      'climate': {col: value},
      'mode': 'recorded' | 'partial' | 'extrapolated',
      'max_known_year': last COMPLETE recorded year for the county,
      'observed_months': months (of the in-progress year) that have real data,
      'estimated_months': months filled from the county trend,
    }

    * recorded     -- a complete historical year: real PRISM values.
    * partial      -- the current, in-progress year: months PRISM has published
                      are used as-is; the remaining months are filled from the
                      county's own trend (same method as future years).
    * extrapolated -- a future year: every month from the county's own trend,
                      limited to MAX_EXTRAPOLATION_YEARS past the last COMPLETE
                      year (so a half-finished year never moves the cap).
    """
    rec = _find_county_climate_raw(county)
    if rec is None:
        return None, (404, {'detail': f'No climate data available for county "{county}".'})
    years = rec['years']
    max_known_year = max(years)
    cols = COUNTY_CLIMATE_COLS

    if year in years:
        row = rec['data'][years.index(year)]
        return {'climate': dict(zip(cols, row)), 'mode': 'recorded',
                'max_known_year': max_known_year,
                'observed_months': [], 'estimated_months': []}, None

    if year > max_known_year + MAX_EXTRAPOLATION_YEARS:
        return None, (400, _extrapolation_limit_error(county, year, max_known_year))

    idx_by_col = {c: i for i, c in enumerate(cols)}
    wanted = [c for c in cols if (only_cols is None or c in only_cols)]

    def trend_for(col):
        i = idx_by_col[col]
        return _column_trend_value(years, [row[i] for row in rec['data']], year)

    partial = rec.get('partial')
    if partial and partial.get('year') == year:
        prow = partial['data']
        climate, filled_cols = {}, set()
        for col in wanted:
            v = prow[idx_by_col[col]]
            if v is None:
                v = trend_for(col)
                filled_cols.add(col)
            if v is not None:
                climate[col] = v
        # A month counts as "recorded" only if all of its variables are real.
        observed, estimated = [], []
        for m in MONTHS:
            month_cols = [c for c in cols if c.endswith('_' + m)]
            if all(prow[idx_by_col[c]] is not None for c in month_cols):
                observed.append(m)
            else:
                estimated.append(m)
        return {'climate': climate, 'mode': 'partial', 'max_known_year': max_known_year,
                'observed_months': observed, 'estimated_months': estimated}, None

    climate = {}
    for col in wanted:
        v = trend_for(col)
        if v is not None:
            climate[col] = v
    return {'climate': climate, 'mode': 'extrapolated', 'max_known_year': max_known_year,
            'observed_months': [], 'estimated_months': []}, None


def _window_months(feat_cols):
    """Month abbreviations a model actually uses, in calendar order."""
    used = {c.rsplit('_', 1)[-1] for c in feat_cols if c != 'CO2_ppm'}
    return [m for m in MONTHS if m in used]


@app.route('/county_climate', methods=['GET'])
def county_climate():
    """Returns one county's climate for a given year (all 12 months x 7
    variables) -- used to auto-fill the manual climate entry fields, and as
    the source for per-county 'Pick a year' predictions. Complete historical
    years are real PRISM data. The current in-progress year uses the months
    PRISM has published so far, with the rest filled from the county's own
    trend. Later years are fully extrapolated (capped)."""
    county = request.args.get('county')
    year = request.args.get('year', type=int)
    if not county or year is None:
        return jsonify({'detail': 'county and year are required.'}), 400

    info, err = _resolve_county_climate(county, year)
    if err:
        return jsonify(err[1]), err[0]

    climate = info['climate'] if info['mode'] == 'recorded' else {
        k: (round(v, 2) if v is not None else None) for k, v in info['climate'].items()}
    body = {'county': county, 'year': year, 'climate': climate,
            'estimated': info['mode'] == 'extrapolated',
            'partial': info['mode'] == 'partial',
            'max_known_year': info['max_known_year']}
    if info['mode'] == 'partial':
        body['observed_months'] = info['observed_months']
        body['estimated_months'] = info['estimated_months']
    return jsonify(body)


@app.route('/predict_year', methods=['POST'])
def predict_year():
    """Predict yield for a given YEAR AND COUNTY, using that county's own
    climate (recorded, current-year-so-far, or extrapolated -- see
    _resolve_county_climate) run through the selected region's trained model.
    This is a real prediction for that specific county, not a regional
    prediction scaled by a historical ratio."""
    data = request.get_json(force=True)
    crop = data.get('crop')
    region = data.get('region')
    window_months_count = data.get('window_months_count')
    yield_type = data.get('yield_type', 'Overall_Yield')
    year = data.get('year')
    county = data.get('county')

    entry, error = get_model_entry(crop, region, window_months_count, yield_type)
    if entry is None:
        return jsonify({'detail': error}), 404

    if not county:
        return jsonify({'detail': 'county is required.'}), 400

    feat_cols = [c for c in entry['feat_cols'] if c != 'CO2_ppm']
    info, err = _resolve_county_climate(county, year, only_cols=set(feat_cols))
    if err:
        return jsonify(err[1]), err[0]
    climate_for_year = info['climate']
    mode = info['mode']

    window = _window_months(entry['feat_cols'])
    observed_in_window = [m for m in window if m in info['observed_months']]
    estimated_in_window = [m for m in window if m in info['estimated_months']]

    if mode == 'recorded':
        note = f'Using actual recorded climate data for {county} County, {year}.'
    elif mode == 'partial':
        if estimated_in_window:
            note = (f'{year} is still in progress. Using {county} County\'s recorded climate for '
                    f'{", ".join(observed_in_window) if observed_in_window else "none of the growing-season months"}'
                    f'; {", ".join(estimated_in_window)} not yet available and estimated from the '
                    f'county\'s own historical trend. The prediction will firm up as more months are published.')
        else:
            note = (f'Using {county} County\'s recorded {year} climate for the full growing window '
                    f'({", ".join(window)}).')
    else:
        note = (f'{year} is beyond {county} County\'s historical record (data available through '
                f'{info["max_known_year"]}). Climate is estimated by extrapolating that county\'s own '
                f'historical trend, not measured data. Treat as approximate.')

    fake_request = {
        'crop': crop, 'region': region, 'window_months_count': window_months_count,
        'yield_type': yield_type, 'climate': climate_for_year, 'co2_ppm': None,
    }
    with app.test_request_context(json=fake_request):
        result_response = predict()
    result = result_response[0].get_json() if isinstance(result_response, tuple) else result_response.get_json()
    result['note'] = note
    result['county'] = county
    result['estimated_climate'] = {k: (round(v, 2) if v is not None else None) for k, v in climate_for_year.items()}
    # 'estimated' only if some month the model actually uses had to be filled in.
    result['climate_estimated'] = mode == 'extrapolated' or (mode == 'partial' and bool(estimated_in_window))
    if mode == 'partial' and estimated_in_window:
        result['climate_partial'] = True
        result['observed_months'] = observed_in_window
        result['estimated_months'] = estimated_in_window
    return jsonify(result)


@app.route('/counties', methods=['GET'])
def counties():
    """Per-county summary stats for the map, plus each county's best-matching
    modeled region (used to auto-select the Region dropdown when a county is
    chosen as the prediction target)."""
    crop = request.args.get('crop')
    if crop not in COUNTIES:
        return jsonify({'detail': f'Unknown or unavailable crop: {crop}'}), 404
    rows = []
    for row in COUNTIES[crop]:
        row = dict(row)
        row['default_region'] = _default_region_for_county(row['County'])
        rows.append(row)
    return jsonify(rows)


@app.route('/county_detail', methods=['GET'])
def county_detail():
    """Full per-county stats and yearly series (yield, precipitation, monthly
    temperature climatology), for the click-to-view detail panel. Only
    available for crops where COUNTY_DETAILS was loaded (currently Cotton)."""
    crop = request.args.get('crop')
    county = request.args.get('county')
    if not crop or not county:
        return jsonify({'detail': 'crop and county query parameters are required.'}), 400
    if crop not in COUNTY_DETAILS:
        return jsonify({'detail': f'No detailed county data available for {crop} yet.'}), 404

    data = COUNTY_DETAILS[crop]
    if county in data:
        return jsonify(data[county])

    # Fall back to a normalized name match (e.g. "DeWitt" vs "De Witt")
    target = _normalize_county_name(county)
    for name, row in data.items():
        if _normalize_county_name(name) == target:
            return jsonify(row)

    return jsonify({'detail': f'No detailed data found for county "{county}".'}), 404


@app.route('/', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'crops_loaded': list(META.keys()),
        'models_currently_cached': list(MODEL_CACHE.keys()),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
