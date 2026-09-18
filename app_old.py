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
CLIMATE_TREND = {}   # crop -> list of yearly records
MODEL_CACHE = OrderedDict()  # filename -> loaded model dict (LRU)


def slugify(text):
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')


# ---- Load only the small metadata/lookup files at startup ----
for crop in CROPS:
    meta_path = os.path.join(BASE_DIR, f'meta_{crop}.json')
    counties_path = os.path.join(BASE_DIR, f'counties_{crop}.json')
    trend_path = os.path.join(BASE_DIR, f'climate_trend_{crop}.json')

    if not os.path.exists(meta_path):
        print(f'WARNING: {meta_path} not found -- {crop} will be unavailable.')
        continue

    with open(meta_path) as f:
        META[crop] = json.load(f)
    with open(counties_path) as f:
        COUNTIES[crop] = json.load(f)
    with open(trend_path) as f:
        CLIMATE_TREND[crop] = json.load(f)
    print(f'Loaded metadata for {crop}: {len(META[crop])} model combinations available, '
          f'{len(COUNTIES[crop])} counties')


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
    return jsonify({'available_models': all_models})


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
                trend = CLIMATE_TREND.get(crop, [])
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


@app.route('/predict_year', methods=['POST'])
def predict_year():
    """Predict yield for a given year. If real climate data exists for that
    year (i.e. it falls within the historical record), use the actual
    recorded values. Only extrapolate the historical trend for years beyond
    the historical record, where no real data exists yet."""
    data = request.get_json(force=True)
    crop = data.get('crop')
    region = data.get('region')
    window_months_count = data.get('window_months_count')
    yield_type = data.get('yield_type', 'Overall_Yield')
    year = data.get('year')

    entry, error = get_model_entry(crop, region, window_months_count, yield_type)
    if entry is None:
        return jsonify({'detail': error}), 404

    trend = CLIMATE_TREND.get(crop, [])
    if not trend:
        return jsonify({'detail': f'No climate trend data for {crop}.'}), 404

    years = np.array([r['Year'] for r in trend])
    max_known_year = int(years.max())

    actual_row = next((r for r in trend if r['Year'] == year), None)

    if actual_row is not None:
        # Real recorded climate exists for this year -- use it directly,
        # no extrapolation, no "estimated" caveat needed.
        climate_for_year = {k: v for k, v in actual_row.items() if k != 'Year'}
        note = f'Using actual recorded climate data for {year}.'
    else:
        # No recorded data for this year (it's beyond the historical record) --
        # estimate it by extrapolating the trend, and say so plainly.
        climate_for_year = {}
        for col in entry['feat_cols']:
            if col not in trend[0]:
                continue
            vals = np.array([r[col] for r in trend])
            coeffs = np.polyfit(years, vals, deg=1)
            climate_for_year[col] = float(np.polyval(coeffs, year))
        note = (f'{year} is beyond the historical record (data available through {max_known_year}). '
                f'Climate is estimated by extrapolating the historical trend, not measured data. '
                f'Treat as approximate.')

    fake_request = {
        'crop': crop, 'region': region, 'window_months_count': window_months_count,
        'yield_type': yield_type, 'climate': climate_for_year, 'co2_ppm': climate_for_year.get('CO2_ppm'),
    }
    with app.test_request_context(json=fake_request):
        result_response = predict()
    result = result_response[0].get_json() if isinstance(result_response, tuple) else result_response.get_json()
    result['note'] = note
    result['estimated_climate'] = {k: round(v, 2) for k, v in climate_for_year.items()}
    return jsonify(result)


@app.route('/counties', methods=['GET'])
def counties():
    """Per-county summary stats for the map."""
    crop = request.args.get('crop')
    if crop not in COUNTIES:
        return jsonify({'detail': f'Unknown or unavailable crop: {crop}'}), 404
    return jsonify(COUNTIES[crop])


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
