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
    small number of recently-used models in memory; evicts the oldest when full."""
    meta_entry = find_meta_entry(crop, region, window_months_count, yield_type)
    if meta_entry is None:
        return None

    filename = meta_entry.get('model_filename')
    if not filename:
        return None

    if filename in MODEL_CACHE:
        MODEL_CACHE.move_to_end(filename)  # mark as recently used
        return MODEL_CACHE[filename]

    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath):
        url = f'{MODEL_BASE_URL}/{filename}'
        print(f'Downloading {filename} from {url} ...')
        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f'  ERROR downloading {filename}: {e}')
            return None

    entry = joblib.load(filepath)

    # Evict oldest cached model(s) if we are at capacity, and remove the file
    # from disk too so it does not accumulate across many different requests.
    while len(MODEL_CACHE) >= MAX_CACHED_MODELS:
        old_filename, _ = MODEL_CACHE.popitem(last=False)
        old_path = os.path.join(BASE_DIR, old_filename)
        if os.path.exists(old_path):
            os.remove(old_path)

    MODEL_CACHE[filename] = entry
    return entry


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

    entry = get_model_entry(crop, region, window_months_count, yield_type)
    if entry is None:
        return jsonify({'detail': f'No trained model for crop={crop}, region={region}, '
                                   f'window={window_months_count} months, yield_type={yield_type}.'}), 404

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
    """Predict yield for a future year by extrapolating the historical climate trend."""
    data = request.get_json(force=True)
    crop = data.get('crop')
    region = data.get('region')
    window_months_count = data.get('window_months_count')
    yield_type = data.get('yield_type', 'Overall_Yield')
    year = data.get('year')

    entry = get_model_entry(crop, region, window_months_count, yield_type)
    if entry is None:
        return jsonify({'detail': f'No trained model for crop={crop}, region={region}, '
                                   f'window={window_months_count} months, yield_type={yield_type}.'}), 404

    trend = CLIMATE_TREND.get(crop, [])
    if not trend:
        return jsonify({'detail': f'No climate trend data for {crop}.'}), 404

    years = np.array([r['Year'] for r in trend])
    est_climate = {}
    for col in entry['feat_cols']:
        if col not in trend[0]:
            continue
        vals = np.array([r[col] for r in trend])
        coeffs = np.polyfit(years, vals, deg=1)
        est_climate[col] = float(np.polyval(coeffs, year))

    fake_request = {
        'crop': crop, 'region': region, 'window_months_count': window_months_count,
        'yield_type': yield_type, 'climate': est_climate, 'co2_ppm': est_climate.get('CO2_ppm'),
    }
    with app.test_request_context(json=fake_request):
        result_response = predict()
    result = result_response[0].get_json() if isinstance(result_response, tuple) else result_response.get_json()
    result['note'] = (f'Climate for {year} is estimated by extrapolating the historical trend, '
                       'not measured data. Treat as approximate.')
    result['estimated_climate'] = {k: round(v, 2) for k, v in est_climate.items()}
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
