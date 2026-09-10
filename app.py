"""
Texas Yield Explorer — Flask backend for SmarterASP.NET hosting.

This does NOT train anything. It loads the model files produced by
TexasYieldExplorer_TrainAndExport.ipynb (run once in Colab) and serves
predictions from them.

Expected files in the same folder as this script (from deploy_package.zip):
  models_Corn.joblib, models_Cotton.joblib
  counties_Corn.json, counties_Cotton.json
  meta_Corn.json, meta_Cotton.json
  climate_trend_Corn.json, climate_trend_Cotton.json
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROPS = ['Corn', 'Cotton']

app = Flask(__name__)
CORS(app)  # allow the web app (hosted separately or elsewhere) to call this API

# ---- Load everything once, at startup ----
MODELS = {}          # crop -> {key: {'model', 'scaler', 'feat_cols', 'n'}}
META = {}            # crop -> list of dicts (from meta_{crop}.json)
COUNTIES = {}        # crop -> list of dicts
CLIMATE_TREND = {}   # crop -> list of yearly records

for crop in CROPS:
    models_path = os.path.join(BASE_DIR, f'models_{crop}.joblib')
    meta_path = os.path.join(BASE_DIR, f'meta_{crop}.json')
    counties_path = os.path.join(BASE_DIR, f'counties_{crop}.json')
    trend_path = os.path.join(BASE_DIR, f'climate_trend_{crop}.json')

    if not os.path.exists(models_path):
        print(f'WARNING: {models_path} not found — {crop} will be unavailable until uploaded.')
        continue

    MODELS[crop] = joblib.load(models_path)
    with open(meta_path) as f:
        META[crop] = json.load(f)
    with open(counties_path) as f:
        COUNTIES[crop] = json.load(f)
    with open(trend_path) as f:
        CLIMATE_TREND[crop] = json.load(f)
    print(f'Loaded {crop}: {len(MODELS[crop])} models, {len(COUNTIES[crop])} counties')


def get_model_entry(crop, region, window_months_count, yield_type):
    if crop not in MODELS:
        return None
    key = f'{region}|{window_months_count}|{yield_type}'
    return MODELS[crop].get(key)


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
    return jsonify({'status': 'ok', 'crops_loaded': list(MODELS.keys())})


if __name__ == '__main__':
    # For local testing only. On SmarterASP.NET, IIS/wfastcgi runs this via web.config instead.
    app.run(host='0.0.0.0', port=5000, debug=False)
