import hashlib
import hmac
import json
import os
import secrets
import time
from base64 import urlsafe_b64encode

from flask import Flask, jsonify, request
from flask_cors import CORS

from pipeline import (compute_drift_report, evaluate_adversarial_resilience,
                      load_or_train_bundle, predict_probability,
                      prepare_prediction_frame)

app = Flask(__name__)
CORS(app)
MODEL_BUNDLE = load_or_train_bundle()


def _oracle_signers():
    configured = []
    raw_config = os.getenv('ORACLE_SIGNERS', '').strip()
    if raw_config:
        for entry in raw_config.split(','):
            signer_id, _, secret = entry.partition(':')
            signer_id = signer_id.strip()
            secret = secret.strip()
            if signer_id and secret:
                configured.append({'id': signer_id, 'secret': secret})
    elif os.getenv('ORACLE_SHARED_SECRET'):
        configured.append({'id': 'oracle-1', 'secret': os.getenv('ORACLE_SHARED_SECRET', '')})
    return configured


def _oracle_summary():
    signers = _oracle_signers()
    required_quorum = min(len(signers), max(1, int(os.getenv('ORACLE_QUORUM', '1')))) if signers else 0
    return {
        'mode': 'hmac-quorum' if signers else 'unsigned-local',
        'signerCount': len(signers),
        'requiredQuorum': required_quorum,
        'decisionTtlMs': max(5_000, int(os.getenv('ORACLE_DECISION_TTL_MS', '30000'))),
    }


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _format_number(value):
    return 'null' if value is None else f'{float(value):.10f}'


def _prediction_payload(payload):
    data = dict(payload or {})
    if 'features' in data:
        features = data['features']
        data = {
            'clickFrequency': features[0] if isinstance(features, list) and len(features) > 0 else features.get('clickFrequency', 1),
            'timeInterval': features[1] if isinstance(features, list) and len(features) > 1 else features.get('timeInterval', 30_000),
            'deviceType': features[2] if isinstance(features, list) and len(features) > 2 else features.get('deviceType', 'desktop'),
            **(features if isinstance(features, dict) else {}),
        }
    return data


def _features_hash(payload):
    return hashlib.sha256(_canonical_json(_prediction_payload(payload)).encode('utf-8')).hexdigest()


def _oracle_signature_message(decision):
    return _canonical_json({
        'confidence': _format_number(decision.get('confidence')),
        'expiresAt': decision.get('expiresAt'),
        'featuresHash': decision.get('featuresHash'),
        'issuedAt': decision.get('issuedAt'),
        'model': decision.get('model'),
        'requestId': decision.get('requestId'),
        'result': str(decision.get('result')),
        'threshold': _format_number(decision.get('threshold')),
    })


def _sign_decision(decision):
    message = _oracle_signature_message(decision).encode('utf-8')
    return [
        {
            'signerId': signer['id'],
            'signature': urlsafe_b64encode(
                hmac.new(signer['secret'].encode('utf-8'), message, hashlib.sha256).digest()
            ).decode('utf-8').rstrip('='),
        }
        for signer in _oracle_signers()
    ]


def _build_prediction_response(payload):
    frame = prepare_prediction_frame(payload, MODEL_BUNDLE['feature_defaults'], MODEL_BUNDLE.get('feature_columns'))
    probability = predict_probability(MODEL_BUNDLE['model'], frame)
    prediction = int(probability >= MODEL_BUNDLE['threshold'])
    return {
        'result': prediction,
        'confidence': float(round(probability, 10)),
        'threshold': float(round(MODEL_BUNDLE['threshold'], 10)),
        'message': 'Valid click' if prediction == 1 else 'Suspicious click detected',
        'model': MODEL_BUNDLE['model_name'],
    }


def bundle_summary():
    monitoring = MODEL_BUNDLE.get('monitoring', {})
    return {
        'status': 'ok',
        'message': 'TalkingData fraud-screening API',
        'model': MODEL_BUNDLE['model_name'],
        'threshold': MODEL_BUNDLE['threshold'],
        'metrics': MODEL_BUNDLE['metrics'],
        'dataset': MODEL_BUNDLE['dataset'],
        'labelingStrategy': MODEL_BUNDLE['labeling_strategy'],
        'availableModels': MODEL_BUNDLE['all_results'],
        'crossValidation': MODEL_BUNDLE.get('cross_validation'),
        'evaluation': MODEL_BUNDLE.get('evaluation'),
        'monitoring': {
            'featureCount': len((monitoring.get('featureBaselines') or {}).keys()),
            'driftThresholds': monitoring.get('driftThresholds', {}),
        },
        'oracle': _oracle_summary(),
        'artifacts': {
            'metadata': 'artifacts/model_metadata.json',
            'rocCurve': 'artifacts/roc_curve.svg',
            'prCurve': 'artifacts/pr_curve.svg',
            'confusionMatrix': 'artifacts/confusion_matrix.svg',
        },
    }


@app.route('/')
@app.route('/health')
def health():
    return jsonify(bundle_summary())


@app.route('/model-metadata')
def model_metadata():
    return jsonify(bundle_summary())


@app.route('/train', methods=['POST'])
def train():
    global MODEL_BUNDLE
    payload = request.get_json(silent=True) or {}
    MODEL_BUNDLE = load_or_train_bundle(
        force_retrain=True,
        dataset_name=payload.get('datasetName'),
        max_rows=payload.get('maxRows'),
    )
    return jsonify({'message': 'Model retrained successfully', **bundle_summary()})


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(_build_prediction_response(_prediction_payload(payload)))
    except Exception as error:
        return jsonify({'message': 'Invalid request format', 'error': str(error)}), 400


@app.route('/predict-signed', methods=['POST'])
def predict_signed():
    payload = request.get_json(silent=True) or {}
    request_id = str(payload.get('requestId') or secrets.token_hex(12))
    oracle = _oracle_summary()
    try:
        prediction_payload = _prediction_payload(payload)
        response = _build_prediction_response(prediction_payload)
        issued_at_ms = int(time.time() * 1000)
        decision = {
            **response,
            'requestId': request_id,
            'issuedAt': time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime(issued_at_ms / 1000)),
            'expiresAt': time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime((issued_at_ms + oracle['decisionTtlMs']) / 1000)),
            'featuresHash': _features_hash(prediction_payload),
            'quorum': oracle['requiredQuorum'],
        }
        decision['signatures'] = _sign_decision(decision)
        decision['oracle'] = oracle
        return jsonify(decision)
    except Exception as error:
        return jsonify({'message': 'Invalid request format', 'error': str(error)}), 400


@app.route('/drift/report', methods=['POST'])
def drift_report():
    payload = request.get_json(silent=True) or {}
    samples = payload.get('samples') if isinstance(payload.get('samples'), list) else [_prediction_payload(payload)]
    try:
        return jsonify(compute_drift_report(samples, MODEL_BUNDLE))
    except Exception as error:
        return jsonify({'message': 'Unable to compute drift report', 'error': str(error)}), 400


@app.route('/adversarial-evaluation', methods=['POST'])
def adversarial_evaluation():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(evaluate_adversarial_resilience(_prediction_payload(payload), MODEL_BUNDLE))
    except Exception as error:
        return jsonify({'message': 'Unable to compute adversarial evaluation', 'error': str(error)}), 400


if __name__ == '__main__':
    print('Starting TalkingData-backed fraud-screening API on http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=True)

