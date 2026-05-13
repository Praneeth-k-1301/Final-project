"""
Focused experiment sweep for AdChain ML pipeline.
Trains all model+sampler combinations and saves results to sweep_result.json.
"""
import os
import json
import sys
import traceback
from pathlib import Path

# Use 300K rows to capture more minority examples (~507 positives in train)
os.environ.setdefault('MAX_TRAIN_ROWS', '300000')

STATUS_PATH = Path(__file__).parent / 'sweep_status.txt'
RESULT_PATH = Path(__file__).parent / 'sweep_result.json'

def log(msg):
    print(msg, flush=True)
    with open(STATUS_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

STATUS_PATH.write_text('', encoding='utf-8')
log('Starting sweep...')

try:
    from pipeline import train_and_persist_bundle, _serializable_results
    log('Pipeline imported OK.')

    log('Loading data and training all experiments...')
    bundle = train_and_persist_bundle()
    log('Training complete.')

    # Build serializable summary
    all_results = bundle.get('all_results', {})
    rows = []
    for exp_name, r in all_results.items():
        m = r.get('metrics', {})
        vm = r.get('validation_metrics', {})
        sc = r.get('sampling_config', {})
        rows.append({
            'experiment': exp_name,
            'model': r.get('model_name', exp_name),
            'feature_set': r.get('feature_set', 'extended_runtime'),
            'sampler': sc.get('name', ''),
            'threshold': round(float(r.get('threshold', 0.5)), 4),
            'precision': round(float(m.get('precision', 0)), 4),
            'recall': round(float(m.get('recall', 0)), 4),
            'f1': round(float(m.get('f1', 0)), 4),
            'roc_auc': round(float(m.get('roc_auc', 0)), 4),
            'average_precision': round(float(m.get('average_precision', 0)), 4),
            'val_f1': round(float(vm.get('f1', 0)), 4),
            'val_roc_auc': round(float(vm.get('roc_auc', 0)), 4),
        })

    rows_sorted = sorted(rows, key=lambda x: x['f1'], reverse=True)

    best = bundle.get('metrics', {})
    cm = bundle.get('confusion_matrix', {})
    cv = bundle.get('cross_validation', {})

    result = {
        'best_model': bundle.get('model_name', ''),
        'best_feature_set': bundle.get('dataset', {}).get('selectedFeatureSet', ''),
        'best_threshold': round(float(bundle.get('threshold', 0.5)), 4),
        'best_metrics': {k: round(float(v), 4) for k, v in best.items() if isinstance(v, (int, float))},
        'confusion_matrix': cm,
        'cross_validation': cv,
        'dataset': {k: v for k, v in bundle.get('dataset', {}).items() if not isinstance(v, dict)},
        'experiment_table': rows_sorted,
        'total_experiments': len(rows),
    }

    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding='utf-8')
    log(f'Results saved to {RESULT_PATH}')
    log(f'Best model: {result["best_model"]}  F1={result["best_metrics"].get("f1")}  ROC-AUC={result["best_metrics"].get("roc_auc")}')
    log('SWEEP COMPLETE')
    print(json.dumps(result, indent=2))

except Exception as e:
    tb = traceback.format_exc()
    log(f'SWEEP FAILED: {e}')
    log(tb)
    sys.exit(1)

