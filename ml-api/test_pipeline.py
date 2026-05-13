import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import pipeline
from pipeline import (COMPACT_FEATURE_COLUMNS, EXTENDED_RUNTIME_FEATURE_COLUMNS,
                      FEATURE_COLUMNS, MAX_TRAINING_IMBALANCE_RATIO,
                      TARGET_TRAINING_IMBALANCE_RATIO, cross_validate_model,
                      compute_drift_report,
                      derive_label_thresholds, engineer_training_frame,
                      evaluate_adversarial_resilience, model_candidates,
                      prepare_prediction_frame,
                      rebalance_training_data, resolve_feature_columns)

SAMPLE_ROWS = [
    {'ip': 1, 'app': 3, 'device': 1, 'os': 13, 'channel': 111, 'click_time': '2017-11-06 14:32:21', 'is_attributed': 0},
    {'ip': 1, 'app': 3, 'device': 1, 'os': 13, 'channel': 111, 'click_time': '2017-11-06 14:32:25', 'is_attributed': 0},
    {'ip': 2, 'app': 5, 'device': 2, 'os': 18, 'channel': 200, 'click_time': '2017-11-06 14:40:00', 'is_attributed': 1},
]


class DummyModel:
    def predict_proba(self, features):
        click_frequency = np.asarray(features['clickFrequency'], dtype=float)
        burst_score = np.asarray(features['burstScore'], dtype=float)
        time_interval = np.asarray(features['timeInterval'], dtype=float)
        raw = 0.25 + (click_frequency / 20.0) + (burst_score / 15.0) - (time_interval / 200000.0)
        positive = np.clip(raw, 0.01, 0.99)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])


class BrokenModel:
    def predict_proba(self, features):
        raise AttributeError('monotonic_cst')


class PipelineTestCase(unittest.TestCase):
    def test_engineering_produces_compact_columns(self):
        # Default feature set is now 'extended_runtime'; test compact explicitly
        features, labels, info = engineer_training_frame(
            pd.DataFrame(SAMPLE_ROWS), feature_set='compact',
        )
        self.assertEqual(list(features.columns), COMPACT_FEATURE_COLUMNS)
        self.assertEqual(len(features), 3)
        self.assertEqual(labels.tolist(), [0, 0, 1])
        self.assertIn('thresholds', info)
        self.assertEqual(info['rows'], 3)
        self.assertAlmostEqual(info['observedAttributionRate'], 1 / 3)
        self.assertEqual(info['thresholdSource'], 'self')
        self.assertEqual(float(features.iloc[1]['hourOfDay']), 14.0)
        self.assertEqual(float(features.iloc[1]['clickFrequency']), 2.0)
        self.assertEqual(float(features.iloc[1]['timeInterval']), 4000.0)
        self.assertEqual(float(features.iloc[1]['ipClickCount']), 2.0)
        self.assertEqual(float(features.iloc[1]['appClickCount']), 2.0)
        self.assertEqual(float(features.iloc[1]['ipAppCount']), 2.0)
        self.assertAlmostEqual(float(features.iloc[1]['burstScore']), 0.5)

    def test_engineering_produces_extended_columns(self):
        features, labels, info = engineer_training_frame(
            pd.DataFrame(SAMPLE_ROWS), feature_set='extended_runtime',
        )
        self.assertEqual(list(features.columns), EXTENDED_RUNTIME_FEATURE_COLUMNS)
        self.assertEqual(len(features), 3)
        self.assertIn('timeSinceLastClickPerIp', features.columns)
        self.assertIn('clickCountLast10Seconds', features.columns)
        self.assertIn('clicksPerDeviceLast60Seconds', features.columns)
        self.assertIn('uniqueAppsPerIp', features.columns)
        self.assertIn('burstClickScore', features.columns)
        self.assertIn('uniqueDevicesPerIp', features.columns)
        self.assertEqual(info['featureSet'], 'extended_runtime')
        self.assertEqual(float(features.iloc[1]['timeSinceLastClickPerIp']), 4000.0)
        self.assertEqual(float(features.iloc[1]['clicksPerDeviceLast60Seconds']), 2.0)
        self.assertEqual(float(features.iloc[1]['uniqueAppsPerIp']), 1.0)

    def test_resolve_feature_columns(self):
        self.assertEqual(resolve_feature_columns('compact'), COMPACT_FEATURE_COLUMNS)
        self.assertEqual(resolve_feature_columns('extended_runtime'), EXTENDED_RUNTIME_FEATURE_COLUMNS)
        # Default is now extended_runtime
        self.assertEqual(resolve_feature_columns(None), EXTENDED_RUNTIME_FEATURE_COLUMNS)

    def test_engineering_accepts_external_thresholds_and_records_source(self):
        raw = pd.DataFrame([
            {'ip': 10, 'app': 7, 'device': 1, 'os': 13, 'channel': 111, 'click_time': '2017-11-06 14:32:21', 'is_attributed': 0},
            {'ip': 10, 'app': 7, 'device': 1, 'os': 13, 'channel': 111, 'click_time': '2017-11-06 14:32:28', 'is_attributed': 0},
            {'ip': 11, 'app': 8, 'device': 2, 'os': 19, 'channel': 112, 'click_time': '2017-11-06 15:01:00', 'is_attributed': 1},
        ])
        base_features, _, _ = engineer_training_frame(raw)
        thresholds = derive_label_thresholds(base_features)
        _, _, info = engineer_training_frame(raw, thresholds=thresholds, threshold_source='train_split')
        self.assertEqual(info['rows'], 3)
        self.assertEqual(info['thresholdSource'], 'train_split')
        self.assertEqual(info['thresholds'], thresholds)

    def test_cross_validation_uses_nested_calibration_split(self):
        rows = []
        for index in range(30):
            rows.append({
                'ip': index % 6, 'app': 10 if index % 3 == 0 else 20,
                'device': 1 if index % 2 == 0 else 2, 'os': 13 if index % 2 == 0 else 19,
                'channel': 111 if index % 3 == 0 else 222,
                'click_time': f'2017-11-06 14:{index:02d}:00',
                'is_attributed': 1 if index % 4 == 0 else 0,
            })
        features, labels, _ = engineer_training_frame(pd.DataFrame(rows))
        report = cross_validate_model(model_candidates()['logistic_regression'], features, labels)
        self.assertGreaterEqual(report['folds'], 5)
        self.assertIn('f1Mean', report['summary'])
        self.assertEqual(len(report['perFold']), report['folds'])

    def test_rebalance_training_data_caps_majority_ratio(self):
        features = pd.DataFrame({col: [float(i) for i in range(62)] for col in FEATURE_COLUMNS})
        labels = pd.Series([0] * 60 + [1] * 2)
        balanced_features, balanced_labels, summary = rebalance_training_data(
            features, labels, sampling_method='random_oversample',
        )
        self.assertTrue(summary['applied'])
        self.assertEqual(len(features), 62)
        self.assertLessEqual(summary['rebalancedImbalanceRatio'], MAX_TRAINING_IMBALANCE_RATIO)
        counts = balanced_labels.value_counts()
        self.assertLessEqual(float(counts.max() / counts.min()), MAX_TRAINING_IMBALANCE_RATIO)

    def test_rebalance_training_data_supports_smote(self):
        features = pd.DataFrame({col: [float(i) for i in range(68)] for col in FEATURE_COLUMNS})
        labels = pd.Series([0] * 60 + [1] * 8)
        balanced_features, balanced_labels, summary = rebalance_training_data(
            features, labels, max_majority_ratio=5, target_majority_ratio=3, sampling_method='smote',
        )
        self.assertTrue(summary['applied'])
        self.assertEqual(summary['originalClassCounts'], {'0': 60, '1': 8})
        counts = balanced_labels.value_counts()
        self.assertGreater(int(counts.loc[1]), 8)
        self.assertLessEqual(float(counts.max() / counts.min()), 3.0)

    def test_prediction_frame_compact(self):
        defaults = {col: 1.0 for col in FEATURE_COLUMNS}
        frame = prepare_prediction_frame({
            'clickFrequency': 4, 'timeInterval': 1500, 'deviceType': 'mobile',
            'click_time': '2017-11-11 16:45:00',
        }, defaults, COMPACT_FEATURE_COLUMNS)
        self.assertEqual(frame.shape, (1, len(COMPACT_FEATURE_COLUMNS)))
        self.assertEqual(float(frame.iloc[0]['clickFrequency']), 4.0)
        self.assertEqual(float(frame.iloc[0]['deviceCode']), 1.0)
        self.assertAlmostEqual(float(frame.iloc[0]['burstScore']), 4.0 / 1.5)

    def test_prediction_frame_extended(self):
        defaults = {col: 1.0 for col in EXTENDED_RUNTIME_FEATURE_COLUMNS}
        frame = prepare_prediction_frame({
            'clickFrequency': 4, 'timeInterval': 1500, 'deviceType': 'mobile',
            'click_time': '2017-11-11 16:45:00',
        }, defaults, EXTENDED_RUNTIME_FEATURE_COLUMNS)
        self.assertEqual(frame.shape, (1, len(EXTENDED_RUNTIME_FEATURE_COLUMNS)))
        self.assertIn('timeSinceLastClickPerIp', frame.columns)
        self.assertIn('clickCountLast10Seconds', frame.columns)
        self.assertIn('clicksPerDeviceLast60Seconds', frame.columns)
        self.assertIn('uniqueAppsPerIp', frame.columns)
        self.assertIn('burstClickScore', frame.columns)

    def test_compute_drift_report_flags_large_feature_shift(self):
        feature_defaults = {col: 1.0 for col in EXTENDED_RUNTIME_FEATURE_COLUMNS}
        baseline_frame = pd.DataFrame([
            {col: 1.0 for col in EXTENDED_RUNTIME_FEATURE_COLUMNS},
            {**{col: 1.0 for col in EXTENDED_RUNTIME_FEATURE_COLUMNS}, 'clickFrequency': 2.0, 'burstScore': 1.5},
            {**{col: 1.0 for col in EXTENDED_RUNTIME_FEATURE_COLUMNS}, 'clickFrequency': 3.0, 'burstScore': 2.0},
        ])
        bundle = {
            'feature_defaults': feature_defaults,
            'feature_columns': EXTENDED_RUNTIME_FEATURE_COLUMNS,
            'monitoring': {
                'featureBaselines': {
                    column: {
                        'mean': float(baseline_frame[column].mean()),
                        'median': float(baseline_frame[column].median()),
                        'std': max(float(baseline_frame[column].std(ddof=0)), 0.1),
                        'p10': float(baseline_frame[column].quantile(0.1)),
                        'p90': float(baseline_frame[column].quantile(0.9)),
                    }
                    for column in EXTENDED_RUNTIME_FEATURE_COLUMNS
                },
            },
        }

        report = compute_drift_report([
            {'clickFrequency': 20, 'timeInterval': 500, 'deviceType': 'mobile', 'burstScore': 18},
            {'clickFrequency': 18, 'timeInterval': 700, 'deviceType': 'mobile', 'burstScore': 16},
        ], bundle)

        self.assertIn(report['status'], {'warning', 'critical'})
        self.assertGreaterEqual(report['sampleCount'], 2)
        self.assertTrue(any(item['feature'] == 'clickFrequency' for item in report['flaggedFeatures']))

    def test_evaluate_adversarial_resilience_returns_scenarios(self):
        bundle = {
            'model': DummyModel(),
            'threshold': 0.5,
            'feature_defaults': {col: 1.0 for col in EXTENDED_RUNTIME_FEATURE_COLUMNS},
            'feature_columns': EXTENDED_RUNTIME_FEATURE_COLUMNS,
        }

        report = evaluate_adversarial_resilience({
            'clickFrequency': 6,
            'timeInterval': 2000,
            'deviceType': 'mobile',
            'burstScore': 4.5,
            'burstClickScore': 6.0,
        }, bundle)

        self.assertIn('baseline', report)
        self.assertEqual(len(report['scenarios']), 4)
        self.assertGreaterEqual(report['robustnessScore'], 0.0)
        self.assertLessEqual(report['robustnessScore'], 1.0)

    def test_load_or_train_bundle_upgrades_legacy_bundle_without_retraining(self):
        legacy_bundle = {
            'schema_version': 9,
            'model': DummyModel(),
            # Simulate an older LightGBM bundle that should be upgraded in-place
            # rather than triggering a fresh rebuild.
            'model_name': 'LightGBM (case 1:10)',
            'threshold': 0.5,
            'metrics': {'accuracy': 0.9},
            'all_results': {'dummy': {'accuracy': 0.9}},
            'feature_columns': EXTENDED_RUNTIME_FEATURE_COLUMNS,
            'feature_defaults': {col: 1.0 for col in EXTENDED_RUNTIME_FEATURE_COLUMNS},
            'dataset': {'datasetName': 'legacy.csv'},
            'labeling_strategy': 'legacy',
            'cross_validation': {'folds': 5},
            'evaluation': {'featureSet': 'extended_runtime'},
            'confusion_matrix': [[8, 1], [2, 5]],
            'roc_curve': {'fpr': [0.0, 1.0], 'tpr': [0.0, 1.0]},
            'pr_curve': {'precision': [1.0, 0.5], 'recall': [0.0, 1.0]},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir)
            model_path = artifacts_dir / 'model_bundle.pkl'
            metadata_path = artifacts_dir / 'model_metadata.json'
            model_path.write_bytes(pickle.dumps(legacy_bundle))

            # Patch in separate nested contexts (older Python & Windows
            # line-continuation rules can be picky about backslashes).
            with mock.patch.object(pipeline, 'ARTIFACTS_DIR', artifacts_dir):
                with mock.patch.object(pipeline, 'MODEL_PATH', model_path):
                    with mock.patch.object(pipeline, 'METADATA_PATH', metadata_path):
                        with mock.patch.object(
                            pipeline,
                            'build_lightgbm_case_control_bundle',
                            side_effect=AssertionError('unexpected rebuild'),
                        ):
                            with mock.patch.object(
                                pipeline,
                                'train_and_persist_bundle',
                                side_effect=AssertionError('unexpected retrain'),
                            ):
                                upgraded = pipeline.load_or_train_bundle()

            self.assertEqual(upgraded['schema_version'], pipeline.BUNDLE_SCHEMA_VERSION)
            self.assertIn('monitoring', upgraded)
            self.assertEqual(
                set(upgraded['monitoring']['featureBaselines'].keys()),
                set(EXTENDED_RUNTIME_FEATURE_COLUMNS),
            )
            self.assertEqual(
                upgraded['monitoring']['driftThresholds']['warningMeanShiftZ'],
                pipeline.DRIFT_WARN_Z_SCORE,
            )
            self.assertEqual(
                upgraded['monitoring']['driftThresholds']['criticalMeanShiftZ'],
                pipeline.DRIFT_CRITICAL_Z_SCORE,
            )
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            self.assertEqual(metadata['schema_version'], pipeline.BUNDLE_SCHEMA_VERSION)

    def test_load_or_train_bundle_retrains_when_loaded_model_cannot_predict(self):
        legacy_bundle = {
            'schema_version': pipeline.BUNDLE_SCHEMA_VERSION,
            'model': BrokenModel(),
            'model_name': 'broken',
            'threshold': 0.5,
            'metrics': {'accuracy': 0.0},
            'feature_columns': EXTENDED_RUNTIME_FEATURE_COLUMNS,
            'feature_defaults': {col: 1.0 for col in EXTENDED_RUNTIME_FEATURE_COLUMNS},
            'confusion_matrix': [[0, 1], [1, 0]],
            'roc_curve': {'fpr': [0.0, 1.0], 'tpr': [0.0, 1.0]},
            'pr_curve': {'precision': [1.0, 0.0], 'recall': [0.0, 1.0]},
        }
        retrained_bundle = {'schema_version': pipeline.BUNDLE_SCHEMA_VERSION, 'model': DummyModel()}

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir)
            model_path = artifacts_dir / 'model_bundle.pkl'
            metadata_path = artifacts_dir / 'model_metadata.json'
            model_path.write_bytes(pickle.dumps(legacy_bundle))

            with mock.patch.object(pipeline, 'ARTIFACTS_DIR', artifacts_dir), \
                    mock.patch.object(pipeline, 'MODEL_PATH', model_path), \
                    mock.patch.object(pipeline, 'METADATA_PATH', metadata_path), \
                    # When the existing model fails the prediction sanity check,
                    # load_or_train_bundle should fall back to building a fresh
                    # LightGBM bundle via build_lightgbm_case_control_bundle.
                    mock.patch.object(pipeline, 'build_lightgbm_case_control_bundle', return_value=retrained_bundle) as rebuild_mock:
                loaded = pipeline.load_or_train_bundle(max_rows=1234)

            self.assertIs(loaded, retrained_bundle)
        rebuild_mock.assert_called_once_with(dataset_name=None, max_rows=1234)


if __name__ == '__main__':
    unittest.main()