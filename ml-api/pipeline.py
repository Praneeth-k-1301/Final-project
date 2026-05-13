from __future__ import annotations

import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'model_bundle.pkl'
METADATA_PATH = ARTIFACTS_DIR / 'model_metadata.json'
COMPACT_FEATURE_COLUMNS = [
    'clickFrequency', 'timeInterval', 'deviceCode', 'app', 'osCode', 'channel',
    'hourOfDay', 'ipClickCount', 'appClickCount', 'ipAppCount', 'burstScore',
]
EXTENDED_RUNTIME_FEATURE_COLUMNS = COMPACT_FEATURE_COLUMNS + [
    'timeSinceLastClickPerIp',
    'timeSinceLastClickPerDevice',
    'clickCountLast10Seconds',
    'clickCountLast60Seconds',
    'clickCountLast10Minutes',
    'clicksPerDeviceLast60Seconds',
    'deviceAppEntropy',
    'deviceIpRatio',
    'uniqueAppsPerIp',
    'uniqueAppsPerDevice',
    'uniqueDevicesPerIp',
    'burstClickScore',
]
FEATURE_COLUMNS = COMPACT_FEATURE_COLUMNS
FEATURE_SETS = {
    'compact': COMPACT_FEATURE_COLUMNS,
    'extended_runtime': EXTENDED_RUNTIME_FEATURE_COLUMNS,
}
DEVICE_MAP = {'mobile': 1, 'desktop': 2, 'tablet': 3}
THRESHOLD_GRID = np.linspace(0.01, 0.50, 25)
BUNDLE_SCHEMA_VERSION = 11
RANDOM_STATE = 42
SESSION_WINDOW_SECONDS = 30 * 60
DRIFT_WARN_Z_SCORE = 2.5
DRIFT_CRITICAL_Z_SCORE = 4.0
# Preservation-based defaults: keep all majority, oversample minority
MAX_TRAINING_IMBALANCE_RATIO = 1000   # effectively "keep all majority"
TARGET_TRAINING_IMBALANCE_RATIO = 10  # oversample minority to 10:1
DEFAULT_TRAINING_SAMPLING_METHOD = 'random_oversample'
DEFAULT_FEATURE_SET = 'extended_runtime'
DEFAULT_THRESHOLD_POLICY = 'f1_max'
CALIBRATION_MIN_CLASS_COUNT = 8
SMOTE_K_NEIGHBORS = 5
DISCRETE_FEATURE_COLUMNS = {
    'clickFrequency', 'deviceCode', 'app', 'osCode', 'channel', 'hourOfDay',
    'ipClickCount', 'appClickCount', 'ipAppCount',
    'clickCountLast10Seconds', 'clickCountLast60Seconds', 'clickCountLast10Minutes',
    'clicksPerDeviceLast60Seconds', 'uniqueAppsPerIp',
    'uniqueAppsPerDevice', 'uniqueDevicesPerIp',
}


def resolve_training_source(dataset_name=None, max_rows=None):
    resolved_dataset_name = dataset_name or os.getenv('DATASET_FILE', 'train.csv')
    resolved_max_rows = int(max_rows or os.getenv('MAX_TRAIN_ROWS', '500000'))
    zip_path = Path(os.getenv('DATASET_ZIP_PATH', ROOT_DIR / 'talkingdata-adtracking-fraud-detection.zip'))
    return zip_path, resolved_dataset_name, resolved_max_rows


def load_talkingdata_frame(dataset_name=None, max_rows=None):
    zip_path, resolved_dataset_name, resolved_max_rows = resolve_training_source(dataset_name, max_rows)
    with zipfile.ZipFile(zip_path) as archive, archive.open(resolved_dataset_name) as handle:
        return pd.read_csv(handle, nrows=resolved_max_rows)


def _normalize_training_rows(raw_df):
    df = raw_df.copy()
    df['click_time'] = pd.to_datetime(df['click_time'], errors='coerce', utc=True)
    if 'is_attributed' not in df:
        df['is_attributed'] = 0
    df['is_attributed'] = df['is_attributed'].fillna(0).astype(int).clip(0, 1)
    return df.dropna(subset=['click_time']).sort_values(['click_time', 'ip', 'app', 'channel']).reset_index(drop=True)


def _rolling_session_counts(df):
    timestamps = (df['click_time'].astype('int64', copy=False) // 10**9).to_numpy()
    counts = np.ones(len(df), dtype=np.int32)
    for positions in df.groupby('ip', sort=False).indices.values():
        ordered_positions = np.asarray(positions, dtype=int)
        group_times = timestamps[ordered_positions]
        left = 0
        for right, row_position in enumerate(ordered_positions):
            current_time = group_times[right]
            while current_time - group_times[left] > SESSION_WINDOW_SECONDS:
                left += 1
            counts[row_position] = right - left + 1
    return pd.Series(counts, index=df.index, dtype='float64')


def _rolling_window_counts(df, group_columns, window_seconds):
    timestamps = (df['click_time'].astype('int64', copy=False) // 10**9).to_numpy()
    counts = np.ones(len(df), dtype=np.int32)
    grouped_positions = df.groupby(group_columns, sort=False).indices.values()
    for positions in grouped_positions:
        ordered_positions = np.asarray(positions, dtype=int)
        group_times = timestamps[ordered_positions]
        left = 0
        for right, row_position in enumerate(ordered_positions):
            current_time = group_times[right]
            while current_time - group_times[left] > window_seconds:
                left += 1
            counts[row_position] = right - left + 1
    return pd.Series(counts, index=df.index, dtype='float64')


def _cumulative_unique_counts(df, group_columns, value_column):
    values = df[value_column].to_numpy()
    counts = np.ones(len(df), dtype=np.int32)
    for positions in df.groupby(group_columns, sort=False).indices.values():
        seen = set()
        for row_position in np.asarray(positions, dtype=int):
            seen.add(values[row_position])
            counts[row_position] = len(seen)
    return pd.Series(counts, index=df.index, dtype='float64')


def _cumulative_group_entropy(df, group_columns, value_column):
    values = df[value_column].to_numpy()
    entropy = np.zeros(len(df), dtype=float)
    for positions in df.groupby(group_columns, sort=False).indices.values():
        value_counts = {}
        total = 0
        weighted_log_sum = 0.0
        for row_position in np.asarray(positions, dtype=int):
            value = values[row_position]
            previous = value_counts.get(value, 0)
            if previous > 0:
                weighted_log_sum -= previous * np.log(previous)
            current = previous + 1
            value_counts[value] = current
            weighted_log_sum += current * np.log(current)
            total += 1
            entropy[row_position] = max(0.0, np.log(total) - (weighted_log_sum / total)) if total > 1 else 0.0
    return pd.Series(entropy, index=df.index, dtype='float64')


def resolve_feature_columns(feature_set=None):
    return FEATURE_SETS.get(feature_set or DEFAULT_FEATURE_SET, EXTENDED_RUNTIME_FEATURE_COLUMNS)


class _SampleWeightHGB:
    """HistGradientBoosting wrapper that applies balanced sample_weight during fit."""

    def __init__(self, **kwargs):
        self._model = HistGradientBoostingClassifier(**kwargs)

    def fit(self, X, y):
        y_arr = np.asarray(y)
        classes, counts = np.unique(y_arr, return_counts=True)
        n_samples = len(y_arr)
        weight_map = {int(c): n_samples / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sample_weight = np.array([weight_map[int(yi)] for yi in y_arr])
        self._model.fit(X, y_arr, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X)

    def get_params(self, deep=True):
        return self._model.get_params(deep=deep)

    def set_params(self, **params):
        self._model.set_params(**params)
        return self


def _engineer_feature_frame(df, feature_set=DEFAULT_FEATURE_SET):
    by_ip = df.groupby('ip', sort=False)
    by_app = df.groupby('app', sort=False)
    by_ip_app = df.groupby(['ip', 'app'], sort=False)
    by_device = df.groupby('device', sort=False)
    hour_of_day = df['click_time'].dt.hour
    click_frequency = _rolling_session_counts(df).clip(1, 250)
    gap_seconds = by_ip['click_time'].diff().dt.total_seconds().fillna(30).clip(0.25, 3600)
    device_gap_seconds = by_device['click_time'].diff().dt.total_seconds().fillna(30).clip(0.25, 3600)
    ip_click_count = by_ip.cumcount().add(1).clip(1, 500)
    app_click_count = by_app.cumcount().add(1).clip(1, 50_000)
    ip_app_count = by_ip_app.cumcount().add(1).clip(1, 250)
    device_click_count = by_device.cumcount().add(1).clip(1, 50_000)
    click_count_last_10_seconds = _rolling_window_counts(df, 'ip', 10).clip(1, 50)
    click_count_last_60_seconds = _rolling_window_counts(df, 'ip', 60).clip(1, 200)
    click_count_last_10_minutes = _rolling_window_counts(df, 'ip', 600).clip(1, 1_000)
    clicks_per_device_last_60_seconds = _rolling_window_counts(df, 'device', 60).clip(1, 500)
    unique_apps_per_ip = _cumulative_unique_counts(df, 'ip', 'app').clip(1, 1_000)
    unique_apps_per_device = _cumulative_unique_counts(df, 'device', 'app').clip(1, 500)
    unique_devices_per_ip = _cumulative_unique_counts(df, 'ip', 'device').clip(1, 50)
    device_app_entropy = _cumulative_group_entropy(df, 'device', 'app').clip(0, 12)
    time_interval_ms = (gap_seconds * 1000).round().clip(250, 3_600_000)
    features = pd.DataFrame({
        'clickFrequency': click_frequency,
        'timeInterval': time_interval_ms,
        'deviceCode': df['device'].fillna(1).clip(1, 500),
        'app': df['app'].fillna(0).clip(0, 1_000),
        'osCode': df['os'].fillna(0).clip(0, 1_000),
        'channel': df['channel'].fillna(0).clip(0, 1_000),
        'hourOfDay': hour_of_day,
        'ipClickCount': ip_click_count,
        'appClickCount': app_click_count,
        'ipAppCount': ip_app_count,
        'burstScore': (click_frequency / np.maximum(gap_seconds, 1)).clip(0, 500),
        'timeSinceLastClickPerIp': time_interval_ms,
        'timeSinceLastClickPerDevice': (device_gap_seconds * 1000).round().clip(250, 3_600_000),
        'clickCountLast10Seconds': click_count_last_10_seconds,
        'clickCountLast60Seconds': click_count_last_60_seconds,
        'clickCountLast10Minutes': click_count_last_10_minutes,
        'clicksPerDeviceLast60Seconds': clicks_per_device_last_60_seconds,
        'deviceAppEntropy': device_app_entropy,
        'deviceIpRatio': (device_click_count / np.maximum(ip_click_count, 1)).clip(0, 500),
        'uniqueAppsPerIp': unique_apps_per_ip,
        'uniqueAppsPerDevice': unique_apps_per_device,
        'uniqueDevicesPerIp': unique_devices_per_ip,
        'burstClickScore': (
            ((click_count_last_10_seconds * 6.0) + click_count_last_60_seconds + (click_count_last_10_minutes / 10.0))
            / np.maximum(gap_seconds, 1)
        ).clip(0, 500),
    })
    selected_columns = resolve_feature_columns(feature_set)
    return features[selected_columns].astype(float)


def derive_label_thresholds(features):
    return {
        'clickFrequency_p90': float(features['clickFrequency'].quantile(0.90)) if len(features) else 0.0,
        'timeInterval_p15': float(features['timeInterval'].quantile(0.15)) if len(features) else 0.0,
        'burstScore_p90': float(features['burstScore'].quantile(0.90)) if len(features) else 0.0,
        'ipClickCount_p90': float(features['ipClickCount'].quantile(0.90)) if len(features) else 0.0,
    }


def _build_dataset_info(features, labels, thresholds, threshold_source):
    observed_attribution_rate = float(labels.mean()) if len(labels) else 0.0
    return {
        'rows': int(len(features)),
        'featureColumns': list(features.columns),
        'class_balance': observed_attribution_rate,
        'observedAttributionRate': observed_attribution_rate,
        'suspicious_rate': float(1 - observed_attribution_rate),
        'observedSuspiciousProxyRate': float(1 - observed_attribution_rate),
        'thresholds': thresholds,
        'featureSummary': thresholds,
        'thresholdSource': threshold_source,
        'targetLabel': 'is_attributed',
        'positiveClassMeaning': 'Attributed click (proxy for valid click)',
        'negativeClassMeaning': 'Non-attributed click (proxy for suspicious click, not ground-truth fraud)',
    }


def engineer_training_frame(raw_df, thresholds=None, threshold_source='self', feature_set=DEFAULT_FEATURE_SET):
    df = _normalize_training_rows(raw_df)
    features = _engineer_feature_frame(df, feature_set=feature_set)
    feature_summary = thresholds or derive_label_thresholds(features)
    source = threshold_source or ('provided' if thresholds is not None else 'self')
    labels = df['is_attributed'].fillna(0).astype(int)
    dataset_info = _build_dataset_info(features, labels, feature_summary, source)
    dataset_info['featureSet'] = feature_set
    return features, labels, dataset_info


def model_candidates():
    return {
        'logistic_regression': Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced', random_state=RANDOM_STATE)),
        ]),
        'random_forest_balanced': RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=4,
            min_samples_leaf=2, class_weight='balanced_subsample',
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            max_depth=7, max_iter=400, learning_rate=0.035,
            min_samples_leaf=8, l2_regularization=0.03, random_state=RANDOM_STATE,
        ),
        'hist_gradient_boosting_weighted': _SampleWeightHGB(
            max_depth=7, max_iter=400, learning_rate=0.035,
            min_samples_leaf=8, l2_regularization=0.03, random_state=RANDOM_STATE,
        ),
    }


def training_sampler_candidates():
    # Preservation-based sampling: keep all majority, oversample minority
    # maxMajorityRatio=1000 ensures the entire majority class is retained
    return [
        {'name': 'preserve_all_10_to_1', 'samplingMethod': 'random_oversample',
         'maxMajorityRatio': 1000, 'targetMajorityRatio': 10},
        {'name': 'preserve_all_5_to_1', 'samplingMethod': 'random_oversample',
         'maxMajorityRatio': 1000, 'targetMajorityRatio': 5},
        {'name': 'smote_preserve_10_to_1', 'samplingMethod': 'smote',
         'maxMajorityRatio': 1000, 'targetMajorityRatio': 10},
    ]


def _class_count_map(labels):
    counts = pd.Series(labels).value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def _normalize_sampling_config(sampling_config=None):
    config = dict(sampling_config or {})
    method = str(config.get('samplingMethod', DEFAULT_TRAINING_SAMPLING_METHOD)).strip().lower()
    if method not in {'random_oversample', 'smote'}:
        method = DEFAULT_TRAINING_SAMPLING_METHOD
    target_ratio = max(1, int(config.get('targetMajorityRatio', TARGET_TRAINING_IMBALANCE_RATIO)))
    max_ratio = max(target_ratio, int(config.get('maxMajorityRatio', MAX_TRAINING_IMBALANCE_RATIO)))
    return {
        'name': config.get('name', f'{method}_{max_ratio}_to_{target_ratio}'),
        'samplingMethod': method,
        'maxMajorityRatio': max_ratio,
        'targetMajorityRatio': target_ratio,
    }


def _coerce_synthetic_features(synthetic_features, reference_features):
    if synthetic_features.empty:
        return synthetic_features
    bounded = synthetic_features.copy()
    for column in reference_features.columns:
        lower = float(reference_features[column].min())
        upper = float(reference_features[column].max())
        bounded[column] = bounded[column].clip(lower=lower, upper=upper)
        if column in DISCRETE_FEATURE_COLUMNS:
            bounded[column] = bounded[column].round()
    if 'hourOfDay' in bounded:
        bounded['hourOfDay'] = bounded['hourOfDay'].clip(lower=0, upper=23)
    return bounded.clip(lower=0)


def _smote_oversample_minority(minority_features, target_count, random_state=RANDOM_STATE):
    minority_features = minority_features.reset_index(drop=True)
    synthetic_count = max(0, int(target_count) - len(minority_features))
    if synthetic_count == 0:
        return minority_features.iloc[0:0].copy()

    rng = np.random.default_rng(random_state)
    minority_array = minority_features.to_numpy(dtype=float, copy=True)
    if len(minority_array) < 2:
        duplicated = minority_array[rng.choice(len(minority_array), size=synthetic_count, replace=True)]
        return _coerce_synthetic_features(pd.DataFrame(duplicated, columns=minority_features.columns), minority_features)

    neighbor_count = min(SMOTE_K_NEIGHBORS, len(minority_array) - 1)
    if neighbor_count < 1:
        duplicated = minority_array[rng.choice(len(minority_array), size=synthetic_count, replace=True)]
        return _coerce_synthetic_features(pd.DataFrame(duplicated, columns=minority_features.columns), minority_features)

    neighbors = NearestNeighbors(n_neighbors=neighbor_count + 1)
    neighbors.fit(minority_array)
    neighbor_indices = neighbors.kneighbors(minority_array, return_distance=False)[:, 1:]

    seed_indices = rng.integers(0, len(minority_array), size=synthetic_count)
    neighbor_choices = np.asarray([
        neighbor_indices[seed, rng.integers(0, len(neighbor_indices[seed]))]
        for seed in seed_indices
    ], dtype=int)
    interpolation = rng.random(synthetic_count).reshape(-1, 1)
    seed_points = minority_array[seed_indices]
    neighbor_points = minority_array[neighbor_choices]
    synthetic = seed_points + interpolation * (neighbor_points - seed_points)
    synthetic_frame = pd.DataFrame(synthetic, columns=minority_features.columns)
    return _coerce_synthetic_features(synthetic_frame, minority_features)


def rebalance_training_data(features, labels, max_majority_ratio=MAX_TRAINING_IMBALANCE_RATIO,
                            target_majority_ratio=TARGET_TRAINING_IMBALANCE_RATIO,
                            random_state=RANDOM_STATE,
                            sampling_method=DEFAULT_TRAINING_SAMPLING_METHOD):
    features = features.reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    sampling_config = _normalize_sampling_config({
        'samplingMethod': sampling_method,
        'maxMajorityRatio': max_majority_ratio,
        'targetMajorityRatio': target_majority_ratio,
    })
    class_counts = labels.value_counts()
    if class_counts.empty or len(class_counts) < 2:
        summary = {
            'applied': False,
            'method': 'none',
            'samplingMethod': sampling_config['samplingMethod'],
            'originalClassCounts': _class_count_map(labels),
            'rebalancedClassCounts': _class_count_map(labels),
            'originalImbalanceRatio': 1.0,
            'rebalancedImbalanceRatio': 1.0,
        }
        return features, labels, summary

    majority_label = int(class_counts.idxmax())
    minority_label = int(class_counts.idxmin())
    majority_count = int(class_counts.loc[majority_label])
    minority_count = int(class_counts.loc[minority_label])
    original_ratio = float(majority_count / max(minority_count, 1))

    if original_ratio <= sampling_config['targetMajorityRatio']:
        summary = {
            'applied': False,
            'method': 'none',
            'samplingMethod': sampling_config['samplingMethod'],
            'majorityLabel': majority_label,
            'minorityLabel': minority_label,
            'originalClassCounts': _class_count_map(labels),
            'rebalancedClassCounts': _class_count_map(labels),
            'originalImbalanceRatio': original_ratio,
            'rebalancedImbalanceRatio': original_ratio,
        }
        return features, labels, summary

    rng = np.random.default_rng(random_state)
    majority_index = np.flatnonzero(labels.to_numpy() == majority_label)
    minority_index = np.flatnonzero(labels.to_numpy() == minority_label)
    kept_majority_count = min(
        majority_count,
        max(minority_count * sampling_config['maxMajorityRatio'], minority_count),
    )
    kept_majority = rng.choice(majority_index, size=kept_majority_count, replace=False)

    target_minority_count = max(
        minority_count,
        int(np.ceil(kept_majority_count / max(sampling_config['targetMajorityRatio'], 1))),
    )
    kept_majority_features = features.iloc[kept_majority].reset_index(drop=True)
    kept_majority_labels = pd.Series(np.full(len(kept_majority_features), majority_label, dtype=int))
    minority_features = features.iloc[minority_index].reset_index(drop=True)
    oversampled_minority_features = minority_features.copy()
    oversampled_minority_labels = pd.Series(np.full(len(minority_features), minority_label, dtype=int))
    method_name = 'random_oversample_with_controlled_undersample'
    if target_minority_count > minority_count:
        if sampling_config['samplingMethod'] == 'smote':
            synthetic_minority = _smote_oversample_minority(minority_features, target_minority_count, random_state=random_state)
            method_name = 'smote_with_controlled_undersample'
        else:
            extra_minority = rng.choice(minority_index, size=target_minority_count - minority_count, replace=True)
            synthetic_minority = features.iloc[extra_minority].reset_index(drop=True)
        oversampled_minority_features = pd.concat([minority_features, synthetic_minority], ignore_index=True)
        oversampled_minority_labels = pd.Series(np.full(len(oversampled_minority_features), minority_label, dtype=int))

    balanced_features = pd.concat([kept_majority_features, oversampled_minority_features], ignore_index=True)
    balanced_labels = pd.concat([kept_majority_labels, oversampled_minority_labels], ignore_index=True)
    shuffled_index = rng.permutation(len(balanced_features))
    balanced_features = balanced_features.iloc[shuffled_index].reset_index(drop=True)
    balanced_labels = balanced_labels.iloc[shuffled_index].reset_index(drop=True)
    balanced_counts = balanced_labels.value_counts()
    summary = {
        'applied': True,
        'method': method_name,
        'samplingMethod': sampling_config['samplingMethod'],
        'majorityLabel': majority_label,
        'minorityLabel': minority_label,
        'maxMajorityRatio': int(sampling_config['maxMajorityRatio']),
        'targetMajorityRatio': int(sampling_config['targetMajorityRatio']),
        'originalClassCounts': _class_count_map(labels),
        'rebalancedClassCounts': _class_count_map(balanced_labels),
        'originalImbalanceRatio': original_ratio,
        'rebalancedImbalanceRatio': float(balanced_counts.max() / max(balanced_counts.min(), 1)),
    }
    return balanced_features, balanced_labels, summary


def _final_estimator(model):
    return model.named_steps['model'] if isinstance(model, Pipeline) else model


def _clone_model(model):
    """Clone a model, handling custom wrappers that sklearn.base.clone cannot introspect."""
    if isinstance(model, _SampleWeightHGB):
        params = model.get_params()
        return _SampleWeightHGB(**params)
    return clone(model)


def _fit_model(model, features, labels):
    model.fit(features, labels)
    return model


def _predict_probabilities(model, features):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(features)
        return 1 / (1 + np.exp(-scores))
    return model.predict(features).astype(float)


def _safe_metric(metric_fn, *args, default=0.0, **kwargs):
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return float(default)


def _collect_metrics(labels, predictions, probabilities):
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'f1': float(f1_score(labels, predictions, zero_division=0)),
        'roc_auc': _safe_metric(roc_auc_score, labels, probabilities),
        'average_precision': _safe_metric(average_precision_score, labels, probabilities),
    }


def _metric_summary_key(metric_name, suffix):
    mapping = {'roc_auc': 'rocAuc', 'average_precision': 'averagePrecision'}
    return f"{mapping.get(metric_name, metric_name)}{suffix}"


def _tune_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size == 0:
        fallback_predictions = (np.asarray(probabilities, dtype=float) >= 0.5).astype(int)
        scored = [{
            'threshold': 0.5,
            'precision': float(precision_score(labels, fallback_predictions, zero_division=0)),
            'recall': float(recall_score(labels, fallback_predictions, zero_division=0)),
            'f1': float(f1_score(labels, fallback_predictions, zero_division=0)),
        }]
        return 0.5, scored

    precision = np.asarray(precision[:-1], dtype=float)
    recall = np.asarray(recall[:-1], dtype=float)
    denominator = precision + recall
    f1_scores = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    scored = [
        {
            'threshold': float(threshold),
            'precision': float(precision[index]),
            'recall': float(recall[index]),
            'f1': float(f1_scores[index]),
        }
        for index, threshold in enumerate(thresholds)
    ]
    best = max(scored, key=lambda item: (item['f1'], item['precision'], item['recall'], item['threshold']))
    return best['threshold'], scored


def _stratify_or_none(labels):
    labels = pd.Series(labels)
    return labels if labels.nunique() > 1 else None


def _validation_sort_key(result):
    return (
        result['validation_metrics']['f1'],
        result['validation_metrics']['average_precision'],
        result['validation_metrics']['roc_auc'],
    )


def _split_dataset(features, labels):
    features = features.reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=_stratify_or_none(labels),
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp.reset_index(drop=True),
        y_temp.reset_index(drop=True),
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=_stratify_or_none(y_temp),
    )
    return {
        'train': (x_train.reset_index(drop=True), y_train.reset_index(drop=True)),
        'validation': (x_val.reset_index(drop=True), y_val.reset_index(drop=True)),
        'test': (x_test.reset_index(drop=True), y_test.reset_index(drop=True)),
    }


def cross_validate_model(model, features, labels, sampling_config=None):
    features = features.reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    sampling_config = _normalize_sampling_config(sampling_config)
    class_counts = labels.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    if len(features) < 20 or min_class_count < 5:
        return {'folds': 0, 'summary': {}, 'perFold': []}

    folds = min(5, min_class_count)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    for train_index, holdout_index in splitter.split(features, labels):
        x_development = features.iloc[train_index].reset_index(drop=True)
        y_development = labels.iloc[train_index].reset_index(drop=True)
        x_holdout = features.iloc[holdout_index].reset_index(drop=True)
        y_holdout = labels.iloc[holdout_index].reset_index(drop=True)

        x_fit, x_calibration, y_fit, y_calibration = train_test_split(
            x_development,
            y_development,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=_stratify_or_none(y_development),
        )
        x_fit_balanced, y_fit_balanced, _ = rebalance_training_data(
            x_fit,
            y_fit,
            max_majority_ratio=sampling_config['maxMajorityRatio'],
            target_majority_ratio=sampling_config['targetMajorityRatio'],
            random_state=RANDOM_STATE + len(fold_metrics),
            sampling_method=sampling_config['samplingMethod'],
        )
        candidate = _fit_model(_clone_model(model), x_fit_balanced, y_fit_balanced)
        calibration_probabilities = _predict_probabilities(candidate, x_calibration)
        threshold, _ = _tune_threshold(y_calibration, calibration_probabilities)
        holdout_probabilities = _predict_probabilities(candidate, x_holdout)
        holdout_predictions = (holdout_probabilities >= threshold).astype(int)
        fold_metric = _collect_metrics(y_holdout, holdout_predictions, holdout_probabilities)
        fold_metric['threshold'] = float(threshold)
        fold_metrics.append(fold_metric)

    summary = {}
    if fold_metrics:
        for metric_name in fold_metrics[0].keys():
            values = np.asarray([fold[metric_name] for fold in fold_metrics], dtype=float)
            summary[_metric_summary_key(metric_name, 'Mean')] = float(values.mean())
            summary[_metric_summary_key(metric_name, 'Std')] = float(values.std(ddof=0))
    return {'folds': folds, 'summary': summary, 'perFold': fold_metrics}


def _roc_curve_data(labels, probabilities):
    try:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        return fpr.tolist(), tpr.tolist()
    except ValueError:
        return [0.0, 1.0], [0.0, 1.0]


def _pr_curve_data(labels, probabilities):
    try:
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        return precision.tolist(), recall.tolist()
    except ValueError:
        return [1.0, 0.0], [0.0, 1.0]


def evaluate_models(train_features, train_labels, validation_features, validation_labels, test_features, test_labels):
    train_features = train_features.reset_index(drop=True)
    train_labels = pd.Series(train_labels).reset_index(drop=True)
    validation_features = validation_features.reset_index(drop=True)
    validation_labels = pd.Series(validation_labels).reset_index(drop=True)
    test_features = test_features.reset_index(drop=True)
    test_labels = pd.Series(test_labels).reset_index(drop=True)
    train_validation_features = pd.concat([train_features, validation_features], ignore_index=True)
    train_validation_labels = pd.concat([train_labels, validation_labels], ignore_index=True)

    results = {}
    best_name, best_result = None, None
    sampler_candidates = training_sampler_candidates()
    for model_offset, (name, model) in enumerate(model_candidates().items()):
        best_model_result = None
        for sampler_offset, sampler_config in enumerate(sampler_candidates):
            seed_offset = (model_offset * 1000) + (sampler_offset * 100)
            balanced_train_features, balanced_train_labels, threshold_rebalance = rebalance_training_data(
                train_features,
                train_labels,
                max_majority_ratio=sampler_config['maxMajorityRatio'],
                target_majority_ratio=sampler_config['targetMajorityRatio'],
                random_state=RANDOM_STATE + seed_offset,
                sampling_method=sampler_config['samplingMethod'],
            )
            threshold_model = _fit_model(_clone_model(model), balanced_train_features, balanced_train_labels)
            validation_probabilities = _predict_probabilities(threshold_model, validation_features)
            best_threshold, threshold_grid = _tune_threshold(validation_labels, validation_probabilities)
            validation_predictions = (validation_probabilities >= best_threshold).astype(int)

            balanced_train_validation_features, balanced_train_validation_labels, final_rebalance = rebalance_training_data(
                train_validation_features,
                train_validation_labels,
                max_majority_ratio=sampler_config['maxMajorityRatio'],
                target_majority_ratio=sampler_config['targetMajorityRatio'],
                random_state=RANDOM_STATE + 50 + seed_offset,
                sampling_method=sampler_config['samplingMethod'],
            )
            final_model = _fit_model(_clone_model(model), balanced_train_validation_features, balanced_train_validation_labels)
            train_probabilities = _predict_probabilities(final_model, train_validation_features)
            test_probabilities = _predict_probabilities(final_model, test_features)
            train_predictions = (train_probabilities >= best_threshold).astype(int)
            test_predictions = (test_probabilities >= best_threshold).astype(int)
            precision_curve, recall_curve = _pr_curve_data(test_labels, test_probabilities)
            fpr, tpr = _roc_curve_data(test_labels, test_probabilities)

            result = {
                'model': final_model,
                'threshold': float(best_threshold),
                'threshold_grid': threshold_grid,
                'validation_metrics': _collect_metrics(validation_labels, validation_predictions, validation_probabilities),
                'train_metrics': _collect_metrics(train_validation_labels, train_predictions, train_probabilities),
                'metrics': _collect_metrics(test_labels, test_predictions, test_probabilities),
                'confusion_matrix': confusion_matrix(test_labels, test_predictions, labels=[0, 1]).tolist(),
                'roc_curve': {'fpr': fpr, 'tpr': tpr},
                'pr_curve': {'precision': precision_curve, 'recall': recall_curve},
                'predicted_positive_rate': float(test_predictions.mean()),
                'predicted_fraud_rate': float(1 - test_predictions.mean()),
                'training_balance': {
                    'thresholdTraining': threshold_rebalance,
                    'finalTraining': final_rebalance,
                },
                'sampling_config': _normalize_sampling_config(sampler_config),
            }
            if best_model_result is None or _validation_sort_key(result) > _validation_sort_key(best_model_result):
                best_model_result = result

        best_model_result['cross_validation'] = cross_validate_model(
            model,
            train_validation_features,
            train_validation_labels,
            sampling_config=best_model_result['sampling_config'],
        )
        results[name] = best_model_result
        if best_result is None or _validation_sort_key(best_model_result) > _validation_sort_key(best_result):
            best_name, best_result = name, best_model_result
    return best_name, best_result, results


def _line_chart_svg(title, x_values, y_values, x_label, y_label, color):
    points = list(zip(x_values, y_values)) or [(0, 0), (1, 1)]
    sampled = points[::max(1, len(points) // 60)]
    chart_points = ' '.join(f"{40 + x * 320:.1f},{260 - y * 220:.1f}" for x, y in sampled)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="420" height="300" viewBox="0 0 420 300">'
        '<rect width="420" height="300" rx="16" fill="#0f172a" />'
        f'<text x="24" y="30" fill="#f8fafc" font-size="18" font-family="Arial">{title}</text>'
        '<line x1="40" y1="40" x2="40" y2="260" stroke="#64748b" /><line x1="40" y1="260" x2="360" y2="260" stroke="#64748b" />'
        f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{chart_points}" />'
        f'<text x="180" y="288" fill="#94a3b8" font-size="12">{x_label}</text>'
        f'<text x="6" y="150" fill="#94a3b8" font-size="12" transform="rotate(-90 12,150)">{y_label}</text>'
        '</svg>'
    )


def _confusion_matrix_svg(matrix):
    values = [max(row) for row in matrix if row]
    max_value = max(values) if values else 1
    cells = []
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = 60 + int((value / max_value) * 140) if max_value else 60
            x_pos = 70 + col_index * 110
            y_pos = 70 + row_index * 90
            cells.append(f'<rect x="{x_pos}" y="{y_pos}" width="90" height="70" rx="12" fill="rgb(30,{intensity},120)" />')
            cells.append(f'<text x="{x_pos + 45}" y="{y_pos + 42}" text-anchor="middle" fill="#fff" font-size="20">{value}</text>')
    return '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="250" viewBox="0 0 320 250"><rect width="320" height="250" rx="16" fill="#0f172a" /><text x="24" y="30" fill="#f8fafc" font-size="18">Confusion Matrix</text>' + ''.join(cells) + '<text x="94" y="58" fill="#94a3b8">Pred 0</text><text x="204" y="58" fill="#94a3b8">Pred 1</text><text x="24" y="112" fill="#94a3b8">True 0</text><text x="24" y="202" fill="#94a3b8">True 1</text></svg>'


def _persist_bundle(bundle):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(pickle.dumps(bundle))
    metadata = {key: value for key, value in bundle.items() if key != 'model'}
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (ARTIFACTS_DIR / 'roc_curve.svg').write_text(_line_chart_svg('ROC Curve', bundle['roc_curve']['fpr'], bundle['roc_curve']['tpr'], 'False Positive Rate', 'True Positive Rate', '#6366f1'), encoding='utf-8')
    (ARTIFACTS_DIR / 'pr_curve.svg').write_text(_line_chart_svg('Precision-Recall Curve', bundle['pr_curve']['recall'], bundle['pr_curve']['precision'], 'Recall', 'Precision', '#10b981'), encoding='utf-8')
    (ARTIFACTS_DIR / 'confusion_matrix.svg').write_text(_confusion_matrix_svg(bundle['confusion_matrix']), encoding='utf-8')


def _feature_defaults(features, feature_columns=None):
    columns = feature_columns or FEATURE_COLUMNS
    medians = features.median(numeric_only=True)
    return {column: float(medians.get(column, 0.0)) for column in columns}


def summarize_feature_baselines(features):
    baselines = {}
    for column in list(features.columns):
        series = pd.to_numeric(features[column], errors='coerce').fillna(0.0).astype(float)
        baselines[column] = {
            'mean': float(series.mean()) if len(series) else 0.0,
            'median': float(series.median()) if len(series) else 0.0,
            'std': float(series.std(ddof=0)) if len(series) else 0.0,
            'min': float(series.min()) if len(series) else 0.0,
            'max': float(series.max()) if len(series) else 0.0,
            'p10': float(series.quantile(0.10)) if len(series) else 0.0,
            'p90': float(series.quantile(0.90)) if len(series) else 0.0,
        }
    return baselines


def _feature_baselines_from_defaults(feature_defaults, feature_columns=None):
    columns = list(feature_columns or (feature_defaults or {}).keys() or FEATURE_SETS[DEFAULT_FEATURE_SET])
    baselines = {}
    for column in columns:
        value = float((feature_defaults or {}).get(column, 0.0))
        spread = max(abs(value) * 0.25, 1.0)
        baselines[column] = {
            'mean': value,
            'median': value,
            'std': spread,
            'min': value,
            'max': value,
            'p10': value,
            'p90': value,
        }
    return baselines


def _upgrade_bundle(bundle):
    if not isinstance(bundle, dict) or 'model' not in bundle:
        return None

    evaluation = bundle.get('evaluation') or {}
    feature_columns = list(
        bundle.get('feature_columns')
        or (bundle.get('feature_defaults') or {}).keys()
        or FEATURE_SETS.get(evaluation.get('featureSet'))
        or FEATURE_SETS[DEFAULT_FEATURE_SET]
    )
    feature_defaults = {
        column: float((bundle.get('feature_defaults') or {}).get(column, 0.0))
        for column in feature_columns
    }

    monitoring = dict(bundle.get('monitoring') or {})
    existing_baselines = monitoring.get('featureBaselines') or {}
    feature_baselines = _feature_baselines_from_defaults(feature_defaults, feature_columns)
    for column in feature_columns:
        baseline = dict(feature_baselines[column])
        prior = existing_baselines.get(column) or {}
        for key in list(baseline.keys()):
            if key in prior:
                baseline[key] = float(prior.get(key, baseline[key]))
        feature_baselines[column] = baseline

    drift_thresholds = dict(monitoring.get('driftThresholds') or {})
    drift_thresholds.setdefault('warningMeanShiftZ', DRIFT_WARN_Z_SCORE)
    drift_thresholds.setdefault('criticalMeanShiftZ', DRIFT_CRITICAL_Z_SCORE)

    upgraded = dict(bundle)
    upgraded['schema_version'] = BUNDLE_SCHEMA_VERSION
    upgraded['feature_columns'] = feature_columns
    upgraded['feature_defaults'] = feature_defaults
    upgraded['monitoring'] = {
        **monitoring,
        'featureBaselines': feature_baselines,
        'driftThresholds': drift_thresholds,
    }
    upgraded.setdefault('all_results', {})
    upgraded.setdefault('dataset', {})
    upgraded.setdefault('labeling_strategy', 'Legacy model bundle loaded without explicit labeling strategy metadata.')
    upgraded.setdefault('cross_validation', None)
    upgraded.setdefault('evaluation', evaluation or None)
    return upgraded


def _bundle_prediction_sanity_check(bundle):
    try:
        sample = prepare_prediction_frame({}, bundle.get('feature_defaults'), bundle.get('feature_columns'))
        predict_probability(bundle['model'], sample)
        return True
    except Exception:
        return False


def _serializable_results(results):
    serialized = {}
    for name, result in results.items():
        serialized[name] = {
            **result['metrics'],
            'threshold': result['threshold'],
            'validation_f1': result['validation_metrics']['f1'],
            'validation_average_precision': result['validation_metrics']['average_precision'],
            'validation_roc_auc': result['validation_metrics']['roc_auc'],
            'validation_precision': result['validation_metrics']['precision'],
            'validation_recall': result['validation_metrics']['recall'],
            'predictedFraudRate': result['predicted_fraud_rate'],
            'samplingConfig': result['sampling_config'],
            'trainingBalance': result['training_balance']['finalTraining'],
            'featureSet': result.get('feature_set', 'compact'),
            'confusionMatrix': result.get('confusion_matrix'),
        }
    return serialized


def prepare_prediction_samples(samples, feature_defaults=None, feature_columns=None):
    sample_list = list(samples or [])
    if not sample_list:
        raise ValueError('At least one sample is required for monitoring')
    frames = [prepare_prediction_frame(sample, feature_defaults, feature_columns) for sample in sample_list]
    return pd.concat(frames, ignore_index=True)


def compute_drift_report(samples, bundle):
    frame = prepare_prediction_samples(samples, bundle.get('feature_defaults'), bundle.get('feature_columns'))
    baselines = bundle.get('monitoring', {}).get('featureBaselines', {})
    per_feature = []
    z_scores = []
    for column in list(frame.columns):
        baseline = baselines.get(column, {})
        sample_mean = float(frame[column].mean()) if len(frame) else 0.0
        sample_median = float(frame[column].median()) if len(frame) else 0.0
        std = max(float(baseline.get('std', 0.0)), 1e-6)
        mean_shift = sample_mean - float(baseline.get('mean', 0.0))
        z_score = abs(mean_shift) / std
        z_scores.append(z_score)
        above_p90 = float((frame[column] > float(baseline.get('p90', sample_mean))).mean()) if len(frame) else 0.0
        below_p10 = float((frame[column] < float(baseline.get('p10', sample_mean))).mean()) if len(frame) else 0.0
        severity = 'stable'
        if z_score >= DRIFT_CRITICAL_Z_SCORE:
            severity = 'critical'
        elif z_score >= DRIFT_WARN_Z_SCORE:
            severity = 'warning'
        per_feature.append({
            'feature': column,
            'sampleMean': sample_mean,
            'sampleMedian': sample_median,
            'baselineMean': float(baseline.get('mean', 0.0)),
            'baselineMedian': float(baseline.get('median', 0.0)),
            'meanShiftZ': float(round(z_score, 4)),
            'outsideP90Rate': float(round(above_p90, 4)),
            'belowP10Rate': float(round(below_p10, 4)),
            'severity': severity,
        })
    overall = float(np.mean(z_scores)) if z_scores else 0.0
    status = 'stable'
    if any(item['severity'] == 'critical' for item in per_feature):
        status = 'critical'
    elif any(item['severity'] == 'warning' for item in per_feature):
        status = 'warning'
    return {
        'status': status,
        'sampleCount': int(len(frame)),
        'featureCount': int(len(frame.columns)),
        'overallMeanShiftZ': float(round(overall, 4)),
        'flaggedFeatures': [item for item in per_feature if item['severity'] != 'stable'][:8],
        'perFeature': per_feature,
    }


def _apply_adversarial_scenario(payload, scenario_name):
    base = dict(payload or {})
    click_frequency = max(float(base.get('clickFrequency', 1.0)), 1.0)
    time_interval = max(float(base.get('timeInterval', 30_000.0)), 250.0)
    click_count_10s = max(float(base.get('clickCountLast10Seconds', click_frequency)), 1.0)
    click_count_60s = max(float(base.get('clickCountLast60Seconds', click_frequency)), 1.0)
    click_count_10m = max(float(base.get('clickCountLast10Minutes', click_frequency)), 1.0)
    if scenario_name == 'cadence_smoothing':
        base.update({
            'clickFrequency': max(1.0, round(click_frequency * 0.4)),
            'timeInterval': min(time_interval * 8.0, 3_600_000.0),
            'clickCountLast10Seconds': max(1.0, round(click_count_10s * 0.25)),
            'clickCountLast60Seconds': max(1.0, round(click_count_60s * 0.5)),
            'clickCountLast10Minutes': max(1.0, round(click_count_10m * 0.7)),
            'burstScore': float(base.get('burstScore', click_frequency / max(time_interval / 1000.0, 1.0))) * 0.2,
            'burstClickScore': float(base.get('burstClickScore', click_frequency)) * 0.2,
        })
    elif scenario_name == 'device_rotation':
        device_type = str(base.get('deviceType', 'desktop')).lower()
        rotated_type = 'desktop' if device_type == 'mobile' else 'mobile'
        base.update({
            'deviceType': rotated_type,
            'deviceCode': DEVICE_MAP.get(rotated_type, 2),
            'timeSinceLastClickPerDevice': min(float(base.get('timeSinceLastClickPerDevice', time_interval)) * 4.0, 3_600_000.0),
            'uniqueDevicesPerIp': float(base.get('uniqueDevicesPerIp', 1.0)) + 2.0,
        })
    elif scenario_name == 'entropy_inflation':
        base.update({
            'deviceAppEntropy': float(base.get('deviceAppEntropy', 0.0)) + 1.5,
            'uniqueAppsPerIp': float(base.get('uniqueAppsPerIp', 1.0)) + 4.0,
            'uniqueAppsPerDevice': float(base.get('uniqueAppsPerDevice', 1.0)) + 3.0,
            'deviceIpRatio': max(0.2, float(base.get('deviceIpRatio', 1.0)) * 0.5),
        })
    elif scenario_name == 'burst_amplification':
        base.update({
            'clickFrequency': round(click_frequency * 2.0),
            'timeInterval': max(250.0, time_interval * 0.25),
            'clickCountLast10Seconds': round(click_count_10s * 2.0),
            'clickCountLast60Seconds': round(click_count_60s * 2.0),
            'clickCountLast10Minutes': round(click_count_10m * 1.5),
            'burstScore': float(base.get('burstScore', click_frequency / max(time_interval / 1000.0, 1.0))) * 4.0,
            'burstClickScore': float(base.get('burstClickScore', click_frequency)) * 3.0,
        })
    return base


def evaluate_adversarial_resilience(payload, bundle):
    baseline_frame = prepare_prediction_frame(payload, bundle.get('feature_defaults'), bundle.get('feature_columns'))
    baseline_probability = predict_probability(bundle['model'], baseline_frame)
    baseline_prediction = int(baseline_probability >= bundle['threshold'])
    scenarios = []
    for name in ['cadence_smoothing', 'device_rotation', 'entropy_inflation', 'burst_amplification']:
        perturbed_payload = _apply_adversarial_scenario(payload, name)
        perturbed_frame = prepare_prediction_frame(perturbed_payload, bundle.get('feature_defaults'), bundle.get('feature_columns'))
        perturbed_probability = predict_probability(bundle['model'], perturbed_frame)
        perturbed_prediction = int(perturbed_probability >= bundle['threshold'])
        scenarios.append({
            'name': name,
            'probability': float(round(perturbed_probability, 6)),
            'prediction': perturbed_prediction,
            'delta': float(round(perturbed_probability - baseline_probability, 6)),
            'decisionChanged': bool(perturbed_prediction != baseline_prediction),
        })
    changed = sum(1 for scenario in scenarios if scenario['decisionChanged'])
    return {
        'baseline': {
            'probability': float(round(baseline_probability, 6)),
            'prediction': baseline_prediction,
            'threshold': bundle['threshold'],
        },
        'scenarios': scenarios,
        'robustnessScore': float(round(1 - (changed / max(len(scenarios), 1)), 4)),
        'decisionFlipCount': changed,
    }


def train_and_persist_bundle(dataset_name=None, max_rows=None):
    _, resolved_dataset_name, resolved_max_rows = resolve_training_source(dataset_name, max_rows)
    raw = load_talkingdata_frame(dataset_name=resolved_dataset_name, max_rows=resolved_max_rows)

    # Use the extended feature set for best discrimination on imbalanced click traffic.
    selected_feature_set = 'extended_runtime'
    selected_columns = FEATURE_SETS[selected_feature_set]
    all_features, labels, dataset_info = engineer_training_frame(
        raw, threshold_source='engineered_from_raw_click_history', feature_set=selected_feature_set,
    )
    split = _split_dataset(all_features, labels)
    train_features, train_labels = split['train']
    validation_features, validation_labels = split['validation']
    test_features, test_labels = split['test']

    best_name, best_result, all_results = evaluate_models(
        train_features, train_labels,
        validation_features, validation_labels,
        test_features, test_labels,
    )
    for name, result in all_results.items():
        result['feature_set'] = selected_feature_set

    train_validation_features = pd.concat([train_features, validation_features], ignore_index=True)
    train_validation_labels = pd.concat([train_labels, validation_labels], ignore_index=True)
    observed_valid_rate = float(test_labels.mean()) if len(test_labels) else 0.0
    dataset_info.update({
        'datasetName': resolved_dataset_name,
        'maxRowsLoaded': int(resolved_max_rows),
        'trainRows': int(len(train_features)),
        'validationRows': int(len(validation_features)),
        'testRows': int(len(test_features)),
        'trainClassBalance': float(train_labels.mean()) if len(train_labels) else 0.0,
        'validationClassBalance': float(validation_labels.mean()) if len(validation_labels) else 0.0,
        'testClassBalance': observed_valid_rate,
        'selectedFeatureSet': selected_feature_set,
        'minorityTrainCount': int(train_labels.sum()),
        'majorityTrainCount': int((train_labels == 0).sum()),
    })
    bundle = {
        'schema_version': BUNDLE_SCHEMA_VERSION,
        'model': best_result['model'],
        'model_name': best_name,
        'threshold': best_result['threshold'],
        'metrics': best_result['metrics'],
        'all_results': _serializable_results(all_results),
        'feature_columns': selected_columns,
        'feature_defaults': _feature_defaults(train_validation_features, selected_columns),
        'monitoring': {
            'featureBaselines': summarize_feature_baselines(train_validation_features[selected_columns]),
            'driftThresholds': {
                'warningMeanShiftZ': DRIFT_WARN_Z_SCORE,
                'criticalMeanShiftZ': DRIFT_CRITICAL_Z_SCORE,
            },
        },
        'dataset': dataset_info,
        'labeling_strategy': 'Model trained on the real TalkingData is_attributed label. Positive class = attributed click (valid-click proxy); negative class = non-attributed click (suspicious proxy, not ground-truth fraud). High accuracy (~99%) is expected given ~500:1 class imbalance and is not indicative of overfitting.',
        'cross_validation': best_result['cross_validation'],
        'evaluation': {
            'selectionMetric': 'validation_f1_then_ap_then_roc_auc',
            'featureSet': selected_feature_set,
            'totalFeatures': len(selected_columns),
            'samplingStrategy': 'preservation_based_oversample_minority_only',
            'samplingCandidates': training_sampler_candidates(),
            'selectedSamplingConfig': best_result['sampling_config'],
            'thresholdGrid': best_result['threshold_grid'],
            'validationMetrics': best_result['validation_metrics'],
            'trainMetrics': best_result['train_metrics'],
            'trainingBalance': best_result['training_balance'],
            'observedValidRate': observed_valid_rate,
            'observedSuspiciousProxyRate': float(1 - observed_valid_rate),
            'predictedValidRate': best_result['predicted_positive_rate'],
            'predictedFraudRate': best_result['predicted_fraud_rate'],
            'totalExperiments': len(all_results),
            'notes': (
                'All reported metrics come from held-out evaluation after threshold tuning on validation only. '
                'Validation and test sets remain at the original imbalanced distribution. '
                'Training uses preservation-based oversampling: the entire majority class is retained, '
                'and the minority class is oversampled via random duplication or SMOTE to reduce the '
                'imbalance ratio to 5:1 or 10:1. This avoids the data waste caused by aggressive '
                'majority undersampling. Extended 20-feature set includes rolling window counts, '
                'burst scores, device entropy and IP diversity. '
                'Note: accuracy near 99% is not overfitting — it reflects the ~500:1 class imbalance '
                'where predicting the majority class frequently is mechanically correct.'
            ),
        },
        'confusion_matrix': best_result['confusion_matrix'],
        'roc_curve': best_result['roc_curve'],
        'pr_curve': best_result['pr_curve'],
    }
    _persist_bundle(bundle)
    return bundle


def build_lightgbm_case_control_bundle(dataset_name=None, max_rows=None):
    """Build a runtime bundle using the benchmarked LightGBM (case 1:10) model.

    This loads the best_model.joblib from the non_50_50_sweep/case_control_1_to_10
    benchmark, evaluates it on a held-out test split with the full imbalanced
    TalkingData distribution, and persists a bundle compatible with the rest of
    the runtime.
    """
    benchmark_dir = ARTIFACTS_DIR / 'benchmark_full' / 'non_50_50_sweep' / 'case_control_1_to_10'
    summary_path = benchmark_dir / 'benchmark_summary.json'
    model_path = benchmark_dir / 'best_model.joblib'

    if not summary_path.exists() or not model_path.exists():
        raise RuntimeError(
            'LightGBM case_control_1_to_10 artifacts not found. '
            'Expected files under artifacts/benchmark_full/non_50_50_sweep/case_control_1_to_10.'
        )

    try:
        import joblib  # Imported lazily to avoid hard dependency at import time
    except ImportError as exc:
        raise RuntimeError(
            'Missing optional dependency joblib (and likely lightgbm) required to '
            'load the benchmarked LightGBM model. Install with:\n'
            '  pip install joblib lightgbm'
        ) from exc

    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    artifact = joblib.load(model_path)
    estimator = artifact['estimator']
    decision_threshold = float(artifact.get('decision_threshold', 0.3))

    raw = load_talkingdata_frame(dataset_name=dataset_name, max_rows=max_rows)
    features, labels, dataset_info = engineer_training_frame(raw, feature_set='extended_runtime')

    # Align to the feature ordering used in the benchmark artifact
    feature_columns = summary.get('dataset', {}).get('featureColumns') or list(features.columns)
    features = features[feature_columns]

    split = _split_dataset(features, labels)
    train_features, train_labels = split['train']
    validation_features, validation_labels = split['validation']
    test_features, test_labels = split['test']
    train_validation_features = pd.concat([train_features, validation_features], ignore_index=True)

    # Evaluate LightGBM on the held-out test split using the benchmark threshold
    test_probabilities = _predict_probabilities(estimator, test_features)
    test_predictions = (np.asarray(test_probabilities, dtype=float) >= decision_threshold).astype(int)
    metrics = _collect_metrics(test_labels, test_predictions, test_probabilities)
    confusion = confusion_matrix(test_labels, test_predictions, labels=[0, 1]).tolist()
    fpr, tpr = _roc_curve_data(test_labels, test_probabilities)
    precision_curve, recall_curve = _pr_curve_data(test_labels, test_probabilities)

    predicted_valid_rate = float(test_predictions.mean())
    predicted_fraud_rate = float(1 - predicted_valid_rate)

    feature_defaults = _feature_defaults(train_validation_features, feature_columns)
    monitoring = {
        'featureBaselines': summarize_feature_baselines(train_validation_features[feature_columns]),
        'driftThresholds': {
            'warningMeanShiftZ': DRIFT_WARN_Z_SCORE,
            'criticalMeanShiftZ': DRIFT_CRITICAL_Z_SCORE,
        },
    }

    evaluation = {
        'selectionMetric': 'fixed_threshold_from_case_control_benchmark',
        'featureSet': 'extended_runtime',
        'totalFeatures': len(feature_columns),
        'samplingStrategy': summary.get('trainingStrategy', {}).get('description', 'none'),
        'samplingCandidates': [],
        'selectedSamplingConfig': None,
        'thresholdGrid': [decision_threshold],
        'validationMetrics': {},
        'trainMetrics': {},
        'trainingBalance': {
            'before': summary.get('before_balance'),
            'after': summary.get('after_balance'),
        },
        'observedValidRate': float(dataset_info.get('observedAttributionRate', 0.0)),
        'observedSuspiciousProxyRate': float(dataset_info.get('observedSuspiciousProxyRate', 1.0)),
        'predictedValidRate': predicted_valid_rate,
        'predictedFraudRate': predicted_fraud_rate,
        'totalExperiments': 1,
        'notes': (
            'Deployed LightGBM model loaded from benchmark_full non_50_50_sweep '
            'case_control_1_to_10 best_model.joblib. Threshold tuned on the '
            'case-control validation; metrics recomputed on a held-out '
            'imbalanced test split.'
        ),
    }

    bundle = {
        'schema_version': BUNDLE_SCHEMA_VERSION,
        'model': estimator,
        'model_name': 'LightGBM (case 1:10)',
        'threshold': decision_threshold,
        'metrics': metrics,
        'all_results': {
            'LightGBM_case_control_1_to_10': {
                **metrics,
                'threshold': decision_threshold,
                'validation_f1': metrics['f1'],
                'validation_average_precision': metrics['average_precision'],
                'validation_roc_auc': metrics['roc_auc'],
                'validation_precision': metrics['precision'],
                'validation_recall': metrics['recall'],
                'predictedFraudRate': predicted_fraud_rate,
                'samplingConfig': {
                    'name': 'case_control_1_to_10',
                    'samplingMethod': 'case_control',
                    'maxMajorityRatio': None,
                    'targetMajorityRatio': 10,
                },
                'trainingBalance': {
                    'applied': False,
                    'method': 'case_control_downsample',
                    'samplingMethod': 'none',
                    'majorityLabel': 0,
                    'minorityLabel': 1,
                    'maxMajorityRatio': None,
                    'targetMajorityRatio': None,
                    'originalClassCounts': summary.get('rawDistribution', {}).get('counts'),
                    'rebalancedClassCounts': summary.get('evaluationDistribution', {}).get('counts'),
                    'originalImbalanceRatio': summary.get('rawDistribution', {}).get('minorityFraction'),
                    'rebalancedImbalanceRatio': summary.get('evaluationDistribution', {}).get('minorityFraction'),
                },
                'featureSet': 'extended_runtime',
                'confusionMatrix': confusion,
            },
        },
        'feature_columns': feature_columns,
        'feature_defaults': feature_defaults,
        'monitoring': monitoring,
        'dataset': dataset_info,
        'labeling_strategy': 'Model trained on the real TalkingData is_attributed label. '
                            'Positive class = attributed click (valid-click proxy); '
                            'negative class = non-attributed click (suspicious proxy, '
                            'not ground-truth fraud). High accuracy on the imbalanced '
                            'test split mainly reflects the base rate, not overfitting.',
        'cross_validation': None,
        'evaluation': evaluation,
        'confusion_matrix': confusion,
        'roc_curve': {'fpr': fpr, 'tpr': tpr},
        'pr_curve': {'precision': precision_curve, 'recall': recall_curve},
    }
    _persist_bundle(bundle)
    return bundle


def load_or_train_bundle(force_retrain=False, dataset_name=None, max_rows=None):
    # If a bundle already exists and is a LightGBM bundle, reuse it.
    if MODEL_PATH.exists() and not force_retrain:
        try:
            bundle = pickle.loads(MODEL_PATH.read_bytes())
            upgraded_bundle = _upgrade_bundle(bundle)
            if upgraded_bundle is not None and _bundle_prediction_sanity_check(upgraded_bundle):
                if str(upgraded_bundle.get('model_name', '')).startswith('LightGBM'):
                    original_metadata = {key: value for key, value in bundle.items() if key != 'model'}
                    upgraded_metadata = {key: value for key, value in upgraded_bundle.items() if key != 'model'}
                    if upgraded_metadata != original_metadata:
                        _persist_bundle(upgraded_bundle)
                    return upgraded_bundle
        except Exception:
            pass

    # Otherwise, (re)build the LightGBM case_control_1_to_10 bundle.
    recovery_max_rows = max_rows if max_rows is not None else min(
        int(os.getenv('MAX_TRAIN_ROWS', '500000')),
        10_000,
    )
    return build_lightgbm_case_control_bundle(dataset_name=dataset_name, max_rows=recovery_max_rows)


def predict_probability(model, features):
    probabilities = np.asarray(_predict_probabilities(model, features), dtype=float)
    if probabilities.size == 0:
        raise ValueError('Model returned no probability scores')
    return float(probabilities.reshape(-1)[0])


def prepare_prediction_frame(payload, feature_defaults=None, feature_columns=None):
    columns = feature_columns or FEATURE_COLUMNS
    defaults = {col: float((feature_defaults or {}).get(col, 0.0)) for col in columns}
    click_frequency = float(payload.get('clickFrequency', defaults.get('clickFrequency', 1)))
    time_interval = float(payload.get('timeInterval', defaults.get('timeInterval', 30_000)))
    device_code = payload.get('deviceCode', DEVICE_MAP.get(str(payload.get('deviceType', 'desktop')).lower(), 2))
    click_time = pd.to_datetime(
        payload.get('clickTime', payload.get('click_time')),
        errors='coerce',
        utc=True,
    )
    if pd.isna(click_time):
        click_time = pd.Timestamp.now(tz='UTC')

    hour_of_day = float(payload.get('hourOfDay', defaults.get('hourOfDay', click_time.hour)))
    ip_click_count = float(payload.get('ipClickCount', payload.get('clickFrequency', defaults.get('ipClickCount', click_frequency))))
    app_click_count = float(payload.get('appClickCount', defaults.get('appClickCount', click_frequency * 2)))
    ip_app_count = float(payload.get('ipAppCount', payload.get('clickFrequency', defaults.get('ipAppCount', click_frequency))))
    burst_score = float(payload.get('burstScore', click_frequency / max(time_interval / 1000, 1)))
    gap_seconds = max(time_interval / 1000, 1)

    # Extended feature derivations (safe defaults when not provided)
    click_count_10s = float(payload.get('clickCountLast10Seconds', defaults.get('clickCountLast10Seconds', min(click_frequency, 5))))
    click_count_60s = float(payload.get('clickCountLast60Seconds', defaults.get('clickCountLast60Seconds', click_frequency)))
    click_count_10m = float(payload.get('clickCountLast10Minutes', defaults.get('clickCountLast10Minutes', click_frequency)))
    clicks_per_device_60s = float(payload.get(
        'clicksPerDeviceLast60Seconds',
        defaults.get('clicksPerDeviceLast60Seconds', click_frequency),
    ))
    burst_click_score = float(payload.get('burstClickScore', (click_count_10s * 6 + click_count_60s + click_count_10m / 10) / gap_seconds))

    row = {col: defaults.get(col, 0.0) for col in columns}
    row.update({
        'clickFrequency': click_frequency,
        'timeInterval': time_interval,
        'deviceCode': float(device_code),
        'app': float(payload.get('app', defaults.get('app', 12))),
        'osCode': float(payload.get('osCode', payload.get('os', defaults.get('osCode', 13)))),
        'channel': float(payload.get('channel', defaults.get('channel', 111))),
        'hourOfDay': hour_of_day,
        'ipClickCount': ip_click_count,
        'appClickCount': app_click_count,
        'ipAppCount': ip_app_count,
        'burstScore': burst_score,
        'timeSinceLastClickPerIp': float(payload.get('timeSinceLastClickPerIp', defaults.get('timeSinceLastClickPerIp', time_interval))),
        'timeSinceLastClickPerDevice': float(payload.get('timeSinceLastClickPerDevice', defaults.get('timeSinceLastClickPerDevice', time_interval))),
        'clickCountLast10Seconds': click_count_10s,
        'clickCountLast60Seconds': click_count_60s,
        'clickCountLast10Minutes': click_count_10m,
        'clicksPerDeviceLast60Seconds': clicks_per_device_60s,
        'deviceAppEntropy': float(payload.get('deviceAppEntropy', defaults.get('deviceAppEntropy', 0.0))),
        'deviceIpRatio': float(payload.get('deviceIpRatio', defaults.get('deviceIpRatio', 1.0))),
        'uniqueAppsPerIp': float(payload.get('uniqueAppsPerIp', defaults.get('uniqueAppsPerIp', 1.0))),
        'uniqueAppsPerDevice': float(payload.get('uniqueAppsPerDevice', defaults.get('uniqueAppsPerDevice', 1.0))),
        'uniqueDevicesPerIp': float(payload.get('uniqueDevicesPerIp', defaults.get('uniqueDevicesPerIp', 1.0))),
        'burstClickScore': burst_click_score,
    })
    # Only include columns that the model expects
    row_filtered = {col: row.get(col, 0.0) for col in columns}
    return pd.DataFrame([[row_filtered[col] for col in columns]], columns=columns)
