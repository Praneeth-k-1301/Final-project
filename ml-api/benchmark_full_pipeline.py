import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             average_precision_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from pipeline import RANDOM_STATE, engineer_training_frame, load_talkingdata_frame

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    RandomOverSampler = None
    SMOTE = None
    ImbPipeline = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts' / 'benchmark_full'
PIP_INSTALL_COMMAND = 'pip install imbalanced-learn lightgbm xgboost optuna seaborn joblib'
SEARCH_ITERATIONS = 30
CV_FOLDS = 5
SEARCH_N_JOBS = 1
SMOTE_K_NEIGHBORS = 3
CASE_CONTROL_NEGATIVE_RATIO = 1.0
PARTIAL_CASE_CONTROL_RATIOS = (2.0, 5.0, 10.0)
PREVALENCE_SUBSET_ROWS = 10000
LARGER_PREVALENCE_SUBSET_ROWS = 20000
TRAIN_ONLY_OVERSAMPLE_RATIO = 10.0
MIN_CV_POSITIVES_PER_FOLD = 5
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
MIN_TRAIN_POSITIVES = 2
MIN_VALIDATION_POSITIVES = 1
MIN_TEST_POSITIVES = 1
SPLIT_MAX_ATTEMPTS = 25
PREVALENCE_SEARCH_SCORING = 'average_precision'
DEFAULT_SEARCH_SCORING = 'roc_auc'
PERMUTATION_SAMPLE_SIZE = 1000
PERMUTATION_REPEATS = 2


def missing_optional_dependencies():
    missing = []
    if SMOTE is None:
        missing.append('imbalanced-learn')
    if LGBMClassifier is None:
        missing.append('lightgbm')
    if XGBClassifier is None:
        missing.append('xgboost')
    return missing


def class_distribution(labels):
    counts = pd.Series(labels).value_counts().sort_index()
    minority_fraction = float(counts.min() / counts.sum()) if len(counts) > 1 else 0.0
    return {
        'counts': {str(int(label)): int(count) for label, count in counts.items()},
        'minorityFraction': minority_fraction,
    }


def should_apply_smote(labels, imbalance_threshold=0.30):
    summary = class_distribution(labels)
    return len(summary['counts']) > 1 and summary['minorityFraction'] < imbalance_threshold


def sample_case_control_subset(features, labels, negative_to_positive_ratio=CASE_CONTROL_NEGATIVE_RATIO,
                               random_state=RANDOM_STATE):
    features = features.reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    positive_index = np.flatnonzero(labels.to_numpy() == 1)
    negative_index = np.flatnonzero(labels.to_numpy() == 0)
    if len(positive_index) == 0 or len(negative_index) == 0:
        raise ValueError('Case-control sampling requires both positive and negative classes.')

    requested_negatives = max(1, int(np.ceil(len(positive_index) * float(negative_to_positive_ratio))))
    sampled_negative_count = min(len(negative_index), requested_negatives)
    rng = np.random.default_rng(random_state)
    sampled_negative_index = rng.choice(negative_index, size=sampled_negative_count, replace=False)
    sampled_index = np.concatenate([positive_index, sampled_negative_index])
    rng.shuffle(sampled_index)

    sampled_features = features.iloc[sampled_index].reset_index(drop=True)
    sampled_labels = labels.iloc[sampled_index].reset_index(drop=True)
    metadata = {
        'strategy': 'case_control',
        'negativeToPositiveRatio': float(negative_to_positive_ratio),
        'sampledRows': int(len(sampled_labels)),
        'positiveRows': int((sampled_labels == 1).sum()),
        'negativeRows': int((sampled_labels == 0).sum()),
    }
    return sampled_features, sampled_labels, metadata


def _positive_count(labels):
    labels = np.asarray(labels, dtype=int)
    return int((labels == 1).sum())


def _safe_metric(metric_fn, *args, default=float('nan'), **kwargs):
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return float(default)


def _resolve_split_count(total_rows, split_size):
    if 0 < float(split_size) < 1:
        return int(round(total_rows * float(split_size)))
    return int(split_size)


def split_train_validation_test(features, labels, test_size=TEST_SIZE, validation_size=VALIDATION_SIZE,
                                min_train_positives=MIN_TRAIN_POSITIVES,
                                min_validation_positives=MIN_VALIDATION_POSITIVES,
                                min_test_positives=MIN_TEST_POSITIVES,
                                max_attempts=SPLIT_MAX_ATTEMPTS):
    features = features.reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    total_rows = len(labels)
    total_positives = _positive_count(labels)
    minimum_required = int(min_train_positives + min_validation_positives + min_test_positives)
    if total_positives < minimum_required:
        raise ValueError(
            f'Need at least {minimum_required} positives to satisfy split constraints; found {total_positives}.'
        )
    test_count = _resolve_split_count(total_rows, test_size)
    validation_count = _resolve_split_count(total_rows, validation_size)
    train_count = total_rows - test_count - validation_count
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError(
            f'Invalid split sizes for {total_rows} rows: train={train_count}, '
            f'validation={validation_count}, test={test_count}.'
        )
    for attempt in range(max_attempts):
        split_seed = RANDOM_STATE + attempt
        x_train_validation, x_test, y_train_validation, y_test = train_test_split(
            features,
            labels,
            test_size=test_count,
            random_state=split_seed,
            stratify=labels,
        )
        x_train, x_validation, y_train, y_validation = train_test_split(
            x_train_validation.reset_index(drop=True),
            y_train_validation.reset_index(drop=True),
            test_size=validation_count,
            random_state=split_seed,
            stratify=y_train_validation,
        )
        split_map = {
            'train': (x_train.reset_index(drop=True), y_train.reset_index(drop=True)),
            'validation': (x_validation.reset_index(drop=True), y_validation.reset_index(drop=True)),
            'test': (x_test.reset_index(drop=True), y_test.reset_index(drop=True)),
        }
        positive_counts = {name: _positive_count(split_labels) for name, (_, split_labels) in split_map.items()}
        if (
            positive_counts['train'] >= int(min_train_positives)
            and positive_counts['validation'] >= int(min_validation_positives)
            and positive_counts['test'] >= int(min_test_positives)
        ):
            return split_map
    raise ValueError(
        'Unable to produce a stratified train/validation/test split with the required positive counts '
        f'after {max_attempts} attempts.'
    )


def sample_prevalence_faithful_subset(features, labels, max_rows=None, random_state=RANDOM_STATE):
    features = features.reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    if max_rows is None or len(labels) <= int(max_rows):
        metadata = build_prevalence_faithful_metadata(labels)
        metadata['sourceRows'] = int(len(labels))
        return features, labels, metadata
    sampled_features, _, sampled_labels, _ = train_test_split(
        features,
        labels,
        train_size=int(max_rows),
        random_state=random_state,
        stratify=labels,
    )
    sampled_features = sampled_features.reset_index(drop=True)
    sampled_labels = sampled_labels.reset_index(drop=True)
    metadata = build_prevalence_faithful_metadata(sampled_labels)
    metadata.update({
        'strategy': 'prevalence_faithful_stratified_subset',
        'sourceRows': int(len(labels)),
    })
    return sampled_features, sampled_labels, metadata


def build_preprocessor(features):
    categorical_columns = features.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numeric_columns = [column for column in features.columns if column not in categorical_columns]
    transformers = []
    if numeric_columns:
        transformers.append(('num', SklearnPipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), numeric_columns))
    if categorical_columns:
        transformers.append(('cat', SklearnPipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder='drop'), numeric_columns, categorical_columns


def training_positive_ratio(labels):
    counts = pd.Series(labels).value_counts()
    positive_count = int(counts.get(1, 0))
    negative_count = int(counts.get(0, 0))
    if positive_count <= 0:
        return 1.0
    return float(max(negative_count, 1) / positive_count)


def prevalence_weight_grid(labels):
    ratio = max(1.0, training_positive_ratio(labels))
    values = {
        round(max(1.0, np.sqrt(ratio)), 6),
        round(max(1.0, ratio / 4.0), 6),
        round(max(1.0, ratio / 2.0), 6),
        round(ratio, 6),
    }
    return sorted(values)


def balanced_hist_gradient_fit_params(labels, training_strategy=None):
    if normalize_training_strategy(training_strategy)['kind'] != 'none':
        return {}
    sample_weights = compute_sample_weight(class_weight='balanced', y=np.asarray(labels, dtype=int))
    return {'classifier__sample_weight': sample_weights}


def build_model_configs(labels=None, prevalence_faithful=False, training_strategy=None):
    weight_grid = prevalence_weight_grid(labels) if prevalence_faithful and labels is not None else [1.0]
    rf_class_weights = ['balanced_subsample']
    hgb_fit_params_builder = None
    if prevalence_faithful and labels is not None:
        ratio = max(weight_grid)
        rf_class_weights = [
            'balanced',
            'balanced_subsample',
            {0: 1.0, 1: max(1.0, np.sqrt(ratio))},
            {0: 1.0, 1: ratio},
        ]
        hgb_fit_params_builder = balanced_hist_gradient_fit_params
    configs = {
        'HistGradientBoostingClassifier': {
            'estimator': HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'classifier__max_iter': [100, 200, 300],
                'classifier__learning_rate': [0.03, 0.05, 0.1],
                'classifier__max_depth': [None, 5, 8],
                'classifier__max_leaf_nodes': [15, 31, 63],
                'classifier__min_samples_leaf': [5, 10, 20],
                'classifier__l2_regularization': [0.0, 0.03, 0.1],
            },
            'fit_params_builder': hgb_fit_params_builder,
        },
        'RandomForestClassifier': {
            'estimator': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            'params': {
                'classifier__n_estimators': [200, 300, 400],
                'classifier__max_depth': [None, 8, 14],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__max_features': ['sqrt', 'log2'],
                'classifier__class_weight': rf_class_weights,
            },
        },
    }
    configs['LightGBM'] = {
        'estimator': None if LGBMClassifier is None else LGBMClassifier(
            objective='binary',
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=1,
        ),
        'params': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.03, 0.05, 0.1],
            'classifier__num_leaves': [15, 31, 63],
            'classifier__max_depth': [-1, 5, 8],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0],
            'classifier__min_child_samples': [5, 10, 20],
            'classifier__reg_alpha': [0.0, 0.5, 1.0],
            'classifier__reg_lambda': [0.0, 1.0, 5.0],
            'classifier__scale_pos_weight': weight_grid,
        },
        'error': None if LGBMClassifier is not None else f'LightGBM is unavailable. Run: {PIP_INSTALL_COMMAND}',
    }
    configs['XGBoost'] = {
        'estimator': None if XGBClassifier is None else XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            objective='binary:logistic',
            n_jobs=1,
            tree_method='hist',
            verbosity=0,
        ),
        'params': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.03, 0.05, 0.1],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_child_weight': [1, 5, 10],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0],
            'classifier__gamma': [0.0, 1.0],
            'classifier__reg_alpha': [0, 0.1, 1],
            'classifier__reg_lambda': [1, 2, 5],
            'classifier__max_delta_step': [0, 1, 5],
            'classifier__scale_pos_weight': weight_grid,
        },
        'error': None if XGBClassifier is not None else f'XGBoost is unavailable. Run: {PIP_INSTALL_COMMAND}',
    }
    return configs


def normalize_training_strategy(training_strategy=None):
    strategy = dict(training_strategy or {})
    kind = str(strategy.get('kind', 'none')).strip().lower()
    if kind not in {'none', 'sample_weight', 'random_oversample', 'smote'}:
        kind = 'none'
    target_ratio = strategy.get('targetNegativeToPositiveRatio')
    if kind in {'random_oversample', 'smote'}:
        target_ratio = max(1.0, float(target_ratio or TRAIN_ONLY_OVERSAMPLE_RATIO))
    return {
        'name': strategy.get('name', kind),
        'kind': kind,
        'description': strategy.get('description', ''),
        'targetNegativeToPositiveRatio': target_ratio,
    }


def minority_to_majority_ratio(negative_to_positive_ratio):
    return min(1.0, 1.0 / max(float(negative_to_positive_ratio), 1.0))


def build_sampler_for_training_strategy(training_strategy):
    training_strategy = normalize_training_strategy(training_strategy)
    if training_strategy['kind'] == 'random_oversample':
        if RandomOverSampler is None or ImbPipeline is None:
            raise ImportError(f'RandomOverSampler requested but imbalanced-learn is unavailable. Run: {PIP_INSTALL_COMMAND}')
        return 'oversample', RandomOverSampler(
            random_state=RANDOM_STATE,
            sampling_strategy=minority_to_majority_ratio(training_strategy['targetNegativeToPositiveRatio']),
        )
    if training_strategy['kind'] == 'smote':
        if SMOTE is None or ImbPipeline is None:
            raise ImportError(f'SMOTE requested but imbalanced-learn is unavailable. Run: {PIP_INSTALL_COMMAND}')
        return 'smote', SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=SMOTE_K_NEIGHBORS,
            sampling_strategy=minority_to_majority_ratio(training_strategy['targetNegativeToPositiveRatio']),
        )
    return None, None


def make_training_pipeline(preprocessor, estimator, training_strategy=None):
    training_strategy = normalize_training_strategy(training_strategy)
    sampler_name, sampler = build_sampler_for_training_strategy(training_strategy)
    if sampler is not None:
        return ImbPipeline([
            ('preprocessor', preprocessor),
            (sampler_name, sampler),
            ('classifier', estimator),
        ])
    return SklearnPipeline([
        ('preprocessor', preprocessor),
        ('classifier', estimator),
    ])


def fit_params_for_training_strategy(training_strategy, labels):
    training_strategy = normalize_training_strategy(training_strategy)
    if training_strategy['kind'] != 'sample_weight':
        return {}
    sample_weights = compute_sample_weight(class_weight='balanced', y=np.asarray(labels, dtype=int))
    return {'classifier__sample_weight': sample_weights}


def determine_cv_folds(labels, preferred_folds=CV_FOLDS, min_positives_per_fold=MIN_CV_POSITIVES_PER_FOLD):
    counts = pd.Series(labels).value_counts()
    if counts.empty or len(counts) < 2:
        raise ValueError('Cross-validation requires both classes in the training split.')
    minority_count = int(counts.min())
    if minority_count < 2:
        raise ValueError('Cross-validation requires at least two minority samples in the training split.')
    safe_by_density = max(2, minority_count // max(int(min_positives_per_fold), 1))
    return int(min(preferred_folds, minority_count, safe_by_density))


def get_feature_names(preprocessor):
    names = preprocessor.get_feature_names_out()
    return [name.split('__', 1)[1] if '__' in name else name for name in names]


def extract_feature_importance(best_estimator, feature_names, x_test, y_test):
    classifier = best_estimator.named_steps['classifier']
    if hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_[0]) if classifier.coef_.ndim == 2 else np.abs(classifier.coef_)
    elif hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    else:
        transformed = best_estimator.named_steps['preprocessor'].transform(x_test)
        sample_size = min(len(y_test), PERMUTATION_SAMPLE_SIZE)
        permutation = permutation_importance(
            classifier,
            transformed[:sample_size],
            np.asarray(y_test)[:sample_size],
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            scoring='roc_auc',
            n_jobs=1,
        )
        importances = permutation.importances_mean
    series = pd.Series(importances, index=feature_names, dtype='float64')
    return series.sort_values(ascending=False).head(15)


def build_prevalence_faithful_metadata(labels):
    labels = pd.Series(labels).reset_index(drop=True)
    return {
        'strategy': 'prevalence_faithful',
        'sampledRows': int(len(labels)),
        'positiveRows': int((labels == 1).sum()),
        'negativeRows': int((labels == 0).sum()),
    }


def build_non_5050_strategy_plans():
    return [
        {
            'name': 'prevalence_10k_baseline',
            'paperOption': 'natural_imbalance',
            'description': 'Prevalence-faithful 10k subset with no training rebalancing.',
            'datasetStrategy': {'kind': 'prevalence_faithful', 'maxRows': PREVALENCE_SUBSET_ROWS},
            'trainingStrategy': {'name': 'none', 'kind': 'none', 'description': 'No class weighting or resampling.'},
        },
        {
            'name': 'prevalence_20k_baseline',
            'paperOption': 'larger_prevalence_subset',
            'description': 'Larger prevalence-faithful 20k subset with no training rebalancing.',
            'datasetStrategy': {'kind': 'prevalence_faithful', 'maxRows': LARGER_PREVALENCE_SUBSET_ROWS},
            'trainingStrategy': {'name': 'none', 'kind': 'none', 'description': 'No class weighting or resampling.'},
        },
        {
            'name': 'case_control_1_to_2',
            'paperOption': 'partial_case_control',
            'description': 'Case-control evaluation with one positive and two negatives.',
            'datasetStrategy': {'kind': 'case_control', 'negativeToPositiveRatio': 2.0},
            'trainingStrategy': {'name': 'none', 'kind': 'none', 'description': 'No class weighting or resampling.'},
        },
        {
            'name': 'case_control_1_to_5',
            'paperOption': 'partial_case_control',
            'description': 'Case-control evaluation with one positive and five negatives.',
            'datasetStrategy': {'kind': 'case_control', 'negativeToPositiveRatio': 5.0},
            'trainingStrategy': {'name': 'none', 'kind': 'none', 'description': 'No class weighting or resampling.'},
        },
        {
            'name': 'case_control_1_to_10',
            'paperOption': 'partial_case_control',
            'description': 'Case-control evaluation with one positive and ten negatives.',
            'datasetStrategy': {'kind': 'case_control', 'negativeToPositiveRatio': 10.0},
            'trainingStrategy': {'name': 'none', 'kind': 'none', 'description': 'No class weighting or resampling.'},
        },
        {
            'name': 'prevalence_20k_class_weighted',
            'paperOption': 'class_weight_only',
            'description': 'Larger prevalence-faithful subset with balanced sample weights during training.',
            'datasetStrategy': {'kind': 'prevalence_faithful', 'maxRows': LARGER_PREVALENCE_SUBSET_ROWS},
            'trainingStrategy': {
                'name': 'balanced_sample_weight',
                'kind': 'sample_weight',
                'description': 'Pass balanced sample weights during model fitting without resampling rows.',
            },
        },
        {
            'name': 'prevalence_20k_train_oversample_1_to_10',
            'paperOption': 'train_only_rebalancing',
            'description': 'Larger prevalence-faithful subset with train-only random oversampling to a 10:1 negative-to-positive ratio.',
            'datasetStrategy': {'kind': 'prevalence_faithful', 'maxRows': LARGER_PREVALENCE_SUBSET_ROWS},
            'trainingStrategy': {
                'name': 'random_oversample_10_to_1',
                'kind': 'random_oversample',
                'targetNegativeToPositiveRatio': TRAIN_ONLY_OVERSAMPLE_RATIO,
                'description': 'Duplicate minority training rows only until the train split reaches a 10:1 ratio.',
            },
        },
    ]


def prepare_strategy_dataset(features, labels, dataset_strategy):
    dataset_strategy = dict(dataset_strategy or {})
    strategy_kind = str(dataset_strategy.get('kind', 'prevalence_faithful')).strip().lower()
    if strategy_kind == 'case_control':
        return sample_case_control_subset(
            features,
            labels,
            negative_to_positive_ratio=float(dataset_strategy.get('negativeToPositiveRatio', CASE_CONTROL_NEGATIVE_RATIO)),
        )
    return sample_prevalence_faithful_subset(
        features,
        labels,
        max_rows=dataset_strategy.get('maxRows'),
    )


def plot_confusion_heatmap(model_name, matrix, artifact_dir):
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axis)
    axis.set_title(f'{model_name} Confusion Matrix')
    axis.set_xlabel('Predicted label')
    axis.set_ylabel('True label')
    output_path = artifact_dir / f'{model_name.lower()}_confusion_matrix.png'
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return str(output_path)


def plot_feature_importance(model_name, importance_series, label, artifact_dir):
    figure, axis = plt.subplots(figsize=(8, 6))
    importance_series.sort_values().plot(kind='barh', ax=axis, color='#4f46e5')
    axis.set_title(f'{model_name} Top 15 {label}')
    axis.set_xlabel(label)
    axis.set_ylabel('Feature')
    output_path = artifact_dir / f'{model_name.lower()}_feature_importance.png'
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return str(output_path)


def plot_roc_curves(results, artifact_dir):
    figure, axis = plt.subplots(figsize=(8, 6))
    for name, result in results.items():
        axis.plot(result['roc_curve']['fpr'], result['roc_curve']['tpr'], label=f"{name} (AUC={result['metrics']['ROC-AUC']:.4f})")
    axis.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    axis.set_title('ROC Curve Comparison')
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.legend(loc='lower right')
    output_path = artifact_dir / 'roc_curve_comparison.png'
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return str(output_path)


def plot_pr_curves(results, artifact_dir):
    figure, axis = plt.subplots(figsize=(8, 6))
    for name, result in results.items():
        axis.plot(result['pr_curve']['recall'], result['pr_curve']['precision'], label=name)
    axis.set_title('Precision-Recall Curve Comparison')
    axis.set_xlabel('Recall')
    axis.set_ylabel('Precision')
    axis.legend(loc='lower left')
    output_path = artifact_dir / 'precision_recall_comparison.png'
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return str(output_path)


def resampled_distribution_for_reporting(x_train, y_train, training_strategy=None):
    training_strategy = normalize_training_strategy(training_strategy)
    summary_before = class_distribution(y_train)
    if training_strategy['kind'] in {'none', 'sample_weight'}:
        training_metadata = {'trainingStrategy': training_strategy}
        if training_strategy['kind'] == 'sample_weight':
            sample_weights = compute_sample_weight(class_weight='balanced', y=np.asarray(y_train, dtype=int))
            training_metadata['sampleWeightByClass'] = {
                str(int(label)): float(np.mean(sample_weights[np.asarray(y_train, dtype=int) == int(label)]))
                for label in np.unique(np.asarray(y_train, dtype=int))
            }
        return summary_before, summary_before, training_metadata
    preprocessor, _, _ = build_preprocessor(x_train)
    transformed = preprocessor.fit_transform(x_train)
    _, sampler = build_sampler_for_training_strategy(training_strategy)
    if sampler is None:
        return summary_before, summary_before, {'trainingStrategy': training_strategy}
    _, rebalanced_labels = sampler.fit_resample(transformed, y_train)
    return summary_before, class_distribution(rebalanced_labels), {'trainingStrategy': training_strategy}


def tune_decision_threshold(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size == 0:
        return 0.5
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
    return best['threshold']


def collect_binary_metrics(labels, predictions, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        'Accuracy': accuracy_score(labels, predictions),
        'Precision': precision_score(labels, predictions, zero_division=0),
        'Recall': recall_score(labels, predictions, zero_division=0),
        'F1': f1_score(labels, predictions, zero_division=0),
        'ROC-AUC': _safe_metric(roc_auc_score, labels, probabilities),
        'PR-AUC': _safe_metric(average_precision_score, labels, probabilities, default=0.0),
        'Kappa': cohen_kappa_score(labels, predictions),
        'MCC': matthews_corrcoef(labels, predictions),
    }


def evaluate_model(name, config, x_train, y_train, x_validation, y_validation, x_test, y_test, cv,
                   training_strategy, artifact_dir, search_scoring=DEFAULT_SEARCH_SCORING):
    if config.get('error'):
        raise ImportError(config['error'])
    preprocessor, _, _ = build_preprocessor(x_train)
    pipeline = make_training_pipeline(preprocessor, config['estimator'], training_strategy)
    fit_params = fit_params_for_training_strategy(training_strategy, y_train)
    fit_params_builder = config.get('fit_params_builder')
    if callable(fit_params_builder):
        fit_params.update(fit_params_builder(y_train, training_strategy))
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=config['params'],
        n_iter=SEARCH_ITERATIONS,
        scoring=search_scoring,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=SEARCH_N_JOBS,
        refit=True,
    )
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', message='X does not have valid feature names.*')
        search.fit(x_train, y_train, **fit_params)
    train_time = time.time() - start_time
    best_estimator = search.best_estimator_
    validation_probabilities = best_estimator.predict_proba(x_validation)[:, 1]
    decision_threshold = tune_decision_threshold(y_validation, validation_probabilities)
    validation_predictions = (validation_probabilities >= decision_threshold).astype(int)
    probabilities = best_estimator.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions)
    best_index = search.best_index_
    if np.unique(np.asarray(y_test, dtype=int)).size < 2:
        roc_fpr, roc_tpr = np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])
    else:
        roc_fpr, roc_tpr, _ = roc_curve(y_test, probabilities)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, probabilities)
    feature_names = get_feature_names(best_estimator.named_steps['preprocessor'])
    importance = extract_feature_importance(best_estimator, feature_names, x_test, y_test)
    return {
        'best_estimator': best_estimator,
        'best_params': search.best_params_,
        'decision_threshold': float(decision_threshold),
        'search_scoring': search_scoring,
        'validation_metrics': collect_binary_metrics(y_validation, validation_predictions, validation_probabilities),
        'metrics': collect_binary_metrics(y_test, predictions, probabilities),
        'classification_report': classification_report(y_test, predictions, zero_division=0),
        'confusion_matrix': matrix.tolist(),
        'confusion_matrix_plot': plot_confusion_heatmap(name, matrix, artifact_dir),
        'roc_curve': {'fpr': roc_fpr.tolist(), 'tpr': roc_tpr.tolist()},
        'pr_curve': {'precision': pr_precision.tolist(), 'recall': pr_recall.tolist()},
        'cv_scores': {
            'mean': float(search.cv_results_['mean_test_score'][best_index]),
            'std': float(search.cv_results_['std_test_score'][best_index]),
        },
        'train_time_seconds': train_time,
        'feature_importance_plot': plot_feature_importance(
            name,
            importance,
            'Coefficient magnitude' if hasattr(best_estimator.named_steps['classifier'], 'coef_') else 'Importance',
            artifact_dir,
        ),
    }


def select_speed_accuracy_tradeoff(comparison_table):
    if comparison_table.empty:
        return None
    auc_score = (comparison_table['ROC-AUC'] - comparison_table['ROC-AUC'].min())
    auc_denominator = comparison_table['ROC-AUC'].max() - comparison_table['ROC-AUC'].min()
    if auc_denominator:
        auc_score = auc_score / auc_denominator
    else:
        auc_score = pd.Series(1.0, index=comparison_table.index)
    if 'PR-AUC' in comparison_table.columns:
        pr_score = (comparison_table['PR-AUC'] - comparison_table['PR-AUC'].min())
        pr_denominator = comparison_table['PR-AUC'].max() - comparison_table['PR-AUC'].min()
        if pr_denominator:
            pr_score = pr_score / pr_denominator
        else:
            pr_score = pd.Series(1.0, index=comparison_table.index)
        auc_score = 0.5 * auc_score + 0.5 * pr_score
    speed_raw = np.log1p(comparison_table['Train_Time(s)'])
    if speed_raw.max() != speed_raw.min():
        speed_score = 1 - ((speed_raw - speed_raw.min()) / (speed_raw.max() - speed_raw.min()))
    else:
        speed_score = pd.Series(1.0, index=comparison_table.index)
    blended = 0.7 * auc_score + 0.3 * speed_score
    return comparison_table.loc[blended.idxmax(), 'Model']


def build_strategy_comparison_table(summary):
    rows = []
    for name, evaluation in summary.get('evaluations', {}).items():
        comparison_table = evaluation.get('comparisonTable', [])
        if not comparison_table:
            continue
        best_row = {
            ('Best_Model_Flag' if key == 'Best_Model' else key): value
            for key, value in comparison_table[0].items()
        }
        design = evaluation.get('evaluationDesign', {})
        rows.append({
            'Strategy': name,
            'PaperOption': evaluation.get('paperOption'),
            'Best_Model_Name': evaluation.get('bestModel'),
            'Training_Strategy': evaluation.get('trainingStrategy', {}).get('name'),
            'Sampled_Rows': design.get('sampledRows'),
            'Positive_Rows': design.get('positiveRows'),
            'Negative_Rows': design.get('negativeRows'),
            'CV_Folds': evaluation.get('cvFolds'),
            **best_row,
        })
    if not rows:
        return pd.DataFrame()
    comparison = pd.DataFrame(rows)
    sort_columns = [
        column for column in ['F1', 'PR-AUC', 'ROC-AUC', 'Precision', 'Recall']
        if column in comparison.columns
    ]
    return comparison.sort_values(sort_columns, ascending=False).reset_index(drop=True)


def is_prevalence_faithful_design(evaluation_design):
    strategy = str((evaluation_design or {}).get('strategy', '')).strip().lower()
    return strategy.startswith('prevalence_faithful')


def run_evaluation_mode(evaluation_name, features, labels, dataset_info, evaluation_design, raw_distribution,
                        training_strategy=None, artifact_root=None, paper_option=None):
    artifact_root = artifact_root or ARTIFACTS_DIR
    artifact_dir = artifact_root / evaluation_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    training_strategy = normalize_training_strategy(training_strategy)
    evaluation_distribution = class_distribution(labels)
    splits = split_train_validation_test(features, labels)
    x_train, y_train = splits['train']
    x_validation, y_validation = splits['validation']
    x_test, y_test = splits['test']
    split_distribution = {name: class_distribution(split_labels) for name, (_, split_labels) in splits.items()}
    before_balance, after_balance, training_metadata = resampled_distribution_for_reporting(
        x_train,
        y_train,
        training_strategy,
    )
    cv_folds = determine_cv_folds(y_train)
    prevalence_faithful = is_prevalence_faithful_design(evaluation_design)
    search_scoring = PREVALENCE_SEARCH_SCORING if prevalence_faithful else DEFAULT_SEARCH_SCORING
    print(f'\n===== {evaluation_name.upper()} EVALUATION =====')
    print('Paper option:', paper_option)
    print('Evaluation design:', evaluation_design)
    print('Training strategy:', training_strategy)
    print('Evaluation dataset distribution:', evaluation_distribution)
    print('Split distribution:', split_distribution)
    print('Class distribution before balancing:', before_balance)
    print('Class distribution after balancing:', after_balance)
    print('Cross-validation folds:', cv_folds)
    print('Search scoring:', search_scoring)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    failures = {}
    model_configs = build_model_configs(y_train, prevalence_faithful=prevalence_faithful, training_strategy=training_strategy)
    for name, config in model_configs.items():
        print(f'\nTraining {name} for {evaluation_name}...')
        try:
            results[name] = evaluate_model(
                name,
                config,
                x_train,
                y_train,
                x_validation,
                y_validation,
                x_test,
                y_test,
                cv,
                training_strategy,
                artifact_dir,
                search_scoring=search_scoring,
            )
            print('Best params:', json.dumps(results[name]['best_params'], indent=2))
            print(f"Validation-selected threshold: {results[name]['decision_threshold']:.3f}")
            print('Classification report:\n', results[name]['classification_report'])
        except Exception as exc:
            failures[name] = str(exc)
            print(f'{name} failed during {evaluation_name}: {exc}')
    if not results:
        raise RuntimeError(f'No models completed successfully for {evaluation_name}.')
    comparison_table = pd.DataFrame([
        {
            'Model': name,
            **result['metrics'],
            'Validation_F1': result['validation_metrics']['F1'],
            'Validation_PR-AUC': result['validation_metrics']['PR-AUC'],
            'Threshold': result['decision_threshold'],
            'CV_Mean': result['cv_scores']['mean'],
            'CV_Std': result['cv_scores']['std'],
            'Train_Time(s)': result['train_time_seconds'],
        }
        for name, result in results.items()
    ]).sort_values(['F1', 'PR-AUC', 'ROC-AUC', 'Precision', 'Recall'], ascending=False).reset_index(drop=True)
    comparison_table['Best_Model'] = comparison_table['Model'].eq(comparison_table.iloc[0]['Model'])
    best_model_name = comparison_table.iloc[0]['Model']
    speed_accuracy_model = select_speed_accuracy_tradeoff(comparison_table)
    best_model_path = artifact_dir / 'best_model.joblib'
    joblib.dump({
        'estimator': results[best_model_name]['best_estimator'],
        'decision_threshold': results[best_model_name]['decision_threshold'],
        'evaluationName': evaluation_name,
        'evaluationDesign': evaluation_design,
        'rawDistribution': raw_distribution,
    }, best_model_path)
    summary = {
        'dataset': dataset_info,
        'rawDistribution': raw_distribution,
        'evaluationName': evaluation_name,
        'paperOption': paper_option,
        'evaluationDesign': evaluation_design,
        'evaluationDistribution': evaluation_distribution,
        'trainingStrategy': training_strategy,
        'searchScoring': search_scoring,
        'before_balance': before_balance,
        'after_balance': after_balance,
        'splitDistribution': split_distribution,
        'trainingMetadata': training_metadata,
        'cvFolds': cv_folds,
        'bestModel': best_model_name,
        'bestModelPath': str(best_model_path),
        'bestSpeedAccuracyTradeoff': speed_accuracy_model,
        'comparisonTable': comparison_table.to_dict(orient='records'),
        'failures': failures,
        'plots': {
            'roc': plot_roc_curves(results, artifact_dir),
            'pr': plot_pr_curves(results, artifact_dir),
        },
        'results': {name: {key: value for key, value in result.items() if key != 'best_estimator'} for name, result in results.items()},
    }
    (artifact_dir / 'comparison_table.csv').write_text(comparison_table.to_csv(index=False), encoding='utf-8')
    (artifact_dir / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(f'\nFinal comparison table for {evaluation_name}:')
    print(comparison_table.to_string(index=False))
    print(f'\nBest overall model ({evaluation_name}): {best_model_name}')
    print(f'Best speed/accuracy tradeoff ({evaluation_name}): {speed_accuracy_model}')
    if failures:
        print('Model failures:', json.dumps(failures, indent=2))
    return summary


def run_non_5050_strategy_sweep(features, labels, dataset_info, raw_distribution):
    artifact_root = ARTIFACTS_DIR / 'non_50_50_sweep'
    artifact_root.mkdir(parents=True, exist_ok=True)
    strategy_plans = build_non_5050_strategy_plans()
    summary = {
        'dataset': dataset_info,
        'rawDistribution': raw_distribution,
        'strategyPlans': strategy_plans,
        'evaluations': {},
    }
    (artifact_root / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    for plan in strategy_plans:
        evaluation_features, evaluation_labels, evaluation_design = prepare_strategy_dataset(
            features,
            labels,
            plan['datasetStrategy'],
        )
        summary['evaluations'][plan['name']] = run_evaluation_mode(
            plan['name'],
            evaluation_features,
            evaluation_labels,
            dataset_info,
            evaluation_design,
            raw_distribution,
            training_strategy=plan['trainingStrategy'],
            artifact_root=artifact_root,
            paper_option=plan['paperOption'],
        )
        comparison = build_strategy_comparison_table(summary)
        if not comparison.empty:
            (artifact_root / 'strategy_comparison.csv').write_text(comparison.to_csv(index=False), encoding='utf-8')
            summary['strategyComparison'] = comparison.to_dict(orient='records')
        (artifact_root / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    return summary


def main(dataset_name=None, max_rows=None, run_balanced=True, run_imbalanced=True, imbalanced_max_rows=None,
         run_strategy_sweep=False):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid')
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings('ignore', message='X does not have valid feature names.*')
    missing = missing_optional_dependencies()
    if missing:
        print('Missing optional packages detected:', ', '.join(missing))
        print('Install with:', PIP_INSTALL_COMMAND)
    raw = load_talkingdata_frame(dataset_name=dataset_name, max_rows=max_rows)
    features, labels, dataset_info = engineer_training_frame(raw, feature_set='extended_runtime')
    raw_distribution = class_distribution(labels)
    print('Raw dataset distribution:', raw_distribution)
    if run_strategy_sweep:
        return run_non_5050_strategy_sweep(features, labels, dataset_info, raw_distribution)
    summary = {
        'dataset': dataset_info,
        'rawDistribution': raw_distribution,
        'evaluations': {},
    }
    (ARTIFACTS_DIR / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    if run_balanced:
        benchmark_features, benchmark_labels, benchmark_design = sample_case_control_subset(features, labels)
        summary['evaluations']['balanced_case_control'] = run_evaluation_mode(
            'balanced_case_control',
            benchmark_features,
            benchmark_labels,
            dataset_info,
            benchmark_design,
            raw_distribution,
        )
        (ARTIFACTS_DIR / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    if run_imbalanced:
        imbalanced_features, imbalanced_labels, imbalanced_design = sample_prevalence_faithful_subset(
            features,
            labels,
            max_rows=imbalanced_max_rows,
        )
        summary['evaluations']['imbalanced_prevalence'] = run_evaluation_mode(
            'imbalanced_prevalence',
            imbalanced_features,
            imbalanced_labels,
            dataset_info,
            imbalanced_design,
            raw_distribution,
        )
        (ARTIFACTS_DIR / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    return summary


if __name__ == '__main__':
    main()