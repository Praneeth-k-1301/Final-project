from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from benchmark_full_pipeline import (
    LGBMClassifier,
    build_preprocessor,
    collect_binary_metrics,
    sample_case_control_subset,
    sample_prevalence_faithful_subset,
    split_train_validation_test,
    tune_decision_threshold,
)
from pipeline import RANDOM_STATE, engineer_training_frame, load_talkingdata_frame


ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "benchmark_full" / "non_50_50_sweep"
MODEL_PATH = ARTIFACT_ROOT / "case_control_1_to_10" / "best_model.joblib"
SUMMARY_PATH = ARTIFACT_ROOT / "case_control_1_to_10" / "benchmark_summary.json"

CASE_CONTROL_NAME = "case_control_1_to_10"
PREVALENCE_NAME = "prevalence_20k_baseline"

BURST_COLUMNS = [
    "clickCountLast10Seconds",
    "clicksPerDeviceLast60Seconds",
    "burstClickScore",
    "burstScore",
]
IP_AGG_COLUMNS = [
    "uniqueAppsPerIp",
    "uniqueDevicesPerIp",
    "ipClickCount",
    "ipAppCount",
]
TEMPORAL_COLUMNS = [
    "timeSinceLastClickPerIp",
    "hourOfDay",
    "timeInterval",
]


def _safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _evaluate_threshold(estimator, x: pd.DataFrame, y: pd.Series, threshold: float) -> dict:
    probabilities = estimator.predict_proba(x)[:, 1]
    predictions = (probabilities >= float(threshold)).astype(int)
    return {
        "Precision": float(precision_score(y, predictions, zero_division=0)),
        "Recall": float(recall_score(y, predictions, zero_division=0)),
        "F1": float(f1_score(y, predictions, zero_division=0)),
        "AUC": _safe_roc_auc(y, probabilities),
    }


def _format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{value:.3f}"


def _print_markdown_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def _load_feature_frame(dataset_name: str | None, max_rows: int | None) -> tuple[pd.DataFrame, pd.Series]:
    raw = load_talkingdata_frame(dataset_name=dataset_name, max_rows=max_rows)
    features, labels, _ = engineer_training_frame(raw, feature_set="extended_runtime")
    return features.reset_index(drop=True), labels.reset_index(drop=True)


def _prepare_benchmark_datasets(features: pd.DataFrame, labels: pd.Series) -> dict:
    cc_x, cc_y, _ = sample_case_control_subset(
        features,
        labels,
        negative_to_positive_ratio=10.0,
        random_state=RANDOM_STATE,
    )
    pv_x, pv_y, _ = sample_prevalence_faithful_subset(
        features,
        labels,
        max_rows=20000,
        random_state=RANDOM_STATE,
    )
    return {
        CASE_CONTROL_NAME: (cc_x.reset_index(drop=True), cc_y.reset_index(drop=True)),
        PREVALENCE_NAME: (pv_x.reset_index(drop=True), pv_y.reset_index(drop=True)),
    }


def run_threshold_comparison(all_features: pd.DataFrame, all_labels: pd.Series) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    model_artifact = joblib.load(MODEL_PATH)
    estimator = model_artifact["estimator"]

    feature_order = list(all_features.columns)
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        summary_columns = summary.get("dataset", {}).get("featureColumns")
        if summary_columns:
            feature_order = list(summary_columns)

    datasets = _prepare_benchmark_datasets(all_features, all_labels)
    thresholds = [0.30, 0.175]

    rows: list[list[str]] = []
    benchmark_label_map = {
        CASE_CONTROL_NAME: "case_control",
        PREVALENCE_NAME: "prevalence_20k",
    }

    for dataset_name in [CASE_CONTROL_NAME, PREVALENCE_NAME]:
        x_data, y_data = datasets[dataset_name]
        x_eval = x_data[feature_order]
        for threshold in thresholds:
            metrics = _evaluate_threshold(estimator, x_eval, y_data, threshold)
            rows.append([
                benchmark_label_map[dataset_name],
                f"{threshold:.3f}",
                _format_float(metrics["Precision"]),
                _format_float(metrics["Recall"]),
                _format_float(metrics["F1"]),
                _format_float(metrics["AUC"]),
            ])

    print("\nTASK 1 — THRESHOLD COMPARISON TABLE\n")
    _print_markdown_table(
        ["Benchmark", "Threshold", "Precision", "Recall", "F1", "AUC"],
        rows,
    )


def _train_lightgbm_once(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_params: dict,
) -> dict:
    if LGBMClassifier is None:
        raise ImportError("lightgbm is not available. Install it with: pip install lightgbm")

    preprocessor, _, _ = build_preprocessor(x_train)
    preprocessor.fit(x_train)
    x_train_t = preprocessor.transform(x_train)
    x_val_t = preprocessor.transform(x_validation)
    x_test_t = preprocessor.transform(x_test)

    estimator = LGBMClassifier(**model_params)
    estimator.fit(x_train_t, np.asarray(y_train, dtype=int))

    validation_prob = estimator.predict_proba(x_val_t)[:, 1]
    decision_threshold = float(tune_decision_threshold(y_validation, validation_prob))

    test_prob = estimator.predict_proba(x_test_t)[:, 1]
    test_pred = (test_prob >= decision_threshold).astype(int)
    metrics = collect_binary_metrics(y_test, test_pred, test_prob)
    return {
        "threshold": decision_threshold,
        "metrics": metrics,
    }


def run_ablation_study(all_features: pd.DataFrame, all_labels: pd.Series) -> None:
    datasets = _prepare_benchmark_datasets(all_features, all_labels)
    cc_features, cc_labels = datasets[CASE_CONTROL_NAME]
    splits = split_train_validation_test(cc_features, cc_labels)
    x_train, y_train = splits["train"]
    x_validation, y_validation = splits["validation"]
    x_test, y_test = splits["test"]

    full_baseline_f1 = 0.800
    full_baseline_auc = 0.919
    full_baseline_precision = 0.905
    full_baseline_recall = 0.717

    base_params = {"random_state": RANDOM_STATE, "objective": "binary", "verbose": -1, "n_jobs": 1}
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        saved_estimator = artifact.get("estimator")
        if saved_estimator is not None and hasattr(saved_estimator, "get_params"):
            extracted = dict(saved_estimator.get_params())
            extracted["random_state"] = RANDOM_STATE
            extracted["n_jobs"] = 1
            base_params.update(extracted)

    ablations = [
        ("Burst indicators", BURST_COLUMNS),
        ("IP aggregation", IP_AGG_COLUMNS),
        ("Temporal features", TEMPORAL_COLUMNS),
    ]

    rows: list[list[str]] = [[
        "None (full model)",
        _format_float(full_baseline_precision),
        _format_float(full_baseline_recall),
        _format_float(full_baseline_f1),
        _format_float(full_baseline_auc),
    ]]

    for label, removed_columns in ablations:
        keep_columns = [col for col in x_train.columns if col not in removed_columns]
        result = _train_lightgbm_once(
            x_train[keep_columns],
            y_train,
            x_validation[keep_columns],
            y_validation,
            x_test[keep_columns],
            y_test,
            base_params,
        )
        m = result["metrics"]
        rows.append([
            label,
            _format_float(float(m["Precision"])),
            _format_float(float(m["Recall"])),
            _format_float(float(m["F1"])),
            _format_float(float(m["ROC-AUC"])),
        ])

    print("\nTASK 2 — ABLATION STUDY (metrics on case_control_1_to_10)\n")
    print("Removed features by run:")
    print(f"- Burst indicators: {', '.join(BURST_COLUMNS)}")
    print(f"- IP aggregation: {', '.join(IP_AGG_COLUMNS)}")
    print(f"- Temporal features: {', '.join(TEMPORAL_COLUMNS)}")
    print()
    _print_markdown_table(
        ["Features Removed", "Precision", "Recall", "F1", "AUC"],
        rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold comparison + feature ablation report for TalkingData LightGBM benchmark artifacts."
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="CSV name inside TalkingData zip (default: pipeline.py behavior, usually train.csv).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional max rows to load from TalkingData source.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_features, all_labels = _load_feature_frame(args.dataset_name, args.max_rows)
    run_threshold_comparison(all_features, all_labels)
    run_ablation_study(all_features, all_labels)


if __name__ == "__main__":
    main()
