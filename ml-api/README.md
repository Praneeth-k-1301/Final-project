# ML API

This service exposes a Flask API for click-fraud scoring backed by the TalkingData ad-tracking dataset.

## What it does

- loads data from `talkingdata-adtracking-fraud-detection.zip`
- engineers dataset-aligned behavioral features
- evaluates multiple classical ML models
- selects the best model by validation performance
- persists the trained bundle and evaluation artifacts
- serves prediction and metadata endpoints for the backend and dashboard

## Core files

- `app.py` – Flask routes and bundle serving
- `pipeline.py` – data loading, feature engineering, training, evaluation, persistence
- `test_pipeline.py` – targeted unit tests
- `artifacts/` – persisted model bundle, metadata, and SVG plots

## Feature schema

The trained bundle uses these feature columns:

- `clickFrequency`
- `timeInterval`
- `deviceCode`
- `app`
- `osCode`
- `channel`
- `hourOfDay`
- `ipClickCount`
- `appClickCount`
- `ipAppCount`
- `burstScore`

## Training strategy

The dataset does not provide a direct fraud label, so the current pipeline uses a **weakly supervised valid-click proxy** derived from:

- TalkingData attribution outcomes, and
- burst/anomaly heuristics built from clickstream behavior.

The pipeline compares multiple candidate models and persists the best-performing bundle with threshold tuning for F1.

## Endpoints

- `GET /` – service summary
- `GET /health` – same metadata as the root route
- `GET /model-metadata` – trained model name, threshold, metrics, dataset details, artifact paths
- `POST /train` – retrain the bundle; accepts optional `datasetName` and `maxRows`
- `POST /predict` – score a click payload

`/predict` accepts either:

- a structured feature object, or
- a legacy `features` array for backward compatibility

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The service listens on `http://localhost:5000`.

## Requirements

Installed from `requirements.txt`:

- Flask
- flask-cors
- numpy
- pandas
- scikit-learn

## Artifacts written by training

- `artifacts/model_bundle.pkl`
- `artifacts/model_metadata.json`
- `artifacts/roc_curve.svg`
- `artifacts/pr_curve.svg`
- `artifacts/confusion_matrix.svg`

## Validation status

Completed successfully:

- `python -m unittest test_pipeline.py`
- bounded real-data training run against the local TalkingData archive

## Notes

- `load_or_train_bundle()` loads persisted artifacts when available and only retrains when needed.
- The default local dataset is `train_sample.csv` unless `DATASET_FILE` or an explicit `datasetName` is supplied.
- This service is designed for reproducible local experimentation rather than production-scale distributed training.