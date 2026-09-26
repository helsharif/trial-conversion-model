# trial-conversion-model

Predicts, from a trial's first 3 days of behavior, whether the trial will convert to a paid plan at the end of day 14, so the growth team can reach trials that look unlikely to convert while they are still live. The business case and rollout plan are in the model plan document.

## Setup

Use Python 3.13 or newer and run commands from the repository root. Install the dependencies and the package, including MLflow and boto3:

```
uv sync
```

The training data is not committed to the repository. Copy `.env.example` to `.env` and replace the password placeholder with the one from the course's Tools & Setup lesson, then materialize the extract:

```
uv run scripts/fetch_data.py
```

This writes `data/01_raw/trial_snapshot.csv`. Training uses a processed table built from this local snapshot rather than querying the live database. Re-running the fetch script replaces the raw snapshot; retain the same extract when comparing experiments.

### Configure MLflow and AWS

The `.env.example` template also includes these settings. Fill them in alongside the database configuration:

| Environment variable | Purpose |
| --- | --- |
| `MLFLOW_TRACKING_URI` | Tracking server address; the template uses `http://127.0.0.1:5001`. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Credentials used by boto3 for S3 access. Temporary credentials also require `AWS_SESSION_TOKEN`. |
| `AWS_DEFAULT_REGION` | AWS region used by boto3. |

Keep credentials in the ignored `.env` file or use boto3's standard AWS credential providers. The training module loads `.env` when imported.

For the local tracking address in the template, start MLflow in a separate terminal and leave it running:

```bash
uv run mlflow server --host 127.0.0.1 --port 5001
```

The tracking server must be reachable before training starts: importing the training module selects the `trial_conversion_model` experiment. Open [MLflow](http://127.0.0.1:5001) to inspect runs.

Training uploads to the existing bucket `futureproofds-trial-conversion-artifacts-bucket-001`. This bucket name is hard-coded in `src/trial_conversion_model/train.py` and `scripts/check_s3.py`, not read from an environment variable. Your AWS identity needs permission to upload the objects under `models/`. Check bucket access with:

```bash
uv run scripts/check_s3.py
```

This calls S3 `HeadBucket` and prints a success message if the bucket is accessible. It does not test object-upload permissions.

## Train

Build the processed training table and train in one command:

```bash
uv run scripts/build_and_train.py
```

Or run the two stages separately:

```bash
uv run scripts/build.py
uv run scripts/train.py
```

`build.py` writes `data/03_processed/training_data.csv`. `train.py` requires that file to exist and uses it as-is; it does not fetch data or rebuild features.

Training fits an XGBoost classifier with a stratified 75%/25% train/test split and `random_state=42`. The current settings are 500 estimators, depth 3, learning rate 0.01, minimum child weight 8, row/column sampling of 0.9, and `eval_metric="auc"`.

### MLflow experiment tracking

Each call to `train()` starts a run in the **`trial_conversion_model`** experiment and records:

| Run information | Logged values |
| --- | --- |
| Model parameters | All XGBoost settings in `PARAMS`. |
| Split and feature parameters | `test_size`, `random_state`, and `n_features`. |
| Data version | `data_version`: the later of the processed file's creation and modification timestamps, formatted as `YYYY-MM-DD-HH_MM_SS`, with a metadata-change-time fallback on platforms without creation time. |
| Dataset inputs | `train_data` and `test_data`, with `converted` as the target and `training` / `testing` contexts. These calls log dataset metadata, including schema and digest, rather than uploading full CSV snapshots. |
| Evaluation metric | `test_auc`: held-out ROC-AUC calculated from conversion probabilities. |
| Model | The fitted XGBoost model logged through `mlflow.xgboost.log_model` under `model`. |

Use the MLflow UI to compare runs by parameters, data version, and test AUC. The timestamp-based `data_version` is not a content hash: rebuilding the processed table can change this value even if its rows are unchanged.

### Local outputs and S3 uploads

After logging to MLflow, training saves these files locally and uploads them directly with boto3:

| Local artifact | S3 destination |
| --- | --- |
| `models/model.json` | `s3://futureproofds-trial-conversion-artifacts-bucket-001/models/model.json` |
| `models/metrics.json` | `s3://futureproofds-trial-conversion-artifacts-bucket-001/models/metrics.json` |

`metrics.json` contains `test_auc` rounded to four decimal places, `n_train`, `n_test`, and the ordered feature names. On success, the script prints the upload confirmation and returned metrics.

Each run writes to the same local filenames and S3 keys, replacing the current artifacts; the upload keys do not include an MLflow run ID or timestamp. These direct S3 uploads are separate from MLflow's artifact storage, whose destination depends on the tracking server configuration.

MLflow logging and S3 uploads are required steps in the current training function, with no offline or skip-upload option. Errors propagate to the caller. If an upload fails, local files and the MLflow model may already have been saved, and one S3 object may have been uploaded before the other failed.

## Serve

The model is served over HTTP so the teams that need scores can get them on demand. The service loads the local `models/model.json`; it does not retrieve a model from MLflow or S3. Ensure that file exists before starting the service or building the container. Run the service locally:

```
uv run uvicorn trial_conversion_model.api.main:app --reload
```

Or build and run it as a container, which is how it ships:

```
docker build -t trial-conversion-model .
docker run -p 8000:8000 trial-conversion-model
```

`POST /predict` takes one trial's first-3-day base aggregates and returns its conversion probability plus a low/medium/high band; `GET /health` reports service status. Interactive docs live at `/docs` while the service runs.

### Checking it works

With the service running, three requests and what each should come back with.

A steady trial, spread across the first three days:

```
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "sessions_day1": 3, "sessions_day2": 2, "sessions_day3": 2,
  "listen_sessions_3d": 5, "total_minutes_3d": 180,
  "country": "US", "device_type": "iOS"
}'
```

```
{"conversion_probability":0.8593,"conversion_band":"high"}
```

The same trial's activity crammed into day one:

```
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "sessions_day1": 9, "sessions_day2": 0, "sessions_day3": 0,
  "listen_sessions_3d": 4, "total_minutes_3d": 150,
  "country": "US", "device_type": "iOS"
}'
```

```
{"conversion_probability":0.3385,"conversion_band":"medium"}
```

A request with `sessions_day1` missing, which the contract turns down before any
of your code runs:

```
curl -i -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "sessions_day2": 0, "sessions_day3": 0,
  "listen_sessions_3d": 4, "total_minutes_3d": 150,
  "country": "US", "device_type": "iOS"
}'
```

```
HTTP/1.1 422 Unprocessable Entity
```

## Layout

- `src/trial_conversion_model/`: the package. `data.py` acquires the extract from the database, loads the pipeline's inputs, and derives the data-version timestamp; `features.py` derives the model features from the snapshot's base aggregates and writes the processed training table; `train.py` trains, evaluates, logs the MLflow run, saves local artifacts, and uploads them to S3; `predict.py` scores trials from their base aggregates; `api/` is the FastAPI service (`main.py` builds the app, `routes.py` holds the endpoints, `schemas.py` defines the request and response shapes).
- `scripts/`: thin entry points that call into the package. `fetch_data.py` materializes the extract, `build.py` builds the training table, `train.py` trains from that table, `build_and_train.py` combines building and training, and `check_s3.py` checks bucket access. The logic stays importable and testable in `src/`.
- `notebooks/`: exploration only. Notebooks import from the package; no pipeline logic lives here.
- `data/01_raw/`: the raw extract as pulled from the database (not committed; replaced when fetched again).
- `data/02_interim/`: reserved for intermediate outputs in multi-step pipelines; this project goes straight from raw to processed, so it stays empty.
- `data/03_processed/`: the model-ready training table written by the pipeline (never committed).
- `models/`: local model and metrics files, also uploaded to S3 during training (not committed).
