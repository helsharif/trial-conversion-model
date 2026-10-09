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

Training uploads to the existing bucket `futureproofds-trial-conversion-artifacts-bucket-001`. This bucket name is hard-coded in `train()` and `scripts/check_s3.py`, and is the default `s3_bucket` argument of `train_and_register()`; it is not read from an environment variable. Your AWS identity needs permission to upload the objects under `models/`. The promotion workflow also requires MLflow registry access to read models and runs, register versions, and assign aliases. Check bucket access with:

```bash
uv run scripts/check_s3.py
```

This calls S3 `HeadBucket` and prints a success message if the bucket is accessible. It does not test object-upload permissions.

## Train

Choose the entry point for the desired behavior:

| Command | Behavior |
| --- | --- |
| `uv run scripts/train.py` | Always saves the trained model locally and uploads it to S3; logs the model to MLflow without registering it. |
| `uv run scripts/train_and_register.py` | Saves, registers, and uploads only when the candidate passes the champion promotion rule below. |
| `uv run scripts/build_and_train.py` | Builds the processed table and calls the original, unconditional `train()` workflow. |

To build the processed training table and run the original training workflow in one command:

```bash
uv run scripts/build_and_train.py
```

Or run the two stages separately:

```bash
uv run scripts/build.py
uv run scripts/train.py
```

`build.py` writes `data/03_processed/training_data.csv`. Both training scripts require that file to exist and use it as-is; they do not fetch data or rebuild features.

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

### Original training: local outputs and S3 uploads

After logging to MLflow, `train()` saves these files locally and uploads them directly with boto3:

| Local artifact | S3 destination |
| --- | --- |
| `models/model.json` | `s3://futureproofds-trial-conversion-artifacts-bucket-001/models/model.json` |
| `models/metrics.json` | `s3://futureproofds-trial-conversion-artifacts-bucket-001/models/metrics.json` |

`metrics.json` contains `test_auc` rounded to four decimal places, `n_train`, `n_test`, and the ordered feature names. On success, the script prints the upload confirmation and returned metrics.

Each run writes to the same local filenames and S3 keys, replacing the current artifacts; the upload keys do not include an MLflow run ID or timestamp. These direct S3 uploads are separate from MLflow's artifact storage, whose destination depends on the tracking server configuration.

MLflow logging and S3 uploads are required steps in `train()`, with no offline or skip-upload option. Errors propagate to the caller. If an upload fails, local files and the MLflow model may already have been saved, and one S3 object may have been uploaded before the other failed.

### Train and register a champion

Build the processed table if needed, then run the promotion workflow:

```bash
uv run scripts/build.py
uv run scripts/train_and_register.py
```

The script calls `train_and_register()` in `src/trial_conversion_model/train.py` and prints its returned metrics. It uses the same features, model parameters, stratified split, and test ROC-AUC calculation as `train()`.

Before training, it finds the existing AUC from these sources:

- **Local:** `models/metrics.json`, paired with `models/model.json`.
- **MLflow:** the `test_auc` metric from the run associated with the `champion` alias of registered model `trial_conversion_model`. If there is no champion alias, it uses the highest version number available in the registered model's `latest_versions`.

If both sources exist, the higher AUC is the baseline. If neither contains an existing model, the first candidate is promoted automatically. S3 is an upload destination, not a baseline lookup source. An incomplete local model/metrics pair, invalid or missing incumbent AUC, or registry service error stops the workflow rather than treating the candidate as the first model.

Promotion requires an **absolute AUC improvement of at least `0.005`**:

```text
candidate_test_auc - previous_auc >= 0.005
```

For example, a baseline of `0.80` requires a candidate AUC of `0.805` or higher. The comparison retains full precision and allows only a `1e-12` absolute tolerance for floating-point noise at the boundary. Local metrics produced by the original `train()` remain rounded to four decimal places and are read as stored.

| Outcome | Actions |
| --- | --- |
| Rejected | Logs parameters, dataset metadata, test AUC, and the decision to MLflow. Returns `promoted: false`. Does not log a model artifact, register a version, replace local artifacts, or upload to S3. |
| Promoted | Logs the model artifact, registers a new version, stages local files, uploads both artifacts to the existing S3 keys, replaces the local files, and assigns the new version the `champion` alias. Returns `promoted: true`. |

The MLflow run records `min_auc_improvement`, plus `previous_auc` and `auc_improvement` when a baseline exists. Its `promotion_status` tag is `rejected`, `pending`, `promoted`, or `failed`, depending on the decision and progress of promotion.

Promoted `metrics.json` files contain full-precision `test_auc`, `previous_auc`, `auc_improvement`, `min_auc_improvement`, `n_train`, `n_test`, `features`, `run_id`, `promoted`, `registered_model_name`, and `model_version`. The first promotion has `null` for the previous AUC and improvement. The default local filenames and S3 keys are the same as those in the table above.

Python callers can override `model_dir`, `registered_model_name`, and `s3_bucket`. The thin script uses the defaults; the threshold is fixed at `0.005` inside the function. A custom `model_dir` changes only local storage, while S3 keys remain `models/model.json` and `models/metrics.json`.

Use comparable evaluation data across runs: this workflow compares historical AUC values and does not re-evaluate the incumbent on the candidate's test set or enforce matching data versions. Run promotions serially. MLflow, local files, and S3 are not updated in a single transaction. Upload failures preserve the local champion and registry alias, but may leave a registered candidate and one updated S3 object. Later failures can also leave local artifacts or the alias out of sync. Inspect and reconcile those states before retrying; a retry may use a changed baseline.

The original `train.py` and `build_and_train.py` commands still overwrite local and S3 artifacts unconditionally and do not update the champion alias. Use `train_and_register.py` when artifacts should follow the promotion rule.

### Test the promotion workflow

```bash
uv run pytest tests/test_train_and_register.py -q
```

These tests mock MLflow, S3, and model training. They cover initial promotion, the exact threshold, rejection, registry fallback, conflicting baselines, invalid metrics, registry errors, and upload failure behavior without live registration or uploads.

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

The model loads once when the API module is imported at startup. Restart the service after replacing the local model to serve the updated artifact.

| Endpoint | Behavior |
| --- | --- |
| `POST /predict` | Scores one trial's first-3-day base aggregates, records the request and prediction, and returns `conv_prob` and `conv_band`. |
| `GET /health` | Returns `status: "ok"` and `version`, the installed `trial-conversion-model` package version. |
| `GET /prediction-logs` | Displays recorded predictions as an HTML table, newest first. |
| `GET /docs` | Opens the interactive API documentation. |

`conv_prob` is rounded to four decimal places before assigning `conv_band`: **low** for values below `0.33`, **medium** for values from `0.33` up to but excluding `0.66`, and **high** for values of `0.66` or above. All seven request fields shown below are required; session counts and total minutes must be nonnegative.

### Prediction logging

`main.py` creates `logs/` relative to the service's working directory and configures INFO-level console logging with timestamps. Each successful prediction produces a readable console message and appends one JSON record to `logs/predictions.jsonl`. The dedicated prediction logger writes only the JSON message to the file and does not propagate it to the console logger.

Each record contains a Unix timestamp in seconds, the validated request fields nested under `request`, and the returned `conv_prob` and `conv_band`. Records accumulate across service restarts; the current implementation does not rotate or truncate the file.

Open [prediction logs](http://127.0.0.1:8000/prediction-logs) while running locally, or use the same path on the deployed host. The page shows the total number of records, timestamps formatted as `YYYY-MM-DD HH:MM:SS` from Unix time (UTC), each request feature in its own column, and the probability and band in the final two columns. Floating-point values display with four decimal places. An empty or missing file displays `No prediction logs found.` The page reads the entire file on each visit; refresh it to see new predictions.

Container logs are written inside the container. Mount persistent storage at the container's working-directory `logs/` path if records must survive container replacement. The log viewer exposes the recorded request features and predictions, and these routes do not implement authentication.

### Checking it works

With the service running, try these three requests. The probabilities below illustrate the response format; actual values depend on the loaded model. Successful requests also appear in the prediction log viewer.

A steady trial, spread across the first three days:

```
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "sessions_day1": 3, "sessions_day2": 2, "sessions_day3": 2,
  "listen_sessions_3d": 5, "total_minutes_3d": 180,
  "country": "US", "device_type": "iOS"
}'
```

```
{"conv_prob":0.8593,"conv_band":"high"}
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
{"conv_prob":0.3385,"conv_band":"medium"}
```

A request with `sessions_day1` missing is rejected by request validation before the prediction handler runs, so it does not create a prediction record:

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
- `scripts/`: thin entry points that call into the package. `fetch_data.py` materializes the extract, `build.py` builds the training table, `train.py` trains from that table, `train_and_register.py` runs training with conditional champion promotion, `build_and_train.py` combines building and unconditional training, and `check_s3.py` checks bucket access. The logic stays importable and testable in `src/`.
- `tests/`: automated checks for the champion promotion workflow with mocked external services.
- `notebooks/`: exploration only. Notebooks import from the package; no pipeline logic lives here.
- `data/01_raw/`: the raw extract as pulled from the database (not committed; replaced when fetched again).
- `data/02_interim/`: reserved for intermediate outputs in multi-step pipelines; this project goes straight from raw to processed, so it stays empty.
- `data/03_processed/`: the model-ready training table written by the pipeline (never committed).
- `models/`: local model and metrics files, also uploaded to S3 during training (not committed).
- `logs/predictions.jsonl`: append-only records of successful API predictions, displayed by `GET /prediction-logs`.
