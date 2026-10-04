import json
from pathlib import Path

import boto3
import dotenv
import mlflow
import mlflow.data
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#from xgboost import XGBClassifier
from trial_conversion_model.data import data_version, load_processed
from trial_conversion_model.features import TARGET

# import .env variables
dotenv.load_dotenv()

# MLFLOW Experiment
mlflow.set_experiment("trial_conversion_model")

MODEL_DIR = Path("models")
TEST_SIZE = 0.25
RANDOM_STATE = 42

# MODEL PARAMETERS
PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.01,
    "min_child_weight": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "auc",
}

def train(model_dir: Path = MODEL_DIR) -> dict:
    """Train the trial conversion model from the processed training table."""
    table = load_processed() # a DataFrame with features and target
    X = table.drop(columns=[TARGET])
    y = table[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    data_version_str = data_version()

    with mlflow.start_run():

        # Log model parameters to MLflow
        mlflow.log_params(PARAMS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_features", X.shape[1])

        # Log data version as parameter (appears in the Parameters column of your runs table), useful for filtering runs by the version of the training data used for model training.
        mlflow.log_param("data_version", data_version_str)

        # Log the data to MLflow. 
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        train_dataset = mlflow.data.from_pandas(train_data, name="train_data", targets=TARGET)
        test_dataset = mlflow.data.from_pandas(test_data, name="test_data", targets=TARGET)
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(test_dataset, context="testing")

        # Train the model
        model = XGBClassifier(
            **PARAMS
            )
        
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        # Log the test AUC to MLflow
        mlflow.log_metric("test_auc", auc)

        # Log the model to MLflow
        mlflow.xgboost.log_model(model, "model") 

        # Save the model and metrics to disk
        model_dir.mkdir(exist_ok=True)
        model.save_model(model_dir / "model.json")
        metrics = {
            "test_auc": round(float(auc), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features": list(X.columns),
        }
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # Upload the model and metrics to S3 into the "models" folder of the bucket "futureproofds-trial-conversion-artifacts-bucket-001"
        s3_bucket = "futureproofds-trial-conversion-artifacts-bucket-001"
        s3_client = boto3.client("s3")
        s3_client.upload_file( 
            Filename = str(model_dir / "model.json"), 
            Bucket = s3_bucket, 
            Key = "models/model.json"
            )
        s3_client.upload_file( 
            Filename = str(model_dir / "metrics.json"),
            Bucket = s3_bucket,
            Key = "models/metrics.json"
            )
        # print message that the model and metrics have been uploaded to S3
        print(f"Model and metrics uploaded to S3 bucket '{s3_bucket}' in 'models/' folder.")

        return metrics


def train_and_register(
    model_dir: Path = MODEL_DIR,
    registered_model_name: str = "trial_conversion_model",
    s3_bucket: str = "futureproofds-trial-conversion-artifacts-bucket-001",
) -> dict:
    """Promote a candidate only when its test AUC improves by at least 0.005.

    Compare with saved metrics and the registry's champion (or latest version
    when no champion alias exists), using the higher AUC if both exist. The
    first model is promoted automatically. Missing/invalid incumbent metrics
    and service errors fail closed rather than bypassing the promotion gate.

    Historical AUCs assume comparable evaluation data and the same split.
    Run this workflow serially: local files, S3 and MLflow cannot be updated
    in one transaction. Errors propagate; a failed promotion may leave a
    registered candidate or partially updated artifacts requiring a retry.
    """
    import math
    from tempfile import TemporaryDirectory

    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    min_auc_improvement = 0.005
    model_dir = Path(model_dir)
    client = MlflowClient()
    baseline_aucs = {}

    # Only a genuinely missing registered model permits first-run behavior.
    try:
        registered_model = client.get_registered_model(registered_model_name)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        registered_model = None

    incumbent = None
    if registered_model is not None:
        if "champion" in registered_model.aliases:
            incumbent = client.get_model_version_by_alias(
                registered_model_name, "champion"
            )
        elif registered_model.latest_versions:
            incumbent = max(
                registered_model.latest_versions, key=lambda version: int(version.version)
            )
        if incumbent is not None:
            if not incumbent.run_id:
                raise ValueError("Registered incumbent has no run containing test_auc.")
            baseline_aucs["registered"] = client.get_run(
                incumbent.run_id
            ).data.metrics.get("test_auc")

    model_path = model_dir / "model.json"
    metrics_path = model_dir / "metrics.json"
    if model_path.exists():
        if not metrics_path.exists():
            raise ValueError("Saved model exists but its metrics.json is missing.")
        baseline_aucs["local"] = json.loads(metrics_path.read_text()).get("test_auc")
    elif metrics_path.exists():
        raise ValueError("Saved metrics exist but their model.json is missing.")

    for source, value in baseline_aucs.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"The {source} incumbent has no valid test_auc.")
    previous_auc = max(baseline_aucs.values(), default=None)

    table = load_processed()
    X = table.drop(columns=[TARGET])
    y = table[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    with mlflow.start_run() as run:
        mlflow.log_params(PARAMS)
        mlflow.log_params({
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_features": X.shape[1],
            "data_version": data_version(),
            "min_auc_improvement": min_auc_improvement,
        })
        for name, features, target, context in (
            ("train_data", X_train, y_train, "training"),
            ("test_data", X_test, y_test, "testing"),
        ):
            dataset = mlflow.data.from_pandas(
                pd.concat([features, target], axis=1), name=name, targets=TARGET
            )
            mlflow.log_input(dataset, context=context)

        model = XGBClassifier(**PARAMS)
        model.fit(X_train, y_train)
        auc = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
        if not math.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError("Candidate test_auc must be finite and between 0 and 1.")
        mlflow.log_metric("test_auc", auc)
        improvement = None if previous_auc is None else auc - previous_auc
        if previous_auc is not None:
            mlflow.log_metric("previous_auc", previous_auc)
            mlflow.log_metric("auc_improvement", improvement)
        # Tolerate only floating-point subtraction noise at the exact boundary.
        promote = improvement is None or improvement >= min_auc_improvement or math.isclose(
            improvement, min_auc_improvement, rel_tol=0.0, abs_tol=1e-12
        )
        metrics = {
            "test_auc": auc,
            "previous_auc": previous_auc,
            "auc_improvement": improvement,
            "min_auc_improvement": min_auc_improvement,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features": list(X.columns),
            "run_id": run.info.run_id,
            "promoted": False,
        }
        if not promote:
            mlflow.set_tag("promotion_status", "rejected")
            return metrics

        mlflow.set_tag("promotion_status", "pending")
        try:
            model_info = mlflow.xgboost.log_model(model, name="model")
            version = mlflow.register_model(model_info.model_uri, registered_model_name)
            metrics.update({
                "registered_model_name": registered_model_name,
                "model_version": str(version.version),
                "promoted": True,
            })
            model_dir.mkdir(parents=True, exist_ok=True)
            # Stage before replacing the local champion so failed uploads do
            # not overwrite the model used by the prediction service.
            with TemporaryDirectory(dir=model_dir) as staging_dir:
                staged_model = Path(staging_dir) / "model.json"
                staged_metrics = Path(staging_dir) / "metrics.json"
                model.save_model(staged_model)
                staged_metrics.write_text(json.dumps(metrics, indent=2))
                s3_client = boto3.client("s3")
                for artifact in (staged_model, staged_metrics):
                    s3_client.upload_file(
                        Filename=str(artifact), Bucket=s3_bucket,
                        Key=f"models/{artifact.name}",
                    )
                staged_model.replace(model_path)
                staged_metrics.replace(metrics_path)
            client.set_registered_model_alias(
                registered_model_name, "champion", version.version
            )
            mlflow.set_tag("promotion_status", "promoted")
        except Exception:
            mlflow.set_tag("promotion_status", "failed")
            raise

        print(
            f"Promoted version {version.version} of '{registered_model_name}' "
            f"with test AUC {auc:.6f}; saved to '{model_dir}' and "
            f"uploaded to s3://{s3_bucket}/models/."
        )
        return metrics
