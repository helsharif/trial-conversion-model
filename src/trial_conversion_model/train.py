import json
from pathlib import Path

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#from xgboost import XGBClassifier

from trial_conversion_model.data import load_processed
from trial_conversion_model.features import TARGET

import dotenv

import mlflow
import mlflow.xgboost


# import .env variables
dotenv.load_dotenv()

# MLFLOW Experiment
mlflow.set_experiment("trial_conversion_model")

MODEL_DIR = Path("models")
TEST_SIZE = 0.25
RANDOM_STATE = 42

# MODEL PARAMETERS
PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "auc",
}

def train(model_dir: Path = MODEL_DIR) -> dict:
    """Train the trial conversion model from the processed training table."""
    table = load_processed()
    X = table.drop(columns=[TARGET])
    y = table[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    with mlflow.start_run():

        # Log model parameters to MLflow
        mlflow.log_params(PARAMS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_features", X.shape[1])

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
        return metrics
