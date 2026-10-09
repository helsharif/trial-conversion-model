import json
import os
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

from trial_conversion_model.data import RAW_DATA

REPORT_DIR = Path("monitoring")

# The convention the report is built with: the dataset counts as drifted
# when at least half of the features have drifted individually.
DRIFT_SHARE = 0.5


def load_reference(path: Path = RAW_DATA) -> pd.DataFrame:
    """The training extract is the reference: what normal looked like."""
    return pd.read_csv(path)


def run_drift_check(current: pd.DataFrame, reference: pd.DataFrame) -> object:
    """Compare current trials against the reference, feature by feature.

    Labels for live trials are 11+ days away, so input drift is the
    earliest signal that the world has shifted under the model.
    """
    # TODO: This function needs imports that are not in this file yet. Add
    # them as you go.
    #
    # Both frames arrive as raw base aggregates. See build_training_data
    # in features.py and predict_proba in predict.py for how the rest of the
    # project derives the model's features, then stop where they start
    # one-hot encoding: a drift test wants the categorical columns as they are.
    #
    # Build Evidently's Report with one DataDriftPreset, using DRIFT_SHARE as
    # its drift_share, then run it against both prepared frames and return
    # the result.
    raise NotImplementedError


def drifted_share(snapshot: object) -> float:
    """Pull the share of drifted feature columns out of the report."""
    for metric in json.loads(snapshot.json())["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            return float(metric["value"]["share"])
    raise ValueError("drift metric missing from report")


def publish_report(snapshot: object, name: str) -> dict:
    """Write the report locally (HTML for eyes, JSON for machines) and to S3."""
    load_dotenv()
    REPORT_DIR.mkdir(exist_ok=True)
    html_path = REPORT_DIR / f"{name}.html"
    json_path = REPORT_DIR / f"{name}.json"
    snapshot.save_html(str(html_path))
    share = drifted_share(snapshot)
    json_path.write_text(
        json.dumps({"name": name, "drifted_share": share, "drift": share >= DRIFT_SHARE})
    )

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3")
    for path in (html_path, json_path):
        s3.upload_file(str(path), bucket, f"monitoring/{path.name}")
    return {"drifted_share": share, "drift": share >= DRIFT_SHARE}
