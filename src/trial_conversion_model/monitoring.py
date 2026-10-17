import json
import os
from pathlib import Path

import boto3
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset

from trial_conversion_model.data import RAW_DATA
from trial_conversion_model.features import FEATURES, add_features

REPORT_DIR = Path("monitoring")
# The reference sits in the bucket next to models/ and predictions/.
REFERENCE_KEY = "data/trial_snapshot.csv"

# The convention the report is built with: the dataset counts as drifted
# when at least half of the features have drifted individually.
DRIFT_SHARE = 0.5


def load_reference(path: Path = RAW_DATA) -> pd.DataFrame:
    """The training extract is the reference: what normal looked like.

    The jobs image carries no data, so the first run on a fresh machine
    fetches the extract from the bucket where it was published once;
    later runs find it on disk.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        s3 = boto3.client("s3")
        s3.download_file(os.environ["S3_BUCKET"], REFERENCE_KEY, str(path))
    return pd.read_csv(path)


def pull_specific_cohort_from_s3_by_name(
    bucket: str, prefix: str, local_dir: Path, cohort_name: str
) -> Path:
    """Download a specific cohort file from S3 to a local directory.

    The local directory is created if it does not exist. The cohort is
    determined by its name, not by timestamp.
    """
    s3 = boto3.client(
        "s3", region_name="eu-north-1", config=Config(signature_version=UNSIGNED)
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in objects:
        raise ValueError(f"No objects found in {bucket}/{prefix}")

    # Find the object with the specified cohort name
    specific_object = next(
        (o for o in objects["Contents"] if Path(o["Key"]).name == cohort_name), None
    )

    if specific_object is None:
        raise ValueError(f"Cohort {cohort_name} not found in {bucket}/{prefix}")

    local_path = local_dir / Path(specific_object["Key"]).name
    s3.download_file(bucket, specific_object["Key"], str(local_path))
    return local_path


def pull_newest_cohort_from_s3(bucket: str, prefix: str, local_dir: Path) -> Path:
    """Download the latest cohort file from S3 to a local directory.

    The local directory is created if it does not exist. The latest file is
    determined by the last modified timestamp, not by name.
    """
    s3 = boto3.client(
        "s3", region_name="eu-north-1", config=Config(signature_version=UNSIGNED)
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in objects:
        raise ValueError(f"No objects found in {bucket}/{prefix}")
    latest = max(objects["Contents"], key=lambda o: o["LastModified"])
    local_path = local_dir / Path(latest["Key"]).name
    s3.download_file(bucket, latest["Key"], str(local_path))
    return local_path


def pull_oldest_cohort_from_s3(bucket: str, prefix: str, local_dir: Path) -> Path:
    """Download the oldest cohort file from S3 to a local directory.

    The local directory is created if it does not exist. The oldest file is
    determined by the last modified timestamp, not by name.
    """
    s3 = boto3.client(
        "s3", region_name="eu-north-1", config=Config(signature_version=UNSIGNED)
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in objects:
        raise ValueError(f"No objects found in {bucket}/{prefix}")
    oldest = min(objects["Contents"], key=lambda o: o["LastModified"])
    local_path = local_dir / Path(oldest["Key"]).name
    s3.download_file(bucket, oldest["Key"], str(local_path))
    return local_path


def load_cohort(path: Path = RAW_DATA) -> pd.DataFrame:
    """Load the current cohort data to check for drift against the reference."""
    return pd.read_parquet(path)


def run_drift_check(current: pd.DataFrame, reference: pd.DataFrame) -> object:
    """Compare current trials against the reference, feature by feature.

    Labels for live trials are 11+ days away, so input drift is the
    earliest signal that the world has shifted under the model.
    """
    ref = add_features(reference)[FEATURES]
    cur = add_features(current)[FEATURES]
    report = Report([DataDriftPreset(drift_share=DRIFT_SHARE)])
    return report.run(reference_data=ref, current_data=cur)


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
        json.dumps(
            {"name": name, "drifted_share": share, "drift": share >= DRIFT_SHARE}
        )
    )

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3")
    for path in (html_path, json_path):
        s3.upload_file(str(path), bucket, f"monitoring/{path.name}")
    return {"drifted_share": share, "drift": share >= DRIFT_SHARE}


def alert(message: str) -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        print("SLACK_WEBHOOK_URL not set; alert printed only")
        return
    requests.post(webhook, json={"text": message}, timeout=10)
