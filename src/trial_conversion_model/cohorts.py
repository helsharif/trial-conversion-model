from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# The course-side pipeline publishes one dated day-3 cohort file per day.
# The bucket is public and read-only: pulling from it needs no credentials
# and involves no coordination between AWS accounts.
COHORT_BUCKET = "fpds-trial-cohorts"
COHORT_PREFIX = "cohorts/"
COHORT_DIR = Path("data/01_raw/cohorts")


def pull_newest_cohort() -> Path:
    """Download the latest cohort file from S3 to a local directory.
    
    The local directory is created if it does not exist. The latest file is
    determined by the last modified timestamp, not by name.
    """
    s3 = boto3.client(
        "s3", region_name="eu-north-1", config=Config(signature_version=UNSIGNED)
    )
    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    objects = s3.list_objects_v2(Bucket=COHORT_BUCKET, Prefix=COHORT_PREFIX)
    if "Contents" not in objects:
        raise ValueError(f"No objects found in {COHORT_BUCKET}/{COHORT_PREFIX}")
    latest = max(objects["Contents"], key=lambda o: o["LastModified"])
    local_path = COHORT_DIR / Path(latest["Key"]).name
    s3.download_file(COHORT_BUCKET, latest["Key"], str(local_path))
    return local_path


def newest_local_cohort() -> Path:
    """The most recent cohort already pulled to disk."""
    files = sorted(COHORT_DIR.glob("cohort_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no cohort files in {COHORT_DIR}; run scripts/score.py first"
        )
    return files[-1]
