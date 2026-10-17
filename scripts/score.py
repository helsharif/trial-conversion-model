import os
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

from trial_conversion_model.cohorts import pull_newest_cohort

PREDICTIONS_DIR = Path("data/04_predictions")

if __name__ == "__main__":
    load_dotenv()
    cohort_path = pull_newest_cohort()
    cohort = pd.read_parquet(cohort_path)

    # TODO: This script needs imports from your own package that are not in
    # this file yet. Add them as you go.
    #
    # This is the lifecycle team's daily ranked list: every trial needs a
    # conversion_probability from the same predict function your API serves
    # (routes.py shows how it's called), with the trials most likely to
    # cancel at the top. Call what you get back `scored`.

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        PREDICTIONS_DIR / f"scored_{cohort_path.stem.removeprefix('cohort_')}.csv"
    )
    scored.to_csv(out_path, index=False)
    s3 = boto3.client("s3")
    s3.upload_file(
        str(out_path), os.environ["S3_BUCKET"], f"predictions/{out_path.name}"
    )
    print(f"scored {len(scored)} trials from {cohort_path.name} -> {out_path} and S3")
