import sys
from pathlib import Path
import os
import pandas as pd
from trial_conversion_model.monitoring import (
    load_reference,
    run_drift_check,
    publish_report,
    drifted_share,
    pull_specific_cohort_from_s3_by_name,
    pull_newest_cohort_from_s3,
    pull_oldest_cohort_from_s3,
    load_cohort,
)


REFERENCE_PATH = Path("data/01_raw/trial_snapshot.csv")
COHORT_S3_BUCKET = "fpds-trial-cohorts"
COHORT_S3_PREFIX = "cohorts/"
COHORT_LOCAL_DIR = Path("data/01_raw/cohorts")

if __name__ == "__main__":

    # if argument of cohort file name is provided, pull that specific cohort from S3
    # example: uv run scripts/check_drift.py cohort_2026-09-13.parquet
    # example: uv run scripts/check_drift.py cohort_2026-10-09.parquet
    if len(sys.argv) == 2:
        cohort_name = sys.argv[1]
        cohort_path = pull_specific_cohort_from_s3_by_name(COHORT_S3_BUCKET, COHORT_S3_PREFIX, COHORT_LOCAL_DIR, cohort_name)
        cohort_df = load_cohort(cohort_path)

        # reference cohort
        reference_df = load_reference(REFERENCE_PATH)

        # Run drift checks, return snapshot
        cohort_snapshot = run_drift_check(current=cohort_df, reference=reference_df)

        # return drift share
        cohort_drifted_share = drifted_share(cohort_snapshot)

        # publish report
        cohort_result = publish_report(cohort_snapshot, cohort_path.stem)

        # print to console
        verdict_cohort = "DRIFT" if cohort_result["drift"] else "OK"
        print(f"Cohort: {cohort_path.name} - Drifted share: {cohort_drifted_share:.2%} - Verdict: {verdict_cohort}")

    else: # example: uv run scripts/check_drift.py
        # if no argument is provided, pull the oldest and newest cohorts from S3
        # and check them for drift against the reference
        print("No cohort file name provided. Checking oldest and newest cohorts from S3.")

        # reference cohort
        reference_df = load_reference(REFERENCE_PATH)

        # pull oldest cohort from S3
        oldest_cohort_path = pull_oldest_cohort_from_s3(COHORT_S3_BUCKET, COHORT_S3_PREFIX, COHORT_LOCAL_DIR)
        oldest_cohort_df = load_cohort(oldest_cohort_path)

        # pull newest cohort from S3
        newest_cohort_path = pull_newest_cohort_from_s3(COHORT_S3_BUCKET, COHORT_S3_PREFIX, COHORT_LOCAL_DIR)
        newest_cohort_df = load_cohort(newest_cohort_path)

        # Run drift checks, return snapshot
        oldest_cohort_snapshot = run_drift_check(current=oldest_cohort_df, reference=reference_df)
        newest_cohort_snapshot = run_drift_check(current=newest_cohort_df, reference=reference_df)

        # return drift share
        oldest_cohort_drifted_share = drifted_share(oldest_cohort_snapshot)
        newest_cohort_drifted_share = drifted_share(newest_cohort_snapshot)

        # publish reports
        oldest_cohort_result = publish_report(oldest_cohort_snapshot, oldest_cohort_path.stem)
        newest_cohort_result = publish_report(newest_cohort_snapshot, newest_cohort_path.stem)

        # print to console
        verdict_oldest_cohort = "DRIFT" if oldest_cohort_result["drift"] else "OK"
        verdict_newest_cohort = "DRIFT" if newest_cohort_result["drift"] else "OK"

        print(f"Oldest cohort: {oldest_cohort_path.name} - Drifted share: {oldest_cohort_drifted_share:.2%} - Verdict: {verdict_oldest_cohort}")
        print(f"Newest cohort: {newest_cohort_path.name} - Drifted share: {newest_cohort_drifted_share:.2%} - Verdict: {verdict_newest_cohort}")

   
