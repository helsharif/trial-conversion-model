import os
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

from trial_conversion_model.cohorts import pull_newest_cohort
from trial_conversion_model.predict import load_model, predict_proba, to_band

PREDICTIONS_DIR = Path("data/04_predictions")

# Load the model
model = load_model()

if __name__ == "__main__":
    load_dotenv()
    cohort_path = pull_newest_cohort()
    cohort = pd.read_parquet(cohort_path)

    # This is the lifecycle team's daily ranked list: every trial needs a
    # conversion_probability from the same predict function your API serves
    # (routes.py shows how it's called), with the trials most likely to
    # cancel at the top. Call what you get back `scored`.

    # predict for entire cohort
    conv_prob = (predict_proba(model, cohort)).astype(float).round(4) # Step 2: Score and round to 4 decimals
    conv_band = conv_prob.map(to_band) # Step 3: Convert each probability to a band

    # create dataframe with all original cohort columns, plus conv_prob and conv_band
    scored_df = cohort.copy()
    scored_df["conversion_probability"] = conv_prob
    scored_df["conversion_band"] = conv_band

    # sort by conv_prob ascending, so the trials least likely to convert are at the top
    scored_df = scored_df.sort_values(by="conversion_probability", ascending=True)

    # excport to CSV and upload to S3
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        PREDICTIONS_DIR / f"scored_{cohort_path.stem.removeprefix('cohort_')}.csv"
    )
    scored_df.to_csv(out_path, index=False)
    s3 = boto3.client("s3")
    s3.upload_file(
        str(out_path), os.environ["S3_BUCKET"], f"predictions/{out_path.name}"
    )
    print(f"scored {len(scored_df)} trials from {cohort_path.name} -> {out_path} and S3")
