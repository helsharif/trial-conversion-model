"""Fill in the last month as if the system had been running since then.

Run once, before building the dashboard: uv run scripts/backfill.py

It fills the thirty days before today with scored cohorts and clean drift
verdicts, as if the system had been running since then, replacing anything
already in the bucket for those days, so the dashboard has a smooth month
behind it. It is a one-time fill, not part of the system: the daily jobs
are scripts/score.py and scripts/check_drift.py, and this never runs on a
schedule.
"""

import json
import os
from datetime import UTC, datetime, time, timedelta

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from xgboost import XGBClassifier

from trial_conversion_model.predict import load_model, predict_proba

DAYS = 30
COHORT_SIZE = 120
# The model has eight features; a drift share is a count out of eight.
FEATURE_COUNT = 8

# The course's own cohort generator, copied here so this script needs
# nothing from outside this repo. The two copies must stay identical.
GAMMA_SHAPE, GAMMA_SCALE = 2.4, 1.3
BINGE_SHARE = 0.22
COUNTRY_P = {"US": 0.4, "EU": 0.2, "India": 0.2, "Rest": 0.2}
DEVICE_P = {"iOS": 1 / 3, "Android": 1 / 3, "Web": 1 / 3}
DRIFT_GAMMA_SCALE = 0.85
DRIFT_BINGE_SHARE = 0.42
DRIFT_COUNTRY_P = {"US": 0.28, "EU": 0.17, "India": 0.25, "Rest": 0.30}
DRIFT_DEVICE_P = {"iOS": 0.22, "Android": 0.33, "Web": 0.45}
ID_BASE = 900_000


def make_cohort(date: datetime, n: int, drift: bool) -> pd.DataFrame:
    seed = int(date.strftime("%Y%m%d")) + (1 if drift else 0)
    rng = np.random.default_rng(seed)
    gamma_scale = DRIFT_GAMMA_SCALE if drift else GAMMA_SCALE
    binge_share = DRIFT_BINGE_SHARE if drift else BINGE_SHARE
    country_p = DRIFT_COUNTRY_P if drift else COUNTRY_P
    device_p = DRIFT_DEVICE_P if drift else DEVICE_P
    started = date - timedelta(days=3)
    rows = []
    for i in range(n):
        lam = rng.gamma(GAMMA_SHAPE, gamma_scale)
        binge = rng.random() < binge_share
        device = rng.choice(list(device_p), p=list(device_p.values()))
        country = rng.choice(list(country_p), p=list(country_p.values()))
        listen_prop = float(
            np.clip(rng.beta(2.2, 4.5) + (0.10 if device != "Web" else 0), 0, 1)
        )
        if binge:
            rates3 = np.array([lam * 3.2, lam * 0.25, lam * 0.25])
        else:
            rates3 = np.array([lam, lam, lam]) * rng.uniform(0.8, 1.2, 3)
        d3 = rng.poisson(rates3)
        durations = [max(int(rng.exponential(10)) + 1, 1) for _ in range(int(d3.sum()))]
        listens = int(sum(rng.random() < listen_prop for _ in durations))
        rows.append(
            dict(  # noqa: C408
                trial_id=ID_BASE + seed % 10_000 * 1_000 + i,
                user_id=ID_BASE + seed % 10_000 * 1_000 + i,
                trial_started_at=started
                + timedelta(seconds=int(rng.integers(0, 86400))),
                country=country,
                device_type=device,
                sessions_day1=int(d3[0]),
                sessions_day2=int(d3[1]),
                sessions_day3=int(d3[2]),
                listen_sessions_3d=listens,
                total_minutes_3d=float(sum(durations)),
            )
        )
    return pd.DataFrame(rows)


def score(model: XGBClassifier, cohort: pd.DataFrame) -> pd.DataFrame:
    """Rank a cohort, least likely to convert first."""
    # The ranking must match the daily job, so these are the same two lines
    # as scripts/score.py.
    scored = cohort.assign(conversion_probability=predict_proba(model, cohort))
    return scored.sort_values("conversion_probability", ascending=True)


def ok_verdict(date: datetime) -> dict:
    """A clean drift verdict in the same shape the daily drift check writes."""
    # The real check is not run: on generated data of 120 rows it sits near
    # the 50% threshold, and this history has to be clean by construction.
    # Its own rng, so the share never depends on how many rows the cohort drew.
    rng = np.random.default_rng(int(date.strftime("%Y%m%d")) + 100)
    k = int(rng.integers(0, 3))
    return {
        "name": f"drift_cohort_{date:%Y-%m-%d}",
        "drifted_share": k / FEATURE_COUNT,
        "drift": False,
        # So nobody later mistakes a filled-in verdict for a real check;
        # the dashboard ignores it.
        "source": "backfill",
    }


def main() -> None:
    load_dotenv()
    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3")
    model = load_model()

    # The course dates its cohorts in UTC, and a laptop in the Americas in
    # the evening is already on tomorrow's date there.
    today = datetime.now(UTC).date()
    days = [today - timedelta(days=n) for n in range(DAYS, 0, -1)]

    for day in days:
        # Both writes replace whatever is already there for that day, so the
        # past month always comes out whole and a rerun gives the same month.
        day_start = datetime.combine(day, time())
        s3.put_object(
            Bucket=bucket,
            Key=f"monitoring/drift_cohort_{day:%Y-%m-%d}.json",
            Body=json.dumps(ok_verdict(day_start)).encode(),
            ContentType="application/json",
        )
        cohort = make_cohort(day_start, COHORT_SIZE, drift=False)
        scored = score(model, cohort)
        s3.put_object(
            Bucket=bucket,
            Key=f"predictions/scored_{day:%Y-%m-%d}.csv",
            Body=scored.to_csv(index=False).encode(),
            ContentType="text/csv",
        )
        print(f"{day}: {len(scored)} trials scored, verdict OK")

    print(f"Backfilled {len(days)} days from {days[0]} to {days[-1]}.")


if __name__ == "__main__":
    main()
