from pathlib import Path

import pandas as pd

from trial_conversion_model.data import PROCESSED_DATA, RAW_DATA, load_raw

CATEGORICAL = ["country", "device_type"]
TARGET = "converted"
FEATURES = [
    "sessions_3d",
    "active_days_3d",
    "day1_share",
    "listen_share",
    "avg_session_minutes",
    "total_minutes_3d",
    "country",
    "device_type",
]
DAY_COLUMNS = ["sessions_day1", "sessions_day2", "sessions_day3"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the model features from the snapshot's base aggregates.

    Trials with no sessions in the first 3 days produce divide-by-zero shares;
    zero engagement is real information, so those become 0 rather than NaN.
    Base aggregates are:
    - sessions_day1, sessions_day2, sessions_day3
    - listen_sessions_3d, total_minutes_3d
    - country, device_type
    """
    df = df.copy()
    df["sessions_3d"] = df[DAY_COLUMNS].sum(axis=1)
    df["active_days_3d"] = (df[DAY_COLUMNS] > 0).sum(axis=1)
    df["day1_share"] = (df["sessions_day1"] / df["sessions_3d"]).fillna(0)
    df["listen_share"] = (df["listen_sessions_3d"] / df["sessions_3d"]).fillna(0)
    df["avg_session_minutes"] = (df["total_minutes_3d"] / df["sessions_3d"]).fillna(0)
    return df


def build_training_data(
    raw_path: Path = RAW_DATA, out_path: Path = PROCESSED_DATA
) -> pd.DataFrame:
    """Turn the raw extract into the model-ready training table and persist it."""
    df = add_features(load_raw(raw_path))
    table = pd.get_dummies(df[FEATURES], columns=CATEGORICAL)
    table[TARGET] = df[TARGET]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    return table
