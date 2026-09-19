from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from trial_conversion_model.features import CATEGORICAL, FEATURES, add_features

MODEL_PATH = Path("models/model.json")


def load_model(path: Path = MODEL_PATH) -> XGBClassifier:
    """Load the trained model from its native XGBoost artifact."""
    model = XGBClassifier()
    model.load_model(path)
    return model


def predict_proba(model: XGBClassifier, aggregates: pd.DataFrame) -> pd.Series:
    """Score trials from their base aggregates; one probability per row.

    The rows go through the same add_features as training. A small batch
    rarely carries every country and device, so the dummy columns are
    reindexed against the model's training columns; the categories the
    batch does not have become explicit zeros.

    Base aggregates are:
    - sessions_day1, sessions_day2, sessions_day3
    - listen_sessions_3d, total_minutes_3d
    - country, device_type

    add_features() and reindexing will result in feature columns:
    sessions_3d, active_days_3d, day1_share, 
    listen_share, avg_session_minutes, total_minutes_3d,
    country_EU,	country_India, country_Rest, country_US,
    device_type_Android, device_type_Web, device_type_iOS
    """
    df = add_features(aggregates)
    rows = pd.get_dummies(df[FEATURES], columns=CATEGORICAL)
    rows = rows.reindex(columns=model.get_booster().feature_names, fill_value=0)
    return pd.Series(model.predict_proba(rows)[:, 1], index=aggregates.index)
