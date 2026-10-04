import pandas as pd

from trial_conversion_model.features import add_features

TRIAL_NO_SESSIONS = {
    "sessions_day1": 0,
    "sessions_day2": 0,
    "sessions_day3": 0,
    "listen_sessions_3d": 0,
    "total_minutes_3d": 0.0,
    "country": "US",
    "device_type": "iOS",
}

def test_zero_session_trial_gets_zero_shares_not_nan():
    #   1. Build a one-row DataFrame for a trial with nothing in it: zero
    #      sessions on each of the three days, zero listen sessions, and
    #      zero total minutes.
    #   2. Pass it through add_features.
    #   3. Assert that day1_share, listen_share and avg_session_minutes each
    #      come back as 0 rather than a missing value.
    aggregates = pd.DataFrame([TRIAL_NO_SESSIONS])
    df = add_features(aggregates)

    # Verify zero-session features are 0 instead of NaN.
    assert df.loc[0, "day1_share"] == 0, f"Expected day1_share to be 0, but got {df.loc[0, 'day1_share']}"
    assert df.loc[0, "listen_share"] == 0, f"Expected listen_share to be 0, but got {df.loc[0, 'listen_share']}"
    assert df.loc[0, "avg_session_minutes"] == 0, f"Expected avg_session_minutes to be 0, but got {df.loc[0, 'avg_session_minutes']}"
