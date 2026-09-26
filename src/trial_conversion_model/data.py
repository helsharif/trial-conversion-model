import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

import datetime

RAW_DATA = Path("data/01_raw/trial_snapshot.csv")
PROCESSED_DATA = Path("data/03_processed/training_data.csv")

QUERY = "SELECT * FROM ml.trial_snapshot_latest"


def fetch(out_path: Path = RAW_DATA) -> None:
    """Materialize the training extract into data/01_raw.

    Training always runs from this file, never from the live table, so the
    training data cannot shift between runs. All connection details come
    from the environment; code never knows which database it is pointed at.
    """
    load_dotenv()
    engine = create_engine(
        "postgresql://{user}:{password}@{host}:{port}/{name}".format(
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            host=os.environ["DB_HOST"],
            port=os.environ["DB_PORT"],
            name=os.environ["DB_NAME"],
        )
    )
    df = pd.read_sql(QUERY, engine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")


def load_raw(path: Path = RAW_DATA) -> pd.DataFrame:
    """Load the trial snapshot extract pulled from ml.trial_snapshot_latest."""
    return pd.read_csv(path)


def load_processed(path: Path = PROCESSED_DATA) -> pd.DataFrame:
    """Load the model-ready training table produced by features.build_training_data."""
    return pd.read_csv(path)

def data_version(path: Path = PROCESSED_DATA) -> str:
    """Return the file creation timestamp or the last modified timestamp (whichever is later) of the processed training table.
    timestamp formatted as a string in the format YYYY-MM-DD-HH_MM_SS. This can be used to track the version of the training data used for model training."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    stat = path.stat()

    # st_birthtime is only available on macOS/BSD/Windows.
    # Fall back to st_ctime on Linux (metadata change time, closest equivalent).
    birth_time = getattr(stat, "st_birthtime", stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_ctime)

    # 1. Get the latest timestamp float
    latest_epoch = max(birth_time, stat.st_mtime)

    # 2. Convert the float timestamp to a datetime object
    dt = datetime.datetime.fromtimestamp(latest_epoch)

    # 3. Format into the requested YYYY-MM-DD-HH_MM_SS string
    return dt.strftime("%Y-%m-%d-%H_%M_%S")
