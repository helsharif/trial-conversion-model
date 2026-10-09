import json
import logging
from importlib.metadata import version

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from trial_conversion_model.api.schemas import PredictionRequest, PredictionResponse
from trial_conversion_model.predict import load_model, predict_proba

import time # for timestamps

from pathlib import Path


# Get the both the general console logger and jsonl prediction logger configured in main.py.
logger = logging.getLogger(__name__)
prediction_logger = logging.getLogger("predictions")

router = APIRouter()

# Load the model once, when the service starts, not on every request.
model = load_model()


def to_band(probability: float) -> str:
    """Turn a raw probability into a label a human can act on."""
    if probability < 0.33:
        return "low"
    if probability < 0.66:
        return "medium"
    return "high"


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": version("trial-conversion-model")}


@router.get("/prediction-logs", response_class=HTMLResponse)
def get_prediction_logs():
    """Display all recorded predictions as an HTML table."""

    log_path = Path("logs/predictions.jsonl")

    # Check whether the log file exists and contains records.
    if not log_path.exists() or log_path.stat().st_size == 0:
        return HTMLResponse("<h3>No prediction logs found.</h3>")

    # Read the JSONL file into a pandas DataFrame.
    df = pd.read_json(log_path, lines=True)

    # Convert Unix timestamps to readable dates.
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Expand request features into separate columns.
    request_df = pd.json_normalize(df["request"])

    # Combine columns in the order: timestamp, request features, predictions.
    # This places conv_prob and conv_band in the last two columns.
    df = pd.concat(
        [
            df[["timestamp"]],
            request_df,
            df[["conv_prob", "conv_band"]],
        ],
        axis=1,
    )

    # Display newest predictions first.
    df = df.iloc[::-1]

    # Convert the DataFrame into an HTML table.
    table_html = df.to_html(
        index=False,
        classes="prediction-table",
        border=0,
        float_format=lambda x: f"{x:.4f}",
    )

    # Create a simple HTML page with table styling.
    html = f"""
    <html>
    <head>
        <title>Prediction Logs</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 30px;
            }}
            h2 {{
                margin-bottom: 20px;
            }}
            .prediction-table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 14px;
            }}
            .prediction-table th,
            .prediction-table td {{
                border: 1px solid #ddd;
                padding: 10px;
                text-align: left;
            }}
            .prediction-table th {{
                background-color: #f2f2f2;
            }}
            .prediction-table tr:nth-child(even) {{
                background-color: #fafafa;
            }}
            .prediction-table tr:hover {{
                background-color: #eaf3ff;
            }}
        </style>
    </head>
    <body>
        <h2>Trial Conversion Prediction Logs</h2>
        <p>Total predictions: {len(df)}</p>
        {table_html}
    </body>
    </html>
    """

    return HTMLResponse(content=html)


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict conversion probability for a single live trial."""

    timestamp = time.time() # for logging
    row = pd.DataFrame([request.model_dump()]) # turn request into one-row table (DataFrame) your prediction function expects.
    conv_prob = round(float(predict_proba(model, row).iloc[0]), 4) # Step 2: Score and round to 4 decimals
    conv_band = to_band(conv_prob) # Step 3: Convert probability to band

    # Create a JSONL record containing the request and prediction.
    log_record = {
        "timestamp": timestamp,
        "request": request.model_dump(mode="json"),
        "conv_prob": conv_prob,
        "conv_band": conv_band,
    }

    # Display a human-readable version of the prediction in the console logs.
    logger.info(
        "Prediction | Request: %s -> Probability: %.4f | Band: %s",
        request.model_dump(mode="json"),
        conv_prob,
        conv_band,
    )

    # Write the prediction record to the JSONL file.
    prediction_logger.info(json.dumps(log_record))

    return PredictionResponse(
        conv_prob = conv_prob, 
        conv_band = conv_band,
        )
