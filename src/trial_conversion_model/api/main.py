import logging

from fastapi import FastAPI

from trial_conversion_model.api.routes import router

import os # for creating log directory

# create a logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure general application logging to the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Create a dedicated logger for JSONL prediction records.
prediction_logger = logging.getLogger("predictions")
prediction_logger.setLevel(logging.INFO)
prediction_logger.propagate = False

# Append JSON records to the JSONL file.
file_handler = logging.FileHandler(
    os.path.join(log_dir, "predictions.jsonl"),
    mode="a",
    encoding="utf-8",
)
file_handler.setFormatter(logging.Formatter("%(message)s"))
prediction_logger.addHandler(file_handler)

# Create the FastAPI application.
app = FastAPI(title="Beam trial conversion model")

# Register the API endpoints.
app.include_router(router)
