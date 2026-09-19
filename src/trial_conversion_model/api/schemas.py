from pydantic import BaseModel, Field
from typing import Literal


class PredictionRequest(BaseModel):
    """One trial's first-3-day base aggregates, as the caller knows them."""

    # TODO: declare the seven fields a caller must send, one line each,
    # with a type for each. They are the same seven base aggregates your
    # feature code starts from; features.py names them.
    # Base aggregates are:  
    # - sessions_day1, sessions_day2, sessions_day3
    # - listen_sessions_3d, total_minutes_3d
    # - country, device_type
    sessions_day1: int = Field(ge=0, description="Total sessions in the first day")
    sessions_day2: int = Field(ge=0, description="Total sessions in the second day")
    sessions_day3: int = Field(ge=0, description="Total sessions in the third day")
    listen_sessions_3d: int = Field(ge=0, description="Total listening sessions in the first 3 days")
    total_minutes_3d: float = Field(ge=0.0, description="Total minutes of sessions in the first 3 days")
    country: str = Field(description="Country of the user")
    device_type: str = Field(description="Type of device used by the user")


class PredictionResponse(BaseModel):
    """What we send back: a probability and a band a human can act on."""

    # TODO: declare the two fields the caller gets back: the conversion
    # probability, and the low/medium/high band it falls in.
    conv_prob: float = Field(ge=0.0, le=1.0, description="Predicted probability of conversion")
    conv_band: Literal["low", "medium", "high"] = Field(description="Band of conversion probability: low, medium, or high")
