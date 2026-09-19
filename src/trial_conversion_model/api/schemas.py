from pydantic import BaseModel, Field
from typing import Literal


class PredictionRequest(BaseModel):
    """One trial's first-3-day base aggregates, as the caller knows them."""

    # TODO: declare the seven fields a caller must send, one line each,
    # with a type for each. They are the same seven base aggregates your
    # feature code starts from; features.py names them.
    sessions_3d: int = Field(ge=0, description="Total sessions in the first 3 days")
    active_days_3d: int = Field(ge=0, description="Number of days with at least one session in the first 3 days")
    day1_share: float = Field(ge=0.0, le=1.0, description="Share of sessions in the first day")
    listen_share: float = Field(ge=0.0, le=1.0, description="Share of sessions that were listening sessions")
    avg_session_minutes: float = Field(ge=0.0, description="Average session length in minutes")
    total_minutes_3d: float = Field(ge=0.0, description="Total minutes of sessions in the first 3 days")
    country: str = Field(description="Country of the user")
    device_type: str = Field(description="Type of device used by the user")


class PredictionResponse(BaseModel):
    """What we send back: a probability and a band a human can act on."""

    # TODO: declare the two fields the caller gets back: the conversion
    # probability, and the low/medium/high band it falls in.
    conv_prob: float = Field(ge=0.0, le=1.0, description="Predicted probability of conversion")
    conv_band: Literal["low", "medium", "high"] = Field(description="Band of conversion probability: low, medium, or high")
