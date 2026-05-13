from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Validated request contract for online inference."""

    feature1: float = Field(..., ge=0, le=10, description="Synthetic feature 1")
    feature2: float = Field(..., ge=0, le=10, description="Synthetic feature 2")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float | None = None
    model_source: str
