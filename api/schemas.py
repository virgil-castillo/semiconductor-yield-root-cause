"""Pydantic v2 schemas for the yield-risk prediction API."""
from __future__ import annotations

from pydantic import BaseModel


class WaferRequest(BaseModel):
    """Request payload for a single wafer prediction."""

    wafer_id: str | None = None
    features: dict[str, float]


class BatchRequest(BaseModel):
    """Request payload for a batch of wafer predictions."""

    wafers: list[WaferRequest]


class PredictionResponse(BaseModel):
    """Response payload for a single wafer prediction."""

    wafer_id: str | None
    failure_probability: float
    risk_flag: bool
    threshold_used: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Response payload for a batch of wafer predictions."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Response payload for the /health liveness check."""

    status: str
    model_loaded: bool
    model_version: str | None
    threshold: float | None


class ErrorResponse(BaseModel):
    """Response payload for error conditions."""

    detail: str
