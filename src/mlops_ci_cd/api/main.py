from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from mlops_ci_cd.config import get_settings
from mlops_ci_cd.schemas import PredictionRequest, PredictionResponse

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

logger = logging.getLogger(__name__)
settings = get_settings()
MODEL: Any | None = None
MODEL_SOURCE = "not_loaded"


def load_model() -> tuple[Any, str]:
    """Load a model from MLflow if configured, otherwise from a local artifact."""
    if settings.model_uri and mlflow is not None:
        if settings.mlflow_tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        try:
            logger.info("Loading model from MLflow URI: %s", settings.model_uri)
            return mlflow.pyfunc.load_model(settings.model_uri), settings.model_uri
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load MLflow model. Falling back to local artifact: %s", exc)

    if not settings.model_path.exists():
        raise RuntimeError(f"Model artifact not found: {settings.model_path}")

    logger.info("Loading local model artifact: %s", settings.model_path)
    return joblib.load(settings.model_path), str(settings.model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, MODEL_SOURCE
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    MODEL, MODEL_SOURCE = load_model()
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": MODEL is not None, "model_source": MODEL_SOURCE}


@app.get("/ready")
def ready() -> dict[str, str]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        df = pd.DataFrame([payload.model_dump()])
        prediction = int(MODEL.predict(df)[0])
        probability = None
        if hasattr(MODEL, "predict_proba"):
            probability = float(MODEL.predict_proba(df)[0][1])
        return PredictionResponse(prediction=prediction, probability=probability, model_source=MODEL_SOURCE)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
