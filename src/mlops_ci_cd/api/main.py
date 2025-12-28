from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


app = FastAPI(title="MLOps CI/CD Inference API", version="1.0.0")


def _load_model():
    """Load model from MLflow (if configured) or local artifact."""
    mlflow_uri = os.getenv("MODEL_URI", "").strip()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()

    if mlflow_uri and mlflow is not None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        try:
            return mlflow.pyfunc.load_model(mlflow_uri)
        except Exception as e:
            # fall back to local if MLflow loading fails
            print(f"⚠️ Failed to load MLflow model '{mlflow_uri}': {e}. Falling back to local artifact.")

    local_path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
    if not local_path.exists():
        raise RuntimeError(f"Local model artifact not found: {local_path}")

    return joblib.load(local_path)


MODEL = _load_model()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/predict")
def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = pd.DataFrame([features])
        pred = MODEL.predict(df)
        return {"prediction": [int(pred[0])] if hasattr(pred, "__len__") else int(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
