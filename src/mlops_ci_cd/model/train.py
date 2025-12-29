from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


def train_model(data_path: str, model_out: str, metrics_out: str, seed: int = 42) -> None:
    """Train a RandomForestClassifier on the provided dataset and save the model and metrics.
    
    Parameters:
        data_path: Path to the training CSV file.
        model_out: Path to save the trained model artifact.
        metrics_out: Path to save the training metrics JSON file.
        seed: Random seed for reproducibility.
    """
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_out).write_text(json.dumps({"accuracy": acc}, indent=2), encoding="utf-8")

    # MLflow logging (works with local file store too)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if mlflow is not None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "mlops-ci-cd"))
        with mlflow.start_run():
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, artifact_path="model")
            print(f"📦 MLflow run logged (accuracy={acc:.4f}).")
    else:
        print("⚠️ mlflow not installed; skipped MLflow logging.")

    print(f"✅ Model trained (accuracy={acc:.4f}) and saved to {model_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/train.csv")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--metrics-out", default="reports/metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(args.data, args.model_out, args.metrics_out, seed=args.seed)


if __name__ == "__main__":
    main()
