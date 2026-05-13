from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import mlflow
    from mlflow.models.signature import infer_signature
except ImportError:  # pragma: no cover
    mlflow = None
    infer_signature = None

FEATURE_COLUMNS = ["feature1", "feature2"]
TARGET_COLUMN = "target"


def train_model(data_path: str, model_out: str, metrics_out: str, seed: int = 42) -> dict[str, float]:
    """Train a RandomForestClassifier and persist model, schema, and metrics."""
    df = pd.read_csv(data_path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    probas = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probas)),
    }

    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metrics_path = Path(metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    schema_path = metrics_path.parent / "model_schema.json"
    schema_path.write_text(
        json.dumps({"features": FEATURE_COLUMNS, "target": TARGET_COLUMN, "feature_ranges": {"feature1": [0, 10], "feature2": [0, 10]}}, indent=2),
        encoding="utf-8",
    )

    if mlflow is not None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "mlops-ci-cd"))
        with mlflow.start_run():
            mlflow.log_params({"model_type": "RandomForestClassifier", "n_estimators": 200, "max_depth": 8, "seed": seed})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(schema_path), artifact_path="schema")

            # Log an MLflow model signature and input example so downstream
            # registry/deployment systems can validate inference inputs.
            input_example = x_test.iloc[:2]
            signature = infer_signature(x_test, preds)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

    print(f"Model trained and saved to {model_path}. Metrics: {metrics}")
    return metrics


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
