from __future__ import annotations

import argparse
import os

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


def register_latest_run(model_name: str) -> None:
    """Register the most recent run's model artifact in MLflow Model Registry.

    This requires an MLflow tracking server with registry support.
    In local mode (no server), this will print instructions and exit cleanly.
    """
    if mlflow is None:
        print("⚠️ mlflow not installed; skipping registry step.")
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        print("ℹ️ MLFLOW_TRACKING_URI not set; skipping registry step.")
        return

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    exp_name = os.getenv("MLFLOW_EXPERIMENT", "mlops-ci-cd")
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        print(f"ℹ️ No MLflow experiment '{exp_name}' found; nothing to register.")
        return

    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        print("ℹ️ No runs found; nothing to register.")
        return

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from {model_uri} as '{model_name}'...")

    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"✅ Registered: name={result.name}, version={result.version}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="random_forest_model", help="Model registry name")
    args = parser.parse_args()
    register_latest_run(args.name)


if __name__ == "__main__":
    main()
