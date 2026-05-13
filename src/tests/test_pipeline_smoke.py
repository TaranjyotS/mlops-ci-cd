import json

import joblib
import pytest
from fastapi.testclient import TestClient

from mlops_ci_cd.data.generate import generate_dataset
from mlops_ci_cd.data.validate import validate_csv
from mlops_ci_cd.model.train import train_model
from mlops_ci_cd.monitoring.drift import compute_drift


def test_end_to_end_tmp(tmp_path):
    data_path = tmp_path / "train.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    generate_dataset(str(data_path), n_rows=300, seed=1)
    assert validate_csv(str(data_path), tmp_path / "validation.json") is True
    metrics = train_model(str(data_path), str(model_path), str(metrics_path), seed=1)

    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["accuracy"] >= 0.70
    assert set(json.loads(metrics_path.read_text())).issuperset({"accuracy", "precision", "recall", "f1", "roc_auc"})


def test_validation_fails_for_missing_columns(tmp_path):
    bad_data = tmp_path / "bad.csv"
    bad_data.write_text("feature1,target\n1,0\n2,1\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        validate_csv(str(bad_data), tmp_path / "validation.json")


def test_drift_report(tmp_path):
    reference = generate_dataset(str(tmp_path / "reference.csv"), n_rows=200, seed=1)
    current = generate_dataset(str(tmp_path / "current.csv"), n_rows=200, seed=2)

    report = compute_drift(str(reference), str(current), str(tmp_path / "drift.json"), threshold=10)

    assert report["drift_detected"] is False
    assert "feature1" in report["features"]


def test_api_predict(tmp_path, monkeypatch):
    data_path = generate_dataset(str(tmp_path / "train.csv"), n_rows=300, seed=3)
    model_path = tmp_path / "model.joblib"
    train_model(str(data_path), str(model_path), str(tmp_path / "metrics.json"), seed=3)

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    from mlops_ci_cd.api import main

    main.MODEL = joblib.load(model_path)
    main.MODEL_SOURCE = str(model_path)
    client = TestClient(main.app)
    response = client.post("/predict", json={"feature1": 1.2, "feature2": 3.4})

    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]
