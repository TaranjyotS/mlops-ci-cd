from mlops_ci_cd.data.generate import generate_dataset
from mlops_ci_cd.data.validate import validate_csv
from mlops_ci_cd.model.train import train_model


def test_end_to_end_tmp(tmp_path):
    data_path = tmp_path / "train.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    generate_dataset(str(data_path), n_rows=200, seed=1)
    validate_csv(str(data_path))
    train_model(str(data_path), str(model_path), str(metrics_path), seed=1)

    assert model_path.exists(), f"Expected model file not found: {model_path}"
    assert metrics_path.exists(), f"Expected metrics file not found: {metrics_path}"
