# MLOps CI/CD (DVC + Great Expectations + MLflow + FastAPI)

This repo is a runnable **MLOps CI/CD template** that demonstrates:
- **Data generation** (placeholder for real ingestion)
- **Data validation** (Great Expectations + basic checks)
- **Model training** (scikit-learn)
- **Model registry** (optional MLflow Model Registry)
- **Inference API** (FastAPI)
- **Reproducible pipelines** (DVC)

## Quickstart (Windows / macOS / Linux)

### 1) Create a venv & install deps
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2) Initialize DVC and run pipeline
If you are in a Git repo:
```bash
dvc init
dvc repro
```

If you are not using Git yet:
```bash
dvc init --no-scm
dvc repro
```

Outputs:
- `data/raw/train.csv`
- `reports/ge_validation.json`
- `reports/metrics.json`
- `models/model.joblib`

### 3) Run the API
```bash
uvicorn mlops_ci_cd.api.main:app --reload
```

Health:
```bash
curl http://127.0.0.1:8000/health
```

Predict:
```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"feature1\": 1.2, \"feature2\": 3.4}"
```

## MLflow (optional)
By default training logs to a local MLflow store if MLflow is installed.
To log to a server, set:
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT` (optional)

To register the latest run:
```bash
python -m mlops_ci_cd.model.registry --name random_forest_model
```

## Notes
- Great Expectations is configured **without YAML** (EphemeralDataContext) to avoid config-schema issues across versions.
- Replace `mlops_ci_cd/data/generate.py` with your real data ingestion.
