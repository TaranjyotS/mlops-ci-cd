# MLOps CI/CD (DVC + Great Expectations + MLflow + FastAPI)

This project demonstrates a production-ready Machine Learning Operations (MLOps) CI/CD pipeline. It automates the complete ML lifecycle, from data generation and validation to model training, versioning, and deployment-ready artifacts using modern MLOps tooling.

This repo is a runnable **MLOps CI/CD template** that demonstrates:
- **Data generation** (placeholder for real ingestion)
- **Data validation** (Great Expectations + basic checks)
- **Model training** (scikit-learn)
- **Model registry** (MLflow Model Registry)
- **Inference API** (FastAPI)
- **Reproducible pipelines** (DVC)

## Quickstart

### 1) Create a venv & install deps
```bash
python -m venv .venv
./.venv/Scripts/activate
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
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"feature1\": 1.2, \"feature2\": 3.4}"
```

## MLflow
By default training logs to a local MLflow store if MLflow is installed.
To log to a server, set:
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT` (optional)

To register the latest run:
```bash
python -m mlops_ci_cd.model.registry --name random_forest_model
```

## MLflow Experiment Tracking

This project uses **MLflow** to track experiments, metrics, and trained models during the `train` stage of the DVC pipeline.

By default, MLflow logs experiments locally. You can optionally connect the pipeline to a remote or containerized MLflow tracking server.

---

### Option 1: Local MLflow Tracking (Recommended for Development)

This option logs experiments to a local `mlruns/` directory.

#### Set tracking URI
```bash
export MLFLOW_TRACKING_URI=file:./mlruns
```

```powershell
$env:MLFLOW_TRACKING_URI="file:./mlruns"
```
#### Run training
```bash
dvc repro train
```
#### Launch MLflow UI
```bash
mlflow ui
```
#### Open in browser
 - http://127.0.0.1:5000

### Option 2: Docker-based MLflow Tracking (Production-style)

#### Start MLflow server using Docker
```bash
docker run -d \
  -p 5000:5000 \
  --name mlflow-server \
  ghcr.io/mlflow/mlflow
```

#### Set tracking URI
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
```

#### Run training
```bash
dvc repro train
```

## Notes
- Great Expectations is configured **without YAML** (EphemeralDataContext) to avoid config-schema issues across versions.
- Replace `mlops_ci_cd/data/generate.py` with your real data ingestion.
- If `MLFLOW_TRACKING_URI` is not set, the pipeline automatically skips model registration and logs locally.
- In CI/CD environments, `MLFLOW_TRACKING_URI` should be set via environment variables or secrets.
