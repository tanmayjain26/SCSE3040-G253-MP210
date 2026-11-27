# SCSE3040 MLOps Mini Project: Group G253

**Problem:** MP210, Tea Leaf Plucking (Kochi Tea Growers' Estate)
**Task:** Regression, predict `kg_per_worker`, evaluated with MAE
**Faculty:** Dr. Upasna Joshi | **Batch:** 78

## Members
| Enrolment No. | Name |
|---|---|
| S24CSEU0391 | Samya Ali |
| S24CSEU2033 | Prashant Kapil |
| S24CSEU0589 | Tanish Vashistha |
| S24CSEU0539 | Shyam Nayak Narayan |
| S24CSEU0580 | Tanmay Jain |

## Setup
    python -m venv .venv
    pip install -r requirements.txt
    dvc pull   # or copy the original CSVs into data/raw/

## Pipeline overview

| Part | Where |
|---|---|
| Data versioning (DVC) | `data/raw/*.dvc` |
| Seven data checks | `src/tea/checks.py` |
| Cleaning | `src/tea/clean.py` |
| Model code | `src/tea/train.py` |
| MLflow training and registry | `src/tea/run_training.py`, `mlflow.db` |
| API service | `app/main.py` |
| Tests | `tests/` |
| Docker | `Dockerfile` |
| CI | `.github/workflows/ci.yml` |
| Kubernetes | `k8s/deployment.yaml`, `k8s/service.yaml` |

## How to run

    pip install -r requirements.txt
    python -m pytest                              # run the tests
    python src/tea/checks.py                      # run the seven data checks
    python src/tea/run_training.py                # train, log to MLflow, register @champion
    uvicorn app.main:app --port 8000              # start the API

## API

`POST /predict` with the fields `worker_exp_yr`, `rain_mm_week`, `slope_deg`, `temp_c`,
`bush_age_yr`, `hours_worked`, `flush`, `section`. Rules from the client: `worker_exp_yr`
above 55 and `bush_age_yr` below 1.5 are rejected with status 422.
`GET /health` returns `{"status": "ok"}`.

## Docker and Kubernetes

    docker build -t mp210-api:1.0 .
    docker run -p 8000:8000 mp210-api:1.0
    kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml    # 3 replicas, service port 80 -> 8000
