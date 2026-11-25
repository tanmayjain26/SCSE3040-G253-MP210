"""Train v1 with MLflow tracking, register the best model, set the @champion alias.

Run from the repository root:   python src/tea/run_training.py
"""
import hashlib
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `tea` importable

from tea.checks import load_params  # noqa: E402
from tea.clean import clean_train, prepare_features, split_xy  # noqa: E402
from tea.train import build_pipeline, evaluate  # noqa: E402


def sha256_of(path):
    """SHA-256 of the ORIGINAL file, exactly as the handbook asks."""
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    params = load_params()
    train_path = params["data"]["train"]
    data_hash = sha256_of(train_path)

    # 1. data: clean train, keep a validation part for choosing the best setting
    train = clean_train(pd.read_csv(train_path))
    X, y = split_xy(train)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"])

    # later.csv is only used to report how v1 does on newer data
    later = pd.read_csv(params["data"]["later"])
    X_later, y_later = prepare_features(later), later["kg_per_worker"]

    # 2. MLflow set-up (SQLite file in the repository root)
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    results = []  # (mae_val, run_id, fitted pipeline)
    for run in params["runs"]:
        model = build_pipeline(run["model"], run["params"])
        with mlflow.start_run(run_name=run["name"]) as active:
            model.fit(X_tr, y_tr)
            mae_val = evaluate(model, X_val, y_val)
            mae_later = evaluate(model, X_later, y_later)

            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_param("model", run["model"])
            mlflow.log_params(run["params"])
            mlflow.log_metric("mae_val", mae_val)
            mlflow.log_metric("mae_later", mae_later)
            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
            results.append((mae_val, active.info.run_id, model))
            print(f"{run['name']:<22} mae_val={mae_val:.3f}  mae_later={mae_later:.3f}  run_id={active.info.run_id}")

    # 3. best run = lowest validation MAE; refit not needed, we keep the fitted pipeline
    best_mae, best_run_id, best_model = min(results, key=lambda r: r[0])
    name = params["mlflow"]["registered_model_name"]
    version = mlflow.register_model(f"runs:/{best_run_id}/model", name).version
    MlflowClient().set_registered_model_alias(name, "champion", version)

    # 4. save the champion for the API
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/model.joblib")

    print("\n--- copy these into reports/findings.yaml (tracking:) ---")
    print("experiment_name:      ", params["mlflow"]["experiment_name"])
    print("best_run_id:          ", best_run_id)
    print("registered_model_name:", name)
    print("champion_version:     ", version)
    print(f"best mae_val: {best_mae:.3f}")


if __name__ == "__main__":
    main()
