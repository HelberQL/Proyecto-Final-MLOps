"""
Script principal del pipeline de ML.

Ejecuta:
    python src/train.py

Hace:
    1. Carga config.yaml
    2. Descarga + preprocesa el dataset Bank Customer Churn
    3. Entrena un RandomForestRegressor para predecir Balance
    4. Evalua con MSE, RMSE, MAE y R2
    5. Registra todo en MLflow (parametros, metricas, signature, input_example,
       y el modelo como artefacto reutilizable con mlflow.sklearn.log_model)
    6. Guarda una copia local del modelo en artifacts/model.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor

from src.data import build_dataset
from src.evaluate import regression_metrics
from src.utils import load_config, setup_logging

logger = logging.getLogger("train")


def train_model(X_train, y_train, params: dict) -> RandomForestRegressor:
    """Entrena un RandomForestRegressor con los hiperparametros indicados."""
    logger.info("Entrenando RandomForestRegressor con params=%s", params)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def save_local_artifacts(model, metrics: dict, output_dir: Path, model_filename: str) -> Path:
    """Persiste el modelo y las metricas en disco para uso fuera de MLflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / model_filename
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Modelo guardado en %s", model_path)
    logger.info("Metricas guardadas en %s", metrics_path)
    return model_path


def run_pipeline(config_path: str = "config.yaml") -> dict:
    """Ejecuta el pipeline completo y devuelve un dict con el resumen del run."""
    setup_logging()
    cfg = load_config(config_path)

    # 1. Dataset
    ds = build_dataset(cfg)
    X_train, X_test = ds["X_train"], ds["X_test"]
    y_train, y_test = ds["y_train"], ds["y_test"]
    logger.info("Dataset listo: train=%d, test=%d, features=%d",
                len(X_train), len(X_test), X_train.shape[1])

    # 2. MLflow setup
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    experiment_name = cfg["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI=%s, experimento='%s'", tracking_uri, experiment_name)

    model_params = cfg["model"]["params"]

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run iniciado: %s", run_id)

        # ---- log de parametros ----
        mlflow.log_param("model_name", cfg["model"]["name"])
        mlflow.log_param("test_size", cfg["split"]["test_size"])
        mlflow.log_param("random_state", cfg["split"]["random_state"])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", len(X_train))
        for k, v in model_params.items():
            mlflow.log_param(f"model__{k}", v)

        # ---- entrenamiento ----
        model = train_model(X_train, y_train, model_params)

        # ---- evaluacion (al menos 2 metricas) ----
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_metrics = regression_metrics(y_train, y_pred_train)
        test_metrics = regression_metrics(y_test, y_pred_test)

        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        logger.info("Metricas TRAIN: %s", train_metrics)
        logger.info("Metricas TEST : %s", test_metrics)

        # ---- signature + input_example ----
        # input_example: 5 filas representativas del set de entrenamiento
        input_example = X_train.head(5)
        signature = infer_signature(X_train, y_pred_train)

        # ---- log_model como artefacto reutilizable ----
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=cfg["mlflow"].get("registered_model_name"),
        )

        # ---- artefactos extra: features.json para trazabilidad ----
        features_path = Path("artifacts/features.json")
        features_path.parent.mkdir(parents=True, exist_ok=True)
        features_path.write_text(json.dumps(ds["feature_names"], indent=2))
        mlflow.log_artifact(str(features_path))

        # ---- copia local del modelo (para CI artifact) ----
        artifacts_cfg = cfg["artifacts"]
        save_local_artifacts(
            model,
            {"train": train_metrics, "test": test_metrics},
            Path(artifacts_cfg["output_dir"]),
            artifacts_cfg["model_filename"],
        )

        summary = {
            "run_id": run_id,
            "experiment": experiment_name,
            "tracking_uri": tracking_uri,
            "metrics": {"train": train_metrics, "test": test_metrics},
            "model_params": model_params,
            "n_features": X_train.shape[1],
        }

        # Imprimir resumen amigable al final
        print("\n" + "=" * 60)
        print(" RESUMEN DEL ENTRENAMIENTO ")
        print("=" * 60)
        print(f" run_id      : {run_id}")
        print(f" experimento : {experiment_name}")
        print(f" features    : {X_train.shape[1]}")
        print(f" muestras    : train={len(X_train)} | test={len(X_test)}")
        print(" Metricas TEST:")
        for k, v in test_metrics.items():
            print(f"   - {k:<5}: {v:,.4f}")
        print("=" * 60 + "\n")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Entrena el pipeline ML y registra en MLflow.")
    parser.add_argument("--config", default="config.yaml", help="Ruta al config YAML.")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
