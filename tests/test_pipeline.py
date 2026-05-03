"""
Pruebas basicas del pipeline de ML.

Cada test valida una pieza independiente:
    - test_config_loads          : config.yaml se parsea
    - test_synthetic_sample      : generador de muestra sintetica
    - test_preprocess_*          : limpieza, codificacion, escalado
    - test_split_proportion      : la particion respeta test_size
    - test_metrics_keys          : evaluate.regression_metrics retorna las metricas
    - test_train_model_learns    : el modelo aprende (R2 > 0 en train)
    - test_pipeline_smoke        : end-to-end con muestra sintetica

Para correr offline en CI usamos `data/sample_bank_churn.csv` como dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (
    build_dataset,
    generate_synthetic_sample,
    load_raw_dataframe,
    preprocess,
    split_data,
)
from src.evaluate import regression_metrics
from src.train import train_model
from src.utils import load_config

REPO = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO / "data" / "sample_bank_churn.csv"


# ---------------------------------------------------------------------------
# Fixtures: se ejecutan una sola vez por sesion de tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sample_csv() -> Path:
    """Garantiza que existe la muestra sintetica antes de los tests."""
    if not SAMPLE_CSV.exists():
        generate_synthetic_sample(SAMPLE_CSV, n=500)
    return SAMPLE_CSV


@pytest.fixture(scope="session")
def config(sample_csv) -> dict:
    """Carga config.yaml y fuerza el uso de la muestra sintetica como dataset."""
    cfg = load_config(REPO / "config.yaml")
    # Apuntamos raw_path a la muestra (asi no se descarga nada)
    cfg["data"]["raw_path"] = str(sample_csv)
    return cfg


@pytest.fixture(scope="session")
def raw_df(config) -> pd.DataFrame:
    return load_raw_dataframe(config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_config_loads(config):
    """El YAML debe contener las llaves principales."""
    for key in ("data", "split", "model", "mlflow", "artifacts"):
        assert key in config, f"Falta la seccion '{key}' en config.yaml"
    assert config["data"]["target"] == "Balance"


def test_synthetic_sample_schema(sample_csv):
    """La muestra sintetica debe tener el mismo esquema que Bank Churn."""
    df = pd.read_csv(sample_csv)
    columnas_esperadas = {
        "RowNumber", "CustomerId", "Surname", "CreditScore", "Geography",
        "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
        "IsActiveMember", "EstimatedSalary", "Exited",
    }
    assert columnas_esperadas.issubset(set(df.columns))
    assert len(df) >= 100, "La muestra deberia tener al menos 100 filas"


def test_load_raw_dataframe_no_nulls_on_sample(raw_df):
    """La muestra sintetica no tiene nulos."""
    assert not raw_df.empty
    assert raw_df.isna().sum().sum() == 0


def test_preprocess_drops_id_columns(raw_df, config):
    """Despues del preprocesamiento no deben quedar columnas identificadoras."""
    X, y, features = preprocess(raw_df, config)
    for col in ("RowNumber", "CustomerId", "Surname"):
        assert col not in X.columns
    assert config["data"]["target"] not in X.columns


def test_preprocess_target_is_balance(raw_df, config):
    """y debe ser la serie de Balance y numerica."""
    _, y, _ = preprocess(raw_df, config)
    assert y.name == "Balance"
    assert pd.api.types.is_numeric_dtype(y)


def test_preprocess_one_hot_encoding(raw_df, config):
    """Geography y Gender ya no estan, pero aparecen sus columnas dummy."""
    X, _, features = preprocess(raw_df, config)
    assert "Geography" not in features
    assert "Gender" not in features
    assert any(f.startswith("Geography_") for f in features)
    assert any(f.startswith("Gender_") for f in features)


def test_preprocess_no_nulls_after(raw_df, config):
    """No deben quedar nulos despues del preprocesamiento."""
    X, y, _ = preprocess(raw_df, config)
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0


def test_split_proportion(raw_df, config):
    """El split debe respetar test_size del config."""
    X, y, _ = preprocess(raw_df, config)
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    test_ratio = len(X_test) / (len(X_train) + len(X_test))
    assert abs(test_ratio - config["split"]["test_size"]) < 0.02
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_metrics_keys_present():
    """regression_metrics retorna las 4 metricas esperadas."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    metrics = regression_metrics(y_true, y_pred)
    for key in ("mse", "rmse", "mae", "r2"):
        assert key in metrics
        assert isinstance(metrics[key], float)
    # Para predicciones cercanas R2 debe ser alto y MAE bajo
    assert metrics["r2"] > 0.9
    assert metrics["mae"] < 0.5


def test_train_model_learns(config):
    """El modelo entrenado debe tener R2 train > 0 (no es aleatorio)."""
    ds = build_dataset(config)
    model = train_model(ds["X_train"], ds["y_train"], config["model"]["params"])
    score = model.score(ds["X_train"], ds["y_train"])
    assert score > 0.5, f"El modelo no aprende: R2_train={score:.3f}"


def test_pipeline_smoke(config):
    """Smoke test end-to-end: entrenamiento + prediccion + metricas."""
    ds = build_dataset(config)
    model = train_model(ds["X_train"], ds["y_train"], config["model"]["params"])
    y_pred = model.predict(ds["X_test"])
    assert len(y_pred) == len(ds["y_test"])
    metrics = regression_metrics(ds["y_test"], y_pred)
    # En la muestra sintetica el R2 test deberia ser razonable
    assert metrics["rmse"] >= 0
    assert -1 <= metrics["r2"] <= 1
