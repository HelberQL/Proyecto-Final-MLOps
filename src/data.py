"""
Modulo de carga y preprocesamiento del dataset Bank Customer Churn.

Aunque el dataset se conoce por su tarea original de clasificacion (Exited),
en este proyecto lo usamos como un problema de REGRESION:
predecir el `Balance` (saldo de cuenta) a partir del resto de atributos.

Funciones publicas:
    load_raw_dataframe(config)      -> pd.DataFrame
    preprocess(df, config)          -> X, y, feature_names
    split_data(X, y, config)        -> X_train, X_test, y_train, y_test
    build_dataset(config)           -> dict con todo lo anterior listo para entrenar
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Mirrors publicos conocidos del dataset Bank Customer Churn.
# Se intentan en orden hasta que uno responda 200.
_DEFAULT_MIRRORS: List[str] = [
    "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Bank%20Churn%20Modelling.csv",
    "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Bank%20Churn%20Modelling.csv",
    "https://raw.githubusercontent.com/sharmaroshan/Churn-Modelling-Dataset/master/Churn_Modelling.csv",
]


# ---------------------------------------------------------------------------
# Descarga / carga
# ---------------------------------------------------------------------------
def _download_csv(url: str, dest: Path, timeout: int = 30) -> bool:
    """Descarga un CSV publico y lo guarda en `dest`. Devuelve True si tuvo exito."""
    import requests

    try:
        logger.info("Intentando descargar dataset desde %s", url)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        logger.info("Dataset guardado en %s (%d bytes)", dest, dest.stat().st_size)
        return True
    except Exception as exc:  # noqa: BLE001 - queremos cualquier fallo de red
        logger.warning("Fallo al descargar %s: %s", url, exc)
        return False


def _ensure_dataset_local(config: Dict) -> Path:
    """Garantiza un CSV local. Orden: cache -> mirrors -> sample sintetico."""
    raw_path = Path(config["data"]["raw_path"])

    if raw_path.exists() and raw_path.stat().st_size > 1000:
        logger.info("Usando dataset cacheado en %s", raw_path)
        return raw_path

    # Intentar mirrors
    url_primario = config["data"].get("url")
    candidatos = [u for u in [url_primario, *_DEFAULT_MIRRORS] if u]
    # Eliminar duplicados conservando orden
    vistos = set()
    candidatos = [u for u in candidatos if not (u in vistos or vistos.add(u))]

    for url in candidatos:
        if _download_csv(url, raw_path):
            return raw_path

    # Fallback: dataset sintetico con mismo esquema (solo para entornos sin red)
    sample = Path(__file__).resolve().parent.parent / "data" / "sample_bank_churn.csv"
    if sample.exists():
        logger.warning(
            "No se pudo descargar el dataset. Usando muestra sintetica de %s", sample
        )
        return sample

    raise RuntimeError(
        "No se pudo obtener el dataset desde los mirrors y no hay muestra local. "
        "Verifica tu conexion o coloca el CSV manualmente en "
        f"{raw_path}."
    )


def load_raw_dataframe(config: Dict) -> pd.DataFrame:
    """Carga el DataFrame crudo desde disco/URL."""
    csv_path = _ensure_dataset_local(config)
    df = pd.read_csv(csv_path)
    logger.info("Dataset cargado: %d filas, %d columnas", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
def preprocess(
    df: pd.DataFrame, config: Dict
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Limpia, codifica y escala el DataFrame. Retorna X, y y nombres de columnas."""
    df = df.copy()

    # 1. Eliminar columnas identificadoras
    drop_cols = [c for c in config["data"].get("drop_columns", []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info("Columnas eliminadas: %s", drop_cols)

    target = config["data"]["target"]
    if target not in df.columns:
        raise KeyError(f"Columna objetivo '{target}' no esta en el dataset.")

    # 2. Manejo basico de nulos
    n_nulos = int(df.isna().sum().sum())
    if n_nulos:
        logger.info("Imputando %d valores nulos (numericos=mediana, categoricos=moda)", n_nulos)
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])

    # 3. Separar X / y
    y = df[target].astype(float)
    X = df.drop(columns=[target])

    # 4. One-Hot encoding de columnas categoricas
    cat_cols = [c for c in config["data"].get("categorical_columns", []) if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        logger.info("One-Hot aplicado a %s -> %d columnas finales", cat_cols, X.shape[1])

    # 5. Escalado de columnas numericas (excluyendo binarias creadas por get_dummies)
    bin_cols = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
    num_cols = [c for c in X.columns if c not in bin_cols]
    if num_cols:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        logger.info("StandardScaler aplicado a %d columnas numericas", len(num_cols))

    feature_names = X.columns.tolist()
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_data(X: pd.DataFrame, y: pd.Series, config: Dict):
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Split realizado: train=%d, test=%d (test_size=%.2f)",
        len(X_train),
        len(X_test),
        test_size,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# API de alto nivel
# ---------------------------------------------------------------------------
def build_dataset(config: Dict) -> Dict:
    """Pipeline completo: load -> preprocess -> split. Devuelve un dict listo para entrenar."""
    df = load_raw_dataframe(config)
    X, y, feature_names = preprocess(df, config)
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "n_samples": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Generador de muestra sintetica (utilizado como fallback de tests offline)
# ---------------------------------------------------------------------------
def generate_synthetic_sample(out_path: str | os.PathLike, n: int = 1000, seed: int = 42) -> Path:
    """Genera un CSV con el mismo esquema que Bank Churn Modelling para tests offline."""
    rng = np.random.default_rng(seed)
    n = int(n)
    df = pd.DataFrame(
        {
            "RowNumber": np.arange(1, n + 1),
            "CustomerId": rng.integers(15_000_000, 16_000_000, size=n),
            "Surname": ["Cliente" + str(i) for i in range(n)],
            "CreditScore": rng.integers(350, 850, size=n),
            "Geography": rng.choice(["France", "Spain", "Germany"], size=n, p=[0.5, 0.25, 0.25]),
            "Gender": rng.choice(["Male", "Female"], size=n),
            "Age": rng.integers(18, 92, size=n),
            "Tenure": rng.integers(0, 11, size=n),
            "NumOfProducts": rng.integers(1, 5, size=n),
            "HasCrCard": rng.integers(0, 2, size=n),
            "IsActiveMember": rng.integers(0, 2, size=n),
            "EstimatedSalary": rng.uniform(10_000, 200_000, size=n).round(2),
            "Exited": rng.integers(0, 2, size=n),
        }
    )
    # Balance correlacionado con edad, salario y producto -> regresion no trivial
    base = (
        0.6 * df["EstimatedSalary"]
        + 1500 * df["Age"]
        - 8000 * df["NumOfProducts"]
        + 50 * df["CreditScore"]
        + rng.normal(0, 15_000, size=n)
    )
    df["Balance"] = np.clip(base, 0, 250_000).round(2)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    # Ejecucion directa: genera la muestra sintetica para tests
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    repo = Path(__file__).resolve().parent.parent
    out = generate_synthetic_sample(repo / "data" / "sample_bank_churn.csv", n=1000)
    print(f"Muestra sintetica generada: {out}")
