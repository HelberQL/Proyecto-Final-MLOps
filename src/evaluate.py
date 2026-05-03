"""Calculo de metricas de regresion."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Calcula 4 metricas relevantes de regresion.

    El proyecto exige minimo 2 metricas; aqui reportamos:
        - MSE  : error cuadratico medio
        - RMSE : raiz del MSE (mismas unidades que el target)
        - MAE  : error absoluto medio
        - R2   : coeficiente de determinacion
    """
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
