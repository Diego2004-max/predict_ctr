
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

from preparar_datos import (
    cargar_particion,
    preparar_variables,
    COLUMNAS_CATEGORICAS,
)

RUTA_MODELOS = Path("models")
RUTA_RESULTADOS = Path("results")

RUTA_RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)


def preparar_xgboost(X, categorias):
    X = X.copy()

    for columna in COLUMNAS_CATEGORICAS:
        categorias_entrenamiento = categorias[columna]

        valores = X[columna].where(
            X[columna].isin(categorias_entrenamiento),
            other=pd.NA
        )

        X[columna] = pd.Categorical(
            valores,
            categories=categorias_entrenamiento
        )

    return X


def calcular_metricas(nombre, y, probabilidades):
    return {
        "Modelo": nombre,
        "ROC-AUC": roc_auc_score(
            y, probabilidades
        ),
        "PR-AUC": average_precision_score(
            y, probabilidades
        ),
        "Log Loss": log_loss(
            y, probabilidades
        ),
        "Brier Score": brier_score_loss(
            y, probabilidades
        ),
        "CTR observado (%)": y.mean() * 100,
        "CTR predicho (%)": np.mean(probabilidades) * 100,
    }


def main():
    print("===== CARGANDO VALIDACIÓN =====")

    df_val = cargar_particion("validation")
    X_val, y_val = preparar_variables(df_val)

    print(f"Registros: {len(X_val):,}")

    print("\n===== CARGANDO MODELOS =====")

    paquete_xgb = joblib.load(
        RUTA_MODELOS / "xgboost_ctr.joblib"
    )

    modelo_xgb = paquete_xgb["modelo"]
    categorias = paquete_xgb["categorias"]

    modelo_cat = joblib.load(
        RUTA_MODELOS / "catboost_ctr.joblib"
    )

    print("\n===== GENERANDO PREDICCIONES =====")

    X_val_xgb = preparar_xgboost(
        X_val, categorias
    )

    prob_xgb = modelo_xgb.predict_proba(
        X_val_xgb
    )[:, 1]

    prob_cat = modelo_cat.predict_proba(
        X_val
    )[:, 1]

    resultados = [
        calcular_metricas(
            "XGBoost", y_val, prob_xgb
        ),
        calcular_metricas(
            "CatBoost", y_val, prob_cat
        ),
    ]

    df_resultados = pd.DataFrame(resultados)

    print("\n===== COMPARACIÓN FINAL EN VALIDACIÓN =====")
    print(
        df_resultados.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    ruta_csv = (
        RUTA_RESULTADOS /
        "comparacion_validacion.csv"
    )

    df_resultados.to_csv(
        ruta_csv,
        index=False
    )

    print(f"\nResultados guardados en: {ruta_csv}")


if __name__ == "__main__":
    main()
