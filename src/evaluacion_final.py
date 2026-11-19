
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from preparar_datos import (
    cargar_particion,
    preparar_variables,
    COLUMNAS_CATEGORICAS,
)

RUTA_MODELOS = Path("models")
RUTA_RESULTADOS = Path("results")
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)


def preparar_xgboost(X, categorias):
    X = X.copy()

    for columna in COLUMNAS_CATEGORICAS:
        conocidas = categorias[columna]

        valores = X[columna].where(
            X[columna].isin(conocidas),
            other=pd.NA
        )

        X[columna] = pd.Categorical(
            valores,
            categories=conocidas
        )

    return X


def evaluar_modelo(nombre, y, probabilidades):
    predicciones = (
        probabilidades >= 0.5
    ).astype(int)

    matriz = confusion_matrix(
        y,
        predicciones,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matriz.ravel()

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
        "Precision": precision_score(
            y, predicciones, zero_division=0
        ),
        "Recall": recall_score(
            y, predicciones, zero_division=0
        ),
        "F1": f1_score(
            y, predicciones, zero_division=0
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "CTR observado (%)": y.mean() * 100,
        "CTR predicho (%)": np.mean(
            probabilidades
        ) * 100,
    }


def main():
    print("===== CARGANDO CONJUNTO DE PRUEBA =====")

    df_test = cargar_particion("test")
    X_test, y_test = preparar_variables(df_test)

    print(f"Registros: {len(X_test):,}")
    print(f"CTR observado: {y_test.mean() * 100:.2f}%")

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

    X_test_xgb = preparar_xgboost(
        X_test, categorias
    )

    prob_xgb = modelo_xgb.predict_proba(
        X_test_xgb
    )[:, 1]

    prob_cat = modelo_cat.predict_proba(
        X_test
    )[:, 1]

    resultados = [
        evaluar_modelo(
            "XGBoost", y_test, prob_xgb
        ),
        evaluar_modelo(
            "CatBoost", y_test, prob_cat
        ),
    ]

    df_resultados = pd.DataFrame(resultados)

    print("\n===== RESULTADOS FINALES =====")
    print(
        df_resultados.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    ruta_csv = (
        RUTA_RESULTADOS /
        "evaluacion_final.csv"
    )

    df_resultados.to_csv(
        ruta_csv,
        index=False
    )

    print(f"\nResultados guardados en: {ruta_csv}")

    # Guardar probabilidades para análisis posterior
    df_predicciones = pd.DataFrame({
        "label": y_test.to_numpy(),
        "prob_xgboost": prob_xgb,
        "prob_catboost": prob_cat,
    })

    ruta_predicciones = (
        RUTA_RESULTADOS /
        "predicciones_test.csv"
    )

    df_predicciones.to_csv(
        ruta_predicciones,
        index=False
    )

    print(
        f"Predicciones guardadas en: "
        f"{ruta_predicciones}"
    )


if __name__ == "__main__":
    main()
