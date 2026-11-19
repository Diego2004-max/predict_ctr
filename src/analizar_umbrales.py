
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
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


def evaluar_umbrales(nombre, y, probabilidades):
    resultados = []

    umbrales = np.arange(
        0.01, 0.51, 0.01
    )

    for umbral in umbrales:
        predicciones = (
            probabilidades >= umbral
        ).astype(int)

        resultados.append({
            "Modelo": nombre,
            "Umbral": round(float(umbral), 2),
            "Precision": precision_score(
                y, predicciones, zero_division=0
            ),
            "Recall": recall_score(
                y, predicciones, zero_division=0
            ),
            "F1": f1_score(
                y, predicciones, zero_division=0
            ),
            "Predicciones positivas": int(
                predicciones.sum()
            ),
        })

    return resultados


def main():
    print("===== CARGANDO VALIDACIÓN =====")

    df = cargar_particion("validation")
    X, y = preparar_variables(df)

    paquete_xgb = joblib.load(
        RUTA_MODELOS / "xgboost_ctr.joblib"
    )

    modelo_xgb = paquete_xgb["modelo"]
    categorias = paquete_xgb["categorias"]

    modelo_cat = joblib.load(
        RUTA_MODELOS / "catboost_ctr.joblib"
    )

    X_xgb = preparar_xgboost(
        X, categorias
    )

    prob_xgb = modelo_xgb.predict_proba(
        X_xgb
    )[:, 1]

    prob_cat = modelo_cat.predict_proba(
        X
    )[:, 1]

    resultados = []

    resultados.extend(
        evaluar_umbrales(
            "XGBoost", y, prob_xgb
        )
    )

    resultados.extend(
        evaluar_umbrales(
            "CatBoost", y, prob_cat
        )
    )

    df_resultados = pd.DataFrame(resultados)

    print("\n===== MEJORES UMBRALES SEGÚN F1 =====")

    for nombre in ["XGBoost", "CatBoost"]:
        datos = df_resultados[
            df_resultados["Modelo"] == nombre
        ]

        mejor = datos.loc[
            datos["F1"].idxmax()
        ]

        print(f"\n{nombre}")
        print(f"Umbral: {mejor['Umbral']:.2f}")
        print(f"Precision: {mejor['Precision']:.4f}")
        print(f"Recall: {mejor['Recall']:.4f}")
        print(f"F1: {mejor['F1']:.4f}")
        print(
            "Predicciones positivas: "
            f"{int(mejor['Predicciones positivas']):,}"
        )

    ruta_csv = (
        RUTA_RESULTADOS /
        "analisis_umbrales.csv"
    )

    df_resultados.to_csv(
        ruta_csv,
        index=False
    )

    print(
        f"\nResultados guardados en: {ruta_csv}"
    )


if __name__ == "__main__":
    main()
