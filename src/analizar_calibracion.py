
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve

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


def main():
    print("===== CARGANDO VALIDACIÓN =====")

    df = cargar_particion("validation")
    X, y = preparar_variables(df)

    print(f"Registros: {len(X):,}")

    paquete_xgb = joblib.load(
        RUTA_MODELOS / "xgboost_ctr.joblib"
    )

    modelo_xgb = paquete_xgb["modelo"]
    categorias = paquete_xgb["categorias"]

    modelo_cat = joblib.load(
        RUTA_MODELOS / "catboost_ctr.joblib"
    )

    X_xgb = preparar_xgboost(X, categorias)

    probabilidades = {
        "XGBoost": modelo_xgb.predict_proba(
            X_xgb
        )[:, 1],
        "CatBoost": modelo_cat.predict_proba(
            X
        )[:, 1],
    }

    print("\n===== CALCULANDO CALIBRACIÓN =====")

    plt.figure(figsize=(8, 6))

    # Línea ideal
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Calibración perfecta"
    )

    for nombre, prob in probabilidades.items():
        fraccion_positivos, media_predicha = (
            calibration_curve(
                y,
                prob,
                n_bins=10,
                strategy="quantile"
            )
        )

        plt.plot(
            media_predicha,
            fraccion_positivos,
            marker="o",
            label=nombre
        )

        print(f"\n{nombre}")
        print(
            f"Probabilidad media: "
            f"{np.mean(prob) * 100:.2f}%"
        )
        print(
            f"CTR observado: "
            f"{y.mean() * 100:.2f}%"
        )

    plt.xlabel("Probabilidad media predicha")
    plt.ylabel("Frecuencia real de clics")
    plt.title("Calibración de XGBoost vs CatBoost")
    plt.legend()
    plt.grid(alpha=0.3)

    # Ampliar la zona de probabilidades relevantes
    plt.xlim(0, 0.15)
    plt.ylim(0, 0.15)

    plt.tight_layout()

    ruta_grafica = (
        RUTA_RESULTADOS /
        "curva_calibracion.png"
    )

    plt.savefig(
        ruta_grafica,
        dpi=300
    )

    plt.close()

    print(
        f"\nGráfica guardada en: {ruta_grafica}"
    )


if __name__ == "__main__":
    main()
