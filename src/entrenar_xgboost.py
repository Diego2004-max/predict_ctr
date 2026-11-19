
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
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

RUTA_MODELO = Path("models/xgboost_ctr.joblib")



def preparar_para_xgboost(X_train, X_val):
    X_train = X_train.copy()
    X_val = X_val.copy()

    for columna in COLUMNAS_CATEGORICAS:
        # Aprender categorías solo con entrenamiento
        categorias = pd.Index(
            X_train[columna].unique()
        )

        tipo = pd.CategoricalDtype(
            categories=categorias
        )

        # Convertir entrenamiento
        X_train[columna] = pd.Categorical(
            X_train[columna],
            categories=categorias
        )

        # Identificar categorías desconocidas
        desconocidas = ~X_val[columna].isin(categorias)

        cantidad = int(desconocidas.sum())

        if cantidad > 0:
            print(
                f"{columna}: "
                f"{cantidad:,} valores con categorías "
                f"no vistas en entrenamiento"
            )

        # Convertir las categorías desconocidas a NaN
        valores_validacion = X_val[columna].where(
            ~desconocidas,
            other=pd.NA
        )

        X_val[columna] = pd.Categorical(
            valores_validacion,
            categories=categorias
        )

    return X_train, X_val



def main():
    print("===== CARGANDO DATOS =====")

    df_train = cargar_particion("train")
    df_val = cargar_particion("validation")

    X_train, y_train = preparar_variables(df_train)
    X_val, y_val = preparar_variables(df_val)

    X_train, X_val = preparar_para_xgboost(
        X_train, X_val
    )

    print(f"Entrenamiento: {len(X_train):,} registros")
    print(f"Validación: {len(X_val):,} registros")

    print("\n===== ENTRENANDO XGBOOST =====")

    modelo = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    modelo.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print("Entrenamiento finalizado.")

    print("\n===== EVALUACIÓN EN VALIDACIÓN =====")

    probabilidades = modelo.predict_proba(X_val)[:, 1]

    metricas = {
        "ROC-AUC": roc_auc_score(
            y_val, probabilidades
        ),
        "PR-AUC": average_precision_score(
            y_val, probabilidades
        ),
        "Log Loss": log_loss(
            y_val, probabilidades
        ),
        "Brier Score": brier_score_loss(
            y_val, probabilidades
        ),
    }

    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor:.6f}")

    print(
        f"CTR observado: {y_val.mean() * 100:.2f}%"
    )
    print(
        f"CTR promedio predicho: "
        f"{np.mean(probabilidades) * 100:.2f}%"
    )

    RUTA_MODELO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "modelo": modelo,
            "categorias": {
                columna: list(X_train[columna].cat.categories)
                for columna in COLUMNAS_CATEGORICAS
            },
        },
        RUTA_MODELO,
    )

    print(f"\nModelo guardado en: {RUTA_MODELO}")


if __name__ == "__main__":
    main()
