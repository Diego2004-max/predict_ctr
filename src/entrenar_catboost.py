
from pathlib import Path

import joblib
import numpy as np

from catboost import CatBoostClassifier
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

RUTA_MODELO = Path("models/catboost_ctr.joblib")


def main():
    print("===== CARGANDO DATOS =====")

    df_train = cargar_particion("train")
    df_val = cargar_particion("validation")

    X_train, y_train = preparar_variables(df_train)
    X_val, y_val = preparar_variables(df_val)

    print(f"Entrenamiento: {len(X_train):,} registros")
    print(f"Validación: {len(X_val):,} registros")

    print("\n===== ENTRENANDO CATBOOST =====")

    modelo = CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        cat_features=COLUMNAS_CATEGORICAS,
        random_seed=42,
        thread_count=-1,
        verbose=50,
        allow_writing_files=False,
    )

    modelo.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=30,
    )

    print("\nEntrenamiento finalizado.")

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
        "CTR promedio predicho: "
        f"{np.mean(probabilidades) * 100:.2f}%"
    )

    RUTA_MODELO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        modelo,
        RUTA_MODELO,
    )

    print(f"\nModelo guardado en: {RUTA_MODELO}")


if __name__ == "__main__":
    main()
