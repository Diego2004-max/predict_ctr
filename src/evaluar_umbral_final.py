
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ==========================================
# CONFIGURACIÓN
# ==========================================

RUTA_PROYECTO = Path(__file__).resolve().parent.parent

RUTA_PREDICCIONES = (
    RUTA_PROYECTO / "results" / "predicciones_test.csv"
)

RUTA_SALIDA = (
    RUTA_PROYECTO / "results" / "evaluacion_umbral_catboost.csv"
)

UMBRAL = 0.06


# ==========================================
# EVALUACIÓN FINAL
# ==========================================

def main():

    print("===== EVALUACIÓN DE CATBOOST =====")

    if not RUTA_PREDICCIONES.exists():
        print("ERROR: No se encontró el archivo de predicciones.")
        print(RUTA_PREDICCIONES)
        return

    df = pd.read_csv(RUTA_PREDICCIONES)

    y_real = df["label"].astype(int)

    probabilidades = df["prob_catboost"]

    y_predicho = (probabilidades >= UMBRAL).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_real,
        y_predicho,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_real,
        y_predicho,
        zero_division=0,
    )

    recall = recall_score(
        y_real,
        y_predicho,
        zero_division=0,
    )

    f1 = f1_score(
        y_real,
        y_predicho,
        zero_division=0,
    )

    print(f"\nUmbral utilizado: {UMBRAL:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    print("\n===== MATRIZ DE CONFUSIÓN =====")

    print(f"Verdaderos negativos (TN): {tn}")
    print(f"Falsos positivos (FP): {fp}")
    print(f"Falsos negativos (FN): {fn}")
    print(f"Verdaderos positivos (TP): {tp}")

    resultados = pd.DataFrame(
        [{
            "Modelo": "CatBoost",
            "Umbral": UMBRAL,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }]
    )

    RUTA_SALIDA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resultados.to_csv(
        RUTA_SALIDA,
        index=False,
    )

    print("\nResultados guardados en:")
    print(RUTA_SALIDA)


if __name__ == "__main__":
    main()
