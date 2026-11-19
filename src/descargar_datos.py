
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Configuración
DATASET_NAME = "criteo/CriteoClickLogs"
NUM_FILAS = 100_000
TAMANO_LOTE = 5_000

RUTA_SALIDA = Path("data/raw/criteo_muestra.parquet")


def descargar_muestra():
    print("Conectando con Hugging Face...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    registros = []
    total = 0

    print("Descargando muestra de Criteo...")

    for fila in dataset:
        registros.append(fila)
        total += 1

        if total % TAMANO_LOTE == 0:
            print(f"Registros procesados: {total:,}")

        if total >= NUM_FILAS:
            break

    if not registros:
        raise RuntimeError("No se recibieron registros del dataset.")

    df = pd.DataFrame(registros)

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RUTA_SALIDA, index=False)

    print("\nDescarga finalizada.")
    print(f"Filas: {len(df):,}")
    print(f"Columnas: {len(df.columns)}")
    print(f"Archivo guardado en: {RUTA_SALIDA}")
    print("\nPrimeras filas:")
    print(df.head())


if __name__ == "__main__":
    descargar_muestra()
