
from pathlib import Path

import pandas as pd

RUTA_DATOS = Path("data/processed")

COLUMNAS_NUMERICAS = [
    f"integer_feature_{i}" for i in range(1, 14)
]

COLUMNAS_CATEGORICAS = [
    f"categorical_feature_{i}" for i in range(1, 27)
]

COLUMNAS_MODELO = (
    COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS
)


def cargar_particion(nombre):
    archivos = sorted(
        RUTA_DATOS.glob(f"{nombre}_*.parquet")
    )

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos para {nombre}"
        )

    df = pd.concat(
        [pd.read_parquet(archivo) for archivo in archivos],
        ignore_index=True
    )

    return df


def preparar_variables(df):
    X = df[COLUMNAS_MODELO].copy()
    y = df["label"].astype(int).copy()

    # Convertir variables numéricas
    for columna in COLUMNAS_NUMERICAS:
        X[columna] = pd.to_numeric(
            X[columna],
            errors="coerce"
        ).astype("float32")

    # Preparar variables categóricas
    for columna in COLUMNAS_CATEGORICAS:
        X[columna] = (
            X[columna]
            .fillna("__MISSING__")
            .astype(str)
        )

    return X, y


def main():
    for nombre in ["train", "validation", "test"]:
        print(f"\n===== {nombre.upper()} =====")

        df = cargar_particion(nombre)
        X, y = preparar_variables(df)

        print(f"Registros: {len(X):,}")
        print(f"Variables predictoras: {X.shape[1]}")
        print(f"CTR observado: {y.mean() * 100:.2f}%")

        print(
            "Valores faltantes categóricos restantes:",
            X[COLUMNAS_CATEGORICAS].isna().sum().sum()
        )

        print(
            "Valores faltantes numéricos:",
            X[COLUMNAS_NUMERICAS].isna().sum().sum()
        )

        print(
            "Tipos de datos:",
            X.dtypes.value_counts().to_dict()
        )

    print("\nPreparación de variables verificada.")


if __name__ == "__main__":
    main()
