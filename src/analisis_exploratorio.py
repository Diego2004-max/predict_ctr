
import pandas as pd

# 1. Cargar el dataset
ruta = "data/raw/criteo_muestra.parquet"
df = pd.read_parquet(ruta)

print("\n===== INFORMACIÓN GENERAL =====")
print(f"Filas: {df.shape[0]:,}")
print(f"Columnas: {df.shape[1]}")

# 2. Identificar tipos de datos
print("\n===== TIPOS DE DATOS =====")
print(df.dtypes.value_counts())

# 3. Distribución de la variable objetivo
print("\n===== DISTRIBUCIÓN DE CLICS =====")
conteo = df["label"].value_counts().sort_index()
porcentaje = df["label"].value_counts(
    normalize=True
).sort_index() * 100

for clase, cantidad in conteo.items():
    print(
        f"Clase {clase}: {cantidad:,} registros "
        f"({porcentaje[clase]:.2f}%)"
    )

print(f"\nCTR observado: {df['label'].mean() * 100:.2f}%")

# 4. Valores faltantes
print("\n===== VALORES FALTANTES =====")
faltantes = df.isna().sum().sort_values(ascending=False)
print(faltantes[faltantes > 0].head(15))

# 5. Duplicados exactos
print("\n===== REGISTROS DUPLICADOS =====")
print(f"Duplicados exactos: {df.duplicated().sum():,}")

# 6. Estadísticas de variables numéricas
columnas_numericas = [
    col for col in df.columns
    if col.startswith("integer_feature_")
]

print("\n===== ESTADÍSTICAS NUMÉRICAS =====")
print(
    df[columnas_numericas]
    .describe()
    .round(2)
    .T
)

print("\nAnálisis exploratorio finalizado.")
