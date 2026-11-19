
from collections import Counter
from huggingface_hub import list_repo_files

NOMBRE_DATASET = "criteo/CriteoClickLogs"

print("Consultando archivos del dataset...")

archivos = list_repo_files(
    repo_id=NOMBRE_DATASET,
    repo_type="dataset"
)

# Identificar fechas en las rutas data/day=AAAA-MM-DD/
conteo_por_fecha = Counter()

for archivo in archivos:
    if archivo.startswith("data/day=") and archivo.endswith(".parquet"):
        partes = archivo.split("/")
        fecha = partes[1].replace("day=", "")
        conteo_por_fecha[fecha] += 1

print("\n===== FECHAS DISPONIBLES =====")

for fecha, cantidad in sorted(conteo_por_fecha.items()):
    print(f"{fecha}: {cantidad} archivos Parquet")

print("\n===== RESUMEN =====")
print(f"Total de días: {len(conteo_por_fecha)}")
print(f"Total de archivos Parquet: {sum(conteo_por_fecha.values())}")

if conteo_por_fecha:
    fechas = sorted(conteo_por_fecha)
    print(f"Primera fecha: {fechas[0]}")
    print(f"Última fecha: {fechas[-1]}")
