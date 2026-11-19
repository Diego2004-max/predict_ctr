
from datasets import load_dataset
from huggingface_hub import list_repo_files

NOMBRE_DATASET = "criteo/CriteoClickLogs"

print("===== ARCHIVOS DEL DATASET =====")

archivos = list_repo_files(
    repo_id=NOMBRE_DATASET,
    repo_type="dataset"
)

print(f"Total de archivos: {len(archivos)}")

print("\nPrimeros 15 archivos:")
for archivo in archivos[:15]:
    print(archivo)

print("\n===== ESTRUCTURA DE LOS DATOS =====")

dataset = load_dataset(
    NOMBRE_DATASET,
    split="train",
    streaming=True
)

print("\nColumnas disponibles:")
for columna in dataset.column_names:
    print(columna)

print("\nInformación de las particiones:")
print(dataset.info.splits)

print("\nInspección finalizada.")
