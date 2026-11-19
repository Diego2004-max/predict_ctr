
from huggingface_hub import hf_hub_download
import joblib

REPOSITORIO = "diego2004/prediccion-ctr-modelos"


def probar_modelo(nombre_archivo):

    print(f"\nDescargando: {nombre_archivo}")

    ruta = hf_hub_download(
        repo_id=REPOSITORIO,
        filename=nombre_archivo,
        repo_type="model",
    )

    print("Descarga completada.")
    print("Archivo:", ruta)

    modelo = joblib.load(ruta)

    print("Carga correcta.")
    print("Tipo de objeto:", type(modelo))

    if isinstance(modelo, dict):
        print("Claves:", list(modelo.keys()))


def main():

    print("===== PRUEBA DE MODELOS HUGGING FACE =====")

    probar_modelo("catboost_ctr.joblib")

    probar_modelo("xgboost_ctr.joblib")

    print("\nAmbos modelos se cargaron correctamente.")


if __name__ == "__main__":
    main()
