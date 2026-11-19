
from preparar_datos import cargar_particion
from predictor import cargar_modelo, predecir_ctr


def main():

    print("===== PRUEBA DE PREDICCIÓN DE AMBOS MODELOS =====")

    # Utilizamos un registro real de entrenamiento.
    df = cargar_particion("train")
    registro = df.iloc[0].to_dict()

    for nombre in ["CatBoost", "XGBoost"]:

        print(f"\nProbando {nombre}...")

        modelo = cargar_modelo(nombre)

        probabilidad = predecir_ctr(
            modelo,
            registro,
            nombre,
        )

        print(
            f"Probabilidad de clic: "
            f"{probabilidad * 100:.4f}%"
        )

    print("\nPrueba completada.")


if __name__ == "__main__":
    main()
