
import joblib
import pandas as pd

from huggingface_hub import hf_hub_download

from preparar_datos import (
    preparar_variables,
    COLUMNAS_NUMERICAS,
    COLUMNAS_CATEGORICAS,
)


REPOSITORIO_HF = "diego2004/prediccion-ctr-modelos"


def cargar_modelo(nombre_modelo="CatBoost"):
    """
    Descarga y carga el modelo seleccionado
    desde Hugging Face Hub.
    """

    nombres_archivos = {
        "CatBoost": "catboost_ctr.joblib",
        "XGBoost": "xgboost_ctr.joblib",
    }

    if nombre_modelo not in nombres_archivos:
        raise ValueError(
            f"Modelo no reconocido: {nombre_modelo}"
        )

    archivo = nombres_archivos[nombre_modelo]

    ruta = hf_hub_download(
        repo_id=REPOSITORIO_HF,
        filename=archivo,
        repo_type="model",
    )

    return joblib.load(ruta)


def preparar_registro(registro):
    """
    Aplica el preprocesamiento original
    a un registro publicitario.
    """

    df = pd.DataFrame([registro])

    # Se agrega label solo para reutilizar
    # la función preparar_variables.
    # No se utiliza como predictor.
    df["label"] = 0

    X, _ = preparar_variables(df)

    return X


def predecir_ctr(modelo, registro, nombre_modelo="CatBoost"):
    """
    Calcula la probabilidad real de clic
    utilizando CatBoost o XGBoost.
    """

    X = preparar_registro(registro)

    if nombre_modelo == "CatBoost":

        probabilidad = float(
            modelo.predict_proba(X)[0, 1]
        )

    elif nombre_modelo == "XGBoost":

        if not isinstance(modelo, dict):
            raise TypeError(
                "El archivo XGBoost debe contener "
                "un diccionario con 'modelo' y 'categorias'."
            )

        modelo_xgb = modelo["modelo"]
        categorias = modelo["categorias"]

        # Convertir las variables categóricas
        # usando las categorías conocidas
        # durante el entrenamiento.
        for columna in COLUMNAS_CATEGORICAS:

            categorias_conocidas = categorias[columna]

            X[columna] = pd.Categorical(
                X[columna],
                categories=categorias_conocidas,
            )

        probabilidad = float(
            modelo_xgb.predict_proba(X)[0, 1]
        )

    else:

        raise ValueError(
            f"Modelo no reconocido: {nombre_modelo}"
        )

    return probabilidad
