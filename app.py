
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =====================================================
# 1. CONFIGURACIÓN DE RUTAS
# =====================================================

RUTA_PROYECTO = Path(__file__).resolve().parent
RUTA_SRC = RUTA_PROYECTO / "src"
RUTA_RESULTADOS = RUTA_PROYECTO / "results"

sys.path.insert(0, str(RUTA_SRC))

from predictor import cargar_modelo, predecir_ctr
from preparar_datos import (
    cargar_particion,
    COLUMNAS_NUMERICAS,
    COLUMNAS_CATEGORICAS,
)


# =====================================================
# 2. CONFIGURACIÓN DE STREAMLIT
# =====================================================

st.set_page_config(
    page_title="Sistema de Predicción de CTR",
    page_icon="📊",
    layout="wide",
)

UMBRALES = {"CatBoost": 0.06, "XGBoost": 0.01}


# =====================================================
# 3. FUNCIONES AUXILIARES
# =====================================================

@st.cache_resource
def obtener_modelo(nombre_modelo):
    """Descarga y carga el modelo real desde Hugging Face Hub."""
    return cargar_modelo(nombre_modelo)


@st.cache_data
def obtener_ejemplos():
    """Carga 100 registros reales de Criteo."""

    ruta_ejemplos = (
        RUTA_PROYECTO
        / "data"
        / "ejemplos"
        / "registros_ejemplo.parquet"
    )

    if ruta_ejemplos.exists():
        return pd.read_parquet(ruta_ejemplos)

    # Alternativa para ejecutar el proyecto localmente.
    df = cargar_particion("train")
    return df.head(100).copy()


@st.cache_data
def obtener_resultados():
    """Carga las métricas finales de ambos modelos."""

    ruta = RUTA_RESULTADOS / "evaluacion_final.csv"

    if not ruta.exists():
        return None

    return pd.read_csv(ruta)


@st.cache_data
def obtener_evaluacion_umbral():
    """Carga la evaluación de CatBoost con umbral 6 %."""

    ruta = (
        RUTA_RESULTADOS
        / "evaluacion_umbral_catboost.csv"
    )

    if not ruta.exists():
        return None

    return pd.read_csv(ruta)


def generar_firma_registro(registro):
    """Identifica los valores actuales del formulario."""

    return tuple(
        (nombre, str(valor))
        for nombre, valor in registro.items()
    )


def mostrar_resultado_prediccion(probabilidad, nombre_modelo):
    """Muestra la probabilidad y el umbral validado para cada modelo."""
    umbral = UMBRALES[nombre_modelo]
    st.caption(f"Modelo utilizado: {nombre_modelo}")

    st.subheader("Resultado de la predicción")

    col1, col2 = st.columns(2)

    col1.metric(
        "Probabilidad estimada de clic",
        f"{probabilidad * 100:.2f}%",
    )

    col2.metric(
        "Umbral de clasificación",
        f"{umbral * 100:.0f}%",
    )

    st.progress(
        min(max(float(probabilidad), 0.0), 1.0)
    )

    if probabilidad >= umbral:

        st.success(
            "Clasificación orientativa: "
            "mayor probabilidad relativa de clic."
        )

    else:

        st.info(
            "Clasificación orientativa: "
            "menor probabilidad relativa de clic."
        )

    st.caption(
        f"El umbral del {umbral * 100:.0f} % de {nombre_modelo} "
        "se seleccionó mediante F1 en validación. "
        "La probabilidad estimada no garantiza que ocurra un clic."
    )


# =====================================================
# 4. ENCABEZADO PRINCIPAL
# =====================================================

RUTA_CSS = RUTA_PROYECTO / "ui.css"
if RUTA_CSS.exists():
    st.html(RUTA_CSS)
st.html(
    """
    <header class="ctr-banner">
        <small>Minería de datos · Criteo Click Logs</small>
        <h1>Predicción de CTR</h1>
        <p>Estimación de la probabilidad de clic en publicidad digital
        con XGBoost y CatBoost.</p>
    </header>
    """
)


# =====================================================
# 5. NAVEGACIÓN
# =====================================================

st.sidebar.html(
    """
    <div class="ctr-brand">
        <span class="ctr-mark" aria-hidden="true"><i></i><i></i><i></i></span>
        <div><small>Minería de datos</small><b>Analítica de CTR</b></div>
    </div>
    """
)

seccion = st.sidebar.radio(
    "Selecciona una sección",
    [
        "Inicio",
        "Simulador de anuncios",
        "Predicción de CTR",
        "Dashboard de rendimiento",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Proyecto final de Minería de Datos"
)


# =====================================================
# SECCIÓN 1: INICIO
# =====================================================

if seccion == "Inicio":

    st.header("Resumen del proyecto")

    st.write(
        """
        Este proyecto utiliza el conjunto de datos
        **Criteo Click Logs** para estimar
        probabilidades de clic en publicidad digital.

        Al comparar **XGBoost** y **CatBoost**, se seleccionó CatBoost por
        su mejor rendimiento probabilístico en el conjunto de prueba.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Registros analizados",
        "140.000",
    )

    col2.metric(
        "Modelos entrenados",
        "2",
    )

    col3.metric(
        "Modelo principal",
        "CatBoost",
    )

    st.subheader("Objetivo del proyecto")

    st.write(
        """
        Desarrollar un sistema que permita estimar
        la probabilidad de clic en anuncios digitales
        mediante técnicas de minería de datos
        y aprendizaje automático.
        """
    )

    st.subheader("Características del conjunto de datos")

    st.write(
        """
        - 13 características numéricas.
        - 26 características categóricas anonimizadas.
        - Variable objetivo binaria: `label`.
        - 100.000 registros de entrenamiento.
        - 20.000 registros de validación.
        - 20.000 registros de prueba.
        """
    )

    st.caption(
        "Los 140.000 registros corresponden "
        "a la muestra utilizada en este proyecto, "
        "no al tamaño total del dataset Criteo."
    )


# =====================================================
# SECCIÓN 2: SIMULADOR DE ANUNCIOS
# =====================================================

elif seccion == "Simulador de anuncios":

    st.header("Simulador de anuncios")

    st.write(
        """
        Selecciona un registro publicitario
        y modifica sus características para explorar
        cómo cambia la probabilidad estimada de clic.
        """
    )

    modelo_seleccionado = st.selectbox(
        "Modelo de predicción",
        ["CatBoost", "XGBoost"],
        key="modelo_simulador",
    )

    try:
        ejemplos = obtener_ejemplos()

    except Exception as error:
        st.error(
            f"No se pudieron cargar los ejemplos: {error}"
        )
        st.stop()

    indice = st.selectbox(
        "Selecciona un registro de ejemplo",
        options=list(range(len(ejemplos))),
        format_func=lambda i: f"Registro {i + 1}",
        key="registro_simulador",
    )

    registro_base = ejemplos.iloc[indice]
    registro = {}

    # -------------------------------------------------
    # CARACTERÍSTICAS NUMÉRICAS
    # -------------------------------------------------

    st.subheader("Características numéricas")

    with st.expander(
        "Editar las 13 variables numéricas",
        expanded=True,
    ):

        columnas = st.columns(3)

        for i, nombre in enumerate(COLUMNAS_NUMERICAS):

            valor_original = registro_base[nombre]
            es_faltante = pd.isna(valor_original)

            valor_inicial = (
                0.0
                if es_faltante
                else float(valor_original)
            )

            with columnas[i % 3]:

                mantener_faltante = st.checkbox(
                    f"{nombre}: valor faltante",
                    value=bool(es_faltante),
                    key=f"faltante_{indice}_{nombre}",
                )

                valor = st.number_input(
                    nombre,
                    value=valor_inicial,
                    disabled=mantener_faltante,
                    key=f"num_{indice}_{nombre}",
                )

                registro[nombre] = (
                    float("nan")
                    if mantener_faltante
                    else float(valor)
                )

    # -------------------------------------------------
    # CARACTERÍSTICAS CATEGÓRICAS
    # -------------------------------------------------

    st.subheader("Características categóricas")

    with st.expander(
        "Editar las 26 variables categóricas"
    ):

        st.caption(
            "Estas variables contienen códigos "
            "anonimizados del dataset Criteo."
        )

        columnas_cat = st.columns(2)

        for i, nombre in enumerate(COLUMNAS_CATEGORICAS):

            valor_original = registro_base[nombre]

            if pd.isna(valor_original):
                valor_original = "__MISSING__"

            with columnas_cat[i % 2]:

                registro[nombre] = st.text_input(
                    nombre,
                    value=str(valor_original),
                    key=f"cat_{indice}_{nombre}",
                )

    # -------------------------------------------------
    # DETECTAR CAMBIOS
    # -------------------------------------------------

    firma_registro = (modelo_seleccionado, generar_firma_registro(registro))

    if (
        st.session_state.get("firma_prediccion")
        != firma_registro
    ):

        st.session_state.pop(
            "ultima_prediccion",
            None,
        )

    # -------------------------------------------------
    # GENERAR PREDICCIÓN
    # -------------------------------------------------

    st.divider()

    if st.button(
        "🔍 Predecir probabilidad de clic",
        type="primary",
        key="boton_simulador",
    ):

        try:

            modelo = obtener_modelo(modelo_seleccionado)

            probabilidad = predecir_ctr(
                modelo,
                registro,
                modelo_seleccionado,
            )

            st.session_state[
                "ultima_prediccion"
            ] = probabilidad

            st.session_state[
                "firma_prediccion"
            ] = firma_registro

        except Exception as error:

            st.session_state.pop(
                "ultima_prediccion",
                None,
            )

            st.error(
                f"No se pudo generar la predicción: {error}"
            )

    # -------------------------------------------------
    # MOSTRAR RESULTADO
    # -------------------------------------------------

    if (
        "ultima_prediccion" in st.session_state
        and st.session_state.get(
            "firma_prediccion"
        ) == firma_registro
    ):

        st.divider()

        mostrar_resultado_prediccion(
            st.session_state["ultima_prediccion"],
            modelo_seleccionado,
        )


# =====================================================
# SECCIÓN 3: PREDICCIÓN DE CTR
# =====================================================

elif seccion == "Predicción de CTR":

    st.header("Predicción de CTR")

    st.write(
        """
        Selecciona un registro publicitario
        para estimar su probabilidad de clic
        mediante el modelo CatBoost o XGBoost entrenado.
        """
    )

    modelo_seleccionado = st.selectbox(
        "Modelo de predicción",
        ["CatBoost", "XGBoost"],
        key="modelo_prediccion",
    )

    try:
        ejemplos = obtener_ejemplos()

    except Exception as error:
        st.error(
            f"No se pudieron cargar los registros: {error}"
        )
        st.stop()

    indice_pred = st.selectbox(
        "Selecciona un registro",
        options=list(range(len(ejemplos))),
        format_func=lambda i: f"Registro {i + 1}",
        key="registro_prediccion",
    )

    registro_base = ejemplos.iloc[indice_pred]

    registro = {
        nombre: registro_base[nombre]
        for nombre in (
            COLUMNAS_NUMERICAS
            + COLUMNAS_CATEGORICAS
        )
    }

    st.subheader("Información del modelo")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Modelo",
        modelo_seleccionado,
    )

    col2.metric(
        "Variables predictoras",
        "39",
    )

    col3.metric(
        "Umbral orientativo",
        f"{UMBRALES[modelo_seleccionado] * 100:.0f} %",
    )

    # -------------------------------------------------
    # MOSTRAR VARIABLES DEL REGISTRO
    # -------------------------------------------------

    with st.expander(
        "Ver características del registro"
    ):

        datos_registro = pd.DataFrame(
            {
                "Característica": list(registro.keys()),
                "Valor": [
                    str(valor)
                    for valor in registro.values()
                ],
            }
        )

        st.dataframe(
            datos_registro,
            width="stretch",
            hide_index=True,
        )

    # -------------------------------------------------
    # LIMPIAR RESULTADO AL CAMBIAR REGISTRO
    # -------------------------------------------------

    identificador_prediccion = (modelo_seleccionado, indice_pred)
    if (
        st.session_state.get("indice_predicho")
        != identificador_prediccion
    ):

        st.session_state.pop(
            "resultado_prediccion",
            None,
        )

    # -------------------------------------------------
    # GENERAR PREDICCIÓN
    # -------------------------------------------------

    if st.button(
        "🔍 Generar predicción de CTR",
        type="primary",
        key="boton_prediccion",
    ):

        try:

            modelo = obtener_modelo(modelo_seleccionado)

            probabilidad = predecir_ctr(
                modelo,
                registro,
                modelo_seleccionado,
            )

            st.session_state[
                "resultado_prediccion"
            ] = probabilidad

            st.session_state[
                "indice_predicho"
            ] = identificador_prediccion

        except Exception as error:

            st.session_state.pop(
                "resultado_prediccion",
                None,
            )

            st.error(
                f"Error al generar predicción: {error}"
            )

    # -------------------------------------------------
    # MOSTRAR RESULTADO
    # -------------------------------------------------

    if (
        "resultado_prediccion" in st.session_state
        and st.session_state.get(
            "indice_predicho"
        ) == identificador_prediccion
    ):

        st.divider()

        mostrar_resultado_prediccion(
            st.session_state["resultado_prediccion"],
            modelo_seleccionado,
        )


# =====================================================
# SECCIÓN 4: DASHBOARD DE RENDIMIENTO
# =====================================================

elif seccion == "Dashboard de rendimiento":

    st.header("Dashboard de rendimiento")

    st.write(
        """
        Comparación de XGBoost y CatBoost
        utilizando el conjunto de prueba
        de 20.000 registros.
        """
    )

    # -------------------------------------------------
    # CARGAR RESULTADOS
    # -------------------------------------------------

    try:
        df_resultados = obtener_resultados()

    except Exception as error:
        st.error(
            f"Error al cargar resultados: {error}"
        )
        st.stop()

    if df_resultados is None:
        st.warning(
            "No se encontró "
            "results/evaluacion_final.csv"
        )
        st.stop()

    datos_xgb = df_resultados[
        df_resultados["Modelo"] == "XGBoost"
    ].iloc[0]

    datos_cat = df_resultados[
        df_resultados["Modelo"] == "CatBoost"
    ].iloc[0]

    # -------------------------------------------------
    # INDICADORES PRINCIPALES
    # -------------------------------------------------

    st.subheader("Indicadores principales")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "ROC-AUC CatBoost",
        f"{datos_cat['ROC-AUC']:.4f}",
    )

    col2.metric(
        "PR-AUC CatBoost",
        f"{datos_cat['PR-AUC']:.4f}",
    )

    col3.metric(
        "Log Loss CatBoost",
        f"{datos_cat['Log Loss']:.4f}",
    )

    col4.metric(
        "Brier Score CatBoost",
        f"{datos_cat['Brier Score']:.4f}",
    )

    st.divider()

    # -------------------------------------------------
    # COMPARACIÓN DE MÉTRICAS
    # -------------------------------------------------

    st.subheader("Comparación de métricas")

    metricas = [
        "ROC-AUC",
        "PR-AUC",
        "Log Loss",
        "Brier Score",
    ]

    col_izq, col_der = st.columns(2)

    for i, metrica in enumerate(metricas):

        datos_grafica = (
            df_resultados[
                ["Modelo", metrica]
            ]
            .set_index("Modelo")
        )

        with (
            col_izq
            if i % 2 == 0
            else col_der
        ):

            st.markdown(f"**{metrica}**")

            st.bar_chart(
                datos_grafica,
                y=metrica,
                horizontal=True,
                color="#087D76",
                width="stretch",
            )

            if metrica in [
                "ROC-AUC",
                "PR-AUC",
            ]:

                st.caption(
                    "Un valor mayor es mejor."
                )

            else:

                st.caption(
                    "Un valor menor es mejor."
                )

    st.divider()

    # -------------------------------------------------
    # CTR OBSERVADO Y PREDICHO
    # -------------------------------------------------

    st.subheader(
        "CTR observado frente al predicho"
    )

    datos_ctr = df_resultados[
        [
            "Modelo",
            "CTR observado (%)",
            "CTR predicho (%)",
        ]
    ].set_index("Modelo")

    st.bar_chart(
        datos_ctr,
        x_label="Modelo",
        y_label="CTR (%)",
        stack=False,
        color=["#087D76", "#6555B9"],
        width="stretch",
    )

    st.caption(
        "El CTR predicho corresponde al promedio "
        "de las probabilidades estimadas. "
        "La cercanía de los promedios no garantiza "
        "una calibración perfecta."
    )

    st.divider()

    # -------------------------------------------------
    # MATRICES DE CONFUSIÓN CON UMBRAL 0.50
    # -------------------------------------------------

    st.subheader(
        "Matrices de confusión: umbral 50 %"
    )

    col1, col2 = st.columns(2)

    for columna, datos, nombre in [
        (col1, datos_xgb, "XGBoost"),
        (col2, datos_cat, "CatBoost"),
    ]:

        with columna:

            st.markdown(f"**{nombre}**")

            matriz = pd.DataFrame(
                [
                    [
                        int(datos["TN"]),
                        int(datos["FP"]),
                    ],
                    [
                        int(datos["FN"]),
                        int(datos["TP"]),
                    ],
                ],
                index=[
                    "Real: no clic",
                    "Real: clic",
                ],
                columns=[
                    "Predicho: no clic",
                    "Predicho: clic",
                ],
            )

            st.dataframe(
                matriz,
                width="stretch",
            )

    st.caption(
        "Estas matrices utilizan el umbral "
        "estándar de 0,50 y corresponden "
        "al conjunto de prueba."
    )

    st.divider()

    # -------------------------------------------------
    # EVALUACIÓN CATBOOST CON UMBRAL 6 %
    # -------------------------------------------------

    st.subheader(
        "Evaluación de CatBoost con umbral del 6 %"
    )

    st.write(
        """
        El umbral de **0,06** fue seleccionado
        utilizando el conjunto de validación,
        buscando maximizar la métrica F1.

        Posteriormente, este umbral se evaluó
        sobre el conjunto de prueba.
        """
    )

    try:
        df_umbral = obtener_evaluacion_umbral()

    except Exception as error:
        st.error(
            f"Error al cargar evaluación de umbral: {error}"
        )
        df_umbral = None

    if df_umbral is not None and not df_umbral.empty:

        datos_umbral = df_umbral.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Precision",
            f"{datos_umbral['Precision']:.4f}",
        )

        col2.metric(
            "Recall",
            f"{datos_umbral['Recall']:.4f}",
        )

        col3.metric(
            "F1",
            f"{datos_umbral['F1']:.4f}",
        )

        st.markdown(
            "**Matriz de confusión de CatBoost al 6 %**"
        )

        matriz_umbral = pd.DataFrame(
            [
                [
                    int(datos_umbral["TN"]),
                    int(datos_umbral["FP"]),
                ],
                [
                    int(datos_umbral["FN"]),
                    int(datos_umbral["TP"]),
                ],
            ],
            index=[
                "Real: no clic",
                "Real: clic",
            ],
            columns=[
                "Predicho: no clic",
                "Predicho: clic",
            ],
        )

        st.dataframe(
            matriz_umbral,
            width="stretch",
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Clics reales identificados",
            int(datos_umbral["TP"]),
        )

        col2.metric(
            "Clics reales no identificados",
            int(datos_umbral["FN"]),
        )

        st.info(
            f"Con el umbral del 6 %, CatBoost identificó "
            f"{int(datos_umbral['TP'])} de "
            f"{int(datos_umbral['TP'] + datos_umbral['FN'])} clics reales "
            f"del conjunto de prueba y generó "
            f"{int(datos_umbral['FP'])} falsos positivos."
        )

    else:

        st.warning(
            "No se encontró "
            "results/evaluacion_umbral_catboost.csv"
        )

    st.divider()

    # -------------------------------------------------
    # CURVA DE CALIBRACIÓN
    # -------------------------------------------------

    st.subheader("Análisis de calibración")

    st.write(
        """
        La curva de calibración compara
        las probabilidades estimadas con
        las frecuencias reales de clic.

        Los puntos cercanos a la diagonal
        indican mejor calibración.
        """
    )

    ruta_calibracion = (
        RUTA_RESULTADOS
        / "curva_calibracion.png"
    )

    if ruta_calibracion.exists():

        st.image(
            str(ruta_calibracion),
            caption=(
                "Curva de calibración de XGBoost "
                "y CatBoost en validación."
            ),
            width="stretch",
        )

        st.info(
            "En el conjunto de validación, "
            "CatBoost mostró una calibración "
            "más cercana a la ideal que XGBoost."
        )

    else:

        st.warning(
            "No se encontró "
            "results/curva_calibracion.png"
        )

    st.divider()

    # -------------------------------------------------
    # TABLA DE RESULTADOS COMPLETOS
    # -------------------------------------------------

    st.subheader("Resultados completos")

    st.dataframe(
        df_resultados,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    # -------------------------------------------------
    # CONCLUSIONES
    # -------------------------------------------------

    st.subheader("Conclusiones del análisis")

    st.success(
        "CatBoost obtuvo mejores resultados "
        "que XGBoost en ROC-AUC, PR-AUC, "
        "Log Loss y Brier Score."
    )

    st.write(
        """
        **CatBoost** fue seleccionado como
        modelo principal de predicción de CTR.

        Con el umbral estándar del 50 %,
        CatBoost no identificó clics positivos
        en el conjunto de prueba.

        Al utilizar el umbral del 6 %,
        seleccionado previamente en validación,
        logró identificar una parte de los
        clics reales, aunque con falsos positivos.

        Los resultados corresponden a una muestra
        reducida del conjunto Criteo Click Logs
        y no garantizan el mismo rendimiento
        en un entorno de producción.
        """
    )
