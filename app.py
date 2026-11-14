from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from detector.configuracion import RUTA_MODELO
from detector.inferencia import Clasificador

EJEMPLOS = {
    "Un logro": "I passed my exam and I can't stop smiling.",
    "Una despedida": "I miss my best friend ever since she moved away.",
    "Un gesto de cariño": "I feel so loved and cherished whenever you hold me close.",
    "Una promesa rota": "I'm furious that they broke their promise again.",
    "Un ruido de noche": "I heard a noise outside and I'm scared to look.",
    "Una visita inesperada": "I feel stunned and amazed that they showed up without warning.",
}


@st.cache_resource
def cargar_modelo(ruta: Path) -> Clasificador:
    return Clasificador(ruta)


def cargar_ejemplo(frase: str) -> None:
    st.session_state.frase = frase
    st.session_state.pop("resultado", None)


def grafico_probabilidades(predicciones: list[tuple[str, float]]) -> alt.LayerChart:
    tabla = pd.DataFrame(predicciones, columns=["Emoción", "Probabilidad"])
    tabla["Principal"] = tabla.index == 0
    base = alt.Chart(tabla).encode(
        x=alt.X(
            "Probabilidad:Q", title=None, scale=alt.Scale(domain=[0, 1.15]),
            axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1]),
        ),
        y=alt.Y("Emoción:N", title=None, sort=tabla["Emoción"].tolist(), axis=alt.Axis(labelFontSize=12)),
    )
    barras = base.mark_bar(cornerRadiusEnd=4, size=16).encode(
        color=alt.condition(alt.datum.Principal, alt.value("#167d69"), alt.value("#80b9ab")),
        tooltip=["Emoción:N", alt.Tooltip("Probabilidad:Q", format=".1%")],
    )
    porcentajes = base.mark_text(align="left", dx=6, fontSize=12).encode(
        text=alt.Text("Probabilidad:Q", format=".1%"),
    )
    return (barras + porcentajes).properties(height=215)


def main() -> None:
    st.set_page_config(page_title="Detector de emociones", layout="centered")
    st.title("Detector de emociones")
    st.write("El modelo se entrenó en inglés. Escribe una frase para analizar la emoción que expresa.")
    frase = st.text_area("Frase", key="frase", placeholder="I feel…", height=100).strip()

    with st.expander("Frases de ejemplo"):
        columnas = st.columns(2)
        for indice, (nombre, ejemplo) in enumerate(EJEMPLOS.items()):
            columnas[indice % 2].button(
                nombre, on_click=cargar_ejemplo, args=(ejemplo,), width="stretch",
            )

    if st.button("Analizar", type="primary", key="analizar"):
        st.session_state.pop("resultado", None)
        if not frase:
            st.warning("Escribe una frase antes de pulsar Analizar.")
        else:
            try:
                with st.spinner("Analizando la frase…"):
                    predicciones = cargar_modelo(RUTA_MODELO).predecir(frase)
            except FileNotFoundError:
                st.error("No se encontró la carpeta del modelo. Ejecuta el entrenamiento indicado en el README.")
            except (OSError, ValueError):
                st.error("No fue posible cargar el modelo. Comprueba los pesos, la configuración y el tokenizador.")
            else:
                st.session_state.resultado = (frase, predicciones)

    with st.sidebar:
        st.header("Configuración")
        umbral = st.slider("Umbral de confianza (%)", 0, 100, 60, step=5)
        st.caption("Si la probabilidad principal está por debajo del umbral, se muestra un aviso.")

    resultado = st.session_state.get("resultado")
    if resultado and resultado[0] == frase:
        predicciones = resultado[1]
        emocion, confianza = predicciones[0]
        principal, probabilidad = st.columns(2)
        principal.metric("Emoción detectada", emocion, border=True)
        probabilidad.metric("Confianza del modelo", f"{confianza:.1%}", border=True)
        if confianza * 100 < umbral:
            st.warning("La confianza está por debajo del umbral configurado.")
        st.subheader("Probabilidad por emoción")
        st.altair_chart(grafico_probabilidades(predicciones), width="stretch")


if __name__ == "__main__":
    main()
