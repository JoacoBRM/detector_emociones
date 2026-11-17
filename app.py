import base64
import json
import random
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from detector.configuracion import RUTA_MODELO, RUTA_RESULTADOS
from detector.inferencia import Clasificador

EJEMPLOS = {
    "Un logro": "I passed my exam and I can't stop smiling.",
    "Una despedida": "I miss my best friend ever since she moved away.",
    "Un gesto de cariño": "I feel so loved and cherished whenever you hold me close.",
    "Una promesa rota": "I'm furious that they broke their promise again.",
    "Un ruido de noche": "I heard a noise outside and I'm scared to look.",
    "Una visita inesperada": "I feel stunned and amazed that they showed up without warning.",
}


RUTA_ICONOS = Path(__file__).parent / "iconos"
AVATAR_USUARIO = RUTA_ICONOS / "avatar-usuario.svg"
AVATAR_ASISTENTE = RUTA_ICONOS / "avatar-asistente.svg"
AVATARES_EMOCION = {
    "Tristeza": "tristeza",
    "Alegría": "alegria",
    "Amor": "amor",
    "Ira": "ira",
    "Miedo": "miedo",
    "Sorpresa": "sorpresa",
}


def avatar_respuesta(turno: dict) -> str:
    if "predicciones" not in turno:
        return str(AVATAR_ASISTENTE)
    archivo = AVATARES_EMOCION.get(turno["predicciones"][0][0])
    return str(RUTA_ICONOS / f"avatar-{archivo}.svg") if archivo else str(AVATAR_ASISTENTE)
# Clave del widget -> (archivo SVG de Solar Icons, lado del texto donde se dibuja).
ICONOS_BOTONES = {
    "ajustes": ("configuracion", "before"),
    "acerca": ("acerca", "before"),
    "aleatoria": ("autogenerar", "before"),
    "ejemplos": ("ejemplos", "before"),
    "analizar": ("enviar", "after"),
}


@st.cache_data
def estilos_iconos() -> str:
    reglas = []
    for clave, (archivo, lado) in ICONOS_BOTONES.items():
        contenido = base64.b64encode((RUTA_ICONOS / f"{archivo}.svg").read_bytes()).decode()
        margen = "margin-right" if lado == "before" else "margin-left"
        reglas.append(
            f".st-key-{clave} button [data-testid='stMarkdownContainer'] p::{lado} {{"
            f"content: ''; display: inline-block; width: 1.15em; height: 1.15em; {margen}: 0.5em;"
            "vertical-align: -0.2em; background-color: currentColor;"
            f"-webkit-mask: url(data:image/svg+xml;base64,{contenido}) center / contain no-repeat;"
            f"mask: url(data:image/svg+xml;base64,{contenido}) center / contain no-repeat;}}"
        )
    # «Acerca de» se ancla en la barra superior de Streamlit, a la izquierda de Deploy.
    reglas.append(
        ".st-key-acerca {position: fixed; top: 0.6rem; right: 9rem; width: auto; z-index: 1000000;}"
    )
    return "<style>" + "".join(reglas) + "</style>"


DESCRIPCION_EMOCIONES = {
    "Tristeza": "pérdida, nostalgia o desánimo",
    "Alegría": "felicidad, logro o entusiasmo",
    "Amor": "cariño, ternura o afecto",
    "Ira": "enojo, molestia o indignación",
    "Miedo": "temor, inquietud o amenaza",
    "Sorpresa": "asombro o algo inesperado",
}


def leer_metricas() -> dict | None:
    ruta = RUTA_RESULTADOS / "metricas.json"
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.dialog("Acerca del detector", width="large")
def acerca_de() -> None:
    st.markdown(
        "Aplicación que lee una frase **en inglés** y estima qué emoción expresa. "
        "Muestra la emoción principal, la confianza del modelo y la probabilidad de las seis clases."
    )
    st.subheader("Emociones que detecta", anchor=False)
    for emocion, descripcion in DESCRIPCION_EMOCIONES.items():
        st.markdown(f"- **{emocion}**: {descripcion}")

    st.subheader("El modelo", anchor=False)
    st.markdown(
        "- **Base:** DistilBERT (`distilbert-base-uncased`), ajustado con ajuste fino.\n"
        "- **Datos:** corpus público `dair-ai/emotion` (frases cortas en inglés).\n"
        "- **Entrada:** se truncan las frases a 64 tokens. Otros idiomas funcionan peor."
    )
    metricas = leer_metricas()
    if metricas:
        st.subheader("Rendimiento en el conjunto de prueba", anchor=False)
        st.caption(f"{metricas['frases_prueba']:,} frases que el modelo no vio al entrenar.")
        modelo, referencia = metricas["distilbert"], metricas["tfidf_regresion_logistica"]
        tabla = pd.DataFrame(
            {
                "Exactitud": [f"{modelo['exactitud']:.1%}", f"{referencia['exactitud']:.1%}"],
                "F1 macro": [f"{modelo['f1_macro']:.3f}", f"{referencia['f1_macro']:.3f}"],
            },
            index=["DistilBERT ajustado", "TF-IDF + regresión logística"],
        )
        st.table(tabla)

    st.subheader("Cómo interpretarlo", anchor=False)
    st.markdown(
        "- La **confianza** es la probabilidad que el modelo asigna a la emoción principal; "
        "no garantiza que sea correcta.\n"
        "- El **umbral** (en Configuración) muestra un aviso cuando la confianza queda por debajo.\n"
        "- Una frase puede mezclar emociones. El modelo elige solo una.\n"
        "- Es un proyecto académico de Ciencia de Datos, no una herramienta de diagnóstico."
    )
    st.caption("Iconos: Solar Icon Set de 480 Design (CC BY 4.0).")


@st.cache_resource
def cargar_modelo(ruta: Path) -> Clasificador:
    return Clasificador(ruta)


def cargar_ejemplo(frase: str) -> None:
    st.session_state.frase = frase


def cargar_aleatoria() -> None:
    otras = [frase for frase in EJEMPLOS.values() if frase != st.session_state.get("frase")]
    st.session_state.frase = random.choice(otras)


def enviar() -> None:
    frase = st.session_state.get("frase", "").strip()
    if not frase:
        st.toast("Escribe una frase antes de analizar.")
        return
    st.session_state.pendiente = frase
    st.session_state.frase = ""


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


def analizar(frase: str) -> dict:
    try:
        with st.spinner("Analizando la frase…"):
            predicciones = cargar_modelo(RUTA_MODELO).predecir(frase)
    except FileNotFoundError:
        return {"frase": frase, "error": "No se encontró la carpeta del modelo. Ejecuta el entrenamiento indicado en el README."}
    except (OSError, ValueError):
        return {"frase": frase, "error": "No fue posible cargar el modelo. Comprueba los pesos, la configuración y el tokenizador."}
    return {"frase": frase, "predicciones": predicciones}


def mostrar_respuesta(turno: dict, umbral: int) -> None:
    if "error" in turno:
        st.error(turno["error"])
        return
    predicciones = turno["predicciones"]
    emocion, confianza = predicciones[0]
    principal, probabilidad = st.columns(2)
    principal.metric("Emoción detectada", emocion, border=True)
    probabilidad.metric("Confianza del modelo", f"{confianza:.1%}", border=True)
    if confianza * 100 < umbral:
        st.warning("La confianza está por debajo del umbral configurado.")
    st.altair_chart(grafico_probabilidades(predicciones), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Detector de emociones", layout="centered")
    st.markdown(estilos_iconos(), unsafe_allow_html=True)
    historial = st.session_state.setdefault("historial", [])

    with st.container(key="acerca"):
        if st.button("Acerca de", type="tertiary"):
            acerca_de()

    encabezado, ajustes = st.columns([4, 1.4], vertical_alignment="center")
    encabezado.title("Detector de emociones", anchor=False)
    encabezado.caption("Escribe una frase en inglés y detectará la emoción que transmite.")
    with ajustes.container(key="ajustes"), st.popover("Configuración", width="stretch"):
        umbral = st.slider("Umbral de confianza (%)", 0, 100, 60, step=5, key="umbral")
        st.caption("Si la probabilidad principal está por debajo del umbral, se muestra un aviso.")

    for turno in historial:
        with st.chat_message("user", avatar=str(AVATAR_USUARIO)):
            st.write(turno["frase"])
        with st.chat_message("assistant", avatar=avatar_respuesta(turno)):
            mostrar_respuesta(turno, umbral)

    pendiente = st.session_state.pop("pendiente", None)
    if pendiente:
        with st.chat_message("user", avatar=str(AVATAR_USUARIO)):
            st.write(pendiente)
        turno = analizar(pendiente)
        historial.append(turno)
        with st.chat_message("assistant", avatar=avatar_respuesta(turno)):
            mostrar_respuesta(turno, umbral)

    with st.container(border=True, key="compositor"):
        st.text_area(
            "Frase", key="frase", placeholder="Escribe una frase en inglés… (por ejemplo: I feel…)",
            height=90, label_visibility="collapsed",
        )
        aleatoria, ejemplos, _, enviar_frase = st.columns([1.5, 1.5, 2, 1.3], vertical_alignment="center")
        aleatoria.button("Autogenerar", on_click=cargar_aleatoria, width="stretch", key="aleatoria")
        with ejemplos.container(key="ejemplos"), st.popover("Ejemplos", width="stretch"):
            for nombre, ejemplo in EJEMPLOS.items():
                st.button(nombre, on_click=cargar_ejemplo, args=(ejemplo,), width="stretch", key=f"ej_{nombre}")
        enviar_frase.button("Analizar", type="primary", on_click=enviar, width="stretch", key="analizar")


if __name__ == "__main__":
    main()
