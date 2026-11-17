import base64
import json
import random
import uuid
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from detector.configuracion import EMOCIONES, ID_A_EMOCION, RUTA_MODELO, RUTA_RESULTADOS
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
    "nuevo_chat": ("nuevo-chat", "before"),
    # Prefijo: un botón de borrar por chat, con clave «borrar_<id>».
    "[borrar_]": ("borrar", "before"),
    "modo": ("modo", "before"),
    # La opción activa del selector de modo lleva una marca de verificación.
    "[modoactivo_]": ("check", "after"),
}


@st.cache_data
def estilos_iconos() -> str:
    reglas = []
    for clave, (archivo, lado) in ICONOS_BOTONES.items():
        contenido = base64.b64encode((RUTA_ICONOS / f"{archivo}.svg").read_bytes()).decode()
        margen = "margin-right" if lado == "before" else "margin-left"
        if clave.startswith("["):
            selector = f"[class*='st-key-{clave[1:-1]}']"
        else:
            selector = f".st-key-{clave}"
        separacion = "0" if archivo == "borrar" else "0.5em"
        reglas.append(
            f"{selector} button [data-testid='stMarkdownContainer'] p::{lado} {{"
            f"content: ''; display: inline-block; width: 1.15em; height: 1.15em; {margen}: {separacion};"
            "vertical-align: -0.2em; background-color: currentColor;"
            f"-webkit-mask: url(data:image/svg+xml;base64,{contenido}) center / contain no-repeat;"
            f"mask: url(data:image/svg+xml;base64,{contenido}) center / contain no-repeat;}}"
        )
    # «Acerca de» se ancla en la barra superior de Streamlit, a la izquierda de Deploy.
    reglas.append(
        ".st-key-acerca {position: fixed; top: 0.6rem; right: 9rem; width: auto; z-index: 1000000;}"
    )
    reglas.append(".st-key-lista_chats button {justify-content: flex-start; text-align: left;}")
    reglas.append(
        ".st-key-opciones_modo button > div, .st-key-opciones_modo button > div > span"
        "{display: block; width: 100%;}"
        ".st-key-opciones_modo button [data-testid='stMarkdownContainer'] {width: 100%;}"
        ".st-key-opciones_modo button p {display: flex; justify-content: space-between;"
        "align-items: center; width: 100%; text-align: left;}"
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


MODOS = {
    "una": ("Una frase", "Analiza una frase y muestra todas las probabilidades."),
    "varias": ("Varias frases", "Una frase por línea o un archivo CSV con columna «frase»."),
}
MAX_LOTE = 500
TAMANO_LOTE_MODELO = 64


def elegir_modo(modo: str) -> None:
    st.session_state.modo = modo


def cargar_ejemplo(frase: str) -> None:
    actual = st.session_state.get("frase", "").strip()
    if st.session_state.modo == "varias" and actual:
        if frase not in actual.splitlines():
            st.session_state.frase = f"{actual}\n{frase}"
    else:
        st.session_state.frase = frase


def cargar_aleatoria() -> None:
    if st.session_state.modo == "varias":
        st.session_state.frase = "\n".join(random.sample(list(EJEMPLOS.values()), 4))
        return
    otras = [frase for frase in EJEMPLOS.values() if frase != st.session_state.get("frase")]
    st.session_state.frase = random.choice(otras)


def frases_de_csv(archivo) -> list[str]:
    try:
        tabla = pd.read_csv(archivo, encoding="utf-8-sig")
    except UnicodeDecodeError:
        archivo.seek(0)
        tabla = pd.read_csv(archivo, encoding="latin-1")
    columnas = {str(columna).strip().lower(): columna for columna in tabla.columns}
    if "frase" not in columnas:
        raise ValueError("El CSV debe tener una columna llamada «frase».")
    return [str(frase).strip() for frase in tabla[columnas["frase"]].dropna() if str(frase).strip()]


RUTA_HISTORIAL = Path(__file__).parent / ".historial" / "chats.json"


def cargar_chats() -> list[dict]:
    try:
        datos = json.loads(RUTA_HISTORIAL.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [chat for chat in datos if isinstance(chat, dict) and chat.get("id") and chat.get("turnos")]


def guardar_chats() -> None:
    RUTA_HISTORIAL.parent.mkdir(parents=True, exist_ok=True)
    temporal = RUTA_HISTORIAL.with_suffix(".tmp")
    temporal.write_text(
        json.dumps(st.session_state.chats, ensure_ascii=False, indent=1), encoding="utf-8",
    )
    temporal.replace(RUTA_HISTORIAL)


def chat_actual() -> dict | None:
    return next((chat for chat in st.session_state.chats if chat["id"] == st.session_state.chat_id), None)


def nuevo_chat() -> None:
    st.session_state.chat_id = uuid.uuid4().hex


def abrir_chat(id_chat: str) -> None:
    st.session_state.chat_id = id_chat


def borrar_chat(id_chat: str) -> None:
    st.session_state.chats = [chat for chat in st.session_state.chats if chat["id"] != id_chat]
    guardar_chats()
    if st.session_state.chat_id == id_chat:
        nuevo_chat()


def registrar_turno(turno: dict) -> None:
    chat = chat_actual()
    if chat is None:
        titulo = turno["frase"] if len(turno["frase"]) <= 40 else turno["frase"][:37] + "…"
        chat = {"id": st.session_state.chat_id, "titulo": titulo, "turnos": []}
        st.session_state.chats.insert(0, chat)
    chat["turnos"].append(turno)
    guardar_chats()


def barra_lateral() -> None:
    with st.sidebar:
        st.button("Nuevo chat", on_click=nuevo_chat, width="stretch", key="nuevo_chat")
        st.caption("Historial")
        if not st.session_state.chats:
            st.caption("Aún no hay chats guardados.")
        with st.container(key="lista_chats"):
            for chat in st.session_state.chats:
                texto, borrar = st.columns([5, 1], vertical_alignment="center")
                activo = chat["id"] == st.session_state.chat_id
                texto.button(
                    chat["titulo"], on_click=abrir_chat, args=(chat["id"],), width="stretch",
                    type="secondary" if activo else "tertiary", key=f"chat_{chat['id']}",
                )
                borrar.button(
                    "​", on_click=borrar_chat, args=(chat["id"],), type="tertiary",
                    help="Borrar este chat", key=f"borrar_{chat['id']}",
                )


def enviar() -> None:
    texto = st.session_state.get("frase", "")
    if st.session_state.modo == "una":
        frase = texto.strip()
        if not frase:
            st.toast("Escribe una frase antes de analizar.")
            return
        st.session_state.pendiente = frase
    else:
        frases = [linea.strip() for linea in texto.splitlines() if linea.strip()]
        archivo = st.session_state.get(f"csv_{st.session_state.csv_n}")
        if archivo is not None:
            try:
                frases += frases_de_csv(archivo)
            except (ValueError, pd.errors.EmptyDataError) as error:
                # ParserError también es un ValueError; solo mostramos nuestro mensaje propio.
                mensaje = str(error)
                st.toast(mensaje if mensaje.startswith("El CSV") else "No se pudo leer el CSV.")
                return
        if not frases:
            st.toast("Escribe al menos una frase o sube un archivo CSV.")
            return
        if len(frases) > MAX_LOTE:
            st.toast(f"Solo se analizarán las primeras {MAX_LOTE} frases.")
            frases = frases[:MAX_LOTE]
        st.session_state.pendiente = frases
        st.session_state.csv_n += 1
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


def analizar_lote(frases: list[str]) -> dict:
    resumen = f"Lote de {len(frases)} frases"
    try:
        with st.spinner(f"Analizando {len(frases)} frases…"):
            modelo = cargar_modelo(RUTA_MODELO)
            filas = []
            for inicio in range(0, len(frases), TAMANO_LOTE_MODELO):
                grupo = frases[inicio:inicio + TAMANO_LOTE_MODELO]
                for frase, probabilidades in zip(grupo, modelo.probabilidades(grupo).tolist()):
                    mejor = max(range(len(probabilidades)), key=probabilidades.__getitem__)
                    filas.append(
                        {"frase": frase, "emocion": ID_A_EMOCION[mejor], "confianza": probabilidades[mejor]},
                    )
    except FileNotFoundError:
        return {"frase": resumen, "error": "No se encontró la carpeta del modelo. Ejecuta el entrenamiento indicado en el README."}
    except (OSError, ValueError):
        return {"frase": resumen, "error": "No fue posible cargar el modelo. Comprueba los pesos, la configuración y el tokenizador."}
    return {"frase": resumen, "frases": frases, "lote": filas}


def grafico_distribucion(filas: list[dict]) -> alt.LayerChart:
    cuentas = pd.Series([fila["emocion"] for fila in filas]).value_counts()
    tabla = pd.DataFrame({"Emoción": list(EMOCIONES), "Frases": [int(cuentas.get(e, 0)) for e in EMOCIONES]})
    base = alt.Chart(tabla).encode(
        x=alt.X("Emoción:N", title=None, sort=list(EMOCIONES), axis=alt.Axis(labelAngle=0, labelFontSize=12)),
        y=alt.Y("Frases:Q", title=None, axis=alt.Axis(tickMinStep=1, format="d")),
    )
    barras = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.value("#167d69"), tooltip=["Emoción:N", "Frases:Q"],
    )
    cifras = base.mark_text(dy=-8, fontSize=12).encode(text="Frases:Q")
    return (barras + cifras).properties(height=220)


def mostrar_lote(turno: dict, umbral: int, clave: str) -> None:
    filas = turno["lote"]
    tabla = pd.DataFrame(filas)
    tabla["segura"] = ["sí" if fila["confianza"] * 100 >= umbral else "no" for fila in filas]
    seguras = int((tabla["segura"] == "sí").sum())
    st.caption(f"{len(filas)} frases analizadas · {seguras} con confianza igual o mayor al umbral ({umbral} %).")
    st.dataframe(
        tabla.rename(columns={"frase": "Frase", "emocion": "Emoción", "confianza": "Confianza", "segura": "Segura"}),
        hide_index=True, width="stretch",
        column_config={
            "Confianza": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        },
    )
    st.subheader("Distribución de emociones", anchor=False)
    st.altair_chart(grafico_distribucion(filas), width="stretch")
    st.download_button(
        "Descargar resultados (CSV)", data=tabla.to_csv(index=False).encode("utf-8-sig"),
        file_name="resultados_emociones.csv", mime="text/csv", on_click="ignore", key=f"descarga_{clave}",
    )


def mostrar_frases_lote(turno: dict) -> None:
    frases = turno.get("frases", [])
    st.write(turno["frase"])
    with st.expander("Ver frases enviadas"):
        for frase in frases[:50]:
            st.markdown(f"- {frase}")
        if len(frases) > 50:
            st.caption(f"… y {len(frases) - 50} más.")


def mostrar_respuesta(turno: dict, umbral: int, clave: str = "") -> None:
    if "error" in turno:
        st.error(turno["error"])
        return
    if "lote" in turno:
        mostrar_lote(turno, umbral, clave)
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
    if "chats" not in st.session_state:
        st.session_state.chats = cargar_chats()
        nuevo_chat()
    st.session_state.setdefault("modo", "una")
    st.session_state.setdefault("csv_n", 0)
    chat = chat_actual()
    historial = chat["turnos"] if chat else []

    with st.container(key="acerca"):
        if st.button("Acerca de", type="tertiary"):
            acerca_de()

    encabezado, ajustes = st.columns([4, 1.4], vertical_alignment="center")
    encabezado.title("Detector de emociones", anchor=False)
    encabezado.caption("Escribe una frase en inglés y detectará la emoción que transmite.")
    with ajustes.container(key="ajustes"), st.popover("Configuración", width="stretch"):
        umbral = st.slider("Umbral de confianza (%)", 0, 100, 60, step=5, key="umbral")
        st.caption("Si la probabilidad principal está por debajo del umbral, se muestra un aviso.")

    def mostrar_pregunta(turno: dict) -> None:
        if "frases" in turno:
            mostrar_frases_lote(turno)
        else:
            st.write(turno["frase"])

    for posicion, turno in enumerate(historial):
        with st.chat_message("user", avatar=str(AVATAR_USUARIO)):
            mostrar_pregunta(turno)
        with st.chat_message("assistant", avatar=avatar_respuesta(turno)):
            mostrar_respuesta(turno, umbral, f"{st.session_state.chat_id}_{posicion}")

    pendiente = st.session_state.pop("pendiente", None)
    if pendiente:
        es_lote = isinstance(pendiente, list)
        with st.chat_message("user", avatar=str(AVATAR_USUARIO)):
            if es_lote:
                mostrar_frases_lote({"frase": f"Lote de {len(pendiente)} frases", "frases": pendiente})
            else:
                st.write(pendiente)
        turno = analizar_lote(pendiente) if es_lote else analizar(pendiente)
        registrar_turno(turno)
        posicion = len(chat_actual()["turnos"]) - 1
        with st.chat_message("assistant", avatar=avatar_respuesta(turno)):
            mostrar_respuesta(turno, umbral, f"{st.session_state.chat_id}_{posicion}")

    varias = st.session_state.modo == "varias"
    with st.container(border=True, key="compositor"):
        st.text_area(
            "Frase", key="frase", height=140 if varias else 90, label_visibility="collapsed",
            placeholder=(
                "Escribe una frase por línea, en inglés…\nI feel so proud of you\nThis makes me really angry"
                if varias else "Escribe una frase en inglés… (por ejemplo: I feel…)"
            ),
        )
        if varias:
            st.file_uploader(
                "O sube un archivo CSV con una columna llamada «frase»", type="csv",
                key=f"csv_{st.session_state.csv_n}",
            )
        aleatoria, ejemplos, modo, enviar_frase = st.columns([1.5, 1.4, 1.7, 1.3], vertical_alignment="center")
        aleatoria.button("Autogenerar", on_click=cargar_aleatoria, width="stretch", key="aleatoria")
        with ejemplos.container(key="ejemplos"), st.popover("Ejemplos", width="stretch"):
            for nombre, ejemplo in EJEMPLOS.items():
                st.button(nombre, on_click=cargar_ejemplo, args=(ejemplo,), width="stretch", key=f"ej_{nombre}")
        with modo.container(key="modo"), st.popover(MODOS[st.session_state.modo][0], width="stretch"):
            with st.container(key="opciones_modo"):
                for id_modo, (nombre, descripcion) in MODOS.items():
                    activo = id_modo == st.session_state.modo
                    st.button(
                        nombre, on_click=elegir_modo, args=(id_modo,), width="stretch", type="tertiary",
                        key=f"modoactivo_{id_modo}" if activo else f"modo_{id_modo}",
                    )
                    st.caption(descripcion)
        enviar_frase.button("Analizar", type="primary", on_click=enviar, width="stretch", key="analizar")

    # Se dibuja al final para que el chat recién guardado aparezca en esta misma ejecución.
    barra_lateral()


if __name__ == "__main__":
    main()
