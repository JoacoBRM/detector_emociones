import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

from detector.configuracion import (
    DATASET,
    DATASET_REVISION,
    EMOCIONES,
    IDS,
    ID_A_EMOCION,
    RUTA_MODELO,
    RUTA_RESULTADOS,
    SEMILLA,
)
from detector.datos import cargar_datos
from detector.inferencia import Clasificador


def resumir_predicciones(reales, predichas) -> dict:
    informe = classification_report(
        reales, predichas, labels=IDS, target_names=EMOCIONES, output_dict=True, zero_division=0,
    )
    return {
        "exactitud": accuracy_score(reales, predichas),
        "f1_macro": informe["macro avg"]["f1-score"],
        "por_emocion": {emocion: informe[emocion] for emocion in EMOCIONES},
        "matriz_confusion": confusion_matrix(reales, predichas, labels=IDS).tolist(),
    }


def evaluar(ruta_modelo: Path, salida: Path) -> None:
    if not ruta_modelo.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta del modelo: {ruta_modelo}")
    metadatos = json.loads((ruta_modelo / "entrenamiento.json").read_text(encoding="utf-8"))
    if metadatos["prueba_rapida"] or metadatos["revision_dataset"] != DATASET_REVISION:
        raise ValueError("Genera el modelo con el entrenamiento completo de esta versión del proyecto.")

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    clasificador = Clasificador(ruta_modelo, dispositivo=dispositivo)
    datos = cargar_datos()
    entrenamiento, prueba = datos["train"], datos["test"]
    textos = list(prueba["text"])
    reales = np.asarray(prueba["label"])
    lote = 32
    probabilidades = torch.cat([
        clasificador.probabilidades(textos[inicio:inicio + lote])
        for inicio in range(0, len(textos), lote)
    ]).numpy()
    predichas = probabilidades.argmax(axis=1)

    linea_base = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEMILLA),
    )
    linea_base.fit(list(entrenamiento["text"]), list(entrenamiento["label"]))
    predichas_base = linea_base.predict(textos)

    errores = pd.DataFrame({
        "indice_prueba": np.arange(len(textos)),
        "frase": textos,
        "etiqueta_esperada": [ID_A_EMOCION[i] for i in reales],
        "prediccion": [ID_A_EMOCION[i] for i in predichas],
        "confianza": probabilidades.max(axis=1),
    }).loc[predichas != reales].sort_values("confianza", ascending=False, kind="stable")

    resultados = {
        "dataset": DATASET,
        "revision_dataset": DATASET_REVISION,
        "frases_entrenamiento": len(entrenamiento),
        "frases_validacion": len(datos["validation"]),
        "frases_prueba": len(prueba),
        "epoca_seleccionada": int(metadatos["mejor_validacion"]["epoch"]),
        "f1_macro_validacion": metadatos["mejor_validacion"]["eval_f1_macro"],
        "orden_emociones": list(EMOCIONES),
        "distilbert": resumir_predicciones(reales, predichas),
        "tfidf_regresion_logistica": resumir_predicciones(reales, predichas_base),
    }
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "metricas.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    errores.to_csv(salida / "errores.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    for nombre in ("distilbert", "tfidf_regresion_logistica"):
        metricas = resultados[nombre]
        print(f"{nombre} | Exactitud: {metricas['exactitud']:.2%} | F1 macro: {metricas['f1_macro']:.4f}")
    print(f"{len(errores)} errores de DistilBERT | Resultados guardados en {salida}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el modelo guardado en la partición de prueba.")
    parser.add_argument("--modelo-dir", type=Path, default=RUTA_MODELO)
    parser.add_argument("--salida", type=Path, default=RUTA_RESULTADOS)
    args = parser.parse_args()
    evaluar(args.modelo_dir.resolve(), args.salida.resolve())


if __name__ == "__main__":
    main()
