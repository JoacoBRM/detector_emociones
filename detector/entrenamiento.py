import argparse
import json
from pathlib import Path
from shutil import copytree, ignore_patterns
from tempfile import TemporaryDirectory

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from detector.configuracion import (
    DATASET,
    DATASET_REVISION,
    EMOCION_A_ID,
    IDS,
    ID_A_EMOCION,
    MAX_TOKENS,
    MODELO_BASE,
    MODELO_BASE_REVISION,
    RUTA_EJECUCIONES,
    RUTA_MODELO,
    SEMILLA,
)
from detector.datos import cargar_datos, tokenizar


def calcular_metricas(resultado):
    reales = resultado.label_ids
    predichas = np.argmax(resultado.predictions, axis=-1)
    return {
        "exactitud": accuracy_score(reales, predichas),
        "f1_macro": f1_score(reales, predichas, labels=IDS, average="macro", zero_division=0),
    }


def entrenar(ruta_modelo: Path, epocas: int, lote: int, prueba_rapida: bool) -> None:
    if ruta_modelo.exists():
        raise FileExistsError(f"Ya existe {ruta_modelo}. Indica otra ruta con --modelo-dir.")

    set_seed(SEMILLA)
    datos = cargar_datos()
    entrenamiento, validacion = datos["train"], datos["validation"]
    if prueba_rapida:
        entrenamiento = entrenamiento.shuffle(seed=SEMILLA).select(range(128))
        validacion = validacion.shuffle(seed=SEMILLA).select(range(64))

    tokenizador = AutoTokenizer.from_pretrained(
        MODELO_BASE, revision=MODELO_BASE_REVISION, model_max_length=MAX_TOKENS,
    )
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE,
        revision=MODELO_BASE_REVISION,
        num_labels=len(IDS),
        id2label=ID_A_EMOCION,
        label2id=EMOCION_A_ID,
        use_safetensors=True,
    )
    usa_cuda = torch.cuda.is_available()
    dispositivo = torch.cuda.get_device_name(0) if usa_cuda else "CPU"
    print(f"Entrenamiento: {len(entrenamiento):,} frases | Validación: {len(validacion):,}", flush=True)
    print(f"Dispositivo: {dispositivo}", flush=True)

    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    RUTA_EJECUCIONES.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="checkpoints-", dir=RUTA_EJECUCIONES) as carpeta:
        argumentos = TrainingArguments(
            output_dir=carpeta,
            learning_rate=2e-5,
            per_device_train_batch_size=lote,
            per_device_eval_batch_size=2 * lote,
            num_train_epochs=epocas,
            weight_decay=0.01,
            warmup_steps=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_only_model=True,
            save_total_limit=1,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=50,
            disable_tqdm=True,
            bf16=usa_cuda and torch.cuda.is_bf16_supported(),
            report_to="none",
            seed=SEMILLA,
            data_seed=SEMILLA,
        )
        entrenador = Trainer(
            model=modelo,
            args=argumentos,
            train_dataset=tokenizar(entrenamiento, tokenizador),
            eval_dataset=tokenizar(validacion, tokenizador),
            data_collator=DataCollatorWithPadding(tokenizador),
            processing_class=tokenizador,
            compute_metrics=calcular_metricas,
        )
        entrenador.train()
        checkpoint = Path(entrenador.state.best_model_checkpoint)
        mejor = max(
            (registro for registro in entrenador.state.log_history if "eval_f1_macro" in registro),
            key=lambda registro: registro["eval_f1_macro"],
        )
        metadatos = {
            "modelo_base": MODELO_BASE,
            "revision_modelo_base": MODELO_BASE_REVISION,
            "dataset": DATASET,
            "revision_dataset": DATASET_REVISION,
            "semilla": SEMILLA,
            "epocas": epocas,
            "lote": lote,
            "max_tokens": MAX_TOKENS,
            "frases_entrenamiento": len(entrenamiento),
            "frases_validacion": len(validacion),
            "prueba_rapida": prueba_rapida,
            "dispositivo": dispositivo,
            "mejor_validacion": mejor,
            "historial": entrenador.state.log_history,
        }
        with TemporaryDirectory(prefix=".exportacion-", dir=ruta_modelo.parent) as exportacion:
            destino_temporal = Path(exportacion) / "modelo"
            # Exporta los pesos exactos de la mejor época y el tokenizador del checkpoint.
            copytree(checkpoint, destino_temporal, ignore=ignore_patterns("trainer_state.json", "training_args.bin"))
            (destino_temporal / "entrenamiento.json").write_text(
                json.dumps(metadatos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            destino_temporal.rename(ruta_modelo)

    print(f"Mejor F1 macro en validación: {mejor['eval_f1_macro']:.4f}")
    print(f"Modelo guardado en {ruta_modelo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ajusta DistilBERT para detectar emociones.")
    parser.add_argument("--modelo-dir", type=Path, help="Carpeta de destino del modelo.")
    parser.add_argument("--epocas", type=int, default=2)
    parser.add_argument("--lote", type=int, default=16)
    parser.add_argument("--prueba-rapida", action="store_true", help="Ejecuta una época con pocos datos.")
    args = parser.parse_args()
    if args.epocas < 1 or args.lote < 1:
        parser.error("--epocas y --lote deben ser mayores que cero.")
    if args.prueba_rapida:
        ruta = args.modelo_dir or RUTA_EJECUCIONES / "modelo_prueba"
        if ruta.resolve() == RUTA_MODELO:
            parser.error("Usa otra carpeta para el modelo de la prueba rápida.")
    else:
        ruta = args.modelo_dir or RUTA_MODELO
    try:
        entrenar(ruta.resolve(), 1 if args.prueba_rapida else args.epocas, args.lote, args.prueba_rapida)
    except FileExistsError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
