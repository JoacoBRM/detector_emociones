from collections import Counter

from datasets import DatasetDict, load_dataset

from detector.configuracion import (
    DATASET,
    DATASET_CONFIG,
    DATASET_REVISION,
    EMOCIONES,
    ETIQUETAS_EN,
    IDS,
    MAX_TOKENS,
)


def cargar_datos() -> DatasetDict:
    datos = load_dataset(DATASET, DATASET_CONFIG, revision=DATASET_REVISION)
    textos_reservados = set()
    # Prioridad de prueba y validación cuando una frase aparece en varias particiones.
    for nombre in ("test", "validation", "train"):
        parte = datos[nombre]
        if tuple(parte.features["label"].names) != ETIQUETAS_EN:
            raise ValueError(f"El orden de las etiquetas cambió en {nombre}.")
        claves = [" ".join(frase.casefold().split()) for frase in parte["text"]]
        indices = [i for i, clave in enumerate(claves) if clave not in textos_reservados]
        parte = parte.select(indices)
        if set(parte["label"]) != set(IDS):
            raise ValueError(f"Hay etiquetas faltantes o desconocidas en {nombre}.")
        datos[nombre] = parte
        textos_reservados.update(claves)
    return datos


def tokenizar(datos, tokenizador):
    return datos.map(
        lambda lote: tokenizador(lote["text"], truncation=True, max_length=MAX_TOKENS),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizando frases",
    )


def mostrar_resumen(datos: DatasetDict) -> None:
    for nombre, parte in datos.items():
        conteo = Counter(parte["label"])
        desglose = ", ".join(f"{emocion}: {conteo[i]}" for i, emocion in enumerate(EMOCIONES))
        print(f"{nombre} ({len(parte):,} frases) | {desglose}")


if __name__ == "__main__":
    mostrar_resumen(cargar_datos())
