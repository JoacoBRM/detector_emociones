from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detector.configuracion import ID_A_EMOCION, MAX_TOKENS, RUTA_MODELO


class Clasificador:
    def __init__(self, ruta: Path = RUTA_MODELO, dispositivo: str = "cpu"):
        if not ruta.is_dir():
            raise FileNotFoundError(f"No se encontró la carpeta del modelo: {ruta}")
        self.dispositivo = torch.device(dispositivo)
        self.tokenizador = AutoTokenizer.from_pretrained(ruta, local_files_only=True)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            ruta, local_files_only=True, use_safetensors=True,
        )
        if self.modelo.config.id2label != ID_A_EMOCION:
            raise ValueError("El modelo guardado debe tener las seis etiquetas de emociones.")
        self.modelo.to(self.dispositivo).eval()

    @torch.inference_mode()
    def probabilidades(self, frases: list[str]) -> torch.Tensor:
        entradas = self.tokenizador(
            frases, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt",
        ).to(self.dispositivo)
        logits = self.modelo(**entradas).logits
        return logits.float().softmax(dim=-1).cpu()

    def predecir(self, frase: str) -> list[tuple[str, float]]:
        frase = frase.strip()
        if not frase:
            raise ValueError("Escribe una frase en inglés para analizarla.")
        probabilidades = self.probabilidades([frase])[0].tolist()
        predicciones = [(ID_A_EMOCION[i], valor) for i, valor in enumerate(probabilidades)]
        return sorted(predicciones, key=lambda item: item[1], reverse=True)
