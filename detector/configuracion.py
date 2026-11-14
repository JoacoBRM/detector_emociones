from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_MODELO = RAIZ / "modelo_emociones_distilbert"
RUTA_EJECUCIONES = RAIZ / ".ejecuciones"
RUTA_RESULTADOS = RAIZ / "resultados"

DATASET = "dair-ai/emotion"
DATASET_CONFIG = "split"
DATASET_REVISION = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
MODELO_BASE = "distilbert/distilbert-base-uncased"
MODELO_BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"

ETIQUETAS_EN = ("sadness", "joy", "love", "anger", "fear", "surprise")
EMOCIONES = ("Tristeza", "Alegría", "Amor", "Ira", "Miedo", "Sorpresa")
ID_A_EMOCION = dict(enumerate(EMOCIONES))
EMOCION_A_ID = {emocion: indice for indice, emocion in ID_A_EMOCION.items()}
IDS = list(ID_A_EMOCION)
SEMILLA = 42
MAX_TOKENS = 64
