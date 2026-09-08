"""
config.py
Constantes compartidas por todo el pipeline de embeddings -> retrieval -> generación.

Decisión de diseño: centralizar rutas y nombres en un único módulo en vez de
repetir literales (paths, nombre del modelo, nombre de la colección) en cada
script. Así, si cambiamos de modelo de embeddings o de ubicación de datos,
se edita en un solo sitio y no se corre el riesgo de que embeddings.py y
vector_store.py queden desincronizados (ej. indexar con un modelo y buscar
con otro, lo cual rompería la búsqueda semántica de forma silenciosa: los
vectores comparables tienen que venir del mismo espacio vectorial).
"""

from pathlib import Path

# --- Rutas -----------------------------------------------------------------

# Raíz del proyecto = carpeta padre de src/. Se calcula en vez de asumir un
# cwd fijo, para que los scripts funcionen igual si se invocan como
# `python src/embeddings.py` o como `python -m src.embeddings` desde otra carpeta.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CORPUS_PATH = DATA_PROCESSED_DIR / "corpus.json"

# Artefactos generados por embeddings.py (fase 1). Se guardan por separado
# de corpus.json en vez de modificarlo in-place: mantiene el corpus original
# (ya validado y cerrado, según instrucciones) intacto y de solo lectura, y
# permite regenerar embeddings sin pasar de nuevo por el pipeline de chunking.
EMBEDDINGS_PATH = DATA_PROCESSED_DIR / "chunk_embeddings.npy"
RECORDS_PATH = DATA_PROCESSED_DIR / "chunk_records.json"

# ChromaDB en modo persistente local (no un servidor separado): para un
# proyecto de aprendizaje de un solo usuario, un cliente embebido que
# escribe a disco evita levantar y mantener un proceso servidor aparte.
CHROMA_DB_DIR = DATA_PROCESSED_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "rag_bootcamp_chunks"

# --- Modelo de embeddings ----------------------------------------------------

# Elegido fuera de este proyecto (contexto del bootcamp): multilingüe porque
# el corpus y las preguntas de uso real son en español, y "MiniLM" porque a
# 420MB / 384 dims corre cómodo en CPU sin GPU dedicada, con una pérdida de
# calidad aceptable frente a modelos más grandes para este caso de uso de
# aprendizaje.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# --- Generación --------------------------------------------------------------

ANTHROPIC_MODEL_NAME = "claude-sonnet-5"
