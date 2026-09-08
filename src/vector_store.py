"""
vector_store.py
Fase 4 del pipeline RAG: levantar ChromaDB local, definir el esquema de
metadata por chunk, e indexar los embeddings generados por embeddings.py.

Por qué pasamos los embeddings ya calculados en vez de dejar que ChromaDB
los calcule con su embedding function por defecto: el default de Chroma
(all-MiniLM-L6-v2) es un modelo en inglés distinto al elegido para este
proyecto (paraphrase-multilingual-MiniLM-L12-v2, necesario porque el corpus
y las preguntas son en español). Si no se especifica embedding_function y
se pasan embeddings explícitos tanto al indexar como al consultar, Chroma
nunca necesita invocar un modelo propio - controlamos nosotros qué modelo
genera cada vector, en ambos lados de la comparación.
"""

import json

import chromadb
import numpy as np

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    EMBEDDINGS_PATH,
    RECORDS_PATH,
)

# ChromaDB indexa en lotes; batches muy grandes pueden exceder el límite
# interno de SQLite que usa como backend. 500 es un margen conservador para
# 929 chunks (dos lotes) sin acercarse a ese límite.
INDEX_BATCH_SIZE = 500


def get_client(persist_dir=CHROMA_DB_DIR) -> chromadb.ClientAPI:
    """
    Cliente persistente local (PersistentClient), no un servidor Chroma
    aparte: para uso de un solo proceso/usuario en este proyecto de
    aprendizaje, evita la complejidad operativa de levantar y mantener un
    servidor HTTP de Chroma solo para escribir y leer del mismo disco.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_collection(client: chromadb.ClientAPI, name: str = CHROMA_COLLECTION_NAME):
    """
    Esquema de metadata por chunk (los campos que ya trae corpus.json, menos
    el texto -- que ChromaDB guarda aparte como "document"):
      - pdf_origen (str): PDF de origen; permite filtrar/citar por clase.
      - numero_seccion (str): número de sección dentro de ese PDF (ej. "2.1").
        Por sí solo NO es único entre PDFs (se repite), por eso no se usa
        como ID de Chroma - ver build_chunk_id en embeddings.py.
      - titulo_seccion (str): título legible de la sección, útil para citar
        la fuente en la respuesta generada sin tener que ir a buscarlo.
      - parte (int) / total_partes (int): posición del chunk dentro de su
        sección cuando esta se dividió en varios fragmentos. Junto con
        pdf_origen y numero_seccion, es lo que hace único a cada chunk.
      - id_original (str): el "id" tal cual venía en corpus.json, conservado
        por trazabilidad aunque no se use como ID de Chroma.

    "hnsw:space": "cosine" -- coseno es la métrica estándar para embeddings
    de sentence-transformers (con la que fueron entrenados/evaluados estos
    modelos), y es invariante a la magnitud del vector, que es justo lo que
    nos interesa al comparar significado semántico y no longitud de texto.
    """
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def load_index_inputs(embeddings_path=EMBEDDINGS_PATH, records_path=RECORDS_PATH):
    """Carga los artefactos producidos por embeddings.py, ya alineados por fila."""
    embeddings = np.load(embeddings_path)
    with open(records_path, encoding="utf-8") as f:
        records = json.load(f)

    if len(records) != embeddings.shape[0]:
        raise ValueError(
            f"Desalineación: {len(records)} records vs {embeddings.shape[0]} embeddings."
        )

    return records, embeddings


def index_records(collection, records: list[dict], embeddings: np.ndarray) -> None:
    """
    Indexa records+embeddings en la colección, en lotes de INDEX_BATCH_SIZE.

    Se usa `upsert` en vez de `add`: hace que volver a correr este script
    sobre el mismo corpus (ej. tras regenerar embeddings) actualice los
    chunks existentes por ID en vez de fallar por IDs duplicados o crear
    entradas fantasma - importante durante desarrollo, cuando el corpus o
    el modelo de embeddings pueden cambiar varias veces.
    """
    n = len(records)
    for start in range(0, n, INDEX_BATCH_SIZE):
        end = min(start + INDEX_BATCH_SIZE, n)
        batch = records[start:end]
        collection.upsert(
            ids=[r["chroma_id"] for r in batch],
            embeddings=embeddings[start:end].tolist(),
            documents=[r["texto"] for r in batch],
            metadatas=[
                {
                    "pdf_origen": r["pdf_origen"],
                    "numero_seccion": r["numero_seccion"],
                    "titulo_seccion": r["titulo_seccion"],
                    "parte": r["parte"],
                    "total_partes": r["total_partes"],
                    "id_original": r["id_original"],
                }
                for r in batch
            ],
        )


def main():
    print(f"Abriendo ChromaDB persistente en {CHROMA_DB_DIR} ...")
    client = get_client()
    collection = get_collection(client)

    print(f"Cargando embeddings desde {EMBEDDINGS_PATH} ...")
    records, embeddings = load_index_inputs()
    print(f"  {len(records)} chunks a indexar.")

    index_records(collection, records, embeddings)
    print(f"Colección '{collection.name}' ahora tiene {collection.count()} chunks indexados.")


if __name__ == "__main__":
    main()
