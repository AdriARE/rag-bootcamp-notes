"""
embeddings.py
Fase 3 del pipeline RAG: generación de embeddings para los chunks de
corpus.json (fase de chunking, ya cerrada) y persistencia en un formato
listo para indexar en ChromaDB.

Por qué separar "generar embeddings" (este módulo) de "indexar en ChromaDB"
(vector_store.py) en dos pasos en vez de uno:
- Generar los embeddings es la parte lenta y determinista (carga del modelo
  de ~420MB, inferencia sobre 929 chunks). Indexar es rápido y se puede
  rehacer muchas veces (probar otra métrica de distancia, recrear la
  colección, migrar de Chroma a otro vector store) sin tener que volver a
  pasar el corpus por el modelo cada vez.
- Permite inspeccionar/depurar los vectores de forma independiente (ej.
  verificar dimensión, detectar NaNs) antes de que lleguen a la base de
  vectores.
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CORPUS_PATH,
    EMBEDDINGS_PATH,
    EMBEDDING_MODEL_NAME,
    RECORDS_PATH,
)

# Campos que corpus.json debe traer para cada chunk. "id" es el identificador
# original generado en la fase de chunking/exportación; se conserva como
# trazabilidad pero, como se explica en build_chunk_id, no se usa tal cual
# como identificador en ChromaDB.
CAMPOS_REQUERIDOS = {
    "id",
    "pdf_origen",
    "numero_seccion",
    "titulo_seccion",
    "parte",
    "total_partes",
    "texto",
}


def load_corpus(path=CORPUS_PATH) -> list[dict]:
    """Carga corpus.json y valida que cada chunk tenga los campos esperados."""
    with open(path, encoding="utf-8") as f:
        corpus = json.load(f)

    for i, chunk in enumerate(corpus):
        faltantes = CAMPOS_REQUERIDOS - chunk.keys()
        if faltantes:
            raise ValueError(f"Chunk en posición {i} sin campos {faltantes}: {chunk}")

    return corpus


def build_chunk_id(chunk: dict) -> str:
    """
    Construye el identificador único de chunk que se usará como ID en ChromaDB.

    Se reutiliza directamente el campo "id" de corpus.json (ej. "chunk_0855"),
    en vez de reconstruir un ID a partir de pdf_origen + numero_seccion + parte.

    Ese esquema de tres campos SE INTENTÓ primero, asumiendo que era único por
    construcción del pipeline de chunking - pero resultó ser falso: dentro de
    un mismo PDF, numero_seccion también se repite cuando distintos bloques
    reinician su propia numeración interna. Ejemplo real (Clase 4, SQL): las
    secciones "Bases de datos", "CREATE", "INSERT", "GRANT" y "START
    TRANSACTION" tienen numero_seccion == "1" en el mismo documento, porque
    cada bloque (DDL/DML/DCL/TCL) numera sus sentencias desde 1. Confirmado
    con datos: 155 colisiones de ID al generar embeddings sobre el corpus real.

    El campo "id" de corpus.json sí es fiable aquí porque se genera con un
    contador secuencial global (id_counter) en build_corpus.py, único por
    construcción sobre el corpus completo - no es un campo externo de origen
    desconocido, es parte del mismo pipeline.
    """
    return chunk["id"]


def validate_unique_ids(chunk_ids: list[str]) -> None:
    """
    Falla rápido y explícito si hay colisiones de ID, en vez de dejar que
    ChromaDB sobrescriba silenciosamente un chunk con otro más adelante
    (ChromaDB trata collection.add con un ID repetido como upsert, no como
    error) o de reportar de menos: por eso este es una de las primeras cosas
    que se ejecuta, antes de gastar tiempo generando embeddings.
    """
    vistos = {}
    duplicados = []
    for idx, chunk_id in enumerate(chunk_ids):
        if chunk_id in vistos:
            duplicados.append((chunk_id, vistos[chunk_id], idx))
        else:
            vistos[chunk_id] = idx

    if duplicados:
        detalle = "\n".join(
            f"  '{cid}' en posiciones {i1} y {i2}" for cid, i1, i2 in duplicados
        )
        raise ValueError(
            f"IDs de chunk duplicados ({len(duplicados)}):\n{detalle}\n"
            "Esto indica que la combinación pdf_origen+numero_seccion+parte "
            "ya no es única - revisar la fase de chunking."
        )


def generate_embeddings(
    textos: list[str],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Genera un embedding por texto de entrada, en el mismo orden.

    normalize_embeddings=True: sentence-transformers normaliza cada vector a
    norma 1. Con vectores normalizados, similitud coseno == producto punto,
    lo cual hace que el resultado sea consistente sin importar si el vector
    store de turno usa "cosine" o "dot product" como métrica configurada -
    evita una fuente sutil de bugs si en el futuro se cambia de motor de
    búsqueda vectorial.

    batch_size=32 es el valor por defecto de sentence-transformers y un
    punto de partida razonable en CPU: no es una decisión crítica del
    pipeline (no afecta el resultado, solo la velocidad), así que no se
    documenta como "provisional a revisar" igual que sí se hace con el
    top-k de búsqueda.
    """
    modelo = SentenceTransformer(model_name)
    embeddings = modelo.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def save_embeddings(
    chunk_ids: list[str],
    corpus: list[dict],
    embeddings: np.ndarray,
    embeddings_path=EMBEDDINGS_PATH,
    records_path=RECORDS_PATH,
) -> None:
    """
    Guarda dos artefactos alineados por índice de fila:
    - embeddings_path: array numpy (N, EMBEDDING_DIM) en formato binario .npy.
      Se elige .npy sobre JSON para los vectores porque son datos numéricos
      densos (929 x 384 floats): JSON los representaría como texto, con un
      costo de espacio y de parseo mucho mayor sin ningún beneficio de
      legibilidad (nadie va a leer un vector de 384 dimensiones a ojo).
    - records_path: JSON con la metadata + texto de cada chunk, en el mismo
      orden que las filas del .npy. JSON sí tiene sentido aquí porque son
      datos heterogéneos (strings, ints) pensados para ser leídos por
      vector_store.py e inspeccionados a mano si hace falta depurar.
    """
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)

    records = [
        {
            "chroma_id": chunk_id,
            "id_original": chunk["id"],
            "pdf_origen": chunk["pdf_origen"],
            "numero_seccion": chunk["numero_seccion"],
            "titulo_seccion": chunk["titulo_seccion"],
            "parte": chunk["parte"],
            "total_partes": chunk["total_partes"],
            "texto": chunk["texto"],
        }
        for chunk_id, chunk in zip(chunk_ids, corpus)
    ]
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    print(f"Cargando corpus desde {CORPUS_PATH} ...")
    corpus = load_corpus()
    print(f"  {len(corpus)} chunks cargados.")

    chunk_ids = [build_chunk_id(c) for c in corpus]
    validate_unique_ids(chunk_ids)
    print("  IDs de chunk verificados: todos únicos.")

    print(f"Generando embeddings con '{EMBEDDING_MODEL_NAME}' ...")
    textos = [c["texto"] for c in corpus]
    embeddings = generate_embeddings(textos)
    print(f"  Embeddings generados: shape={embeddings.shape}")

    save_embeddings(chunk_ids, corpus, embeddings)
    print(f"Guardado en:\n  {EMBEDDINGS_PATH}\n  {RECORDS_PATH}")


if __name__ == "__main__":
    main()
