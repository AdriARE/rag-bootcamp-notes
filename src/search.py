"""
search.py
Fase 5 del pipeline RAG: búsqueda semántica sobre la colección de ChromaDB.

La consulta del usuario se embebe con el MISMO modelo usado para indexar
(paraphrase-multilingual-MiniLM-L12-v2). Si se usara un modelo distinto para
la consulta, los vectores no serían comparables aunque ChromaDB no dé
ningún error al respecto - simplemente devolvería resultados sin sentido
en silencio. Por eso get_embedding_model() vive aquí y se reutiliza el
mismo nombre de modelo definido en config.py.
"""

from sentence_transformers import SentenceTransformer

from config import CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME
from vector_store import get_client, get_collection

# --- top_k: valor por defecto PROVISIONAL, pendiente de revisión ------------
#
# 5 es un punto de partida común en sistemas RAG (ni tan bajo que se pierda
# el fragmento relevante si no quedó en el top-1/2, ni tan alto que se
# diluya el contexto con chunks poco relevantes o se dispare el costo/latencia
# de la llamada a Claude). Con chunks de hasta ~1000 caracteres, 5 resultados
# son como mucho ~5000 caracteres de contexto, un tamaño manejable.
#
# Es una elección heurística, no medida contra este corpus todavía: no hay
# evaluación de retrieval (ej. precision@k / recall@k sobre un set de
# preguntas con respuesta esperada) que la respalde. Se deja explícito aquí
# para que se revise y, si hace falta, se ajuste con datos reales.
TOP_K_PROVISIONAL = 5


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Sin cache/singleton a propósito: mantener este módulo simple para un
    pipeline de script/CLI. Si se reutiliza en un servicio de larga vida
    (ej. la futura interfaz de Streamlit), cargar el modelo una sola vez a
    nivel de módulo o de sesión evitaría recargar ~420MB en cada búsqueda.
    """
    return SentenceTransformer(model_name)


def semantic_search(
    query: str,
    top_k: int = TOP_K_PROVISIONAL,
    collection=None,
    model: SentenceTransformer = None,
) -> list[dict]:
    """
    Busca los `top_k` chunks más relevantes semánticamente para `query`.

    top_k es un parámetro explícito y configurable por diseño (no un valor
    fijo dentro de la función): distintos casos de uso (respuesta rápida vs.
    respuesta exhaustiva) necesitan distinto tamaño de contexto, y esa es una
    decisión que debe poder tomar quien llama a la función, no quedar
    escondida en su implementación. Ver TOP_K_PROVISIONAL más arriba para el
    razonamiento (y las limitaciones) del valor por defecto.

    `collection` y `model` son inyectables para no reabrir ChromaDB ni
    recargar el modelo de embeddings en cada llamada (ej. en un bucle de
    CLI interactivo o en tests); si no se pasan, se resuelven con los
    defaults del proyecto.

    Devuelve una lista de dicts, ordenada de más a menos relevante, con:
    texto, distancia (coseno; más bajo = más similar) y la metadata del
    chunk (pdf_origen, numero_seccion, titulo_seccion, parte, total_partes).
    """
    if top_k <= 0:
        raise ValueError(f"top_k debe ser positivo, se recibió {top_k}")

    if collection is None:
        collection = get_collection(get_client(), CHROMA_COLLECTION_NAME)
    if model is None:
        model = get_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    resultados = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
    )

    chunks_encontrados = []
    for texto, distancia, metadata in zip(
        resultados["documents"][0],
        resultados["distances"][0],
        resultados["metadatas"][0],
    ):
        chunks_encontrados.append({"texto": texto, "distancia": distancia, **metadata})

    return chunks_encontrados


def main():
    import sys

    query = " ".join(sys.argv[1:]) or "¿Qué es un modelo de lenguaje?"
    print(f"Consulta: {query!r} (top_k={TOP_K_PROVISIONAL}, valor provisional)\n")

    resultados = semantic_search(query, top_k=TOP_K_PROVISIONAL)
    for i, r in enumerate(resultados, start=1):
        print(f"--- #{i} | distancia={r['distancia']:.4f} | {r['pdf_origen']} "
              f"sección {r['numero_seccion']} - {r['titulo_seccion']} ---")
        print(r["texto"][:200].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
