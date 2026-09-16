"""
generation.py
Fase 6 del pipeline RAG: generación de respuestas con la API de Claude,
usando el contexto recuperado por search.py.

El cliente de Anthropic lee la API key de la variable de entorno
ANTHROPIC_API_KEY automáticamente (comportamiento estándar del SDK, no
hace falta pasarla a mano) - evita tener la key en el código o en config.py.
"""

import anthropic

from config import ANTHROPIC_MODEL_NAME
from prompts_draft import RAG_SYSTEM_PROMPT_DRAFT, build_user_message_draft
from search import TOP_K_PROVISIONAL, semantic_search


def generate_answer_from_chunks(
    query: str,
    chunks: list[dict],
    client: anthropic.Anthropic = None,
) -> dict:
    """
    Llama a Claude con chunks YA recuperados (usa el prompt DRAFT de
    prompts_draft.py). Separada de generate_answer() para que quien ya hizo
    el retrieval (ej. para mostrar las fuentes en una interfaz antes de
    esperar la respuesta generada) no tenga que repetirlo: cada llamada a
    semantic_search recarga el modelo de embeddings si no se le pasa uno
    cacheado, así que evitar una segunda búsqueda redundante no es solo
    prolijidad, ahorra una re-embebida de la consulta.
    """
    if client is None:
        client = anthropic.Anthropic()

    mensaje_usuario = build_user_message_draft(query, chunks)

    respuesta = client.messages.create(
        model=ANTHROPIC_MODEL_NAME,
        max_tokens=1024,
        system=RAG_SYSTEM_PROMPT_DRAFT,
        messages=[{"role": "user", "content": mensaje_usuario}],
    )

    return {
        "respuesta": respuesta.content[0].text,
        "chunks_usados": chunks,
    }


def generate_answer(
    query: str,
    top_k: int = TOP_K_PROVISIONAL,
    collection=None,
    model=None,
    client: anthropic.Anthropic = None,
) -> dict:
    """
    Pipeline completo de una consulta: retrieval (search.py) + generación
    (generate_answer_from_chunks).

    top_k se reexpone aquí con el mismo default provisional de search.py
    (importado, no redefinido, para que no puedan quedar desincronizados)
    porque quien genera una respuesta necesita poder controlar cuánto
    contexto se recupera, igual que quien solo busca.

    `collection` y `model` se reenvían tal cual a semantic_search (ver ahí
    el porqué de inyectarlos: evitar recargar el modelo de embeddings o
    reabrir ChromaDB en cada llamada, clave para un uso interactivo como
    la interfaz de Streamlit).
    """
    chunks = semantic_search(query, top_k=top_k, collection=collection, model=model)
    return generate_answer_from_chunks(query, chunks, client=client)


def main():
    import sys

    query = " ".join(sys.argv[1:]) or "¿Qué es un modelo de lenguaje?"
    resultado = generate_answer(query)

    print(f"Pregunta: {query}\n")
    print(f"Respuesta:\n{resultado['respuesta']}\n")
    print("Fuentes usadas:")
    for c in resultado["chunks_usados"]:
        print(f"  - {c['pdf_origen']} sección {c['numero_seccion']} ({c['titulo_seccion']})")


if __name__ == "__main__":
    main()
