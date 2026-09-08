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


def generate_answer(
    query: str,
    top_k: int = TOP_K_PROVISIONAL,
    client: anthropic.Anthropic = None,
) -> dict:
    """
    Pipeline completo de una consulta: retrieval (search.py) + generación
    (Claude), usando el prompt DRAFT de prompts_draft.py.

    top_k se reexpone aquí con el mismo default provisional de search.py
    (importado, no redefinido, para que no puedan quedar desincronizados)
    porque quien genera una respuesta necesita poder controlar cuánto
    contexto se recupera, igual que quien solo busca.

    Devuelve un dict con la respuesta y los chunks usados como contexto
    (para poder mostrar las fuentes en la interfaz más adelante, o
    depurar por qué una respuesta salió mal), en vez de solo el string de
    respuesta.
    """
    if client is None:
        client = anthropic.Anthropic()

    chunks = semantic_search(query, top_k=top_k)
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
