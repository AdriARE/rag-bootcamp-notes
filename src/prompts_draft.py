"""
prompts_draft.py
Prompt de generación para la fase RAG con Claude.

Se aísla en su propio archivo, y se nombra explícitamente "DRAFT" (en la
constante y en el archivo), porque el prompt es la pieza más subjetiva y
más fácil de iterar de todo el pipeline: no hay una única forma "correcta"
de instruir al modelo, y conviene poder revisarlo/afinarlo sin tocar la
lógica de retrieval ni la de la llamada a la API.

Pendiente de revisión (no probado con casos reales todavía):
- Tono y nivel de detalle esperado en las respuestas.
- Formato exacto de la citación de fuente (¿sección y PDF, o solo PDF?).
- Qué tan estricto debe ser el "solo con el contexto dado" (¿se permite que
  Claude complemente con conocimiento general si el contexto es insuficiente,
  dejándolo explícito, o debe negarse a responder?).
"""

RAG_SYSTEM_PROMPT_DRAFT = """\
Eres un asistente que ayuda a repasar el contenido de un bootcamp de ciencia \
de datos. Respondes preguntas usando ÚNICAMENTE la información de los \
fragmentos de contexto que se te proporcionan a continuación, extraídos del \
material de las clases.

Reglas:
- Si el contexto no contiene información suficiente para responder, dilo \
explícitamente en vez de inventar o completar con conocimiento propio.
- Cuando uses un fragmento, cita de qué clase y sección viene (el nombre del \
PDF y el número/título de sección que se indican junto a cada fragmento).
- Responde en español, de forma clara y concisa.
"""


def build_user_message_draft(query: str, chunks: list[dict]) -> str:
    """
    Arma el mensaje de usuario: cada chunk recuperado con su metadata de
    procedencia (para que Claude pueda citar la fuente, ver reglas del
    system prompt) seguido de la pregunta real.

    Se pasa la metadata en texto plano junto a cada fragmento, en vez de
    como un campo estructurado aparte, porque el modelo consume todo el
    mensaje como texto de todas formas y esto simplifica la implementación
    inicial (DRAFT). Si en la revisión se decide que la citación debe ser
    más confiable/estructurada, una alternativa a considerar es pedirle a
    Claude que devuelva las fuentes en un formato fijo (ej. JSON) - se deja
    fuera de esta primera versión a propósito, para no sobre-diseñar el
    prompt antes de ver cómo se comporta el caso simple.
    """
    bloques_contexto = []
    for i, chunk in enumerate(chunks, start=1):
        etiqueta = f"{chunk['pdf_origen']} - sección {chunk['numero_seccion']} ({chunk['titulo_seccion']})"
        bloques_contexto.append(f"[Fragmento {i} | {etiqueta}]\n{chunk['texto']}")

    contexto = "\n\n".join(bloques_contexto)
    return f"{contexto}\n\n---\n\nPregunta: {query}"
