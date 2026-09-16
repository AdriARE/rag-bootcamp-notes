"""
app.py
Interfaz Streamlit del pipeline RAG: retrieval semántico sobre ChromaDB +
generación con Claude.

Esta app asume que el índice de ChromaDB ya existe (generado corriendo
`python src/embeddings.py` y luego `python src/vector_store.py`). No genera
embeddings ni indexa desde acá a propósito: eso es una operación de varios
minutos que además solo hace falta rehacer cuando cambia el corpus o el
modelo, no en cada carga de página de una app web.

El modelo de embeddings y el cliente de ChromaDB se cargan una sola vez por
proceso con st.cache_resource: recargar el modelo (~420MB) en cada pregunta
haría la interfaz inutilizable. search.py y generation.py ya estaban
preparados para esto (aceptan `collection`/`model` inyectados en vez de
resolverlos ellos mismos en cada llamada).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import anthropic
import streamlit as st

from generation import generate_answer_from_chunks
from search import TOP_K_PROVISIONAL, get_embedding_model, semantic_search
from vector_store import get_client, get_collection

st.set_page_config(page_title="RAG Bootcamp Notes", page_icon="📚")


@st.cache_resource(show_spinner="Cargando modelo de embeddings (paraphrase-multilingual-MiniLM-L12-v2)...")
def load_model():
    return get_embedding_model()


@st.cache_resource(show_spinner="Conectando a ChromaDB...")
def load_collection():
    return get_collection(get_client())


st.title("📚 Asistente RAG - Bootcamp de Ciencia de Datos")
st.caption(
    "Busca en el material de las clases con retrieval semántico y genera "
    "una respuesta con Claude a partir de lo encontrado."
)

model = load_model()
collection = load_collection()
n_chunks = collection.count()

if n_chunks == 0:
    st.warning(
        "La colección de ChromaDB está vacía. Corré `python src/embeddings.py` "
        "y luego `python src/vector_store.py` para generar e indexar los "
        "embeddings del corpus antes de usar esta interfaz.",
        icon="⚠️",
    )
else:
    st.caption(f"Colección indexada: {n_chunks} chunks.")

# top_k configurable desde la UI, con TOP_K_PROVISIONAL (src/search.py) como
# valor inicial. Se expone como control del usuario en vez de una constante
# fija precisamente porque ese default es una heurística sin validar contra
# el corpus real (ver el comentario en search.py) - la persona que use la
# app puede ajustarlo según lo que observe, sin tocar código.
top_k = st.sidebar.slider(
    "Fragmentos a recuperar (top-k)",
    min_value=1,
    max_value=15,
    value=TOP_K_PROVISIONAL,
)
st.sidebar.caption(
    f"El valor inicial ({TOP_K_PROVISIONAL}) es provisional - ver "
    "TOP_K_PROVISIONAL en src/search.py."
)

query = st.text_input("Tu pregunta sobre el material del bootcamp:")
preguntar = st.button("Preguntar", disabled=(n_chunks == 0))

if preguntar and query:
    with st.spinner("Buscando fragmentos relevantes..."):
        chunks = semantic_search(query, top_k=top_k, collection=collection, model=model)

    with st.expander(f"Fragmentos recuperados ({len(chunks)})", expanded=False):
        for i, c in enumerate(chunks, start=1):
            st.markdown(
                f"**#{i} — {c['pdf_origen']} · sección {c['numero_seccion']} "
                f"({c['titulo_seccion']})** · distancia={c['distancia']:.4f}"
            )
            st.text(c["texto"][:500])

    with st.spinner("Generando respuesta con Claude..."):
        try:
            resultado = generate_answer_from_chunks(query, chunks)
        except anthropic.AnthropicError as e:
            # Cubre tanto la falta de ANTHROPIC_API_KEY (el cliente falla al
            # armar el request, no al construirse) como errores de la API
            # (rate limit, modelo inválido, etc.) - en ambos casos la app no
            # debe romperse con un traceback, sino explicar qué pasó.
            st.error(f"No se pudo generar la respuesta con la API de Claude: {e}")
        else:
            st.subheader("Respuesta")
            st.write(resultado["respuesta"])
elif preguntar and not query:
    st.info("Escribí una pregunta antes de buscar.")
