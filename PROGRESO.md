# Progreso: pipeline embeddings → retrieval → generación

## Qué se hizo

Se implementó el pipeline completo pedido, en 5 módulos nuevos dentro de `src/`,
pensado para correrse por CLI (sin interfaz todavía):

1. **`src/config.py`** — constantes compartidas (rutas, nombre del modelo de
   embeddings, nombre de la colección de Chroma, modelo de Claude). Un único
   punto de verdad para que embeddings/indexado/búsqueda no se desincronicen.

2. **`src/embeddings.py`** (Fase 3 - genera embeddings)
   - Carga `data/processed/corpus.json` y valida que cada chunk tenga los
     campos esperados (`id`, `pdf_origen`, `numero_seccion`, `titulo_seccion`,
     `parte`, `total_partes`, `texto`).
   - Genera embeddings con `paraphrase-multilingual-MiniLM-L12-v2`
     (`normalize_embeddings=True`, para que coseno == producto punto).
   - Guarda dos artefactos alineados por índice: `chunk_embeddings.npy`
     (vectores, formato binario) y `chunk_records.json` (metadata + texto).
     Se separan de `corpus.json` (que queda intacto) para poder regenerar
     embeddings sin re-tocar la fase de chunking ya cerrada.

3. **`src/vector_store.py`** (Fases 4 y 5 - ChromaDB + indexado)
   - Cliente `PersistentClient` local (sin servidor aparte).
   - Colección con métrica `cosine` explícita.
   - Metadata por chunk: `pdf_origen`, `numero_seccion`, `titulo_seccion`,
     `parte`, `total_partes`, `id_original` (el `id` de corpus.json,
     conservado solo por trazabilidad).
   - Indexa con `upsert` en lotes de 500 (idempotente: re-correrlo no falla
     ni duplica).

4. **`src/search.py`** (Fase 6 - búsqueda semántica)
   - `semantic_search(query, top_k=..., collection=None, model=None)`.
   - `top_k` es un parámetro explícito, no un valor fijo dentro de la
     función.

5. **`src/prompts_draft.py` + `src/generation.py`** (Fase 7 - generación)
   - Prompt en un archivo separado y nombrado `prompts_draft.py`, con la
     constante `RAG_SYSTEM_PROMPT_DRAFT` — marcado como DRAFT a propósito.
   - `generate_answer(query, top_k=...)` encadena retrieval + llamada a
     Claude (`ANTHROPIC_MODEL_NAME` en `config.py`) y devuelve la respuesta
     junto con los chunks usados (para poder mostrar fuentes más adelante).

Cada módulo tiene un bloque `if __name__ == "__main__":` para poder correrse
como script suelto, ej.:
```
python src/embeddings.py
python src/vector_store.py
python src/search.py "¿qué es un embedding?"
python src/generation.py "¿qué es un embedding?"
```

`requirements.txt` se completó con: `sentence-transformers`, `chromadb`,
`anthropic`, `numpy` (+ `pdfplumber`, ya usado por `extraction.py`).

## Decisiones de diseño (para poder explicarlas en la entrevista)

- **ID único de chunk = `pdf_origen::numero_seccion::parte`, no el `id` que
  ya trae corpus.json.** `numero_seccion` se repite entre PDFs distintos (lo
  sabíamos de antes); además, cuando una sección se divide en varias partes
  (por `split_section` en `chunking.py`), varios chunks comparten
  `pdf_origen` + `numero_seccion` y solo difieren en `parte`. Esa
  combinación de tres campos sí es única *por construcción* del pipeline de
  chunking, así que puedo razonar sobre su unicidad en vez de confiar
  ciegamente en un campo externo. El `id` original se conserva como metadata
  (`id_original`) por trazabilidad, no se descarta.
- **Falla rápido ante colisiones de ID** (`validate_unique_ids`), antes de
  gastar tiempo generando embeddings. Motivo: `collection.add`/`upsert` de
  Chroma no avisa si un ID se repite, simplemente sobrescribe — preferí un
  error explícito a un bug silencioso.
- **Embeddings y metadata se guardan en un artefacto propio, separado de
  `corpus.json` y del indexado en Chroma.** Generar embeddings es la parte
  lenta (carga de modelo + inferencia); indexar es rápido y se puede rehacer
  muchas veces (otra métrica, recrear la colección) sin volver a pasar el
  corpus por el modelo.
- **`normalize_embeddings=True` + métrica `cosine` en Chroma.** Con vectores
  normalizados, coseno = producto punto, así que el resultado no depende de
  qué métrica exponga el vector store — reduce una fuente de bugs si algún
  día se cambia de motor.
- **Se pasan embeddings explícitos a Chroma en vez de dejar que use su
  embedding function por defecto.** El default de Chroma es un modelo en
  inglés (`all-MiniLM-L6-v2`), distinto al elegido para este proyecto
  (multilingüe, por el corpus en español). Pasar el vector ya calculado
  tanto al indexar como al buscar garantiza que ambos lados de la
  comparación vienen del mismo modelo.
- **`upsert` en vez de `add` al indexar.** Permite re-correr el indexado
  tras regenerar embeddings sin que falle por IDs duplicados.
- **`top_k` como parámetro explícito, con default *provisional* documentado
  en el propio código (`TOP_K_PROVISIONAL = 5` en `search.py`).** Es un
  punto de partida heurístico (tamaño de contexto razonable con chunks de
  ~1000 caracteres), *no* validado contra este corpus con una evaluación de
  retrieval real. Lo dejé señalado explícitamente para que lo revises y, si
  hace falta, lo ajustes con datos.
- **Prompt de generación aislado en `prompts_draft.py`, nombrado DRAFT.**
  Es la pieza más subjetiva de todo el pipeline (tono, formato de citación,
  qué tan estricto ser con "solo el contexto dado"). Lo dejé funcional pero
  claramente marcado para que lo revises antes de darlo por definitivo.

## Cómo se probó

Este entorno remoto es un contenedor efímero: no tenía `data/processed/corpus.json`
(la carpeta está en `.gitignore`, es un artefacto local tuyo) ni acceso de red a
`huggingface.co` (bloqueado por la política de red de la sesión — confirmado
como bloqueo de política, no un error transitorio). Por lo tanto **no se
generaron embeddings reales de los 929 chunks en esta sesión**, y el corpus
real nunca estuvo ni pasó por este entorno.

Para no entregar código sin probar, sí se validó el pipeline completo con:
- Un corpus sintético de 8 chunks (mismo esquema que el real, incluyendo a
  propósito `numero_seccion` repetido entre PDFs y una sección dividida en
  `parte`s) para probar la construcción de IDs y la detección de colisiones.
- Un doble de `SentenceTransformer` que devuelve vectores aleatorios
  normalizados de 384 dimensiones (sin red), para poder correr
  `embeddings.py` → `vector_store.py` → `search.py` de punta a punta y
  verificar mecánica (formatos de guardado, indexado en lotes, esquema de
  metadata, forma de la respuesta de búsqueda, `top_k` configurable,
  `ValueError` en `top_k<=0`, construcción del prompt) sin depender del
  modelo real. **Esto no valida la calidad semántica de los resultados**
  (con vectores aleatorios el ranking no tiene significado), solo que el
  código no tiene bugs de plomería.
- La llamada real a la API de Claude (`generation.py`) no se probó: no había
  `ANTHROPIC_API_KEY` en esta sesión.

Ni el corpus sintético ni los artefactos generados quedaron en el repo
(vive todo bajo `data/processed/`, gitignorado, y se limpió al terminar).

## Pendiente de tu revisión

1. **Correr el pipeline real en tu máquina**, donde sí existen
   `data/processed/corpus.json` y acceso a Hugging Face:
   ```
   pip install -r requirements.txt
   python src/embeddings.py      # genera embeddings de los 929 chunks reales
   python src/vector_store.py    # indexa en ChromaDB local
   python src/search.py "una pregunta de prueba"
   ```
   y confirmar que la búsqueda semántica trae resultados sensatos con datos
   reales (esto no se pudo verificar aquí).
2. **Revisar `TOP_K_PROVISIONAL` en `src/search.py`** (actualmente 5) —
   ajustar si hace falta, idealmente con algún criterio empírico
   (ej. mirar unos cuantos resultados con distintas preguntas reales del
   material).
3. **Revisar el prompt en `src/prompts_draft.py`** (`RAG_SYSTEM_PROMPT_DRAFT`
   y `build_user_message_draft`) — tono, formato de citación de fuentes, y
   qué tan estricto debe ser con "solo el contexto dado". Una vez aprobado,
   renombrar quitando "DRAFT" (y actualizar el import en `generation.py`).
4. **Probar `generation.py` con una `ANTHROPIC_API_KEY` real** — no se pudo
   probar la llamada efectiva a la API en esta sesión.
5. Nada de extracción/chunking se tocó (según lo pedido); tampoco se
   construyó la interfaz de Streamlit (`app.py` sigue vacío, a propósito).
