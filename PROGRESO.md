# Progreso: pipeline embeddings → retrieval → generación

## Qué se hizo

Se implementó el pipeline completo pedido, en 5 módulos nuevos dentro de `src/`
más una interfaz Streamlit (`app.py`) que los une:

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
   - `generation.py` separa dos funciones: `generate_answer_from_chunks(query,
     chunks)` (solo llama a Claude con chunks ya recuperados) y
     `generate_answer(query, top_k=..., collection=None, model=None)`
     (retrieval + generación). La separación existe para que quien ya hizo
     el retrieval (típicamente para mostrar las fuentes antes de esperar la
     respuesta) no tenga que repetirlo innecesariamente.

6. **`app.py`** (interfaz Streamlit)
   - Une todo el pipeline en una UI: pregunta → fragmentos recuperados
     (en un expander, con distancia y metadata) → respuesta generada.
   - El modelo de embeddings y el cliente de ChromaDB se cargan una sola
     vez por proceso con `st.cache_resource`, no en cada pregunta (recargar
     ~420MB de modelo por interacción haría la app inusable). Por eso
     `search.py`/`generation.py` ya aceptaban `collection`/`model`
     inyectados desde el principio.
   - `top_k` es un slider en la barra lateral, con `TOP_K_PROVISIONAL`
     (de `search.py`) como valor inicial — resuelve en la práctica el
     "valor provisional pendiente de revisión": en vez de una sola
     constante fija en el código, quien use la app lo puede ajustar sin
     tocar nada.
   - Si `ANTHROPIC_API_KEY` no está seteada, la app lo avisa de entrada
     (en vez de esperar a que falle) y la búsqueda semántica sigue
     funcionando igual — la generación es la única parte que depende de
     la key.

Cada módulo de `src/` además tiene un bloque `if __name__ == "__main__":`
para poder correrse como script suelto, ej.:
```
python src/embeddings.py
python src/vector_store.py
python src/search.py "¿qué es un embedding?"
python src/generation.py "¿qué es un embedding?"
streamlit run app.py
```

`requirements.txt` se completó con: `sentence-transformers`, `chromadb`,
`anthropic`, `numpy`, `streamlit` (+ `pdfplumber`, ya usado por
`extraction.py`).

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
- **`app.py` no genera embeddings ni indexa.** Solo lee una colección de
  Chroma que ya existe. Indexar es una operación de varios minutos que
  además solo hace falta rehacer cuando cambia el corpus o el modelo, no en
  cada carga de página de una app web — mezclar eso en la UI sería lento y
  confuso (¿qué pasa si dos personas abren la app a la vez mientras indexa?).
- **`top_k` se resolvió como control de UI (slider), no como decisión
  final de código.** Es la forma más directa de resolver "es provisional,
  hay que poder ajustarlo": en vez de perseguir un valor "correcto" sin
  tener el corpus real para medirlo, se le dio el control a quien usa la
  app, con el valor heurístico como punto de partida visible.
- **Se detectó (probando la app sin `ANTHROPIC_API_KEY`) que el SDK de
  Anthropic no falla con una subclase de `anthropic.AnthropicError` cuando
  falta la key, sino con un `TypeError` genérico al armar los headers del
  request.** `app.py` primero chequea la variable de entorno de forma
  explícita (mensaje claro, antes de intentar nada) y además captura
  `Exception` en general alrededor de la llamada a la API (no solo
  `AnthropicError`) para no depender de conocer cada tipo de excepción que
  el SDK pueda lanzar en ese límite de sistema externo.

## Cómo se probó

Este entorno remoto es un contenedor efímero: no tenía `data/processed/corpus.json`
(la carpeta está en `.gitignore`, es un artefacto local tuyo) ni acceso de red a
`huggingface.co` (bloqueado por la política de red de la sesión — confirmado
como bloqueo de política, no un error transitorio). Por lo tanto **no se
generaron embeddings reales de los 929 chunks en esta sesión**, y el corpus
real nunca estuvo ni pasó por este entorno.

Para no entregar código sin probar, sí se validó el pipeline completo (CLI
+ interfaz) con:
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
- `app.py` se probó con `streamlit.testing.v1.AppTest` (la forma oficial de
  testear apps Streamlit sin navegador, corriendo el script en el mismo
  proceso), con el mismo doble de modelo y el mismo corpus sintético:
  carga sin excepciones, detecta la colección indexada, el slider de
  `top_k` está presente y funciona (probado con dos valores distintos,
  confirmando que trae exactamente esa cantidad de fragmentos), el botón
  "Preguntar" no rompe la app, y - sin `ANTHROPIC_API_KEY` en esta sesión -
  el error de generación se muestra con `st.error()` en vez de tirar un
  traceback (este último caso reveló y permitió corregir el bug descrito
  arriba sobre el `TypeError` del SDK).
- La llamada real a la API de Claude nunca devolvió una respuesta genuina:
  no había `ANTHROPIC_API_KEY` en esta sesión, así que solo se validó el
  camino de error (que sí es un resultado válido de la prueba, no un
  sustituto de probarla con una key real).

Ni el corpus sintético ni los artefactos generados quedaron en el repo
(vive todo bajo `data/processed/`, gitignorado, y se limpió al terminar;
tampoco el entorno virtual de prueba, `.venv/`).

## Pendiente de tu revisión

1. **Correr el pipeline real en tu máquina**, donde sí existen
   `data/processed/corpus.json` y acceso a Hugging Face:
   ```
   pip install -r requirements.txt
   python src/embeddings.py      # genera embeddings de los 929 chunks reales
   python src/vector_store.py    # indexa en ChromaDB local
   python src/search.py "una pregunta de prueba"
   export ANTHROPIC_API_KEY=...
   streamlit run app.py          # o: python src/generation.py "una pregunta"
   ```
   y confirmar que la búsqueda semántica trae resultados sensatos con datos
   reales (esto no se pudo verificar aquí, solo con vectores aleatorios).
2. **Revisar `TOP_K_PROVISIONAL` en `src/search.py`** (actualmente 5, y
   valor inicial del slider en la app). Ya es ajustable sin tocar código
   desde la UI, pero ese número de partida sigue sin validar contra el
   corpus real — si al usar la app con preguntas reales notás que hace
   falta otro valor por defecto, cambialo ahí.
3. **Revisar el prompt en `src/prompts_draft.py`** (`RAG_SYSTEM_PROMPT_DRAFT`
   y `build_user_message_draft`) — tono, formato de citación de fuentes, y
   qué tan estricto debe ser con "solo el contexto dado". Una vez aprobado,
   renombrar quitando "DRAFT" (y actualizar los imports en `generation.py`
   y `app.py` si corresponde).
4. **Probar la generación con una `ANTHROPIC_API_KEY` real** — en esta
   sesión solo se pudo validar que la app maneja bien la *ausencia* de key
   (mensaje claro en vez de un traceback), no que una respuesta real de
   Claude sea la esperada con el prompt actual.
5. Nada de extracción/chunking se tocó (según lo pedido).
