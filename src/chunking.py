"""
chunking.py
Fase 2 del pipeline RAG: detección de headers de sección y generación
de chunks a partir de ellos.

Pipeline: texto crudo -> limpieza de iconos -> detección de candidatos a
header -> clasificación índice/real -> extracción de secciones -> chunks
con overlap.
"""

import re
from dataclasses import dataclass
from pathlib import Path


# --- Limpieza ---------------------------------------------------------

# Caracteres de la Private Use Area de Unicode (U+E000–U+F8FF): iconos de
# fuente incrustados como texto por el exportador de Colab. No aportan
# información y rompen el anclaje de línea del regex de headers.
PUA_PATTERN = re.compile(r'[\uE000-\uF8FF]')

def clean_text(texto: str) -> str:
    """
    Limpia el texto crudo antes de detectar headers y generar chunks:
    - Caracteres de icono (Private Use Area) que rompen el anclaje de línea.
    - Fórmulas LaTeX duplicadas (artefacto de exportación de MathJax).
    - Líneas largas repetidas (logs de librerías verbosas como LightGBM).
    """
    texto = PUA_PATTERN.sub('', texto)
    texto = clean_duplicated_latex(texto)
    texto = clean_repeated_lines(texto)
    return texto


# Artefacto de exportación de MathJax desde Colab: tanto fórmulas en bloque
# ($$...$$) como inline ($...$) aparecen duplicadas, pegadas inmediatamente
# a sí mismas sin nada entre medias. Ej: "$$ y=x $$$$ y=x $$" o "$y$$y$".
LATEX_DUP_PATTERN = re.compile(r'(\${1,2})([^$]+?)\1\1\2\1')


def clean_duplicated_latex(texto: str) -> str:
    """Colapsa fórmulas LaTeX (bloque o inline) duplicadas inmediatamente."""
    return LATEX_DUP_PATTERN.sub(r'\1\2\1', texto)

MIN_LINEA_REPETIDA = 30


def clean_repeated_lines(texto: str) -> str:
    """
    Colapsa líneas largas que se repiten 3+ veces dentro del mismo texto,
    conservando solo la primera aparición. Pensado para logs verbosos de
    entrenamiento (ej. warnings de LightGBM repetidos docenas de veces)
    que no aportan información nueva al embedding.

    Solo actúa sobre líneas de MIN_LINEA_REPETIDA+ caracteres, para no
    tocar líneas de código cortas que se repiten legítimamente (ej.
    "print(x)"), que no son ruido.
    """
    lineas = texto.split('\n')
    conteos = {}
    for linea in lineas:
        clave = linea.strip()
        if len(clave) >= MIN_LINEA_REPETIDA:
            conteos[clave] = conteos.get(clave, 0) + 1

    vistas = set()
    resultado = []
    for linea in lineas:
        clave = linea.strip()
        if len(clave) >= MIN_LINEA_REPETIDA and conteos[clave] >= 3:
            if clave in vistas:
                continue
            vistas.add(clave)
        resultado.append(linea)
    return '\n'.join(resultado)    


# --- Detección de candidatos a header ----------------------------------

# Patrón de header: "1.", "1.1", "2.3." ... con o sin punto final,
# porque hay inconsistencia en el propio material (a veces "2.1", a veces "1.2.").
# [ \t]+ (no \s+) para que el número y el título tengan que estar en la
# misma línea visual - \s+ permitía saltos de línea y fusionaba números
# de código/output con el header siguiente.
HEADER_PATTERN = re.compile(r'^[ \t]*(\d+(?:\.\d+)*)\.?[ \t]+(.+)$', re.MULTILINE)

# Caracteres permitidos en un título real: letras (incl. tildes/ñ), dígitos
# (para casos como "L1", "L2"), espacios y puntuación básica de prosa.
# Todo lo demás (=, _, [, ], #, comillas, etc.) delata código o output.
TITLE_ALLOWED = re.compile(
    r"^[A-Za-zÀ-ÿ0-9\s,\.:\(\)\-/¿?¡!]+$"
)

# Dos o más números con decimales en el "título" es señal de fila de
# DataFrame impresa (ej. "98.212354 9.910215 0.907587"), no de un título real.
DECIMAL_NUMBER_PATTERN = re.compile(r'\d+\.\d+')


MAX_TITLE_WORDS = 8

def _is_natural_language(titulo: str) -> bool:
    """
    Un título real de sección es prosa: solo letras, dígitos sueltos,
    espacios y puntuación básica - y como mucho un número con decimales,
    y a lo sumo un token puramente numérico. Filas de tabla impresas
    (DataFrames) traen varios números enteros o decimales seguidos, sea
    cual sea el formato (ej. "MAY 363 420 472" o "98.21 9.91 0.90") - dos
    o más tokens numéricos sueltos es señal fiable de tabla, no de título.
    Además, un título real es corto (nombra un tema); una definición o
    explicación enumerada es una oración completa, mucho más larga.
    """
    if not TITLE_ALLOWED.match(titulo):
        return False
    if not any(c.isalpha() for c in titulo):
        return False
    if len(DECIMAL_NUMBER_PATTERN.findall(titulo)) >= 2:
        return False
    tokens_numericos = sum(1 for tok in titulo.split() if tok.replace('.', '', 1).isdigit())
    if tokens_numericos >= 2:
        return False
    if len(titulo.split()) > MAX_TITLE_WORDS:
        return False
    return True


@dataclass
class HeaderMatch:
    numero: str
    titulo: str
    start: int  # posición donde empieza la línea del header
    end: int    # posición donde termina (inicio de la siguiente línea)


def find_header_candidates(texto: str) -> list[HeaderMatch]:
    """Encuentra líneas que matchean el patrón numérico Y parecen texto natural."""
    return [
        HeaderMatch(numero=m.group(1), titulo=m.group(2).strip(),
                    start=m.start(), end=m.end())
        for m in HEADER_PATTERN.finditer(texto)
        if _is_natural_language(m.group(2).strip())
    ]


# --- Clasificación índice vs. header real ------------------------------

CONTENIDO_MARKER = re.compile(r'\bContenido\b')


def _numero_tuple(numero: str) -> tuple[int, ...]:
    """Convierte '2.3' -> (2, 3) para comparar jerarquía, no solo el entero inicial."""
    return tuple(int(p) for p in numero.split("."))


def classify_headers(
    candidatos: list[HeaderMatch],
    texto: str,
    gap_threshold: int = 40,
    streak_min: int = 3,
) -> tuple[list[HeaderMatch], list[HeaderMatch]]:
    """
    Separa headers reales de entradas de índice.

    Estrategia principal: los PDFs del bootcamp marcan el índice con el
    literal "Contenido" justo antes de la lista de secciones. Anclamos ahí
    y consideramos índice toda la racha de candidatos con huecos cortos que
    arranca justo después de ese marcador - SIN exigir que los números
    vayan en orden creciente, porque el propio índice puede tener errores
    de numeración del material original (ej. Clase 17: salta de "5." a
    "7." sin "6.", incluso dentro de "Contenido").

    Respaldo: si no se encuentra "Contenido", o para candidatos fuera de
    esa racha anclada, se usa la lógica de huecos cortos + monotonía
    numérica (con comparación de tuplas para respetar jerarquía padre/hijo,
    ej. "2" -> "2.1" es un avance válido aunque el entero inicial no cambie).
    """
    n = len(candidatos)
    es_indice = [False] * n


    marker = CONTENIDO_MARKER.search(texto)
    if marker:
        start_idx = next(
            (i for i, c in enumerate(candidatos) if c.start >= marker.end()),
            None,
        )
        if start_idx is not None:
            j = start_idx
            anterior = _numero_tuple(candidatos[start_idx].numero)
            while j + 1 < n:
                gap_corto = (candidatos[j + 1].start - candidatos[j].end) < gap_threshold
                siguiente = _numero_tuple(candidatos[j + 1].numero)
                avanza = siguiente > anterior
                if not gap_corto or not avanza:
                    break
                anterior = siguiente
                j += 1
            for k in range(start_idx, j + 1):
                es_indice[k] = True

    if marker is None:
        # Sin "Contenido" localizable, recurrimos a la heurística de huecos +
        # monotonía como único criterio disponible. Con "Contenido" presente,
        # confiamos solo en el bloque anclado: una lista de pasos en prosa
        # (ej. "1. Definir la hipótesis... 4. Decisión...") puede simular un
        # índice por huecos cortos y números crecientes, y arrastrar de forma
        # incorrecta al header real que la sigue si dejamos que esta
        # heurística clasifique también fuera del bloque anclado.
        i = 0
        while i < n:
            if es_indice[i]:
                i += 1
                continue
            j = i
            anterior = _numero_tuple(candidatos[i].numero)
            while j + 1 < n and not es_indice[j + 1]:
                gap_corto = (candidatos[j + 1].start - candidatos[j].end) < gap_threshold
                siguiente = _numero_tuple(candidatos[j + 1].numero)
                avanza = siguiente > anterior
                if not gap_corto or not avanza:
                    break
                anterior = siguiente
                j += 1
            streak_len = j - i + 1
            if streak_len >= streak_min:
                for k in range(i, j + 1):
                    es_indice[k] = True
            i = j + 1

    reales = [c for c, ind in zip(candidatos, es_indice) if not ind]
    indice = [c for c, ind in zip(candidatos, es_indice) if ind]
    return reales, indice


# --- Construcción de chunks ---------------------------------------------

PAGE_MARKER_PATTERN = re.compile(r'--- PÁGINA \d+ ---\n?')


@dataclass
class Chunk:
    numero_seccion: str
    titulo_seccion: str
    texto: str
    parte: int      # índice de sub-chunk dentro de la sección (0 si no se dividió)
    total_partes: int


def extract_sections(reales: list[HeaderMatch], texto: str) -> list[tuple[HeaderMatch, str]]:
    """
    Extrae el texto de cada sección: desde el final del header real hasta
    el inicio del siguiente header real (o el final del documento para
    la última sección).

    Los marcadores de página (insertados en extraction.py para trazabilidad)
    se eliminan aquí, justo antes de que el contenido se convierta en chunks:
    no aportan significado semántico y contaminarían el embedding si se
    dejaran dentro del texto vectorizable.

    Nota: el texto antes del PRIMER header real (portada, nombre del
    docente, etc.) se descarta deliberadamente - no aporta contenido
    de repaso y no pertenece a ninguna sección numerada.
    """
    secciones = []
    for i, header in enumerate(reales):
        inicio = header.end
        fin = reales[i + 1].start if i + 1 < len(reales) else len(texto)
        contenido = texto[inicio:fin].strip()
        contenido = PAGE_MARKER_PATTERN.sub('', contenido).strip()
        secciones.append((header, contenido))
    return secciones


def split_section(
    header: HeaderMatch,
    contenido: str,
    max_chars: int = 1000,
    overlap: int = 180,
) -> list[Chunk]:
    """
    Divide el contenido de una sección en uno o más chunks.

    Si la sección completa cabe bajo max_chars, es un único chunk (caso
    normal: la sección numerada ya es la unidad semántica correcta).

    Si la excede, se corta en fragmentos de max_chars, repitiendo los
    últimos `overlap` caracteres del fragmento anterior al inicio del
    siguiente, para no perder contexto justo en el punto de corte.
    """
    if len(contenido) <= max_chars:
        return [Chunk(header.numero, header.titulo, contenido, parte=0, total_partes=1)]

    partes_texto = []
    pos = 0
    while pos < len(contenido):
        fin = min(pos + max_chars, len(contenido))
        partes_texto.append(contenido[pos:fin])
        if fin == len(contenido):
            break
        pos = fin - overlap  # retrocedemos `overlap` caracteres para el siguiente fragmento

    return [
        Chunk(header.numero, header.titulo, parte_texto, parte=idx, total_partes=len(partes_texto))
        for idx, parte_texto in enumerate(partes_texto)
    ]


def build_chunks(
    reales: list[HeaderMatch],
    texto: str,
    max_chars: int = 1000,
    overlap: int = 180,
) -> list[Chunk]:
    """Pipeline completo: secciones -> chunks, aplicando el límite de tamaño a cada una."""
    chunks = []
    for header, contenido in extract_sections(reales, texto):
        if not contenido:
            continue  # sección sin contenido real (raro, pero posible con headers pegados)
        chunks.extend(split_section(header, contenido, max_chars, overlap))
    return chunks


# --- Prueba manual --------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from extraction import extract_text_from_pdf

    pdf = sorted(Path("data/raw").glob("*.pdf"))[0]
    texto = extract_text_from_pdf(pdf)
    texto = clean_text(texto)

    candidatos = find_header_candidates(texto)
    reales, indice = classify_headers(candidatos, texto)

    chunks = build_chunks(reales, texto)

    print(f"Headers reales: {len(reales)}")
    print(f"Chunks generados: {len(chunks)}\n")

    tamanos = [len(c.texto) for c in chunks]
    print(f"Tamaño chunk: min={min(tamanos)}, max={max(tamanos)}, promedio={sum(tamanos)//len(tamanos)}\n")

    for c in chunks:
        etiqueta = f"{c.numero_seccion} {c.titulo_seccion}"
        if c.total_partes > 1:
            etiqueta += f" (parte {c.parte + 1}/{c.total_partes})"
        print(f"--- {etiqueta} [{len(c.texto)} chars] ---")
        print(c.texto[:200].replace("\n", " "))
        print()
