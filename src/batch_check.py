"""
batch_check.py
Corre el pipeline de chunking sobre todos los PDFs del corpus y reporta
únicamente los casos sospechosos, para no tener que revisar 32 salidas
completas a mano.
"""

import sys
sys.path.insert(0, "src")
from pathlib import Path
from extraction import extract_text_from_pdf
from chunking import clean_text, find_header_candidates, classify_headers, build_chunks

MIN_CHUNK_CHARS = 80
MAX_PARTES_ESPERADO = 4

pdfs = sorted(Path("data/raw").glob("*.pdf"))
resumen = []
alertas = []

for pdf in pdfs:
    try:
        texto = clean_text(extract_text_from_pdf(pdf))
        candidatos = find_header_candidates(texto)
        reales, indice = classify_headers(candidatos, texto)
        chunks = build_chunks(reales, texto)

        if not chunks:
            alertas.append(f"{pdf.name}: 0 chunks generados (0 headers reales)")
            continue

        tamanos = [len(c.texto) for c in chunks]
        max_partes = max(c.total_partes for c in chunks)

        resumen.append({
            "pdf": pdf.name,
            "reales": len(reales),
            "indice": len(indice),
            "chunks": len(chunks),
            "min_tam": min(tamanos),
            "max_tam": max(tamanos),
        })

        if min(tamanos) < MIN_CHUNK_CHARS:
            cortos = [c for c in chunks if len(c.texto) < MIN_CHUNK_CHARS]
            for c in cortos:
                alertas.append(
                    f"{pdf.name}: chunk muy corto ({len(c.texto)} chars) "
                    f"en sección '{c.numero_seccion} {c.titulo_seccion}'"
                )

        if max_partes > MAX_PARTES_ESPERADO:
            alertas.append(
                f"{pdf.name}: sección dividida en {max_partes} partes (revisar si hay headers sin detectar)"
            )

        if len(reales) == 0:
            alertas.append(f"{pdf.name}: 0 headers reales detectados")

    except Exception as e:
        alertas.append(f"{pdf.name}: ERROR durante procesamiento -> {e}")

print(f"PDFs procesados: {len(resumen)} de {len(pdfs)}\n")

print("--- Resumen por PDF ---")
for r in resumen:
    print(f"  {r['pdf']:<55} reales={r['reales']:>3} indice={r['indice']:>3} "
          f"chunks={r['chunks']:>3} tam=[{r['min_tam']:>4}-{r['max_tam']:>4}]")

print(f"\n--- Alertas ({len(alertas)}) ---")
if not alertas:
    print("  Ninguna.")
for a in alertas:
    print(f"  ⚠ {a}")
