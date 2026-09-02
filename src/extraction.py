"""
extraction.py
Extracción de texto plano desde los PDFs de las clases del bootcamp.
Fase 1 del pipeline RAG: solo extraer texto crudo, sin limpiar ni trocear todavía.
"""

from pathlib import Path
import pdfplumber


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extrae el texto de todas las páginas de un PDF y lo concatena
    en un único string, separando páginas con un marcador.

    El marcador de página no es solo informativo: si más adelante detectamos
    problemas de chunking en el límite entre dos páginas, necesitamos poder
    rastrear de dónde vino cada fragmento.
    """
    texto_paginas = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            texto = page.extract_text() or ""  # puede devolver None si la página está vacía
            texto_paginas.append(f"--- PÁGINA {i + 1} ---\n{texto}")

    return "\n\n".join(texto_paginas)


if __name__ == "__main__":
    # Prueba con un solo PDF antes de generalizar al corpus completo
    carpeta_raw = Path("data/raw")
    pdfs = sorted(carpeta_raw.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"No se encontraron PDFs en {carpeta_raw}")

    pdf_prueba = pdfs[0]
    print(f"Probando extracción con: {pdf_prueba.name}\n")

    texto = extract_text_from_pdf(pdf_prueba)

    print(f"Longitud total del texto extraído: {len(texto)} caracteres")
    print("\n--- Primeros 1500 caracteres ---\n")
    print(texto[:1500])