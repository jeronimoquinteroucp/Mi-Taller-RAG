"""Carga de las fuentes de datos propias de la organización (etapas 3 y 4).

TODO etapa 3: separar_front_matter y cargar_documentos.
TODO etapa 4: leer_pdf.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

EXTENSIONES = {".md", ".txt", ".pdf"}


@dataclass
class Documento:
    fuente: str          # nombre del archivo, se usa para citar
    texto: str           # contenido sin el front matter
    metadata: dict = field(default_factory=dict)


def separar_front_matter(contenido: str) -> tuple[dict, str]:
    """Separa el bloque '--- clave: valor ---' del resto del texto."""
    raise NotImplementedError("Etapa 3: separar_front_matter")


def leer_pdf(ruta: Path) -> str:
    """Extrae el texto de un PDF página por página."""
    raise NotImplementedError("Etapa 4: leer_pdf")


def leer_archivo(ruta: Path) -> str:
    if ruta.suffix.lower() == ".pdf":
        return leer_pdf(ruta)
    return ruta.read_text(encoding="utf-8")


def cargar_documentos(carpeta: str | Path) -> list[Documento]:
    """Lee todos los archivos soportados de la carpeta, en orden alfabético."""
    raise NotImplementedError("Etapa 3: cargar_documentos")
