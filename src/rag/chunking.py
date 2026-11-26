"""Etapa 4 · Preparación y segmentación de documentos (chunking).

TODO etapa 4: implemente las funciones siguiendo la guía.
Prueba: pytest tests/test_etapa4_segmentacion.py
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fragmento:
    id: str
    texto: str
    metadata: dict = field(default_factory=dict)


def limpiar_texto(texto: str) -> str:
    raise NotImplementedError("Etapa 4: limpiar_texto")


def dividir_por_secciones(texto: str) -> list[tuple[str, str]]:
    raise NotImplementedError("Etapa 4: dividir_por_secciones")


def dividir_por_caracteres(texto: str, tamano: int, solapamiento: int) -> list[str]:
    raise NotImplementedError("Etapa 4: dividir_por_caracteres")


def fragmentar_documento(documento, tamano: int = 600, solapamiento: int = 100) -> list[Fragmento]:
    raise NotImplementedError("Etapa 4: fragmentar_documento")


def fragmentar_documentos(documentos, tamano: int = 600, solapamiento: int = 100) -> list[Fragmento]:
    raise NotImplementedError("Etapa 4: fragmentar_documentos")
