"""Etapa 2 · Bases de datos vectoriales y búsqueda por similitud.

TODO etapa 2: implemente los dos almacenes siguiendo la guía.
Prueba: pytest tests/test_etapa2_vectores.py
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Resultado:
    id: str
    texto: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0  # similitud coseno: 1 = idéntico, 0 = nada en común


class MemoryVectorStore:
    """Almacén vectorial en memoria con búsqueda exacta (NumPy)."""

    def __init__(self, embedder) -> None:
        raise NotImplementedError("Etapa 2: MemoryVectorStore")

    def agregar(self, ids, textos, metadatas=None) -> None:
        raise NotImplementedError("Etapa 2: MemoryVectorStore.agregar")

    def contar(self) -> int:
        raise NotImplementedError("Etapa 2: MemoryVectorStore.contar")

    def buscar(self, consulta: str, k: int = 4, filtro: dict | None = None) -> list[Resultado]:
        raise NotImplementedError("Etapa 2: MemoryVectorStore.buscar")


class ChromaVectorStore:
    """Almacén vectorial persistente con ChromaDB."""

    def __init__(self, embedder, ruta: str, coleccion: str = "andina") -> None:
        raise NotImplementedError("Etapa 2: ChromaVectorStore")

    def reiniciar(self) -> None:
        raise NotImplementedError("Etapa 2: ChromaVectorStore.reiniciar")

    def agregar(self, ids, textos, metadatas=None) -> None:
        raise NotImplementedError("Etapa 2: ChromaVectorStore.agregar")

    def contar(self) -> int:
        raise NotImplementedError("Etapa 2: ChromaVectorStore.contar")

    def buscar(self, consulta: str, k: int = 4, filtro: dict | None = None) -> list[Resultado]:
        raise NotImplementedError("Etapa 2: ChromaVectorStore.buscar")
