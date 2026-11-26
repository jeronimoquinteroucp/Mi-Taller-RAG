"""Etapa 1 · Representación vectorial de texto.

Un *embedding* convierte un texto en un vector de números de tamaño fijo.
Textos con significado parecido quedan cerca en ese espacio, y "cerca" se
mide con la similitud coseno.

TODO etapa 1: implemente las funciones y clases de este archivo siguiendo la guía.
Prueba: pytest tests/test_etapa1_embeddings.py
"""
from __future__ import annotations

import numpy as np

from .config import get_config

STOPWORDS: set[str] = set()  # TODO etapa 1: palabras vacías, escritas sin tildes


def normalizar(texto: str) -> str:
    """Pasa a minúsculas y quita tildes: 'Vacación' -> 'vacacion'."""
    raise NotImplementedError("Etapa 1: normalizar")


def tokenizar(texto: str) -> list[str]:
    """Palabras normalizadas, sin stopwords, recortadas a 5 letras."""
    raise NotImplementedError("Etapa 1: tokenizar")


class HashEmbedder:
    """Embedding didáctico: bolsa de palabras proyectada con hashing."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texto: str) -> np.ndarray:
        raise NotImplementedError("Etapa 1: HashEmbedder.embed")

    def embed_many(self, textos: list[str]) -> np.ndarray:
        raise NotImplementedError("Etapa 1: HashEmbedder.embed_many")


class SentenceTransformerEmbedder:
    """Embedding neuronal con un modelo multilingüe de sentence-transformers."""

    def __init__(self, modelo: str) -> None:
        raise NotImplementedError("Etapa 1: SentenceTransformerEmbedder")


def get_embedder(backend: str | None = None):
    """Crea el embedder según la configuración (EMBEDDING_BACKEND)."""
    raise NotImplementedError("Etapa 1: get_embedder")


def similitud_coseno(a: np.ndarray, b: np.ndarray) -> float:
    """cos(θ) = (a · b) / (|a| |b|)."""
    raise NotImplementedError("Etapa 1: similitud_coseno")


def mas_similares(consulta: str, textos: list[str], embedder, k: int = 3) -> list[tuple[int, float]]:
    """Los k textos más parecidos a la consulta como (índice, similitud)."""
    raise NotImplementedError("Etapa 1: mas_similares")
