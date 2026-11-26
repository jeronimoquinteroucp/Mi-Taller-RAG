"""Configuración central del proyecto.

Todos los valores se leen de variables de entorno (o del archivo .env),
así el mismo código funciona en el portátil del estudiante, en GitHub
Actions y en producción sin cambiar una línea.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv es opcional: si está instalado, carga el archivo .env
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]


def _env(nombre: str, defecto: str):
    return field(default_factory=lambda: os.getenv(nombre, defecto))


@dataclass
class Config:
    # Etapa 1: cómo convertimos texto en vectores
    embedding_backend: str = _env("EMBEDDING_BACKEND", "hash")  # hash | local
    embedding_model: str = _env(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    # Etapa 2: dónde se guardan los vectores
    chroma_dir: str = _env("CHROMA_DIR", str(RAIZ_PROYECTO / "chroma_db"))
    coleccion: str = _env("CHROMA_COLECCION", "andina")
    # Etapa 3: qué modelo de lenguaje genera la respuesta
    llm_provider: str = _env("LLM_PROVIDER", "ninguno")  # ninguno | ollama | openai | anthropic
    llm_model: str = _env("LLM_MODEL", "")
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    top_k: str = _env("TOP_K", "4")
    umbral_similitud: str = _env("UMBRAL_SIMILITUD", "0.25")
    # Etapa 4: segmentación
    chunk_size: str = _env("CHUNK_SIZE", "600")
    chunk_overlap: str = _env("CHUNK_OVERLAP", "100")
    # Datos del caso
    docs_dir: str = _env("DOCS_DIR", str(RAIZ_PROYECTO / "data" / "documentos"))
    preguntas_eval: str = _env(
        "PREGUNTAS_EVAL", str(RAIZ_PROYECTO / "data" / "evaluacion" / "preguntas.json")
    )

    def __post_init__(self) -> None:
        # Las variables de entorno siempre llegan como texto: convertimos los números.
        self.top_k = int(self.top_k)
        self.umbral_similitud = float(self.umbral_similitud)
        self.chunk_size = int(self.chunk_size)
        self.chunk_overlap = int(self.chunk_overlap)


def get_config() -> Config:
    """Devuelve la configuración leyendo el entorno en este momento."""
    return Config()
