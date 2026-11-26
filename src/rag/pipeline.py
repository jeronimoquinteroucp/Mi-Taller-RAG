"""Etapa 3 · Arquitectura de recuperación aumentada (RAG).

Flujo de una pregunta:

    pregunta ─► embedding ─► búsqueda en la base vectorial (top-k)
             ─► ¿hay contexto suficiente? ──no──► "no encuentro esa información"
             └─sí─► prompt = instrucciones + contexto + pregunta ─► LLM ─► respuesta con fuentes

TODO etapa 3: construir_prompt, PipelineRAG e indexar_documentos.
TODO etapa 4: indexar_fragmentos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .vector_store import Resultado

RESPUESTA_SIN_CONTEXTO = "No encuentro esa información en los documentos de la empresa."

PLANTILLA_PROMPT = ""  # TODO etapa 3


@dataclass
class RespuestaRAG:
    pregunta: str
    texto: str
    fuentes: list[str] = field(default_factory=list)
    resultados: list[Resultado] = field(default_factory=list)
    prompt: str = ""


def formatear_contexto(resultados: list[Resultado]) -> str:
    raise NotImplementedError("Etapa 3: formatear_contexto")


def construir_prompt(pregunta: str, resultados: list[Resultado]) -> str:
    raise NotImplementedError("Etapa 3: construir_prompt")


class PipelineRAG:
    def __init__(self, store, llm, k: int = 4, umbral: float = 0.25) -> None:
        self.store, self.llm, self.k, self.umbral = store, llm, k, umbral

    def recuperar(self, pregunta: str, filtro: dict | None = None) -> list[Resultado]:
        raise NotImplementedError("Etapa 3: PipelineRAG.recuperar")

    def responder(self, pregunta: str, filtro: dict | None = None) -> RespuestaRAG:
        raise NotImplementedError("Etapa 3: PipelineRAG.responder")


def indexar_documentos(store, documentos) -> int:
    raise NotImplementedError("Etapa 3: indexar_documentos")


def indexar_fragmentos(store, documentos, tamano: int = 600, solapamiento: int = 100) -> int:
    raise NotImplementedError("Etapa 4: indexar_fragmentos")
