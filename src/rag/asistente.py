"""Etapa 5 · Caso organizacional: asistentes para Andina Logística.

TODO etapa 5: implemente crear_indice y la clase Asistente siguiendo la guía.
Prueba: pytest tests/test_etapa5_caso.py
"""
from __future__ import annotations

AREAS = {
    "rrhh": "Talento Humano",
    "ti": "Tecnología",
    "finanzas": "Finanzas",
    "clientes": "Servicio al cliente",
}


def crear_indice(store=None, reindexar: bool = False):
    raise NotImplementedError("Etapa 5: crear_indice")


class Asistente:
    def __init__(self, store, llm=None, perfil: str = "interno") -> None:
        raise NotImplementedError("Etapa 5: Asistente")

    def responder(self, pregunta: str, area: str | None = None):
        raise NotImplementedError("Etapa 5: Asistente.responder")
