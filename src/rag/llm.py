"""Etapa 3 · Conexión con el modelo de lenguaje (la "G" de RAG).

Todos los proveedores exponen el mismo método ``generar(prompt) -> str``.
Así el pipeline no sabe ni le importa qué modelo hay detrás.
"""
from __future__ import annotations

import os
import re

import requests

from .config import get_config


class SinModeloLLM:
    """Modo sin modelo: devuelve el fragmento recuperado más relevante.

    Permite probar todo el sistema sin claves de API ni GPU. No redacta:
    solo muestra lo que la recuperación encontró.
    """

    nombre = "ninguno"

    def generar(self, prompt: str) -> str:
        bloques = re.findall(r"\[(\d+)\] \(fuente: ([^)]+)\)\n(.*?)(?=\n\[\d+\] \(fuente:|\n</contexto>)", prompt, re.S)
        if not bloques:
            return "No encuentro esa información en los documentos."
        _, fuente, texto = bloques[0]
        return f"(modo sin modelo) Fragmento más relevante:\n{texto.strip()} [{fuente}]"


class OllamaLLM:
    """Modelo local con Ollama (https://ollama.com). Gratis y sin internet."""

    def __init__(self, modelo: str, url: str) -> None:
        self.nombre = f"ollama:{modelo}"
        self.modelo, self.url = modelo, url.rstrip("/")

    def generar(self, prompt: str) -> str:
        r = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.modelo,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


class OpenAILLM:
    """API de OpenAI (requiere OPENAI_API_KEY)."""

    def __init__(self, modelo: str) -> None:
        self.nombre = f"openai:{modelo}"
        self.modelo = modelo
        self.clave = os.environ["OPENAI_API_KEY"]

    def generar(self, prompt: str) -> str:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.clave}"},
            json={
                "model": self.modelo,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class AnthropicLLM:
    """API de Anthropic, modelos Claude (requiere ANTHROPIC_API_KEY)."""

    def __init__(self, modelo: str) -> None:
        self.nombre = f"anthropic:{modelo}"
        self.modelo = modelo
        self.clave = os.environ["ANTHROPIC_API_KEY"]

    def generar(self, prompt: str) -> str:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.clave,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.modelo,
                "max_tokens": 800,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        r.raise_for_status()
        return "".join(b["text"] for b in r.json()["content"] if b["type"] == "text").strip()


MODELOS_POR_DEFECTO = {
    "ollama": "llama3.2",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}


def get_llm(proveedor: str | None = None):
    """Crea el modelo según LLM_PROVIDER y LLM_MODEL."""
    cfg = get_config()
    proveedor = (proveedor or cfg.llm_provider).lower()
    modelo = cfg.llm_model or MODELOS_POR_DEFECTO.get(proveedor, "")
    if proveedor == "ninguno":
        return SinModeloLLM()
    if proveedor == "ollama":
        return OllamaLLM(modelo, cfg.ollama_url)
    if proveedor == "openai":
        return OpenAILLM(modelo)
    if proveedor == "anthropic":
        return AnthropicLLM(modelo)
    raise ValueError(f"LLM_PROVIDER desconocido: {proveedor!r}")
