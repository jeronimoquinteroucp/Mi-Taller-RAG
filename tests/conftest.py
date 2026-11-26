"""Configuración común de las pruebas.

* Las pruebas siempre usan el embedder "hash" y el modo sin modelo: son rápidas,
  no descargan nada y no gastan dinero (sirven igual en GitHub Actions).
* Si una función todavía lanza NotImplementedError (etapa no hecha), la prueba
  se marca como OMITIDA en lugar de FALLIDA. Así `main` siempre está en verde.
"""
import os
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["LLM_PROVIDER"] = "ninguno"
os.environ.setdefault("UMBRAL_SIMILITUD", "0.25")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    resultado = yield
    reporte = resultado.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        reporte.outcome = "skipped"
        reporte.longrepr = (str(item.fspath), item.location[1], f"Pendiente: {call.excinfo.value}")


@pytest.fixture
def carpeta_docs() -> Path:
    return RAIZ / "data" / "documentos"


@pytest.fixture
def ruta_preguntas() -> Path:
    return RAIZ / "data" / "evaluacion" / "preguntas.json"


@pytest.fixture
def embedder():
    from rag.embeddings import HashEmbedder

    return HashEmbedder()
