"""Etapa 0 · El entorno está listo."""
import importlib
import sys


def test_version_de_python():
    assert sys.version_info >= (3, 10), "Se necesita Python 3.10 o superior"


def test_dependencias_instaladas():
    for paquete in ["numpy", "chromadb", "pypdf", "requests"]:
        importlib.import_module(paquete)


def test_documentos_del_caso(carpeta_docs):
    archivos = sorted(p.name for p in carpeta_docs.glob("*.md"))
    assert len(archivos) >= 6, f"Faltan documentos en {carpeta_docs}"


def test_preguntas_de_evaluacion(ruta_preguntas):
    import json

    preguntas = json.loads(ruta_preguntas.read_text(encoding="utf-8"))
    assert len(preguntas) >= 15
    assert {"pregunta", "fuente", "clave"} <= set(preguntas[0])
