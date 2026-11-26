"""Etapas 3 y 4 · Construir el índice vectorial persistente con los documentos.

Uso:
    python scripts/indexar.py --estrategia documento     # etapa 3: un vector por documento
    python scripts/indexar.py --estrategia fragmentos    # etapa 4: un vector por fragmento
"""
import argparse
import time

import _ruta  # noqa: F401
from rag.config import get_config
from rag.documentos import cargar_documentos
from rag.embeddings import get_embedder
from rag.vector_store import ChromaVectorStore

parser = argparse.ArgumentParser()
parser.add_argument("--estrategia", choices=["documento", "fragmentos"], default="fragmentos")
args = parser.parse_args()

cfg = get_config()
inicio = time.perf_counter()
documentos = cargar_documentos(cfg.docs_dir)
store = ChromaVectorStore(get_embedder(), cfg.chroma_dir, cfg.coleccion)
store.reiniciar()  # siempre desde cero para no mezclar estrategias

if args.estrategia == "documento":
    from rag.pipeline import indexar_documentos

    n = indexar_documentos(store, documentos)
else:
    from rag.pipeline import indexar_fragmentos

    n = indexar_fragmentos(store, documentos, cfg.chunk_size, cfg.chunk_overlap)

print(f"Documentos leídos: {len(documentos)}")
print(f"Vectores guardados ({args.estrategia}): {n}")
print(f"Colección '{cfg.coleccion}' en {cfg.chroma_dir}")
print(f"Tiempo: {time.perf_counter() - inicio:.1f} s")
