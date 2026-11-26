"""Etapa 4 · Medir la recuperación: documento completo vs. fragmentos.

Uso:
    python scripts/evaluar.py
    python scripts/evaluar.py --tamanos 300 600 1000 --solapamiento 100
    python scripts/evaluar.py --backend local --detalle
"""
import argparse

import _ruta  # noqa: F401
from rag.config import get_config
from rag.documentos import cargar_documentos
from rag.embeddings import get_embedder
from rag.evaluacion import cargar_preguntas, evaluar_recuperacion
from rag.pipeline import indexar_documentos, indexar_fragmentos
from rag.vector_store import MemoryVectorStore

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=None)
parser.add_argument("-k", type=int, default=3)
parser.add_argument("--tamanos", type=int, nargs="+", default=[600])
parser.add_argument("--solapamiento", type=int, default=100)
parser.add_argument("--detalle", action="store_true", help="muestra las preguntas falladas")
args = parser.parse_args()

cfg = get_config()
embedder = get_embedder(args.backend)
documentos = cargar_documentos(cfg.docs_dir)
preguntas = cargar_preguntas(cfg.preguntas_eval)

configuraciones = [("documento completo", None)] + [(f"fragmentos de {t}", t) for t in args.tamanos]
print(f"Embedder: {type(embedder).__name__} · {len(preguntas)} preguntas · k={args.k}\n")
print(f"{'estrategia':<22}{'vectores':>9}{'acierto@1':>11}{f'acierto@{args.k}':>11}{'MRR':>7}{f'clave@{args.k}':>10}{'contexto':>10}")
for nombre, tamano in configuraciones:
    store = MemoryVectorStore(embedder)
    if tamano is None:
        n = indexar_documentos(store, documentos)
    else:
        n = indexar_fragmentos(store, documentos, tamano, min(args.solapamiento, tamano // 2))
    m = evaluar_recuperacion(store, preguntas, k=args.k)
    print(
        f"{nombre:<22}{n:>9}{m['acierto@1']:>11.0%}{m[f'acierto@{args.k}']:>11.0%}"
        f"{m['mrr']:>7.2f}{m[f'clave@{args.k}']:>10.0%}{m['caracteres_contexto']:>9.0f}c"
    )
    if args.detalle:
        for d in m["detalle"]:
            if d["posicion"] != 1 or not d["clave"]:
                print(f"      ✗ {d['pregunta']} → posición {d['posicion']}, top1 {d['top1']}")
