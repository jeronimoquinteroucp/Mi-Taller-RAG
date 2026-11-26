"""Etapa 2 · Indexar párrafos en ChromaDB y buscar por similitud.

Uso:
    python scripts/etapa2_buscar.py "¿cuál es el servidor de la VPN?"
    python scripts/etapa2_buscar.py "permisos" --area rrhh
"""
import argparse

import _ruta  # noqa: F401
from rag.config import get_config
from rag.embeddings import get_embedder
from rag.vector_store import ChromaVectorStore

parser = argparse.ArgumentParser()
parser.add_argument("consulta")
parser.add_argument("-k", type=int, default=3)
parser.add_argument("--area", default=None, help="rrhh | ti | finanzas | clientes")
args = parser.parse_args()

cfg = get_config()
store = ChromaVectorStore(get_embedder(), cfg.chroma_dir, "etapa2_parrafos")

if store.contar() == 0:
    from pathlib import Path

    ids, textos, metas = [], [], []
    for ruta in sorted(Path(cfg.docs_dir).glob("*.md")):
        contenido = ruta.read_text(encoding="utf-8")
        _, cabecera, cuerpo = contenido.split("---", 2)  # front matter: --- area: ... ---
        area = cabecera.split("area:")[1].split()[0]
        for n, parrafo in enumerate(p.strip() for p in cuerpo.split("\n\n")):
            if len(parrafo) > 40 and not parrafo.startswith("#"):
                ids.append(f"{ruta.name}#{n}")
                textos.append(parrafo)
                metas.append({"fuente": ruta.name, "area": area})
    store.agregar(ids, textos, metas)
    print(f"Indexados {store.contar()} párrafos en {cfg.chroma_dir}\n")

filtro = {"area": args.area} if args.area else None
for r in store.buscar(args.consulta, k=args.k, filtro=filtro):
    print(f"{r.score:5.2f}  [{r.metadata['fuente']}]  {r.texto[:110]}...")
