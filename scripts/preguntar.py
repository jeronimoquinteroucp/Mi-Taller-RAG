"""Etapa 3 · Hacer una pregunta al sistema RAG desde la terminal.

Uso:
    python scripts/preguntar.py "¿Cuál es el servidor de la VPN?"
    python scripts/preguntar.py "¿Cuántos días de vacaciones tengo?" --ver-prompt
    python scripts/preguntar.py "permisos" --area rrhh
"""
import argparse

import _ruta  # noqa: F401
from rag.config import get_config
from rag.embeddings import get_embedder
from rag.llm import get_llm
from rag.pipeline import PipelineRAG
from rag.vector_store import ChromaVectorStore

parser = argparse.ArgumentParser()
parser.add_argument("pregunta")
parser.add_argument("--area", default=None)
parser.add_argument("--ver-prompt", action="store_true", help="muestra el prompt enviado al modelo")
args = parser.parse_args()

cfg = get_config()
store = ChromaVectorStore(get_embedder(), cfg.chroma_dir, cfg.coleccion)
if store.contar() == 0:
    raise SystemExit("El índice está vacío. Ejecute primero: python scripts/indexar.py")

llm = get_llm()
pipeline = PipelineRAG(store, llm, k=cfg.top_k, umbral=cfg.umbral_similitud)
respuesta = pipeline.responder(args.pregunta, filtro={"area": args.area} if args.area else None)

print(f"Modelo: {getattr(llm, 'nombre', type(llm).__name__)}\n")
print("Fragmentos recuperados:")
for r in pipeline.store.buscar(args.pregunta, k=cfg.top_k):
    marca = "✓" if r.score >= cfg.umbral_similitud else "✗ (bajo el umbral)"
    print(f"  {r.score:5.2f} {marca}  {r.id}")
if args.ver_prompt and respuesta.prompt:
    print("\n----- PROMPT -----\n" + respuesta.prompt + "\n------------------")
print(f"\nRespuesta:\n{respuesta.texto}")
if respuesta.fuentes:
    print(f"\nFuentes: {', '.join(respuesta.fuentes)}")
