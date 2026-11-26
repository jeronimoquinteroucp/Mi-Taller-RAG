"""Etapa 1 · Explorar embeddings y similitud coseno.

Uso:
    python scripts/etapa1_embeddings.py               # embedder didáctico (hash)
    python scripts/etapa1_embeddings.py --backend local  # modelo neuronal
"""
import argparse

import _ruta  # noqa: F401
from rag.embeddings import get_embedder, mas_similares, similitud_coseno, tokenizar

FRASES = [
    "¿Cómo pido mis vacaciones?",
    "Las vacaciones se solicitan en el portal MiAndina",
    "Quiero tomarme unos días de descanso",          # paráfrasis sin palabras en común
    "El servidor de la VPN es vpn.andinalogistica.co",
]

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=None, help="hash | local")
args = parser.parse_args()

embedder = get_embedder(args.backend)
print(f"Embedder: {type(embedder).__name__} · dimensión del vector: {embedder.dim}\n")

print("1) Del texto a los tokens")
print(f"   {FRASES[0]!r} -> {tokenizar(FRASES[0])}\n")

v = embedder.embed(FRASES[0])
no_cero = [(i, round(float(x), 3)) for i, x in enumerate(v) if abs(x) > 1e-6]
print("2) Del texto al vector")
print(f"   posiciones distintas de cero: {len(no_cero)} de {embedder.dim}")
print(f"   primeras: {no_cero[:6]}\n")

print("3) Matriz de similitud coseno")
vectores = embedder.embed_many(FRASES)
print("      " + "".join(f"  F{j + 1}  " for j in range(len(FRASES))))
for i, vi in enumerate(vectores):
    print(f"   F{i + 1} " + "".join(f" {similitud_coseno(vi, vj):5.2f} " for vj in vectores))
for i, f in enumerate(FRASES):
    print(f"   F{i + 1}: {f}")

print("\n4) Búsqueda: ¿qué frase se parece más a 'solicitar vacaciones'?")
for idx, score in mas_similares("solicitar vacaciones", FRASES, embedder, k=3):
    print(f"   {score:5.2f}  {FRASES[idx]}")
