"""Evaluación de la recuperación con un conjunto de preguntas de referencia.

Para cada pregunta sabemos qué archivo la responde (``fuente``) y una frase
que debe aparecer en el contexto recuperado (``clave``). Métricas:

* acierto@1 y acierto@k: ¿la fuente correcta está en el primer / en los k primeros?
* MRR: promedio de 1/posición de la primera fuente correcta (1 = siempre primera).
* clave@k: ¿el texto recuperado contiene la frase que responde?
* caracteres de contexto: cuánto texto mandaríamos al modelo (≈ costo y ruido).
"""
from __future__ import annotations

import json
from pathlib import Path


def cargar_preguntas(ruta: str | Path) -> list[dict]:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def evaluar_recuperacion(store, preguntas: list[dict], k: int = 3) -> dict:
    a1 = ak = clave = rr = caracteres = 0
    detalle = []
    for p in preguntas:
        resultados = store.buscar(p["pregunta"], k=k)
        fuentes = [r.metadata.get("fuente") for r in resultados]
        contexto = "\n".join(r.texto for r in resultados)
        pos = fuentes.index(p["fuente"]) + 1 if p["fuente"] in fuentes else None
        a1 += pos == 1
        ak += pos is not None
        rr += 1 / pos if pos else 0
        tiene_clave = p["clave"].lower() in contexto.lower()
        clave += tiene_clave
        caracteres += len(contexto)
        detalle.append({"pregunta": p["pregunta"], "posicion": pos, "clave": tiene_clave, "top1": fuentes[:1]})
    n = len(preguntas)
    return {
        "n": n,
        "k": k,
        "acierto@1": a1 / n,
        f"acierto@{k}": ak / n,
        "mrr": rr / n,
        f"clave@{k}": clave / n,
        "caracteres_contexto": caracteres / n,
        "detalle": detalle,
    }
