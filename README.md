# Asistente RAG con datos propios · Andina Logística

Repositorio base del módulo **4. Recuperación aumentada con datos propios**.
Durante el módulo construirá, etapa por etapa, un asistente que responde preguntas
sobre los documentos internos de una empresa ficticia (Andina Logística S.A.S.) y
cita la fuente de cada respuesta.

## Etapas y ramas

| Etapa | Tema | Rama | Archivo que se completa | Prueba |
|---|---|---|---|---|
| 0 | Entorno | `main` | — | `tests/test_etapa0_entorno.py` |
| 1 | Representación vectorial de texto | `etapa-1-embeddings` | `src/rag/embeddings.py` | `tests/test_etapa1_embeddings.py` |
| 2 | Bases de datos vectoriales | `etapa-2-vectores` | `src/rag/vector_store.py` | `tests/test_etapa2_vectores.py` |
| 3 | Arquitectura RAG | `etapa-3-rag` | `src/rag/documentos.py`, `src/rag/pipeline.py` | `tests/test_etapa3_rag.py` |
| 4 | Preparación y segmentación | `etapa-4-segmentacion` | `src/rag/chunking.py` (+ PDF e índice por fragmentos) | `tests/test_etapa4_segmentacion.py` |
| 5 | Caso organizacional | `etapa-5-caso` | `src/rag/asistente.py`, `app.py` | `tests/test_etapa5_caso.py` |

Las pruebas de etapas pendientes aparecen como **SKIPPED** (omitidas). Cuando termine
una etapa, sus pruebas deben pasar a **PASSED**.

## Puesta en marcha

```bash
git clone https://github.com/<su-usuario>/asistente-rag.git
cd asistente-rag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
pytest                              # etapa 0 en verde, las demás omitidas
```

## Ciclo de trabajo de cada etapa

```bash
git switch main && git pull
git switch -c etapa-N-nombre
# ... escribir el código de la etapa ...
pytest tests/test_etapaN_*.py -v
git add . && git commit -m "Etapa N: descripción"
git push -u origin etapa-N-nombre
# abrir Pull Request → esperar check verde → fusionar
git switch main && git pull
git tag vN.0 && git push --tags
```

## Estructura

```
asistente-rag/
├── app.py                     # interfaz web (etapa 5)
├── data/
│   ├── documentos/            # fuentes de datos propias del caso
│   └── evaluacion/preguntas.json
├── scripts/                   # un script para observar cada etapa
├── src/rag/
│   ├── config.py              # configuración por variables de entorno (dada)
│   ├── llm.py                 # conexión a modelos de lenguaje (dada)
│   ├── evaluacion.py          # métricas de recuperación (dada)
│   ├── embeddings.py          # etapa 1
│   ├── vector_store.py        # etapa 2
│   ├── documentos.py          # etapas 3 y 4
│   ├── pipeline.py            # etapas 3 y 4
│   ├── chunking.py            # etapa 4
│   └── asistente.py           # etapa 5
└── tests/                     # una prueba por etapa
```

## Modelos (opcionales)

El proyecto funciona sin claves ni GPU (`EMBEDDING_BACKEND=hash`, `LLM_PROVIDER=ninguno`).
Para usar modelos reales, edite `.env`:

- Embeddings neuronales: `pip install -r requirements-modelo.txt` y `EMBEDDING_BACKEND=local`.
- Modelo local gratuito: instale [Ollama](https://ollama.com), ejecute `ollama pull llama3.2` y use `LLM_PROVIDER=ollama`.
- OpenAI: `LLM_PROVIDER=openai` y `OPENAI_API_KEY=...`.
- Anthropic: `LLM_PROVIDER=anthropic` y `ANTHROPIC_API_KEY=...`.

Nunca suba el archivo `.env` al repositorio: ya está en `.gitignore`.
