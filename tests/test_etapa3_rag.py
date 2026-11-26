"""Etapa 3 · Arquitectura de recuperación aumentada."""
import pytest


class LLMFalso:
    """Doble de prueba: guarda el prompt recibido y responde algo fijo."""

    def __init__(self):
        self.prompts = []

    def generar(self, prompt):
        self.prompts.append(prompt)
        return "respuesta simulada"


@pytest.fixture
def pipeline_docs(embedder, carpeta_docs):
    from rag.documentos import cargar_documentos
    from rag.pipeline import PipelineRAG, indexar_documentos
    from rag.vector_store import MemoryVectorStore

    store = MemoryVectorStore(embedder)
    indexar_documentos(store, cargar_documentos(carpeta_docs))
    llm = LLMFalso()
    return PipelineRAG(store, llm, k=2, umbral=0.1), llm


def test_front_matter():
    from rag.documentos import separar_front_matter

    meta, texto = separar_front_matter("---\narea: ti\ntitulo: Manual\n---\n# Hola\ntexto")
    assert meta == {"area": "ti", "titulo": "Manual"}
    assert texto.startswith("# Hola")


def test_carga_documentos_con_metadatos(carpeta_docs):
    from rag.documentos import cargar_documentos

    docs = cargar_documentos(carpeta_docs)
    assert len(docs) >= 6
    vac = next(d for d in docs if d.fuente == "politica_vacaciones.md")
    assert vac.metadata["area"] == "rrhh"
    assert vac.metadata["fuente"] == "politica_vacaciones.md"
    assert not vac.texto.startswith("---")


def test_prompt_contiene_instrucciones_contexto_y_pregunta():
    from rag.pipeline import RESPUESTA_SIN_CONTEXTO, construir_prompt
    from rag.vector_store import Resultado

    r = [Resultado("a", "Texto de la política", {"fuente": "politica.md"}, 0.8)]
    prompt = construir_prompt("¿Qué dice la política?", r)
    assert "Texto de la política" in prompt
    assert "politica.md" in prompt
    assert "¿Qué dice la política?" in prompt
    assert RESPUESTA_SIN_CONTEXTO in prompt


def test_responde_con_fuentes(pipeline_docs):
    pipeline, llm = pipeline_docs
    resp = pipeline.responder("¿Cuál es el servidor de la VPN corporativa?")
    assert resp.texto == "respuesta simulada"
    assert resp.fuentes[0] == "manual_soporte_ti.md"
    assert "vpn.andinalogistica.co" in llm.prompts[0]


def test_sin_contexto_no_llama_al_modelo(pipeline_docs):
    from rag.pipeline import RESPUESTA_SIN_CONTEXTO

    pipeline, llm = pipeline_docs
    resp = pipeline.responder("Receta de arepas de huevo")
    assert resp.texto == RESPUESTA_SIN_CONTEXTO
    assert llm.prompts == []


def test_modo_sin_modelo_extrae_fragmento(embedder, carpeta_docs):
    from rag.documentos import cargar_documentos
    from rag.llm import SinModeloLLM
    from rag.pipeline import PipelineRAG, indexar_documentos
    from rag.vector_store import MemoryVectorStore

    store = MemoryVectorStore(embedder)
    indexar_documentos(store, cargar_documentos(carpeta_docs))
    resp = PipelineRAG(store, SinModeloLLM(), k=2).responder("¿Cómo rastrea un cliente su envío?")
    assert "faq_clientes.md" in resp.texto
