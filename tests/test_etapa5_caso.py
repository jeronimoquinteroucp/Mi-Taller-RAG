"""Etapa 5 · Caso organizacional: asistente interno y asistente de clientes."""
import pytest


@pytest.fixture
def indice(embedder, tmp_path, monkeypatch):
    from rag.asistente import crear_indice
    from rag.vector_store import ChromaVectorStore

    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    return crear_indice(ChromaVectorStore(embedder, str(tmp_path / "chroma"), "caso"))


def test_crear_indice_indexa_fragmentos(indice):
    assert indice.contar() > 30


def test_crear_indice_no_duplica(indice):
    from rag.asistente import crear_indice

    antes = indice.contar()
    crear_indice(indice)
    assert indice.contar() == antes


def test_asistente_interno_responde_con_fuente(indice):
    from rag.asistente import Asistente

    resp = Asistente(indice, perfil="interno").responder("¿Cuál es el tope diario de alimentación en Medellín?")
    assert resp.fuentes[0] == "reembolso_gastos.md"
    assert "120.000" in resp.texto


def test_filtro_por_area(indice):
    from rag.asistente import Asistente

    resp = Asistente(indice).responder("¿Cuántos días tengo?", area="rrhh")
    assert resp.resultados and all(r.metadata["area"] == "rrhh" for r in resp.resultados)


def test_asistente_de_clientes_no_ve_informacion_interna(indice):
    from rag.asistente import Asistente

    clientes = Asistente(indice, perfil="clientes")
    for pregunta in ["¿Cuál es el servidor de la VPN?", "¿Cuántos días de vacaciones tienen los empleados?"]:
        resp = clientes.responder(pregunta)
        assert all(r.metadata["area"] == "clientes" for r in resp.resultados)
    resp = clientes.responder("¿Cuál es el servidor de la VPN?", area="ti")
    assert resp.resultados == []


def test_pregunta_fuera_de_dominio(indice):
    from rag.asistente import Asistente
    from rag.pipeline import RESPUESTA_SIN_CONTEXTO

    assert Asistente(indice).responder("¿Quién ganó el mundial de fútbol de 2022?").texto == RESPUESTA_SIN_CONTEXTO


def test_perfil_desconocido(indice):
    from rag.asistente import Asistente

    with pytest.raises(ValueError):
        Asistente(indice, perfil="gerencia")
