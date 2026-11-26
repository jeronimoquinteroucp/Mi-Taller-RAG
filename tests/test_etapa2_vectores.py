"""Etapa 2 · Bases de datos vectoriales y búsqueda por similitud."""
import pytest

TEXTOS = [
    "Las vacaciones se solicitan en MiAndina con 30 días de anticipación",
    "El auxilio de conectividad para teletrabajo es de 90.000 pesos",
    "La VPN corporativa usa el servidor vpn.andinalogistica.co",
    "Los reembolsos de viáticos se pagan los días 15 y 30",
]
IDS = ["vac", "tele", "vpn", "reem"]
METAS = [{"area": "rrhh"}, {"area": "rrhh"}, {"area": "ti"}, {"area": "finanzas"}]


@pytest.fixture(params=["memoria", "chroma"])
def store(request, embedder, tmp_path):
    from rag.vector_store import ChromaVectorStore, MemoryVectorStore

    if request.param == "memoria":
        s = MemoryVectorStore(embedder)
    else:
        s = ChromaVectorStore(embedder, str(tmp_path / "chroma"), "prueba")
    s.agregar(IDS, TEXTOS, METAS)
    return s


def test_cuenta_los_elementos(store):
    assert store.contar() == 4


def test_el_mas_cercano_es_el_correcto(store):
    r = store.buscar("¿cuál es el servidor de la VPN?", k=1)
    assert r[0].id == "vpn"
    assert r[0].texto == TEXTOS[2]


def test_resultados_ordenados_y_con_score(store):
    r = store.buscar("pedir vacaciones", k=3)
    assert len(r) == 3
    assert r[0].id == "vac"
    assert all(-1.0 <= x.score <= 1.0001 for x in r)
    assert r[0].score >= r[1].score >= r[2].score


def test_filtro_por_metadatos(store):
    r = store.buscar("servidor VPN", k=4, filtro={"area": "rrhh"})
    assert r and all(x.metadata["area"] == "rrhh" for x in r)


def test_chroma_persiste_en_disco(embedder, tmp_path):
    from rag.vector_store import ChromaVectorStore

    ruta = str(tmp_path / "persistente")
    ChromaVectorStore(embedder, ruta, "coleccion").agregar(IDS, TEXTOS, METAS)
    reabierta = ChromaVectorStore(embedder, ruta, "coleccion")  # como si reiniciáramos el programa
    assert reabierta.contar() == 4
    reabierta.reiniciar()
    assert reabierta.contar() == 0
