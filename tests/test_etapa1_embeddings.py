"""Etapa 1 · Representación vectorial de texto."""
import numpy as np
import pytest


def test_normalizar_quita_tildes_y_mayusculas():
    from rag.embeddings import normalizar

    assert normalizar("Vacación ÁRBOL") == "vacacion arbol"


def test_tokenizar_quita_stopwords_y_recorta():
    from rag.embeddings import tokenizar

    assert tokenizar("¿Cuántos días de vacaciones tengo?") == ["dias", "vacac"]


def test_vector_tiene_dimension_fija_y_norma_1(embedder):
    v = embedder.embed("Solicitud de vacaciones en MiAndina")
    assert v.shape == (embedder.dim,)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


def test_texto_vacio_da_vector_cero(embedder):
    assert np.linalg.norm(embedder.embed("de la el")) == 0


def test_embed_many_devuelve_una_fila_por_texto(embedder):
    m = embedder.embed_many(["uno dos", "tres cuatro", "cinco seis"])
    assert m.shape == (3, embedder.dim)


def test_similitud_coseno_casos_extremos():
    from rag.embeddings import similitud_coseno

    a = np.array([1.0, 0.0])
    assert similitud_coseno(a, a) == pytest.approx(1.0)
    assert similitud_coseno(a, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert similitud_coseno(a, np.array([0.0, 0.0])) == 0.0


def test_textos_relacionados_son_mas_similares(embedder):
    from rag.embeddings import similitud_coseno

    pregunta = embedder.embed("¿Cómo pido mis vacaciones?")
    relacionado = embedder.embed("Las vacaciones se solicitan en el portal MiAndina")
    no_relacionado = embedder.embed("El servidor de la VPN es vpn.andinalogistica.co")
    assert similitud_coseno(pregunta, relacionado) > similitud_coseno(pregunta, no_relacionado)


def test_mas_similares_ordena_por_puntaje(embedder):
    from rag.embeddings import mas_similares

    textos = [
        "El auxilio de conectividad es de 90.000 pesos",
        "Restablezca la contraseña en el sitio de autoservicio",
        "Los portátiles se renuevan cada cuatro años",
    ]
    top = mas_similares("olvidé mi contraseña", textos, embedder, k=2)
    assert len(top) == 2
    assert top[0][0] == 1
    assert top[0][1] >= top[1][1]


def test_get_embedder_por_defecto_es_hash():
    from rag.embeddings import HashEmbedder, get_embedder

    assert isinstance(get_embedder(), HashEmbedder)
    with pytest.raises(ValueError):
        get_embedder("no-existe")
