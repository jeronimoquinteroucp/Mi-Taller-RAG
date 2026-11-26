"""Etapa 4 · Preparación y segmentación de documentos."""
import pytest


def test_limpiar_texto():
    from rag.chunking import limpiar_texto

    assert limpiar_texto("  Hola   mundo \r\n\n\n\nAdiós\t ") == "Hola mundo\n\nAdiós"


def test_dividir_por_secciones():
    from rag.chunking import dividir_por_secciones

    texto = "# Título\nIntro\n## Uno\nTexto uno\n## Dos\nTexto dos"
    assert dividir_por_secciones(texto) == [("", "Intro"), ("Uno", "Texto uno"), ("Dos", "Texto dos")]


def test_ventanas_respetan_tamano_y_solapan():
    from rag.chunking import dividir_por_caracteres

    texto = " ".join(f"palabra{i}" for i in range(300))
    trozos = dividir_por_caracteres(texto, tamano=200, solapamiento=50)
    assert len(trozos) > 5
    assert all(len(t) <= 200 for t in trozos)
    for a, b in zip(trozos, trozos[1:]):
        assert b.split()[0] in a, "cada trozo debe empezar con texto del anterior (solapamiento)"
    assert all(not t.startswith(" ") and t.split()[0].startswith("palabra") for t in trozos)


def test_parametros_invalidos():
    from rag.chunking import dividir_por_caracteres

    with pytest.raises(ValueError):
        dividir_por_caracteres("abc", tamano=100, solapamiento=100)


def test_fragmentos_con_metadatos_y_encabezado(carpeta_docs):
    from rag.chunking import fragmentar_documentos
    from rag.documentos import cargar_documentos

    frags = fragmentar_documentos(cargar_documentos(carpeta_docs), tamano=600, solapamiento=100)
    assert len(frags) > 30
    assert all(len(f.texto) <= 600 for f in frags)
    assert len({f.id for f in frags}) == len(frags), "los ids deben ser únicos"
    vpn = next(f for f in frags if "vpn.andinalogistica.co" in f.texto)
    assert vpn.metadata["fuente"] == "manual_soporte_ti.md"
    assert vpn.metadata["area"] == "ti"
    assert vpn.metadata["seccion"] == "Conexión VPN"
    assert vpn.texto.startswith("Manual de soporte de tecnología para colaboradores > Conexión VPN")


def test_fragmentos_mejoran_la_recuperacion(embedder, carpeta_docs, ruta_preguntas):
    from rag.documentos import cargar_documentos
    from rag.evaluacion import cargar_preguntas, evaluar_recuperacion
    from rag.pipeline import indexar_documentos, indexar_fragmentos
    from rag.vector_store import MemoryVectorStore

    docs, preguntas = cargar_documentos(carpeta_docs), cargar_preguntas(ruta_preguntas)
    completo, fragmentado = MemoryVectorStore(embedder), MemoryVectorStore(embedder)
    indexar_documentos(completo, docs)
    indexar_fragmentos(fragmentado, docs)
    m_doc = evaluar_recuperacion(completo, preguntas, k=3)
    m_frag = evaluar_recuperacion(fragmentado, preguntas, k=3)
    assert m_frag["acierto@1"] >= m_doc["acierto@1"]
    assert m_frag["clave@3"] >= 0.9
    assert m_frag["caracteres_contexto"] < m_doc["caracteres_contexto"] / 3


def test_carga_pdf(tmp_path):
    from rag.documentos import cargar_documentos

    pdf = tmp_path / "circular.pdf"
    _crear_pdf(pdf, "Circular 07: el parqueadero de la sede cierra a las 9 p. m.")
    docs = cargar_documentos(tmp_path)
    assert docs[0].fuente == "circular.pdf"
    assert "parqueadero" in docs[0].texto


def _crear_pdf(ruta, texto):
    """PDF mínimo de una página, escrito a mano para no depender de otra librería."""
    contenido = f"BT /F1 12 Tf 72 720 Td ({texto}) Tj ET".encode("latin-1")
    objetos = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(contenido) + contenido + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    salida, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, obj in enumerate(objetos, start=1):
        offsets.append(len(salida))
        salida += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    inicio_xref = len(salida)
    salida += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objetos) + 1)
    salida += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    salida += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objetos) + 1, inicio_xref)
    ruta.write_bytes(bytes(salida))
