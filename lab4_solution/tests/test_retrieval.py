"""Pruebas de recuperación sin descargar modelos (BM25, RRF, texto y nDCG en candidatos)."""

import pandas as pd
import pytest

from asistente_compras.evaluation import compare, evaluate_candidates
from asistente_compras.retrieval import (
    BM25Retriever,
    HybridRetriever,
    TextConfig,
    product_texts,
    rrf,
)

CATALOG = pd.DataFrame({
    "product_id": ["a", "b", "c", "d"],
    "title": ["Zapatillas running Nike hombre", "Funda móvil Samsung Galaxy",
              "Zapatos de vestir cuero", "Cargador USB-C rápido"],
    "brand": ["Nike", "", "Clarks", "Anker"],
    "color": ["negro", "", "marrón", ""],
    "bullet_points": ["Suela amortiguada", "", "", "20W"],
    "description": ["", "", "", ""],
})


@pytest.fixture
def bm25():
    return BM25Retriever(CATALOG.product_id, product_texts(CATALOG, TextConfig()))


def test_texto_omite_campos_vacios_y_trunca():
    t = product_texts(CATALOG, TextConfig(fields="titulo_marca"))
    assert t[0] == "Zapatillas running Nike hombre Marca: Nike. Color: negro."
    assert "Marca" not in t[1]  # sin marca ⇒ no se inventa
    assert len(product_texts(CATALOG, TextConfig(max_words=2))[0].split()) == 2


def test_bm25_stemming_espanol(bm25):
    # "zapatilla" (singular) debe encontrar "Zapatillas" gracias al stemming.
    assert bm25.search("zapatilla nike", k=1)[0][0] == "a"
    assert bm25.search("cargadores usb", k=1)[0][0] == "d"


def test_rank_candidates_solo_reordena_los_dados(bm25):
    out = bm25.rank_candidates("funda samsung", ["d", "b", "a"])
    assert sorted(out) == ["a", "b", "d"] and out[0] == "b"


def test_consulta_solo_stopwords_no_falla(bm25):
    assert len(bm25.rank_candidates("de la", ["a", "b"])) == 2


def test_rrf():
    f = rrf([["x", "y"], ["y", "x"]], k=60)
    assert f["x"] == f["y"]
    f = rrf([["x", "y"], ["x", "z"]], k=60)
    assert max(f, key=f.get) == "x"


def test_hibrido_con_dos_bm25_igual_al_bm25(bm25):
    hyb = HybridRetriever(bm25, bm25)
    assert hyb.rank_candidates("zapatos cuero", list("abcd"))[0] == "c"


def test_evaluacion_y_comparacion(bm25):
    queries = pd.DataFrame({"query_id": [1, 2], "query": ["zapatillas nike", "funda samsung"]})
    qrels = {1: {"a": 1.0, "c": 0.1, "d": 0.0}, 2: {"b": 1.0, "d": 0.01}}
    res = evaluate_candidates(bm25, queries, qrels)
    assert res["ndcg@10"].tolist() == [1.0, 1.0]
    tabla = compare({"BM25": res, "copia": res}, baseline="BM25")
    assert tabla.loc["copia", "Δ vs BM25"] == 0


def test_jax_no_reserva_la_gpu():
    import os

    import asistente_compras  # noqa: F401

    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
