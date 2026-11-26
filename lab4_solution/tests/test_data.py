"""Pruebas de la preparación de datos con un mini-ESCI sintético (no requiere descargas)."""

import pandas as pd
import pytest

from asistente_compras.config import DataConfig
from asistente_compras.data import (
    build_catalog,
    check_integrity,
    clean_text,
    make_qrels,
    make_queries,
    select_queries,
)


@pytest.fixture
def examples():
    rows = []
    for qid in range(20):
        split = "test" if qid < 10 else "train"
        for j in range(3):
            label = "I" if qid == 0 else ["E", "S", "I"][j]  # la consulta 0 no tiene relevantes
            rows.append((qid, f"consulta {qid}", f"p{qid}_{j}", label, split))
    return pd.DataFrame(rows, columns=["query_id", "query", "product_id", "esci_label", "split"])


@pytest.fixture
def products(examples):
    ids = list(examples["product_id"]) + [f"extra{i}" for i in range(50)]
    return pd.DataFrame(
        {
            "product_id": ids,
            "product_title": "Título <b>x</b>",
            "product_description": None,
            "product_bullet_point": "a&amp;b",
            "product_brand": "M",
            "product_color": "",
        }
    )


CFG = DataConfig(n_test_queries=5, n_dev_queries=5, min_catalog_size=40)


def test_seleccion_reproducible_y_sin_irrelevantes(examples):
    a, b = select_queries(examples, CFG), select_queries(examples, CFG)
    assert a == b
    assert 0 not in a["test"]
    assert not set(a["test"]) & set(a["dev"])


def test_catalogo_contiene_anotados_y_completa_minimo(examples, products):
    sel = select_queries(examples, CFG)
    qrels = {k: make_qrels(examples, v, CFG) for k, v in sel.items()}
    queries = {k: make_queries(examples, v) for k, v in sel.items()}
    anotados = set(qrels["dev"]["product_id"]) | set(qrels["test"]["product_id"])
    cat = build_catalog(products, anotados, set(products["product_id"]), CFG)
    assert len(cat) == CFG.min_catalog_size
    assert anotados <= set(cat["product_id"])
    check_integrity(cat, qrels, queries)  # no debe lanzar


def test_qrels_ganancias(examples):
    qr = make_qrels(examples, [1], CFG)
    assert dict(zip(qr["esci_label"], qr["gain"], strict=True)) == {"E": 1.0, "S": 0.1, "I": 0.0}


def test_clean_text():
    assert clean_text("<p>Hola&nbsp;  mundo</p>") == "Hola mundo"
    assert clean_text("a&amp;b") == "a&b"
    assert clean_text(None) == ""
    assert clean_text("a<br/>b") == "a b"
