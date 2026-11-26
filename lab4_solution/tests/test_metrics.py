"""Pruebas de nDCG@k con casos calculables a mano."""

import math

from asistente_compras.metrics import mean_ndcg, ndcg_at_k

QREL = {"e1": 1.0, "s1": 0.1, "c1": 0.01, "i1": 0.0}


def test_ranking_ideal_da_uno():
    assert ndcg_at_k(["e1", "s1", "c1", "i1"], QREL) == 1.0


def test_ranking_invertido_menor_que_uno():
    assert ndcg_at_k(["i1", "c1", "s1", "e1"], QREL) < 0.5


def test_valor_a_mano():
    # Exact en la posición 2 y nada más relevante: (1/log2(3)) / ideal
    ideal = 1 + 0.1 / math.log2(3) + 0.01 / 2
    esperado = (1 / math.log2(3)) / ideal
    assert math.isclose(ndcg_at_k(["i1", "e1"], QREL), esperado)


def test_consulta_sin_relevantes_es_nan():
    assert math.isnan(ndcg_at_k(["x"], {"x": 0.0}))


def test_promedio_ignora_indefinidas():
    qrels = {1: QREL, 2: {"x": 0.0}}
    assert mean_ndcg({1: ["e1", "s1", "c1"], 2: ["x"]}, qrels) == 1.0
