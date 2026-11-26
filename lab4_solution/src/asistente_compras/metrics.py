"""Métricas de recuperación.

nDCG@k sobre los candidatos anotados, con las ganancias del enunciado
(E = 1, S = 0,1, C = 0,01, I = 0). Se usa la ganancia lineal ``rel / log2(pos + 1)``,
igual que trec_eval, que es lo que usa el baseline oficial de ESCI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def dcg(gains: Sequence[float], k: int) -> float:
    g = np.asarray(gains[:k], dtype=float)
    if g.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, g.size + 2))
    return float((g / discounts).sum())


def ndcg_at_k(ranking: Sequence[str], qrel: Mapping[str, float], k: int = 10) -> float:
    """nDCG@k de una consulta.

    ``ranking``: product_id ordenados por el sistema. ``qrel``: product_id → ganancia.
    Los productos del ranking sin juicio cuentan como ganancia 0.
    Devuelve NaN si la consulta no tiene ningún producto con ganancia > 0.
    """
    ideal = dcg(sorted(qrel.values(), reverse=True), k)
    if ideal == 0:
        return float("nan")
    return dcg([qrel.get(pid, 0.0) for pid in ranking], k) / ideal


def mean_ndcg(
    rankings: Mapping[int, Sequence[str]],
    qrels: Mapping[int, Mapping[str, float]],
    k: int = 10,
) -> float:
    """Promedio de nDCG@k sobre las consultas de ``qrels`` (ignora las indefinidas)."""
    vals = [ndcg_at_k(rankings.get(q, []), qrels[q], k) for q in qrels]
    return float(np.nanmean(vals))
