"""Evaluación de la recuperación con las etiquetas ESCI (único módulo que lee ``eval/``)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from asistente_compras.config import Paths
from asistente_compras.metrics import ndcg_at_k


def load_qrels(paths: Paths, split: str) -> dict[int, dict[str, float]]:
    """``{query_id: {product_id: ganancia}}`` para ``split`` en {"dev", "test"}."""
    df = pd.read_parquet(paths.eval / f"qrels_{split}.parquet")
    return {int(q): dict(zip(g.product_id, g.gain, strict=True)) for q, g in df.groupby("query_id")}


def evaluate_candidates(
    retriever, queries: pd.DataFrame, qrels: dict[int, dict[str, float]], k: int = 10
) -> pd.DataFrame:
    """nDCG@k por consulta re-ordenando solo sus candidatos anotados (protocolo del enunciado).

    El orden en que llegan los candidatos no influye: cada método los ordena por sí mismo.
    """
    rows = []
    for qid, text in zip(queries.query_id, queries["query"], strict=True):
        cands = list(qrels[int(qid)])
        t0 = time.perf_counter()
        ranking = retriever.rank_candidates(text, cands)
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"query_id": int(qid), "query": text, "n_cand": len(cands),
                     f"ndcg@{k}": ndcg_at_k(ranking, qrels[int(qid)], k), "ms": ms,
                     "top3": ranking[:3]})
    return pd.DataFrame(rows)


def recall_exact_at_k(
    retriever, queries: pd.DataFrame, qrels: dict[int, dict[str, float]], k: int = 50
) -> float:
    """Métrica complementaria (no oficial): fracción de productos Exact que aparecen en los
    primeros ``k`` al buscar en **todo** el catálogo. Mide lo que realmente usa el agente.
    """
    vals = []
    for qid, text in zip(queries.query_id, queries["query"], strict=True):
        exact = {p for p, g in qrels[int(qid)].items() if g == 1.0}
        if exact:
            found = {p for p, _ in retriever.search(text, k)}
            vals.append(len(exact & found) / len(exact))
    return float(np.mean(vals))


def compare(
    results: dict[str, pd.DataFrame], baseline: str, metric: str = "ndcg@10"
) -> pd.DataFrame:
    """Tabla de resultados con IC 95 % (bootstrap) y Wilcoxon pareado frente a ``baseline``.

    Se compara consulta a consulta (las mismas consultas en todos los métodos), como pide el
    enunciado: "utilizando las mismas consultas".
    """
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(0)
    base = results[baseline][metric].to_numpy()
    rows = []
    for name, df in results.items():
        v = df[metric].to_numpy()
        boots = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(2000)]
        diff = v - base
        p = wilcoxon(v, base).pvalue if name != baseline and np.any(diff != 0) else np.nan
        rows.append({
            "método": name,
            metric: v.mean(),
            "IC95 inf": np.percentile(boots, 2.5),
            "IC95 sup": np.percentile(boots, 97.5),
            f"Δ vs {baseline}": v.mean() - base.mean(),
            "gana / empata / pierde": f"{(diff > 1e-9).sum()} / {(abs(diff) <= 1e-9).sum()} / "
                                      f"{(diff < -1e-9).sum()}",
            "p (Wilcoxon)": p,
            "ms/consulta": df["ms"].median(),
        })
    return pd.DataFrame(rows).set_index("método")


def timed(fn, *args, **kwargs):
    """Ejecutar ``fn`` y devolver (resultado, segundos). Para separar carga inicial y consulta."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def per_query_table(results: dict[str, pd.DataFrame], metric: str = "ndcg@10") -> pd.DataFrame:
    """Matriz consulta × método, útil para elegir aciertos y fallos en el análisis."""
    first = next(iter(results.values()))
    out = first[["query_id", "query", "n_cand"]].copy()
    for name, df in results.items():
        out[name] = df[metric].to_numpy()
    return out

