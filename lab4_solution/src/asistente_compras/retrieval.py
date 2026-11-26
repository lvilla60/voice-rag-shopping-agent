"""Recuperación sobre el catálogo: BM25 (línea base), denso (e5), híbrido (RRF) y re-ranker.

Todos los métodos exponen la misma interfaz:

- ``search(query, k)`` → ``[(product_id, score), …]`` sobre **todo** el catálogo. Es lo que usa
  la herramienta ``search_products`` del agente.
- ``rank_candidates(query, pids)`` → ``[product_id, …]`` re-ordenando **solo** los candidatos
  dados. Es lo que mide el nDCG@10 oficial del enunciado (candidatos anotados de cada consulta).

Diseño del sistema propuesto::

    consulta ─┬─ BM25 (léxico: marcas, modelos, códigos) ─┐
              └─ e5 denso (semántico: sinónimos, intención) ┴─ RRF → top-N → re-ranker → ranking

Solo se indexa texto del catálogo. Precio y stock nunca están en el índice: los consulta el
agente en la tienda en el momento (``store.get_price_stock``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Construir el texto de cada producto
# --------------------------------------------------------------------------------------
FIELD_SETS: dict[str, tuple[str, ...]] = {
    "titulo": ("title",),
    "titulo_marca": ("title", "brand", "color"),
    "titulo_marca_bullets": ("title", "brand", "color", "bullet_points"),
    "completo": ("title", "brand", "color", "bullet_points", "description"),
}


@dataclass(frozen=True)
class TextConfig:
    """Qué campos del catálogo forman el documento y cuántas palabras se conservan."""

    fields: str = "titulo_marca_bullets"
    max_words: int = 200

    def key(self) -> str:
        return f"{self.fields}-w{self.max_words}"


def product_texts(catalog: pd.DataFrame, cfg: TextConfig) -> list[str]:
    """Concatenar los campos elegidos ("Marca: X. Color: Y.") y truncar por palabras.

    Los campos vacíos se omiten: un atributo ausente no se rellena (será "desconocido").
    """
    labels = {"brand": "Marca", "color": "Color"}
    out = []
    for row in catalog[list(FIELD_SETS[cfg.fields])].itertuples(index=False):
        parts = []
        for name, value in zip(FIELD_SETS[cfg.fields], row, strict=True):
            if value:
                parts.append(f"{labels[name]}: {value}." if name in labels else str(value))
        words = " ".join(parts).split()
        out.append(" ".join(words[: cfg.max_words]))
    return out


# --------------------------------------------------------------------------------------
# Interfaz común
# --------------------------------------------------------------------------------------
class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]: ...

    def rank_candidates(self, query: str, pids: Sequence[str]) -> list[str]: ...


class _ScoredRetriever:
    """Base para métodos que puntúan todo el catálogo de una vez (BM25 y denso)."""

    name = "base"

    def __init__(self, product_ids: Sequence[str]):
        self.pids = np.asarray(product_ids)
        self.pos = {p: i for i, p in enumerate(self.pids)}

    def scores(self, query: str) -> np.ndarray:  # puntaje para cada producto del catálogo
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        s = self.scores(query)
        top = np.argsort(-s, kind="stable")[:k]
        return [(str(self.pids[i]), float(s[i])) for i in top]

    def rank_candidates(self, query: str, pids: Sequence[str]) -> list[str]:
        s = self.scores(query)
        cand = sorted(pids)  # fijar un orden base ⇒ desempates deterministas
        return sorted(cand, key=lambda p: -s[self.pos[p]])


# --------------------------------------------------------------------------------------
# 1. BM25: línea base léxica
# --------------------------------------------------------------------------------------
class BM25Retriever(_ScoredRetriever):
    """BM25 (Okapi) con stopwords y stemming Snowball en español (``bm25s`` + PyStemmer)."""

    name = "BM25"

    def __init__(self, product_ids: Sequence[str], texts: Sequence[str], k1=1.5, b=0.75):
        import bm25s
        import Stemmer

        super().__init__(product_ids)
        self._bm25s = bm25s
        self.stemmer = Stemmer.Stemmer("spanish")
        tokens = bm25s.tokenize(list(texts), stopwords="es", stemmer=self.stemmer,
                                show_progress=False)
        self.model = bm25s.BM25(k1=k1, b=b)
        self.model.index(tokens, show_progress=False)

    def scores(self, query: str) -> np.ndarray:
        tok = self._bm25s.tokenize([query], stopwords="es", stemmer=self.stemmer,
                                   show_progress=False, return_ids=False)[0]
        if not tok:  # cubrir una consulta formada solo por stopwords
            return np.zeros(len(self.pids))
        return np.asarray(self.model.get_scores(tok))


# --------------------------------------------------------------------------------------
# 2. Denso: embeddings multilingües (e5)
# --------------------------------------------------------------------------------------
def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DenseRetriever(_ScoredRetriever):
    """Bi-encoder e5: consulta y producto se codifican por separado y se compara el coseno.

    Los embeddings del catálogo se calculan una vez y se guardan en ``.npy`` (cache por
    modelo + configuración de texto), así cada sesión de Colab no los recalcula.
    """

    name = "Denso (e5)"

    def __init__(
        self,
        product_ids: Sequence[str],
        texts: Sequence[str],
        model_name: str = "intfloat/multilingual-e5-base",
        cache_dir: Path | None = None,
        cache_key: str = "",
        device: str | None = None,
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        super().__init__(product_ids)
        self.device = device or pick_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        if self.device.startswith("cuda"):  # usar media precisión en GPU: la mitad de memoria
            self.model.half()
        self.model.max_seq_length = 512
        self.emb = self._load_or_encode(texts, model_name, cache_dir, cache_key, batch_size)

    def _load_or_encode(self, texts, model_name, cache_dir, cache_key, batch_size) -> np.ndarray:
        cache = None
        if cache_dir is not None:
            digest = hashlib.sha256(("\n".join(texts)).encode()).hexdigest()[:10]
            slug = model_name.split("/")[-1]
            cache = Path(cache_dir) / f"{slug}_{cache_key}_{digest}.npy"
            if cache.exists():
                return np.load(cache)
        # Anteponer "passage: " a los documentos y "query: " a las consultas (lo exige e5).
        emb = self.model.encode(
            [f"passage: {t}" for t in texts], batch_size=batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
        ).astype(np.float32)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, emb)
        return emb

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(f"query: {query}", normalize_embeddings=True,
                                 convert_to_numpy=True)

    def scores(self, query: str) -> np.ndarray:
        return self.emb @ self.encode_query(query)


# --------------------------------------------------------------------------------------
# 3. Híbrido: Reciprocal Rank Fusion
# --------------------------------------------------------------------------------------
def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """RRF (Cormack et al., SIGIR 2009): suma de 1/(k + posición). Combina por posición, sin
    normalizar puntajes de escalas distintas (BM25 no está acotado; el coseno sí). k = 60 es el
    valor estándar del paper; no se ajusta.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for r, pid in enumerate(ranking, start=1):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (k + r)
    return fused


class HybridRetriever:
    name = "Híbrido (BM25 + e5, RRF)"

    def __init__(self, lexical: _ScoredRetriever, dense: _ScoredRetriever, k_rrf: int = 60,
                 depth: int = 200):
        self.lexical, self.dense, self.k_rrf, self.depth = lexical, dense, k_rrf, depth

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        runs = [[p for p, _ in r.search(query, self.depth)] for r in (self.lexical, self.dense)]
        fused = rrf(runs, self.k_rrf)
        top = sorted(fused, key=lambda p: (-fused[p], p))[:k]
        return [(p, fused[p]) for p in top]

    def rank_candidates(self, query: str, pids: Sequence[str]) -> list[str]:
        runs = [r.rank_candidates(query, pids) for r in (self.lexical, self.dense)]
        fused = rrf(runs, self.k_rrf)
        return sorted(pids, key=lambda p: (-fused[p], p))


# --------------------------------------------------------------------------------------
# 4. Re-ranker: cross-encoder sobre los primeros N
# --------------------------------------------------------------------------------------
class RerankRetriever:
    """Cross-encoder (bge-reranker-v2-m3): lee consulta y producto **juntos**, por eso es más
    preciso que el bi-encoder, pero también más caro. Solo se aplica a los ``top_n`` del
    híbrido; el resto conserva el orden del híbrido.
    """

    name = "Híbrido + re-ranker"
    calibrated = True  # puntajes en [0, 1] (sigmoide del cross-encoder)

    def __init__(
        self,
        first_stage: HybridRetriever,
        product_ids: Sequence[str],
        texts: Sequence[str],
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 50,
        max_length: int = 384,
        device: str | None = None,
        batch_size: int = 32,
    ):
        from sentence_transformers import CrossEncoder

        self.first, self.top_n, self.batch_size = first_stage, top_n, batch_size
        self.text = dict(zip(product_ids, texts, strict=True))
        device = device or pick_device()
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        if device.startswith("cuda"):  # usar fp16 en GPU: ~1,1 GB menos, mismas posiciones
            getattr(self.model, "model", self.model).half()

    def _rerank(self, query: str, ranking: list[str]) -> list[tuple[str, float]]:
        head, tail = ranking[: self.top_n], ranking[self.top_n :]
        scores = self.model.predict([(query, self.text[p]) for p in head],
                                    batch_size=self.batch_size, show_progress_bar=False)
        order = sorted(zip(head, scores, strict=True), key=lambda x: (-x[1], x[0]))
        return [(p, float(s)) for p, s in order] + [(p, float("-inf")) for p in tail]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        first = [p for p, _ in self.first.search(query, self.top_n)]
        return self._rerank(query, first)[:k]

    def rank_candidates(self, query: str, pids: Sequence[str]) -> list[str]:
        return [p for p, _ in self._rerank(query, self.first.rank_candidates(query, pids))]


# --------------------------------------------------------------------------------------
# Configuración congelada en el notebook 03 (elegida con las consultas de desarrollo)
# --------------------------------------------------------------------------------------
FROZEN_BM25 = TextConfig(fields="titulo_marca")
FROZEN_DENSE = TextConfig(fields="titulo_marca_bullets")


def build_system_retriever(catalog: pd.DataFrame, index_dir: Path, rerank: bool = True):
    """Sistema propuesto (híbrido → re-ranker) con la configuración congelada.

    ``rerank=False`` devuelve la variante de la ablación (solo híbrido).
    """
    dense_texts = product_texts(catalog, FROZEN_DENSE)
    bm25 = BM25Retriever(catalog.product_id, product_texts(catalog, FROZEN_BM25))
    dense = DenseRetriever(catalog.product_id, dense_texts, cache_dir=index_dir,
                           cache_key=FROZEN_DENSE.key())
    hybrid = HybridRetriever(bm25, dense)
    if not rerank:
        return hybrid
    return RerankRetriever(hybrid, catalog.product_id, dense_texts)
