"""Preparación de datos: ESCI (español, versión reducida) → catálogo, consultas y qrels.

Flujo:

1. ``download_esci``: baja los parquet originales (≈ 1,2 GB) a ``raw/`` si no están.
2. ``load_examples_es``: pares consulta–producto con ``small_version == 1`` y locale ``es``.
3. ``select_queries``: muestrea consultas de test (evaluación) y train (desarrollo) con
   semilla fija, separadas por ``query_id``.
4. ``build_catalog``: une todos los productos anotados de esas consultas y, si no llegan al
   mínimo, agrega distractores. Limpia HTML de descripciones y bullets.
5. ``prepare``: escribe ``processed/`` (lo que ve el agente), ``eval/`` (etiquetas) y un
   ``manifest.json`` con conteos y huellas para verificar reproducibilidad.

Uso desde terminal: ``uv run lab4-preparar-datos``.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import urllib.request
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from asistente_compras.config import (
    ESCI_BASE_URL,
    ESCI_FILES,
    DataConfig,
    Paths,
    get_paths,
)

PRODUCT_TEXT_COLS = [
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
]


# --------------------------------------------------------------------------------------
# 1. Descargar
# --------------------------------------------------------------------------------------
def download_esci(paths: Paths, force: bool = False) -> dict[str, Path]:
    """Descargar los parquet de ESCI a ``paths.raw``. No volver a bajar lo que ya existe."""
    paths.ensure()
    out: dict[str, Path] = {}
    for key, fname in ESCI_FILES.items():
        dest = paths.raw / fname
        if force or not dest.exists() or dest.stat().st_size == 0:
            tmp = dest.with_suffix(".part")
            print(f"Descargando {fname} …")
            with urllib.request.urlopen(f"{ESCI_BASE_URL}/{fname}") as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.rename(dest)
        out[key] = dest
    return out


# --------------------------------------------------------------------------------------
# 2. Cargar filtrando al leer (el parquet de productos completo pesa ~1,1 GB)
# --------------------------------------------------------------------------------------
def load_examples_es(paths: Paths, locale: str = "es") -> pd.DataFrame:
    """Pares consulta–producto de la versión reducida (Task 1) en el locale pedido."""
    df = pd.read_parquet(
        paths.raw / ESCI_FILES["examples"],
        filters=[("product_locale", "==", locale), ("small_version", "==", 1)],
        columns=["query_id", "query", "product_id", "esci_label", "split"],
    )
    return df.reset_index(drop=True)


def load_products_es(paths: Paths, locale: str = "es") -> pd.DataFrame:
    df = pd.read_parquet(
        paths.raw / ESCI_FILES["products"],
        filters=[("product_locale", "==", locale)],
        columns=["product_id", *PRODUCT_TEXT_COLS],
    )
    return df.drop_duplicates("product_id").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 3. Seleccionar las consultas
# --------------------------------------------------------------------------------------
def select_queries(examples: pd.DataFrame, cfg: DataConfig) -> dict[str, list[int]]:
    """Muestrear ``query_id`` de test y de train con semilla fija.

    Se respeta el ``split`` original de ESCI, que ya separa por consulta. Opcionalmente se
    descartan consultas cuyos candidatos son todos Irrelevant (nDCG indefinido).
    """
    rng = np.random.default_rng(cfg.seed)
    per_query = examples.groupby("query_id").agg(
        split=("split", "first"),
        has_relevant=("esci_label", lambda s: bool((s != "I").any())),
    )
    if cfg.require_relevant:
        per_query = per_query[per_query["has_relevant"]]

    selected: dict[str, list[int]] = {}
    plan = (("test", cfg.n_test_queries, "test"), ("train", cfg.n_dev_queries, "dev"))
    for split, n, name in plan:
        pool = np.sort(per_query.index[per_query["split"] == split].to_numpy())
        if len(pool) < n:
            raise ValueError(f"Solo hay {len(pool)} consultas en {split}; se pidieron {n}.")
        selected[name] = sorted(int(q) for q in rng.choice(pool, size=n, replace=False))
    return selected


# --------------------------------------------------------------------------------------
# 4. Construir el catálogo
# --------------------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Quitar HTML y espacios repetidos. ``None``/NaN → cadena vacía (atributo desconocido)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = html.unescape(_TAG_RE.sub(" ", str(value)))
    return _WS_RE.sub(" ", text).strip()


def build_catalog(
    products: pd.DataFrame,
    annotated_ids: set[str],
    distractor_pool: set[str],
    cfg: DataConfig,
) -> pd.DataFrame:
    """Catálogo = productos anotados de las consultas elegidas (+ distractores si faltan).

    La columna ``is_distractor`` solo sirve para el análisis; el agente no la usa.
    """
    missing = annotated_ids - set(products["product_id"])
    if missing:
        raise ValueError(f"{len(missing)} productos anotados no están en el parquet de productos")

    ids = sorted(annotated_ids)
    n_extra = max(0, cfg.min_catalog_size - len(ids))
    extra: list[str] = []
    if n_extra:
        rng = np.random.default_rng(cfg.seed + 1)
        pool = np.array(sorted(distractor_pool - annotated_ids))
        extra = sorted(rng.choice(pool, size=n_extra, replace=False).tolist())

    cat = products[products["product_id"].isin(set(ids) | set(extra))].copy()
    for col in PRODUCT_TEXT_COLS:
        cat[col] = cat[col].map(clean_text)
    cat["is_distractor"] = cat["product_id"].isin(set(extra))
    cat = cat.rename(
        columns={
            "product_title": "title",
            "product_description": "description",
            "product_bullet_point": "bullet_points",
            "product_brand": "brand",
            "product_color": "color",
        }
    )
    return cat.sort_values("product_id").reset_index(drop=True)


def make_qrels(examples: pd.DataFrame, query_ids: list[int], cfg: DataConfig) -> pd.DataFrame:
    q = examples[examples["query_id"].isin(query_ids)]
    q = q[["query_id", "product_id", "esci_label"]].copy()
    q["gain"] = q["esci_label"].map(cfg.gains)
    return q.sort_values(["query_id", "product_id"]).reset_index(drop=True)


def make_queries(examples: pd.DataFrame, query_ids: list[int]) -> pd.DataFrame:
    """Solo ``query_id`` y texto: es lo único de ESCI que puede ver el agente."""
    q = examples[examples["query_id"].isin(query_ids)][["query_id", "query"]]
    return q.drop_duplicates("query_id").sort_values("query_id").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 5. Orquestar el pipeline
# --------------------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """Huella del *contenido* de una tabla (no de los bytes del archivo).

    Los bytes de un parquet cambian con la versión de pyarrow; el contenido no. Así se
    puede comprobar que la Mac y Colab generan exactamente los mismos datos.
    """
    h = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def check_integrity(
    catalog: pd.DataFrame, qrels: dict[str, pd.DataFrame], queries: dict[str, pd.DataFrame]
) -> None:
    """Verificaciones que exige el enunciado; si alguna falla, se detiene la preparación."""

    def require(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(msg)

    cat_ids = set(catalog["product_id"])
    for name, qr in qrels.items():
        faltan = set(qr["product_id"]) - cat_ids
        require(not faltan, f"{name}: {len(faltan)} productos anotados fuera del catálogo")
    dev_ids, test_ids = set(queries["dev"]["query_id"]), set(queries["test"]["query_id"])
    require(not dev_ids & test_ids, "Hay query_id en dev y test a la vez")
    dev_txt = set(queries["dev"]["query"].str.lower().str.strip())
    test_txt = set(queries["test"]["query"].str.lower().str.strip())
    require(not dev_txt & test_txt, "El mismo texto de consulta aparece en dev y test")
    require(catalog["product_id"].is_unique, "product_id duplicado en el catálogo")


def prepare(cfg: DataConfig | None = None, paths: Paths | None = None) -> dict:
    """Ejecutar todo el pipeline y devolver el manifiesto."""
    cfg = cfg or DataConfig()
    paths = (paths or get_paths()).ensure()

    download_esci(paths)
    examples = load_examples_es(paths, cfg.locale)
    products = load_products_es(paths, cfg.locale)

    sel = select_queries(examples, cfg)
    queries = {k: make_queries(examples, v) for k, v in sel.items()}
    qrels = {k: make_qrels(examples, v, cfg) for k, v in sel.items()}

    annotated = set(qrels["dev"]["product_id"]) | set(qrels["test"]["product_id"])
    catalog = build_catalog(products, annotated, set(examples["product_id"]), cfg)
    check_integrity(catalog, qrels, queries)

    outputs = {
        paths.processed / "catalog.parquet": catalog,
        paths.processed / "queries_dev.parquet": queries["dev"],
        paths.processed / "queries_test.parquet": queries["test"],
        paths.eval / "qrels_dev.parquet": qrels["dev"],
        paths.eval / "qrels_test.parquet": qrels["test"],
    }
    for path, df in outputs.items():
        df.to_parquet(path, index=False)

    manifest = {
        "config": asdict(cfg),
        "counts": {
            "examples_es_small": int(len(examples)),
            "catalog": int(len(catalog)),
            "catalog_distractors": int(catalog["is_distractor"].sum()),
            "queries_dev": int(len(queries["dev"])),
            "queries_test": int(len(queries["test"])),
            "qrels_dev": int(len(qrels["dev"])),
            "qrels_test": int(len(qrels["test"])),
        },
        "label_dist_test": qrels["test"]["esci_label"].value_counts().to_dict(),
        "fingerprints": {path.name: fingerprint(df) for path, df in outputs.items()},
    }
    (paths.root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> None:
    manifest = prepare()
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
