"""Configuración central: rutas, semillas y parámetros de la preparación de datos.

Todas las rutas cuelgan de ``LAB4_DATA_DIR``. En la Mac, por defecto es ``<proyecto>/data``.
En Colab se apunta a Google Drive para no re-descargar ni re-procesar en cada sesión::

    os.environ["LAB4_DATA_DIR"] = "/content/drive/MyDrive/lab4/data"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Resolver la raíz del proyecto (…/lab4_solution) desde src/asistente_compras/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Fijar la semilla global: todo lo aleatorio (consultas, precios, stock) deriva de ella.
SEED = 42

# Fijar las ganancias de relevancia que indica el enunciado para nDCG@10.
ESCI_GAINS: dict[str, float] = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

ESCI_BASE_URL = (
    "https://media.githubusercontent.com/media/amazon-science/esci-data/main/"
    "shopping_queries_dataset"
)
ESCI_FILES = {
    "examples": "shopping_queries_dataset_examples.parquet",
    "products": "shopping_queries_dataset_products.parquet",
}


@dataclass(frozen=True)
class Paths:
    """Estructura de carpetas de datos.

    - ``raw/``: parquet originales de ESCI (sin modificar).
    - ``processed/``: lo que puede leer el agente (catálogo y texto de las consultas).
    - ``eval/``: etiquetas ESCI (qrels). **Solo las lee el módulo de evaluación**; el agente
      nunca recibe esta ruta. Así se cumple la regla de que las etiquetas no sean consultables.
    """

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def eval(self) -> Path:
        return self.root / "eval"

    @property
    def store_db(self) -> Path:
        """Base SQLite de la tienda simulada (se regenera con ``restore_store``)."""
        return self.root / "store" / "tienda.db"

    @property
    def index(self) -> Path:
        """Embeddings del catálogo (.npy) y resultados de recuperación."""
        return self.root / "index"

    def ensure(self) -> Paths:
        for p in (self.raw, self.processed, self.eval, self.store_db.parent, self.index):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    root = Path(os.environ.get("LAB4_DATA_DIR", PROJECT_ROOT / "data"))
    return Paths(root=root)


@dataclass(frozen=True)
class DataConfig:
    """Parámetros de la preparación de datos (§2 del enunciado)."""

    locale: str = "es"
    # Tomar las consultas de evaluación (split=test). El enunciado pide ≥ 200; dejar margen.
    n_test_queries: int = 250
    # Tomar las consultas de desarrollo (split=train): ajustar el método, nunca reportar.
    n_dev_queries: int = 200
    # Exigir el mínimo del enunciado. Si los productos anotados no llegan, completar
    # con distractores (productos ES de otras consultas, sin etiqueta).
    min_catalog_size: int = 5000
    # Una consulta cuyos candidatos son todos Irrelevant tiene IDCG = 0 y su nDCG no está
    # definido; excluirla. No filtrar más (p. ej. "al menos un Exact") para no sesgar.
    require_relevant: bool = True
    seed: int = SEED
    gains: dict[str, float] = field(default_factory=lambda: dict(ESCI_GAINS))
