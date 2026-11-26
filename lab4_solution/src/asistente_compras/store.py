"""Tienda simulada en SQLite: precios, stock, carritos, cotizaciones y pedidos.

Principios de diseño (§3 del enunciado):

- **Datos ficticios con semilla fija y restaurables.** ``restore_store`` regenera la tienda
  desde el catálogo; misma semilla ⇒ mismos precios y stock (se verifica con una huella).
- **Las validaciones viven en el código, no en el LLM.** Cada operación valida cantidades,
  existencia, stock y presupuesto, y devuelve un ``Result`` estructurado con errores legibles
  que el agente puede explicar al usuario.
- **Dinero en céntimos enteros** (sin errores de coma flotante). Moneda: EUR (Amazon ES).
- **Confirmación atómica e idempotente.** ``confirm_order`` re-verifica stock, precio y
  presupuesto dentro de una transacción ``BEGIN IMMEDIATE``: o se aplica todo o nada. Una
  segunda confirmación de la misma cotización devuelve el mismo pedido (``UNIQUE(quote_id)``).
- **Bitácora de operaciones** (``ops_log``): el enunciado pide conservar las operaciones
  ejecutadas en cada escenario.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asistente_compras.config import SEED

CURRENCY = "EUR"
MAX_QTY_PER_LINE = 50

SCHEMA = """
CREATE TABLE products (
    product_id   TEXT PRIMARY KEY,
    price_cents  INTEGER NOT NULL CHECK (price_cents > 0),
    stock        INTEGER NOT NULL CHECK (stock >= 0)
);
CREATE TABLE carts (
    cart_id      TEXT PRIMARY KEY
);
CREATE TABLE cart_items (
    cart_id      TEXT NOT NULL REFERENCES carts(cart_id),
    product_id   TEXT NOT NULL REFERENCES products(product_id),
    qty          INTEGER NOT NULL CHECK (qty > 0),
    PRIMARY KEY (cart_id, product_id)
);
CREATE TABLE quotes (
    quote_id     TEXT PRIMARY KEY,
    cart_id      TEXT NOT NULL REFERENCES carts(cart_id),
    total_cents  INTEGER NOT NULL,
    budget_cents INTEGER,
    status       TEXT NOT NULL CHECK (status IN ('open', 'ordered', 'superseded'))
);
CREATE TABLE quote_items (
    quote_id         TEXT NOT NULL REFERENCES quotes(quote_id),
    product_id       TEXT NOT NULL REFERENCES products(product_id),
    qty              INTEGER NOT NULL CHECK (qty > 0),
    unit_price_cents INTEGER NOT NULL,
    PRIMARY KEY (quote_id, product_id)
);
CREATE TABLE orders (
    order_id     TEXT PRIMARY KEY,
    quote_id     TEXT NOT NULL UNIQUE REFERENCES quotes(quote_id),
    total_cents  INTEGER NOT NULL
);
CREATE TABLE order_items (
    order_id         TEXT NOT NULL REFERENCES orders(order_id),
    product_id       TEXT NOT NULL REFERENCES products(product_id),
    qty              INTEGER NOT NULL CHECK (qty > 0),
    unit_price_cents INTEGER NOT NULL,
    PRIMARY KEY (order_id, product_id)
);
CREATE TABLE ops_log (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    op      TEXT NOT NULL,
    args    TEXT NOT NULL,
    ok      INTEGER NOT NULL,
    result  TEXT NOT NULL
);
"""


# --------------------------------------------------------------------------------------
# Generación de datos ficticios
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StoreConfig:
    """Parámetros de la tienda ficticia. Todo deriva de ``seed``."""

    seed: int = SEED
    # Precio log-normal: mediana ≈ 20 €, cola larga hasta cientos de euros.
    price_median_eur: float = 20.0
    price_sigma: float = 1.0
    price_min_eur: float = 1.99
    price_max_eur: float = 1500.0
    # Stock: una fracción agotada, otra con pocas unidades (para probar cantidades) y el resto
    # con stock holgado.
    p_out_of_stock: float = 0.10
    p_low_stock: float = 0.20
    low_stock_max: int = 3
    high_stock_max: int = 40


def generate_store_data(product_ids: Iterable[str], cfg: StoreConfig) -> pd.DataFrame:
    """Precios y stock deterministas para cada producto (ordenados por ``product_id``)."""
    ids = np.array(sorted(set(product_ids)))
    rng = np.random.default_rng(cfg.seed)
    n = len(ids)

    price = rng.lognormal(mean=np.log(cfg.price_median_eur), sigma=cfg.price_sigma, size=n)
    price = np.clip(price, cfg.price_min_eur, cfg.price_max_eur)
    # Precios "comerciales": terminan en ,99 (p. ej. 23,99 €).
    price_cents = (np.floor(price) * 100 + 99).astype(int)

    u = rng.random(n)
    stock = np.where(
        u < cfg.p_out_of_stock,
        0,
        np.where(
            u < cfg.p_out_of_stock + cfg.p_low_stock,
            rng.integers(1, cfg.low_stock_max + 1, size=n),
            rng.integers(cfg.low_stock_max + 1, cfg.high_stock_max + 1, size=n),
        ),
    )
    return pd.DataFrame({"product_id": ids, "price_cents": price_cents, "stock": stock})


def restore_store(db_path: Path, product_ids: Iterable[str], cfg: StoreConfig | None = None) -> str:
    """Borrar y regenerar la tienda en ``db_path``. Devolver la huella del estado inicial."""
    cfg = cfg or StoreConfig()
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)

    data = generate_store_data(product_ids, cfg)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)
        con.executemany(
            "INSERT INTO products VALUES (?, ?, ?)",
            data.itertuples(index=False, name=None),
        )
        con.commit()
    finally:
        con.close()
    return Store(db_path).fingerprint()


# --------------------------------------------------------------------------------------
# Resultado de cada operación (lo que recibe el agente)
# --------------------------------------------------------------------------------------
@dataclass
class Result:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "errors": self.errors}


def fmt_eur(cents: int) -> str:
    """1234 → '12,34 €' (formato español)."""
    return f"{cents // 100:,}".replace(",", ".") + f",{cents % 100:02d} €"


# --------------------------------------------------------------------------------------
# Operaciones de la tienda
# --------------------------------------------------------------------------------------
class Store:
    """Acceso a la tienda. Cada método público valida, ejecuta y registra en ``ops_log``."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        # isolation_level=None: controlar las transacciones a mano (BEGIN IMMEDIATE).
        self.con = sqlite3.connect(self.db_path, isolation_level=None)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.con.close()

    # ---------- utilidades internas ----------
    def _log(self, op: str, args: dict, res: Result) -> Result:
        self.con.execute(
            "INSERT INTO ops_log (op, args, ok, result) VALUES (?, ?, ?, ?)",
            (op, json.dumps(args, ensure_ascii=False), int(res.ok),
             json.dumps(res.to_dict(), ensure_ascii=False)),
        )
        return res

    def _next_id(self, table: str, col: str, prefix: str) -> str:
        # Generar IDs secuenciales y deterministas (Q0001, O0001…): reproducibles entre corridas.
        n = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return f"{prefix}{n + 1:04d}"

    def _product(self, pid: str) -> sqlite3.Row | None:
        return self.con.execute(
            "SELECT product_id, price_cents, stock FROM products WHERE product_id = ?", (pid,)
        ).fetchone()

    def _cart_lines(self, cart_id: str) -> list[sqlite3.Row]:
        return self.con.execute(
            """SELECT ci.product_id, ci.qty, p.price_cents, p.stock
               FROM cart_items ci JOIN products p USING (product_id)
               WHERE ci.cart_id = ? ORDER BY ci.product_id""",
            (cart_id,),
        ).fetchall()

    # ---------- consulta ----------
    def get_price_stock(self, product_ids: list[str]) -> Result:
        """Precio y stock **actuales** (siempre de la BD, nunca del texto indexado)."""
        items, errors = [], []
        for pid in product_ids:
            row = self._product(pid)
            if row is None:
                errors.append({"code": "unknown_product", "product_id": pid})
                continue
            items.append({
                "product_id": pid,
                "price_cents": row["price_cents"],
                "price": fmt_eur(row["price_cents"]),
                "stock": row["stock"],
                "available": row["stock"] > 0,
            })
        res = Result(ok=not errors, data={"items": items}, errors=errors)
        return self._log("get_price_stock", {"product_ids": product_ids}, res)

    # ---------- carrito ----------
    def set_cart_qty(self, cart_id: str, product_id: str, qty: int) -> Result:
        """Fijar la cantidad de un producto en el carrito (0 = quitar). Validar al escribir."""
        args = {"cart_id": cart_id, "product_id": product_id, "qty": qty}
        errors = []
        if not isinstance(qty, int) or isinstance(qty, bool) or qty < 0:
            errors.append({"code": "invalid_qty", "qty": qty})
        elif qty > MAX_QTY_PER_LINE:
            errors.append({"code": "qty_over_limit", "qty": qty, "max": MAX_QTY_PER_LINE})
        row = self._product(product_id)
        if row is None:
            errors.append({"code": "unknown_product", "product_id": product_id})
        elif not errors and qty > row["stock"]:
            # Agotado ⇒ el agente debe buscar alternativa; insuficiente ⇒ puede bajar la cantidad.
            code = "out_of_stock" if row["stock"] == 0 else "insufficient_stock"
            errors.append({"code": code, "product_id": product_id,
                           "requested": qty, "stock": row["stock"]})
        if errors:
            return self._log("set_cart_qty", args, Result(ok=False, errors=errors))

        self.con.execute("BEGIN IMMEDIATE")
        self.con.execute("INSERT OR IGNORE INTO carts (cart_id) VALUES (?)", (cart_id,))
        if qty == 0:
            self.con.execute(
                "DELETE FROM cart_items WHERE cart_id = ? AND product_id = ?", (cart_id, product_id)
            )
        else:
            self.con.execute(
                """INSERT INTO cart_items (cart_id, product_id, qty) VALUES (?, ?, ?)
                   ON CONFLICT (cart_id, product_id) DO UPDATE SET qty = excluded.qty""",
                (cart_id, product_id, qty),
            )
        # Un carrito modificado invalida sus cotizaciones abiertas.
        self.con.execute(
            "UPDATE quotes SET status = 'superseded' WHERE cart_id = ? AND status = 'open'",
            (cart_id,),
        )
        self.con.execute("COMMIT")
        return self._log("set_cart_qty", args, Result(ok=True, data=self.view_cart(cart_id)))

    def view_cart(self, cart_id: str) -> dict[str, Any]:
        lines = [
            {"product_id": r["product_id"], "qty": r["qty"], "unit_price_cents": r["price_cents"],
             "line_total_cents": r["qty"] * r["price_cents"], "stock": r["stock"]}
            for r in self._cart_lines(cart_id)
        ]
        total = sum(line["line_total_cents"] for line in lines)
        return {"cart_id": cart_id, "lines": lines, "total_cents": total, "total": fmt_eur(total)}

    # ---------- cotización ----------
    def create_quote(self, cart_id: str, budget_cents: int | None = None) -> Result:
        """Congelar precios y cantidades del carrito. Validar stock y presupuesto."""
        args = {"cart_id": cart_id, "budget_cents": budget_cents}
        lines = self._cart_lines(cart_id)
        errors = []
        if not lines:
            errors.append({"code": "empty_cart"})
        for r in lines:
            if r["qty"] > r["stock"]:
                errors.append({"code": "insufficient_stock", "product_id": r["product_id"],
                               "requested": r["qty"], "stock": r["stock"]})
        total = sum(r["qty"] * r["price_cents"] for r in lines)
        if budget_cents is not None and total > budget_cents:
            errors.append({"code": "over_budget", "total_cents": total,
                           "budget_cents": budget_cents, "excess_cents": total - budget_cents})
        if errors:
            return self._log("create_quote", args, Result(ok=False, errors=errors))

        self.con.execute("BEGIN IMMEDIATE")
        self.con.execute(
            "UPDATE quotes SET status = 'superseded' WHERE cart_id = ? AND status = 'open'",
            (cart_id,),
        )
        quote_id = self._next_id("quotes", "quote_id", "Q")
        self.con.execute(
            "INSERT INTO quotes VALUES (?, ?, ?, ?, 'open')",
            (quote_id, cart_id, total, budget_cents),
        )
        self.con.executemany(
            "INSERT INTO quote_items VALUES (?, ?, ?, ?)",
            [(quote_id, r["product_id"], r["qty"], r["price_cents"]) for r in lines],
        )
        self.con.execute("COMMIT")
        data = {"quote_id": quote_id, "total_cents": total, "total": fmt_eur(total),
                "budget_cents": budget_cents, "lines": self.view_cart(cart_id)["lines"]}
        return self._log("create_quote", args, Result(ok=True, data=data))

    # ---------- confirmación ----------
    def confirm_order(self, quote_id: str) -> Result:
        """Convertir una cotización en pedido: todo o nada, y sin duplicados.

        1. Si la cotización ya tiene pedido → devuelve ese pedido (``duplicate=True``).
        2. Re-verifica cada línea: stock suficiente y precio sin cambios; y el presupuesto.
        3. Si algo falló → ROLLBACK, no se toca nada y se devuelven los conflictos.
        4. Si no → descuenta stock, crea el pedido, vacía el carrito y hace COMMIT.
        """
        args = {"quote_id": quote_id}
        self.con.execute("BEGIN IMMEDIATE")  # bloquear escrituras concurrentes hasta COMMIT
        try:
            existing = self.con.execute(
                "SELECT order_id, total_cents FROM orders WHERE quote_id = ?", (quote_id,)
            ).fetchone()
            if existing:
                self.con.execute("ROLLBACK")
                data = {"order_id": existing["order_id"], "total_cents": existing["total_cents"],
                        "total": fmt_eur(existing["total_cents"]), "duplicate": True}
                return self._log("confirm_order", args, Result(ok=True, data=data))

            quote = self.con.execute(
                "SELECT * FROM quotes WHERE quote_id = ?", (quote_id,)
            ).fetchone()
            if quote is None or quote["status"] != "open":
                self.con.execute("ROLLBACK")
                code = "unknown_quote" if quote is None else f"quote_{quote['status']}"
                return self._log("confirm_order", args,
                                 Result(ok=False, errors=[{"code": code, "quote_id": quote_id}]))

            lines = self.con.execute(
                """SELECT qi.product_id, qi.qty, qi.unit_price_cents, p.price_cents, p.stock
                   FROM quote_items qi JOIN products p USING (product_id)
                   WHERE qi.quote_id = ?""",
                (quote_id,),
            ).fetchall()
            errors = []
            for r in lines:
                if r["stock"] < r["qty"]:
                    errors.append({"code": "stock_changed", "product_id": r["product_id"],
                                   "requested": r["qty"], "stock": r["stock"]})
                if r["price_cents"] != r["unit_price_cents"]:
                    errors.append({"code": "price_changed", "product_id": r["product_id"],
                                   "quoted_cents": r["unit_price_cents"],
                                   "current_cents": r["price_cents"]})
            if quote["budget_cents"] is not None and quote["total_cents"] > quote["budget_cents"]:
                errors.append({"code": "over_budget", "total_cents": quote["total_cents"],
                               "budget_cents": quote["budget_cents"]})
            if errors:
                self.con.execute("ROLLBACK")
                return self._log("confirm_order", args, Result(ok=False, errors=errors))

            order_id = self._next_id("orders", "order_id", "O")
            for r in lines:
                self.con.execute(
                    "UPDATE products SET stock = stock - ? WHERE product_id = ?",
                    (r["qty"], r["product_id"]),
                )
            self.con.execute(
                "INSERT INTO orders VALUES (?, ?, ?)", (order_id, quote_id, quote["total_cents"])
            )
            self.con.executemany(
                "INSERT INTO order_items VALUES (?, ?, ?, ?)",
                [(order_id, r["product_id"], r["qty"], r["unit_price_cents"]) for r in lines],
            )
            self.con.execute("UPDATE quotes SET status = 'ordered' WHERE quote_id = ?", (quote_id,))
            self.con.execute("DELETE FROM cart_items WHERE cart_id = ?", (quote["cart_id"],))
            self.con.execute("COMMIT")
        except Exception:
            # Deshacer todo ante cualquier error inesperado (incluido un CHECK de stock negativo).
            if self.con.in_transaction:
                self.con.execute("ROLLBACK")
            raise
        data = {"order_id": order_id, "quote_id": quote_id, "total_cents": quote["total_cents"],
                "total": fmt_eur(quote["total_cents"]), "duplicate": False}
        return self._log("confirm_order", args, Result(ok=True, data=data))

    # ---------- hooks de prueba (no son herramientas del agente) ----------
    def log_event(self, op: str, args: dict) -> None:
        """Registrar en la bitácora una acción del sistema que no es una operación de tienda."""
        self._log(op, args, Result(ok=True))

    def set_stock(self, product_id: str, stock: int) -> None:
        """Evento externo: otro cliente compró o llegó mercadería (escenarios 4 y 6)."""
        self.con.execute("UPDATE products SET stock = ? WHERE product_id = ?", (stock, product_id))
        self._log("external_set_stock", {"product_id": product_id, "stock": stock},
                  Result(ok=True))

    def set_price(self, product_id: str, price_cents: int) -> None:
        self.con.execute(
            "UPDATE products SET price_cents = ? WHERE product_id = ?", (price_cents, product_id)
        )
        self._log("external_set_price", {"product_id": product_id, "price_cents": price_cents},
                  Result(ok=True))

    # ---------- estado y reproducibilidad ----------
    def ops(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM ops_log ORDER BY seq", self.con)

    def summary(self) -> dict[str, Any]:
        q = self.con.execute
        return {
            "products": q("SELECT COUNT(*) FROM products").fetchone()[0],
            "out_of_stock": q("SELECT COUNT(*) FROM products WHERE stock = 0").fetchone()[0],
            "units_in_stock": q("SELECT SUM(stock) FROM products").fetchone()[0],
            "quotes": q("SELECT COUNT(*) FROM quotes").fetchone()[0],
            "orders": q("SELECT COUNT(*) FROM orders").fetchone()[0],
        }

    def fingerprint(self) -> str:
        """Huella del inventario (precios + stock). Igual huella ⇒ mismo estado de tienda."""
        rows = self.con.execute(
            "SELECT product_id, price_cents, stock FROM products ORDER BY product_id"
        ).fetchall()
        return hashlib.sha256(repr([tuple(r) for r in rows]).encode()).hexdigest()[:16]
