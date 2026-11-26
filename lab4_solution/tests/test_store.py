"""Pruebas de la tienda: validaciones en código, atomicidad, idempotencia y restauración."""

import sqlite3

import pytest

from asistente_compras.store import Store, StoreConfig, fmt_eur, generate_store_data, restore_store

IDS = [f"P{i:03d}" for i in range(200)]


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "tienda.db"
    restore_store(db, IDS)
    s = Store(db)
    # Estado conocido para las pruebas, independiente de los valores aleatorios.
    for pid, price, stock in (("P001", 1000, 5), ("P002", 2500, 2), ("P003", 500, 0)):
        s.con.execute(
            "UPDATE products SET price_cents = ?, stock = ? WHERE product_id = ?",
            (price, stock, pid),
        )
    yield s
    s.close()


# ---------- generación y restauración ----------
def test_generacion_determinista():
    a, b = generate_store_data(IDS, StoreConfig()), generate_store_data(IDS, StoreConfig())
    assert a.equals(b)
    assert not a.equals(generate_store_data(IDS, StoreConfig(seed=7)))


def test_generacion_rangos():
    df = generate_store_data([f"X{i}" for i in range(5000)], StoreConfig())
    assert (df.price_cents % 100 == 99).all()
    assert df.stock.min() == 0 and 0.05 < (df.stock == 0).mean() < 0.15


def test_restore_vuelve_al_estado_inicial(tmp_path):
    db = tmp_path / "t.db"
    h0 = restore_store(db, IDS)
    s = Store(db)
    s.set_stock("P010", 0)
    assert s.fingerprint() != h0
    s.close()
    assert restore_store(db, IDS) == h0


# ---------- carrito ----------
@pytest.mark.parametrize("qty", [-1, 2.5, "3", True])
def test_cantidad_invalida_se_rechaza(store, qty):
    res = store.set_cart_qty("c1", "P001", qty)
    assert not res.ok and res.errors[0]["code"] == "invalid_qty"
    assert store.view_cart("c1")["lines"] == []


def test_no_se_agrega_mas_que_el_stock(store):
    res = store.set_cart_qty("c1", "P002", 3)
    assert not res.ok and res.errors[0]["code"] == "insufficient_stock"


def test_producto_agotado_tiene_codigo_propio(store):
    assert store.set_cart_qty("c1", "P003", 1).errors[0]["code"] == "out_of_stock"


def test_producto_inexistente(store):
    assert store.set_cart_qty("c1", "NOEXISTE", 1).errors[0]["code"] == "unknown_product"


def test_cambiar_cantidad_y_quitar(store):
    store.set_cart_qty("c1", "P001", 2)
    store.set_cart_qty("c1", "P001", 4)
    assert store.view_cart("c1")["total_cents"] == 4000
    store.set_cart_qty("c1", "P001", 0)
    assert store.view_cart("c1")["lines"] == []


# ---------- cotización ----------
def test_presupuesto_excedido(store):
    store.set_cart_qty("c1", "P001", 3)  # 30,00 €
    res = store.create_quote("c1", budget_cents=2500)
    assert not res.ok and res.errors[0]["code"] == "over_budget"
    assert res.errors[0]["excess_cents"] == 500


def test_cotizar_carrito_vacio(store):
    assert store.create_quote("vacio").errors[0]["code"] == "empty_cart"


def test_modificar_carrito_invalida_cotizacion(store):
    store.set_cart_qty("c1", "P001", 1)
    q = store.create_quote("c1").data["quote_id"]
    store.set_cart_qty("c1", "P001", 2)
    res = store.confirm_order(q)
    assert not res.ok and res.errors[0]["code"] == "quote_superseded"


# ---------- confirmación (escenario 6 del enunciado) ----------
def test_confirmacion_feliz_descuenta_stock(store):
    store.set_cart_qty("c1", "P001", 2)
    store.set_cart_qty("c1", "P002", 1)
    q = store.create_quote("c1", budget_cents=10000).data["quote_id"]
    res = store.confirm_order(q)
    assert res.ok and not res.data["duplicate"] and res.data["total_cents"] == 4500
    items = store.get_price_stock(["P001", "P002"]).data["items"]
    stock = {i["product_id"]: i["stock"] for i in items}
    assert stock == {"P001": 3, "P002": 1}
    assert store.view_cart("c1")["lines"] == []


def test_confirmar_dos_veces_no_duplica(store):
    store.set_cart_qty("c1", "P001", 1)
    q = store.create_quote("c1").data["quote_id"]
    r1, r2 = store.confirm_order(q), store.confirm_order(q)
    assert r1.data["order_id"] == r2.data["order_id"] and r2.data["duplicate"]
    assert store.summary()["orders"] == 1
    assert store.get_price_stock(["P001"]).data["items"][0]["stock"] == 4  # descontado una vez


def test_cambio_de_stock_entre_cotizar_y_confirmar_no_deja_cambios_parciales(store):
    store.set_cart_qty("c1", "P001", 2)
    store.set_cart_qty("c1", "P002", 2)
    q = store.create_quote("c1").data["quote_id"]
    before = store.fingerprint()
    store.set_stock("P002", 1)  # otro cliente compró: ya no alcanza
    after_external = store.fingerprint()
    res = store.confirm_order(q)
    assert not res.ok and res.errors[0]["code"] == "stock_changed"
    # P001 sí tenía stock, pero NO se descontó: todo o nada.
    assert store.fingerprint() == after_external != before
    assert store.summary()["orders"] == 0


def test_cambio_de_precio_detectado(store):
    store.set_cart_qty("c1", "P001", 1)
    q = store.create_quote("c1").data["quote_id"]
    store.set_price("P001", 1200)
    assert store.confirm_order(q).errors[0]["code"] == "price_changed"


def test_check_de_bd_impide_stock_negativo(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.con.execute("UPDATE products SET stock = -1 WHERE product_id = 'P001'")


def test_bitacora_registra_operaciones(store):
    store.set_cart_qty("c1", "P001", 1)
    store.create_quote("c1")
    assert list(store.ops().op) == ["set_cart_qty", "create_quote"]


def test_formato_euros():
    assert fmt_eur(123456) == "1.234,56 €" and fmt_eur(99) == "0,99 €"
