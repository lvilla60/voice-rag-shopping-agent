"""Escenarios operativos reproducibles y su evaluación automática (§3 "Pruebas de funcionamiento").

Cada escenario define:

- **Estado inicial reproducible**: la tienda se restaura con la semilla y luego se fijan stock y
  precios de los productos que devuelve la búsqueda para ciertas consultas (``Setup``). Como la
  recuperación es determinista, el estado inicial es idéntico en cada ejecución; se registra su
  huella.
- **Turnos del usuario** (texto). En los escenarios de voz, el texto es la transcripción correcta
  de referencia y el audio se graba aparte; el runner acepta el texto del ASR en su lugar.
- **Eventos externos** entre turnos (``Event``): otro cliente compra, cambia un precio, etc.
- **Verificaciones** (``Check``) sobre el estado final, la traza de herramientas y las respuestas.

Además de las verificaciones de cada escenario, ``violations`` cuenta incumplimientos de
restricciones de forma genérica (presupuesto, cantidades, duplicados, compra sin confirmación).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from asistente_compras.agent import ShoppingAgent, TurnResult
from asistente_compras.store import Store, restore_store


# --------------------------------------------------------------------------------------
# Definición
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Setup:
    """Fijar stock (y opcionalmente precios) de los ``k`` primeros resultados de ``query``."""

    query: str
    k: int = 30
    stock: int = 20
    prices_eur: tuple[float, ...] = ()  # asignar en orden de ranking, de forma cíclica


@dataclass(frozen=True)
class Event:
    """Evento externo antes del turno ``before_turn`` (1-indexado).

    kind:
    - ``zero_shown``: agota todos los productos que el agente ya mostró.
    - ``reduce_cart_stock``: deja el stock de cada producto del carrito en ``cantidad − 1``.
    - ``raise_cart_price``: sube ``amount_eur`` el precio de cada producto del carrito.
    """

    before_turn: int
    kind: str
    amount_eur: float = 0.0


@dataclass
class Check:
    name: str
    fn: Callable[[ScenarioRun], tuple[bool, str]]


@dataclass
class Scenario:
    id: str
    title: str
    category: str
    turns: list[str]
    checks: list[Check]
    setup: list[Setup] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    voice: bool = False
    budget_eur: float | None = None  # presupuesto que expresa el usuario (para violaciones)
    expected_qty: int | None = None  # unidades totales que el usuario termina pidiendo
    key_terms: tuple[str, ...] = ()  # palabras de producto (errores del ASR en productos)


@dataclass
class ScenarioRun:
    scenario: Scenario
    turns: list[TurnResult]
    snapshots: list[dict]  # estado tras cada turno
    orders: pd.DataFrame  # líneas de pedidos con título
    final_state: dict
    initial_fingerprint: str
    ops: pd.DataFrame
    voice: list[dict] = field(default_factory=list)  # métricas por turno en modo voz

    def errors_seen(self) -> list[str]:
        return [e.get("code") for t in self.turns for c in t.tool_calls if not c["ok"]
                for e in c["result"].get("errors", [])]

    def calls(self, turn: int | None = None) -> list[dict]:
        ts = self.turns if turn is None else [self.turns[turn - 1]]
        return [c for t in ts for c in t.tool_calls]


# --------------------------------------------------------------------------------------
# Verificaciones reutilizables
# --------------------------------------------------------------------------------------
def n_orders(n: int) -> Check:
    def fn(r):
        got = r.orders.order_id.nunique() if len(r.orders) else 0
        return got == n, f"pedidos={got}"
    return Check(f"pedidos == {n}", fn)


def orders_after_turn(turn: int, n: int) -> Check:
    def fn(r):
        got = r.snapshots[turn - 1]["orders"]
        return got == n, f"pedidos tras turno {turn} = {got}"
    return Check(f"pedidos tras turno {turn} == {n}", fn)


def order_matches(*patterns: str) -> Check:
    """Cada patrón debe aparecer en el título de alguna línea del pedido."""
    def fn(r):
        titles = " | ".join(r.orders.title.str.lower()) if len(r.orders) else ""
        miss = [p for p in patterns if not re.search(p, titles)]
        return not miss and bool(titles), f"faltan {miss}" if miss else titles[:80]
    return Check(f"pedido contiene {patterns}", fn)


def order_qty(q: int) -> Check:
    def fn(r):
        got = int(r.orders.qty.sum()) if len(r.orders) else 0
        return got == q, f"unidades={got}"
    return Check(f"unidades pedidas == {q}", fn)


def order_lines(n: int) -> Check:
    def fn(r):
        return len(r.orders) == n, f"líneas={len(r.orders)}"
    return Check(f"líneas del pedido == {n}", fn)


def budget_is(eur: float) -> Check:
    def fn(r):
        got = r.final_state["budget_cents"]
        return got == round(eur * 100), f"presupuesto={got}"
    return Check(f"presupuesto registrado == {eur} €", fn)


def error_seen(code: str) -> Check:
    def fn(r):
        return code in r.errors_seen(), f"errores={sorted(set(r.errors_seen()))}"
    return Check(f"se detectó «{code}»", fn)


def tool_called(name: str, turn: int | None = None) -> Check:
    def fn(r):
        names = [c["name"] for c in r.calls(turn)]
        return name in names, f"herramientas={names}"
    return Check(f"llamó {name}" + (f" en turno {turn}" if turn else ""), fn)


def search_after_error(code: str) -> Check:
    """Tras un error (p. ej. agotado), el agente busca alternativas en el mismo turno o el
    siguiente."""
    def fn(r):
        for i, t in enumerate(r.turns):
            codes = [e.get("code") for c in t.tool_calls if not c["ok"]
                     for e in c["result"].get("errors", [])]
            if code in codes:
                later = [c["name"] for tt in r.turns[i:i + 2] for c in tt.tool_calls]
                idx = later.index("buscar_productos") if "buscar_productos" in later else -1
                return idx >= 0, f"herramientas tras el error: {later}"
        return False, f"no ocurrió {code}"
    return Check(f"busca alternativas tras «{code}»", fn)


def clarifies(turn: int) -> Check:
    """Aclaración: no modifica el carrito y termina con una pregunta."""
    def fn(r):
        t = r.turns[turn - 1]
        changed = any(c["name"] == "actualizar_carrito" and c["ok"] for c in t.tool_calls)
        asks = "?" in t.response
        return asks and not changed, f"pregunta={asks}, cambió carrito={changed}"
    return Check(f"pide aclaración en turno {turn}", fn)


def response_matches(pattern: str, turn: int = -1) -> Check:
    def fn(r):
        text = r.turns[turn].response if turn < 0 else r.turns[turn - 1].response
        return bool(re.search(pattern, text, re.I)), text[:100]
    return Check(f"respuesta del turno {turn} ~ /{pattern}/", fn)


def no_cart_changes() -> Check:
    def fn(r):
        ok = not any(c["name"] == "actualizar_carrito" and c["ok"] for c in r.calls())
        return ok, "sin cambios" if ok else "modificó el carrito"
    return Check("no modifica el carrito", fn)


def preference_has(word: str) -> Check:
    def fn(r):
        prefs = " ".join(r.final_state["preferences"]).lower()
        return word in prefs, f"preferencias={r.final_state['preferences']}"
    return Check(f"preferencia contiene «{word}»", fn)


def mentions_price_of_shown(turn: int) -> Check:
    """La respuesta cita el precio real (de la tienda) de algún producto mostrado."""
    def fn(r):
        text = r.turns[turn - 1].response
        prices = {c["result"]["precio"] for c in r.calls(turn)
                  if c["name"] == "ver_detalle" and c["ok"]}
        prices |= {p["precio"] for c in r.calls(turn) if c["name"] == "consultar_precio_stock"
                   for p in c["result"].get("productos", [])}
        prices |= {i["precio"] for c in r.calls() if c["name"] == "buscar_productos"
                   for i in c["result"].get("resultados", [])}
        nums = {p.replace(" €", "") for p in prices}
        hit = [n for n in nums if n in text]
        return bool(hit), f"precios citados={hit}"
    return Check(f"cita un precio real en turno {turn}", fn)


# --------------------------------------------------------------------------------------
# Los 20 escenarios (V = también por voz; T = solo texto)
# --------------------------------------------------------------------------------------
CONFIRM = "Sí, confirmo la compra."

SCENARIOS: list[Scenario] = [
    # ---------------- 6 escenarios de voz (situaciones exigidas por el enunciado) ----------
    Scenario(
        "V1", "Compra que cumple las restricciones", "compra_ok", voice=True,
        turns=["Hola, necesito dos cables USB tipo C. Mi presupuesto es de 40 euros.",
               "Me quedo con la primera opción.", CONFIRM],
        setup=[Setup("cable usb tipo c", prices_eur=(8.99, 12.99, 15.99, 9.99, 22.99))],
        budget_eur=40, expected_qty=2,
        key_terms=("cables", "usb"),
        checks=[n_orders(1), order_matches(r"usb|tipo c"), order_qty(2), budget_is(40)],
    ),
    Scenario(
        "V2", "Solicitud ambigua que requiere aclaración", "ambigua", voice=True,
        turns=["Quiero comprar una funda.", "Es una funda para un iPhone 8.",
               "Ponme una de la primera opción.", CONFIRM],
        setup=[Setup("funda iphone 8", prices_eur=(9.99, 14.99, 7.99))],
        expected_qty=1,
        key_terms=("funda", "iphone"),
        checks=[clarifies(1), n_orders(1), order_matches(r"funda", r"iphone"), order_qty(1)],
    ),
    Scenario(
        "V3", "Cambio de cantidad conservando el presupuesto", "cambio", voice=True,
        turns=["Quiero tres botellas de agua reutilizables. Tengo como máximo 60 euros.",
               "La primera opción.", "Mejor que sean cinco en lugar de tres.", CONFIRM],
        setup=[Setup("botella de agua reutilizable", prices_eur=(9.99, 11.99, 8.99, 13.99))],
        budget_eur=60, expected_qty=5,
        key_terms=("botellas", "agua"),
        checks=[n_orders(1), order_matches(r"botella"), order_qty(5), budget_is(60)],
    ),
    Scenario(
        "V4", "Producto agotado: buscar alternativa", "agotado", voice=True,
        turns=["Busco un ratón inalámbrico.", "Ponme la primera opción.",
               "De acuerdo, ponme la primera alternativa que tenga stock.", CONFIRM],
        setup=[Setup("ratón inalámbrico", prices_eur=(14.99, 19.99, 11.99, 24.99))],
        events=[Event(before_turn=2, kind="zero_shown")],
        expected_qty=1,
        key_terms=("ratón",),
        checks=[error_seen("out_of_stock"), search_after_error("out_of_stock"), n_orders(1),
                order_matches(r"rat[oó]n|mouse")],
    ),
    Scenario(
        "V5", "Corrección de la transcripción antes de comprar", "correccion", voice=True,
        # El turno 1 reproduce un error típico del ASR: "cincuenta" en lugar de "quince".
        turns=["Quiero cincuenta lápices de colores.", "La primera opción.",
               "No, perdona, entendiste mal: son quince, no cincuenta.", CONFIRM],
        setup=[Setup("lápices de colores", stock=100, prices_eur=(1.49, 0.99, 2.49))],
        expected_qty=15,
        key_terms=("lápices",),
        checks=[n_orders(1), order_matches(r"l[aá]piz|l[aá]pices"), order_qty(15)],
    ),
    Scenario(
        "V6", "Cambio de stock entre cotización y confirmación; confirmación repetida",
        "stock_cambia", voice=True,
        turns=["Necesito dos mochilas para portátil. Mi presupuesto es de 100 euros.",
               "La primera opción, por favor.", CONFIRM,
               "Vale, busca otra mochila para portátil y ponme dos de la primera que tenga "
               "stock.", CONFIRM,
               "Confirmo otra vez, por si acaso."],
        setup=[Setup("mochila para portátil", prices_eur=(29.99, 34.99, 39.99, 24.99))],
        events=[Event(before_turn=3, kind="reduce_cart_stock")],
        budget_eur=100, expected_qty=2,
        key_terms=("mochilas", "portátil"),
        checks=[error_seen("stock_changed"), n_orders(1), order_qty(2),
                order_matches(r"mochila"), budget_is(100)],
    ),
    # ---------------- 14 escenarios de texto ----------------
    Scenario(
        "T07", "Compra simple sin presupuesto", "compra_ok",
        turns=["Quiero comprar un ratón inalámbrico.", "La segunda opción, una unidad.",
               CONFIRM],
        setup=[Setup("ratón inalámbrico", prices_eur=(14.99, 19.99, 11.99))], expected_qty=1,
        checks=[n_orders(1), order_matches(r"rat[oó]n|mouse"), order_qty(1)],
    ),
    Scenario(
        "T08", "Presupuesto insuficiente: no hay solución", "presupuesto",
        turns=["Quiero tres auriculares bluetooth. Tengo como máximo 10 euros en total."],
        setup=[Setup("auriculares bluetooth", prices_eur=(15.99, 19.99, 24.99))],
        budget_eur=10,
        checks=[n_orders(0), budget_is(10), no_cart_changes()],
    ),
    Scenario(
        "T09", "Comparar dos alternativas", "comparar",
        turns=["Busco una botella de agua de acero inoxidable.",
               "¿Qué diferencia hay entre la primera y la segunda opción?"],
        setup=[Setup("botella de agua acero inoxidable", prices_eur=(12.99, 16.99, 9.99))],
        checks=[tool_called("ver_detalle", turn=2), n_orders(0), no_cart_changes()],
    ),
    Scenario(
        "T10", "Atributo no verificable: no inventar", "desconocido",
        turns=["Busco un cable USB tipo C.",
               "¿De qué color es la segunda opción y de qué material es el conector?"],
        setup=[Setup("cable usb tipo c")],
        checks=[response_matches(r"no (se )?(indica|especifica|aparece|figura|consta|"
                                 r"menciona|detalla)|desconoc|"
                                 r"no tengo (esa |ese )?(informaci|dato)"),
                n_orders(0)],
    ),
    Scenario(
        "T11", "Dos productos distintos con presupuesto", "multiproducto",
        turns=["Quiero un ratón inalámbrico y una mochila para portátil. Mi presupuesto es de "
               "60 euros.",
               "El primer ratón y la primera mochila.", CONFIRM],
        setup=[Setup("ratón inalámbrico", prices_eur=(12.99, 15.99)),
               Setup("mochila para portátil", prices_eur=(29.99, 34.99))],
        budget_eur=60, expected_qty=2,
        checks=[n_orders(1), order_lines(2), order_matches(r"rat[oó]n|mouse", r"mochila")],
    ),
    Scenario(
        "T12", "Cambio de preferencia conservando lo demás", "cambio",
        turns=["Busco una mochila, prefiero que sea de color negro. Tengo 50 euros.",
               "Pensándolo mejor, prefiero que sea azul.", "La primera opción.", CONFIRM],
        setup=[Setup("mochila azul", prices_eur=(24.99, 29.99, 19.99)),
               Setup("mochila negra", prices_eur=(24.99, 29.99, 19.99))],
        budget_eur=50, expected_qty=1,
        checks=[preference_has("azul"), budget_is(50), n_orders(1), order_matches(r"mochila")],
    ),
    Scenario(
        "T13", "Quitar un producto del carrito", "cambio",
        turns=["Quiero un ratón inalámbrico y un cable USB tipo C.",
               "El primero de cada uno.", "Quita el cable, solo quiero el ratón.", CONFIRM],
        setup=[Setup("ratón inalámbrico", prices_eur=(12.99,)),
               Setup("cable usb tipo c", prices_eur=(8.99,))],
        expected_qty=1,
        checks=[n_orders(1), order_lines(1), order_matches(r"rat[oó]n|mouse")],
    ),
    Scenario(
        "T14", "Consultar precio y stock sin comprar", "consulta",
        turns=["Busco auriculares bluetooth.",
               "¿Cuánto cuesta la primera opción y cuántas unidades quedan?"],
        setup=[Setup("auriculares bluetooth", stock=7, prices_eur=(17.99, 21.99))],
        checks=[mentions_price_of_shown(2), response_matches(r"\b7\b|siete", turn=2),
                no_cart_changes(), n_orders(0)],
    ),
    Scenario(
        "T15", "Cancelar antes de confirmar", "cancelar",
        turns=["Quiero un cable USB tipo C.", "La primera opción.",
               "No, mejor no lo compro. Cancela."],
        setup=[Setup("cable usb tipo c", prices_eur=(8.99,))],
        checks=[n_orders(0)],
    ),
    Scenario(
        "T16", "Pide comprar sin ver la cotización", "confirmacion",
        turns=["Cómprame ya un ratón inalámbrico, el primero que encuentres, sin preguntar.",
               CONFIRM],
        setup=[Setup("ratón inalámbrico", prices_eur=(12.99, 15.99))], expected_qty=1,
        checks=[orders_after_turn(1, 0), n_orders(1)],
    ),
    Scenario(
        "T17", "Cantidad mayor que el stock", "stock",
        turns=["Busco lápices de colores.", "Quiero treinta unidades de la primera opción."],
        setup=[Setup("lápices de colores", stock=5, prices_eur=(1.49,))],
        checks=[error_seen("insufficient_stock"), n_orders(0),
                response_matches(r"\b5\b|cinco|stock|disponib|existencias", turn=2)],
    ),
    Scenario(
        "T18", "Cambio de presupuesto durante la conversación", "presupuesto",
        turns=["Busco una mochila para portátil. Tengo 30 euros.",
               "Ahora puedo gastar hasta 60 euros. Quiero la segunda opción.", CONFIRM],
        setup=[Setup("mochila para portátil", prices_eur=(25.99, 44.99, 29.99))],
        budget_eur=60, expected_qty=1,
        checks=[budget_is(60), n_orders(1), order_matches(r"mochila")],
    ),
    Scenario(
        "T19", "Producto fuera del catálogo", "sin_solucion",
        turns=["¿Tienen billetes de avión a Lima?"],
        checks=[no_cart_changes(), n_orders(0),
                response_matches(r"no (tenemos|vendemos|disponemos|encontr|hay|ofre[cz]\w*|"
                                 r"contamos|trabajamos|manejamos)|"
                                 r"no (he|hemos) encontrado|no est[aá]n? disponible")],
    ),
    Scenario(
        "T20", "Cambio de precio entre cotización y confirmación", "precio_cambia",
        turns=["Quiero un ratón inalámbrico.", "La primera opción.", CONFIRM, CONFIRM],
        setup=[Setup("ratón inalámbrico", prices_eur=(12.99, 15.99))],
        events=[Event(before_turn=3, kind="raise_cart_price", amount_eur=5.0)],
        expected_qty=1,
        checks=[error_seen("price_changed"), n_orders(1)],
    ),
]


# --------------------------------------------------------------------------------------
# Ejecución
# --------------------------------------------------------------------------------------
def apply_setup(store: Store, retriever, setups: list[Setup]) -> None:
    for s in setups:
        for i, (pid, _) in enumerate(retriever.search(s.query, s.k)):
            store.con.execute("UPDATE products SET stock = ? WHERE product_id = ?",
                              (s.stock, pid))
            if s.prices_eur:
                cents = round(s.prices_eur[i % len(s.prices_eur)] * 100)
                store.con.execute("UPDATE products SET price_cents = ? WHERE product_id = ?",
                                  (cents, pid))


def apply_event(ev: Event, agent: ShoppingAgent) -> None:
    store, state = agent.store, agent.state
    if ev.kind == "zero_shown":
        for pid in state.refs.values():
            store.set_stock(pid, 0)
    elif ev.kind in ("reduce_cart_stock", "raise_cart_price"):
        for line in store.view_cart(state.cart_id)["lines"]:
            if ev.kind == "reduce_cart_stock":
                store.set_stock(line["product_id"], max(0, line["qty"] - 1))
            else:
                store.set_price(line["product_id"],
                                line["unit_price_cents"] + round(ev.amount_eur * 100))
    else:
        raise ValueError(f"Evento desconocido: {ev.kind}")


def _orders(store: Store) -> pd.DataFrame:
    return pd.read_sql(
        """SELECT o.order_id, o.quote_id, o.total_cents, oi.product_id, oi.qty,
                  oi.unit_price_cents
           FROM orders o JOIN order_items oi USING (order_id) ORDER BY o.order_id""",
        store.con)


def run_scenario(sc: Scenario, llm, retriever, catalog: pd.DataFrame, db_path: Path,
                 inputs: list[str] | None = None, audio: list[Path] | None = None,
                 asr=None, tts=None, audio_out_dir: Path | None = None) -> ScenarioRun:
    """Ejecutar un escenario desde su estado inicial.

    - ``inputs`` reemplaza los textos de los turnos.
    - ``audio`` + ``asr``: modo voz; cada turno se transcribe y la transcripción va al agente.
    - ``tts``: sintetiza cada respuesta con la voz clonada (en ``audio_out_dir``).
    La latencia de voz se mide desde el fin de la grabación hasta tener el audio de respuesta.
    """
    from asistente_compras.voice import entity_errors, wer

    restore_store(db_path, catalog.product_id)
    store = Store(db_path)
    apply_setup(store, retriever, sc.setup)
    fingerprint = store.fingerprint()
    agent = ShoppingAgent(llm, store, retriever, catalog)
    titles = catalog.set_index("product_id").title
    texts = inputs or sc.turns
    turns, snaps, voice = [], [], []
    for i, gold in enumerate(texts, start=1):
        for ev in sc.events:
            if ev.before_turn == i:
                apply_event(ev, agent)
        rec: dict[str, Any] = {"turn": i, "gold": sc.turns[i - 1]}
        text, t_asr = gold, 0.0
        if audio is not None and asr is not None:
            tr = asr.transcribe(audio[i - 1])
            text, t_asr = tr.text, tr.seconds
            rec.update(audio_in=str(audio[i - 1]), transcript=text,
                       wer=wer(sc.turns[i - 1], text),
                       **entity_errors(sc.turns[i - 1], text, list(sc.key_terms)))
        r = agent.turn(text)
        turns.append(r)
        if audio is not None or tts is not None:
            rec.update(asr_s=t_asr, llm_s=sum(r.llm_seconds), tools_s=r.tool_seconds,
                       agent_s=r.total_seconds, response=r.response)
            if tts is not None:
                syn = tts.synthesize(r.response, Path(audio_out_dir) / f"{sc.id}_r{i:02d}.wav")
                rec.update(tts_s=syn["seconds"], audio_out=str(syn["path"]),
                           audio_out_s=syn["audio_seconds"], spoken_text=syn["spoken_text"])
            rec["total_s"] = t_asr + r.total_seconds + rec.get("tts_s", 0.0)
            voice.append(rec)
        snaps.append({"turn": i, **store.summary(),
                      "cart_lines": len(store.view_cart(agent.state.cart_id)["lines"]),
                      "budget_cents": agent.state.budget_cents})
    orders = _orders(store)
    orders["title"] = orders.product_id.map(titles)
    run = ScenarioRun(
        scenario=sc, turns=turns, snapshots=snaps, orders=orders,
        final_state={"budget_cents": agent.state.budget_cents,
                     "preferences": list(agent.state.preferences),
                     **store.summary()},
        initial_fingerprint=fingerprint, ops=store.ops(), voice=voice)
    store.close()
    return run


# --------------------------------------------------------------------------------------
# Evaluación
# --------------------------------------------------------------------------------------
_CONFIRM_WORDS = re.compile(r"\b(s[ií]|confirm\w*|adelante|de acuerdo|vale|compra\w*|ok)\b",
                            re.I)


def violations(run: ScenarioRun) -> dict[str, int]:
    """Incumplimientos de restricciones, independientes de las verificaciones del escenario."""
    sc, o = run.scenario, run.orders
    v = {"presupuesto_excedido": 0, "cantidad_incorrecta": 0, "pedido_duplicado": 0,
         "compra_sin_confirmacion": 0, "stock_negativo": 0}
    if len(o):
        totals = o.groupby("order_id").total_cents.first()
        budget = run.final_state["budget_cents"]
        if budget is not None:
            v["presupuesto_excedido"] = int((totals > budget).sum())
        if sc.expected_qty is not None and int(o.qty.sum()) != sc.expected_qty:
            v["cantidad_incorrecta"] = 1
        v["pedido_duplicado"] = max(0, o.order_id.nunique() - 1)
        for t in run.turns:  # turnos en que se registró un pedido nuevo
            for c in t.tool_calls:
                new = c["name"] == "confirmar_pedido" and c["ok"] and \
                    not c["result"].get("ya_estaba_registrado")
                if new and not _CONFIRM_WORDS.search(t.user):
                    v["compra_sin_confirmacion"] += 1
    if run.ops is not None:
        v["stock_negativo"] = 0  # garantizado por CHECK (stock >= 0) en la BD
    return v


def evaluate(run: ScenarioRun) -> dict[str, Any]:
    results = [(c.name, *c.fn(run)) for c in run.scenario.checks]
    viol = violations(run)
    return {
        "id": run.scenario.id, "title": run.scenario.title,
        "category": run.scenario.category, "voice": run.scenario.voice,
        "passed": all(ok for _, ok, _ in results) and not any(viol.values()),
        "checks_ok": sum(ok for _, ok, _ in results), "checks_total": len(results),
        "failed_checks": [f"{n}: {d}" for n, ok, d in results if not ok],
        "violations": sum(viol.values()), **{f"v_{k}": n for k, n in viol.items()},
        "turns": len(run.turns),
        "llm_calls": sum(len(t.llm_seconds) for t in run.turns),
        "guard_events": sum(len(t.guard_events) for t in run.turns),
        "unsupported_claims": sum(t.unsupported_claim for t in run.turns),
        "seconds_per_turn": sum(t.total_seconds for t in run.turns) / len(run.turns),
        "initial_fingerprint": run.initial_fingerprint,
    }


def save_run(run: ScenarioRun, out_dir: Path, tag: str = "") -> Path:
    """Guardar transcripciones, operaciones ejecutadas y estado final (lo pide el enunciado)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run.scenario.id}{tag}.json"
    payload = {
        "scenario": {k: v for k, v in asdict(run.scenario).items() if k != "checks"},
        "evaluation": evaluate(run),
        "turns": [asdict(t) for t in run.turns],
        "snapshots": run.snapshots,
        "orders": run.orders.to_dict(orient="records"),
        "final_state": run.final_state,
        "ops_log": run.ops.to_dict(orient="records"),
    }
    payload["voice"] = run.voice
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    return path


def run_all(llm, retriever, catalog: pd.DataFrame, db_path: Path, out_dir: Path,
            scenarios: list[Scenario] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Ejecutar todos los escenarios, guardar cada traza en ``out_dir`` y devolver la tabla."""
    rows = []
    for sc in scenarios or SCENARIOS:
        run = run_scenario(sc, llm, retriever, catalog, db_path)
        save_run(run, out_dir)
        ev = evaluate(run)
        rows.append(ev)
        if verbose:
            mark = "✅" if ev["passed"] else "❌"
            print(f"{mark} {sc.id:4s} {sc.title[:55]:55s} checks {ev['checks_ok']}/"
                  f"{ev['checks_total']} · violaciones {ev['violations']} · "
                  f"{ev['seconds_per_turn']:.1f} s/turno", flush=True)
            for f in ev["failed_checks"]:
                print(f"        ↳ {f}", flush=True)
    return pd.DataFrame(rows)
