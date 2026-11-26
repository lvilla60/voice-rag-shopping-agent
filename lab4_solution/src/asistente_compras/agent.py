"""Agente de compras: estado de la conversación, herramientas, validaciones y bucle de decisión.

Reparto de responsabilidades (decisión central del diseño):

- **El LLM decide** qué herramienta usar, extrae entidades (producto, cantidad, presupuesto),
  pide aclaraciones y redacta la respuesta.
- **El código valida y ejecuta**: tipos y rangos de argumentos, referencias a productos,
  stock, presupuesto, y la regla de que un pedido solo se registra **después** de que el
  usuario vio la cotización en un turno anterior. Ninguna restricción depende de que el LLM
  "se acuerde" de cumplirla.
- **El estado vive fuera del prompt** (``ConversationState``) y se re-inyecta en cada llamada
  al LLM como un bloque "ESTADO ACTUAL". Así se conservan presupuesto, preferencias y carrito
  aunque la conversación se alargue.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from asistente_compras.llm import LLM
from asistente_compras.store import Store, fmt_eur

UNKNOWN = "desconocido"

SYSTEM_PROMPT = """Eres un asistente de compras por voz de una tienda online española. Hablas \
con el cliente en español y tus respuestas se convierten a voz.

CÓMO ACTUAR
- Si necesitas información o hacer un cambio, llama a la herramienta en ese mismo mensaje. \
Nunca respondas "un momento" o "voy a buscar": hazlo.
- Nunca digas que agregaste, cotizaste o compraste algo si no lo hiciste con una herramienta \
que respondió ok.
- Puedes llamar varias herramientas seguidas antes de responder.

REGLAS
1. Respuestas breves para voz: máximo 3 oraciones, sin markdown, sin listas ni emojis. Precios \
como "23,99 euros". Presenta como máximo 3 opciones en una oración, numeradas en palabras \
(la primera, la segunda…), con su nombre corto y su precio. No pronuncies las refs.
2. Usa solo información devuelta por las herramientas. Si un dato (marca, color, material, \
talla…) no aparece o figura como "desconocido", di que la ficha no lo indica. Nunca inventes \
precios, stock ni características.
3. Para comprar: actualizar_carrito, luego cotizar, y presenta el resumen (productos, \
cantidades y total) pidiendo confirmación explícita. Llama a confirmar_pedido solo cuando el \
cliente confirme en un mensaje posterior a la cotización.
4. Si la solicitud es ambigua (producto genérico con opciones muy distintas, no queda claro a \
qué producto o cantidad se refiere), pregunta antes de agregar al carrito.
5. Si un producto está agotado o el stock no alcanza, busca alternativas con \
buscar_productos y propónlas; si no hay ninguna adecuada, dilo claramente.
6. En cuanto el cliente indique un presupuesto, llama a fijar_presupuesto; sus preferencias, \
a anotar_preferencia. Si cambia una cantidad o una preferencia, conserva las demás \
restricciones.
7. Si el cliente corrige algo que entendiste mal (cantidad, producto, importe), aplica la \
corrección y vuelve a cotizar antes de confirmar.
8. En las herramientas, refiérete a los productos por su ref (R1, R2…). "La primera opción" \
es la primera de la última búsqueda que presentaste.
9. Si confirmar_pedido falla porque cambió el stock o el precio, explica el problema y \
propón una alternativa."""


# --------------------------------------------------------------------------------------
# Estado de la conversación
# --------------------------------------------------------------------------------------
_CONFIRM_RE = re.compile(r"\bconfirm\w*|^\W*(s[ií]|ok|okay|adelante|dale|procede|hazlo)\b",
                         re.I)
_NEGATION_RE = re.compile(r"\bno\b[^.?!]*\bconfirm|\bcancel\w*|\btodav[ií]a no\b", re.I)


def user_confirms(text: str) -> bool:
    """¿El mensaje del cliente acepta la compra? ("Sí, confirmo", "Confirmo", "Adelante")."""
    return bool(_CONFIRM_RE.search(text or "")) and not _NEGATION_RE.search(text or "")


@dataclass
class ConversationState:
    cart_id: str = "cliente"
    budget_cents: int | None = None
    preferences: list[str] = field(default_factory=list)
    refs: dict[str, str] = field(default_factory=dict)  # "R1" → product_id
    quote: dict[str, Any] | None = None  # {"quote_id", "total_cents", "turn"}
    order: dict[str, Any] | None = None
    turn: int = 0
    last_user: str = ""  # mensaje del cliente en el turno actual

    def ref_for(self, product_id: str) -> str:
        for ref, pid in self.refs.items():
            if pid == product_id:
                return ref
        ref = f"R{len(self.refs) + 1}"
        self.refs[ref] = product_id
        return ref

    def render(self, store: Store, titles: pd.Series) -> str:
        cart = store.view_cart(self.cart_id)
        lines = [
            f"- {self.ref_for(li['product_id'])} × {li['qty']}: "
            f"{titles.get(li['product_id'], '')[:60]} ({fmt_eur(li['line_total_cents'])})"
            for li in cart["lines"]
        ]
        q = self.quote
        return "\n".join([
            "ESTADO ACTUAL (mantenido por el sistema)",
            f"Presupuesto: {fmt_eur(self.budget_cents) if self.budget_cents else 'no indicado'}",
            f"Preferencias: {'; '.join(self.preferences) or 'ninguna'}",
            "Carrito:" if lines else "Carrito: vacío",
            *lines,
            f"Total carrito: {cart['total']}" if lines else "",
            (f"Cotización vigente: {q['quote_id']} por {fmt_eur(q['total_cents'])}"
             if q else "Cotización vigente: ninguna"),
            f"Pedido registrado: {self.order['order_id']}" if self.order else "",
        ]).replace("\n\n", "\n")


# --------------------------------------------------------------------------------------
# Herramientas
# --------------------------------------------------------------------------------------
def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


TOOL_SCHEMAS = [
    _fn("buscar_productos",
        "Busca productos en el catálogo. Devuelve refs, título, marca, precio y stock actuales.",
        {"consulta": {"type": "string", "description": "Qué busca el cliente, en sus palabras"},
         "precio_max_eur": {"type": "number", "description": "Precio unitario máximo (opcional)"},
         "solo_disponibles": {"type": "boolean", "description": "Excluir agotados"}},
        ["consulta"]),
    _fn("ver_detalle", "Ficha completa de un producto (marca, color, características).",
        {"ref": {"type": "string", "description": "Ref del producto, p. ej. R2"}}, ["ref"]),
    _fn("consultar_precio_stock", "Precio y stock actuales de uno o varios productos.",
        {"refs": {"type": "array", "items": {"type": "string"}}}, ["refs"]),
    _fn("actualizar_carrito",
        "Fija la cantidad de un producto en el carrito (0 lo quita). Reemplaza la anterior.",
        {"ref": {"type": "string"}, "cantidad": {"type": "integer"}}, ["ref", "cantidad"]),
    _fn("ver_carrito", "Contenido y total del carrito.", {}, []),
    _fn("fijar_presupuesto", "Registra el presupuesto máximo total del cliente en euros.",
        {"presupuesto_eur": {"type": "number"}}, ["presupuesto_eur"]),
    _fn("anotar_preferencia", "Registra una preferencia del cliente (marca, color, uso…).",
        {"preferencia": {"type": "string"}}, ["preferencia"]),
    _fn("cotizar",
        "Genera la cotización del carrito validando stock y presupuesto. Úsala antes de pedir "
        "la confirmación del cliente.", {}, []),
    _fn("confirmar_pedido",
        "Registra el pedido de la cotización vigente. Solo tras la confirmación explícita del "
        "cliente en un mensaje posterior a la cotización.", {}, []),
]


def _as_int(value: Any) -> int | None:
    """Aceptar 3, 3.0 o "3"; rechazar 2.5, "tres", True."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _as_cents(value: Any) -> int | None:
    try:
        eur = float(str(value).replace("€", "").replace(",", ".").strip())
    except ValueError:
        return None
    return round(eur * 100) if eur > 0 else None


def _err(code: str, **kw) -> dict:
    return {"ok": False, "errors": [{"code": code, **kw}]}


class ShoppingTools:
    """Implementación de las herramientas. Cada una valida sus argumentos antes de actuar."""

    def __init__(self, store: Store, retriever, catalog: pd.DataFrame, state: ConversationState,
                 search_depth: int = 20, max_results: int = 3):
        self.store, self.retriever, self.state = store, retriever, state
        self.catalog = catalog.set_index("product_id")
        self.search_depth, self.max_results = search_depth, max_results

    # ---------- utilidades ----------
    def _pid(self, ref: Any) -> str | None:
        ref = str(ref).strip().upper()
        return self.state.refs.get(ref)

    def _brief(self, pid: str, price_cents: int, stock: int) -> dict:
        row = self.catalog.loc[pid]
        return {"ref": self.state.ref_for(pid),
                "nombre_corto": " ".join(row.title.split()[:8]), "titulo": row.title[:120],
                "marca": row.brand or UNKNOWN, "precio": fmt_eur(price_cents),
                "precio_cents": price_cents, "stock": stock,
                "disponible": stock > 0}

    def _cart(self) -> dict:
        cart = self.store.view_cart(self.state.cart_id)
        out = {"lineas": [{"ref": self.state.ref_for(li["product_id"]), "cantidad": li["qty"],
                           "precio_unitario": fmt_eur(li["unit_price_cents"]),
                           "subtotal": fmt_eur(li["line_total_cents"])}
                          for li in cart["lines"]],
               "total": cart["total"]}
        b = self.state.budget_cents
        if b is not None and cart["total_cents"] > b:
            out["aviso"] = (f"El carrito supera el presupuesto ({fmt_eur(b)}) en "
                            f"{fmt_eur(cart['total_cents'] - b)}")
        return out

    # ---------- herramientas ----------
    def buscar_productos(self, consulta: str = "", precio_max_eur: Any = None,
                         solo_disponibles: Any = False) -> dict:
        if not str(consulta).strip():
            return _err("consulta_vacia")
        max_cents = _as_cents(precio_max_eur) if precio_max_eur not in (None, "", 0) else None
        scored = self.retriever.search(str(consulta), self.search_depth)
        hits = [p for p, _ in scored]
        # Informar la relevancia solo con el re-ranker (único con puntajes calibrados 0–1), para
        # que el agente distinga "encontré opciones" de "nada corresponde a lo pedido".
        calibrated = getattr(self.retriever, "calibrated", False)
        rel = dict(scored)
        info = {i["product_id"]: i for i in self.store.get_price_stock(hits).data["items"]}
        items = []
        for pid in hits:
            ps = info[pid]
            if max_cents is not None and ps["price_cents"] > max_cents:
                continue
            if solo_disponibles is True and ps["stock"] == 0:
                continue
            item = self._brief(pid, ps["price_cents"], ps["stock"])
            if calibrated:
                item["relevancia"] = ("alta" if rel[pid] >= 0.5 else
                                      "media" if rel[pid] >= 0.1 else "baja")
            items.append(item)
            if len(items) == self.max_results:
                break
        nota = "sin resultados con esos filtros" if not items else ""
        if items and calibrated and all(i["relevancia"] == "baja" for i in items):
            nota = "ningún resultado corresponde a lo pedido: dilo y no ofrezcas estos productos"
        return {"ok": True, "resultados": items, "nota": nota}

    def ver_detalle(self, ref: Any = "") -> dict:
        pid = self._pid(ref)
        if pid is None:
            return _err("ref_desconocida", ref=ref)
        row = self.catalog.loc[pid]
        ps = self.store.get_price_stock([pid]).data["items"][0]
        return {"ok": True, "ref": str(ref).upper(), "titulo": row.title,
                "marca": row.brand or UNKNOWN, "color": row.color or UNKNOWN,
                "caracteristicas": row.bullet_points[:600] or UNKNOWN,
                "descripcion": row.description[:400] or UNKNOWN,
                "precio": ps["price"], "stock": ps["stock"]}

    def consultar_precio_stock(self, refs: Any = None) -> dict:
        refs = refs if isinstance(refs, list) else [refs]
        pids, unknown = [], []
        for r in refs:
            (pids.append(p) if (p := self._pid(r)) else unknown.append(r))
        items = [{"ref": self.state.ref_for(i["product_id"]), "precio": i["price"],
                  "stock": i["stock"], "disponible": i["available"]}
                 for i in self.store.get_price_stock(pids).data["items"]]
        out = {"ok": not unknown, "productos": items}
        if unknown:
            out["errors"] = [{"code": "ref_desconocida", "ref": u} for u in unknown]
        return out

    def actualizar_carrito(self, ref: Any = "", cantidad: Any = None) -> dict:
        pid = self._pid(ref)
        if pid is None:
            return _err("ref_desconocida", ref=ref)
        qty = _as_int(cantidad)
        if qty is None:
            return _err("cantidad_invalida", cantidad=cantidad)
        res = self.store.set_cart_qty(self.state.cart_id, pid, qty)
        if not res.ok:
            return {"ok": False, "errors": res.errors}
        self.state.quote = None  # invalidar la cotización anterior: el carrito cambió
        return {"ok": True, "carrito": self._cart()}

    def ver_carrito(self) -> dict:
        return {"ok": True, "carrito": self._cart()}

    def fijar_presupuesto(self, presupuesto_eur: Any = None) -> dict:
        cents = _as_cents(presupuesto_eur)
        if cents is None:
            return _err("presupuesto_invalido", presupuesto_eur=presupuesto_eur)
        self.state.budget_cents = cents
        self.state.quote = None
        return {"ok": True, "presupuesto": fmt_eur(cents), "carrito": self._cart()}

    def anotar_preferencia(self, preferencia: Any = "") -> dict:
        p = str(preferencia).strip()
        if not p:
            return _err("preferencia_vacia")
        if p not in self.state.preferences:
            self.state.preferences.append(p)
        return {"ok": True, "preferencias": self.state.preferences}

    def cotizar(self) -> dict:
        # Mantener la idempotencia: si el carrito no cambió desde la última cotización, devolver
        # la misma. Re-cotizar crearía una cotización nueva "no vista" y bloquearía la confirmación.
        q = self.state.quote
        if q is not None:
            seen = q["turn"] < self.state.turn
            return {"ok": True, "cotizacion": q["quote_id"], "total": fmt_eur(q["total_cents"]),
                    "carrito": self._cart(), "ya_presentada_al_cliente": seen,
                    "siguiente_paso": ("El cliente ya vio esta cotización: si acaba de "
                                       "confirmar, llama a confirmar_pedido.") if seen else
                                      "Presenta el resumen y pide confirmación explícita."}
        res = self.store.create_quote(self.state.cart_id, self.state.budget_cents)
        if not res.ok:
            return {"ok": False, "errors": res.errors, "carrito": self._cart()}
        self.state.quote = {"quote_id": res.data["quote_id"],
                            "total_cents": res.data["total_cents"], "turn": self.state.turn}
        return {"ok": True, "cotizacion": res.data["quote_id"], "total": res.data["total"],
                "carrito": self._cart(),
                "siguiente_paso": "Presenta el resumen y pide confirmación explícita."}

    def confirmar_pedido(self) -> dict:
        q = self.state.quote
        if q is None:
            return _err("sin_cotizacion", detalle="Primero hay que cotizar el carrito.")
        # Regla en código: exigir que el cliente haya visto la cotización en un turno anterior.
        if q["turn"] >= self.state.turn:
            return _err("falta_confirmacion_del_cliente",
                        detalle="Presenta la cotización y espera a que el cliente confirme.")
        # Regla en código: exigir además una confirmación explícita en su mensaje actual. Elegir un
        # producto ("me quedo con la primera") no es aceptar la compra.
        if not user_confirms(self.state.last_user):
            return _err("falta_confirmacion_del_cliente",
                        detalle="El cliente no confirmó en este mensaje. Muestra el total y "
                                "pregunta: ¿Confirmas la compra?")
        res = self.store.confirm_order(q["quote_id"])
        if not res.ok:
            self.state.quote = None
            return {"ok": False, "errors": res.errors,
                    "detalle": "No se registró nada. Informa al cliente y propone alternativas."}
        self.state.order = {"order_id": res.data["order_id"], "total": res.data["total"]}
        return {"ok": True, "pedido": res.data["order_id"], "total": res.data["total"],
                "ya_estaba_registrado": res.data["duplicate"]}

    # ---------- despacho ----------
    def dispatch(self, name: str, args: dict) -> dict:
        fn = getattr(self, name, None) if name in {t["function"]["name"] for t in TOOL_SCHEMAS} \
            else None
        if fn is None:
            return _err("herramienta_desconocida", nombre=name)
        try:
            return fn(**(args or {}))
        except TypeError as e:  # capturar argumentos con nombres inesperados
            return _err("argumentos_invalidos", detalle=str(e))


# --------------------------------------------------------------------------------------
# Control de afirmaciones (validación en código de la respuesta final)
# --------------------------------------------------------------------------------------
_ANNOUNCE_RE = re.compile(
    r"\b(un momento|voy a (buscar|revisar|consultar|agregar|añadir|cotizar)|buscando|"
    r"^busco\b|te busco|buscar[ée]|estoy buscando|"
    r"d[ée]jame (buscar|revisar|consultar)|perm[ií]teme (buscar|revisar|consultar))", re.I)
_CLAIM_CART_RE = re.compile(
    r"\b(he (agregado|añadido|puesto|incluido)|agregu[ée]|añad[ií]|"
    r"(he actualizado|actualic[ée]) (el|tu|su) carrito)", re.I)
# Herramientas que "hacen" algo: un anuncio de búsqueda/acción exige al menos una de ellas.
_ACTION_TOOLS = {"buscar_productos", "ver_detalle", "consultar_precio_stock",
                 "actualizar_carrito", "ver_carrito", "cotizar", "confirmar_pedido"}
_CLAIM_ORDER_RE = re.compile(
    r"\b(pedido (ha sido |fue |queda |quedó )?(registrado|confirmado|realizado)|"
    r"he (registrado|confirmado|realizado) (el|tu|su) (pedido|compra)|"
    r"compra (ha sido |fue )?(realizada|confirmada|registrada))", re.I)


# Presupuesto total expresado por el usuario. Patrones estrictos a propósito: "auriculares de
# máximo 20 euros" es un tope por producto (lo maneja buscar_productos), no un presupuesto.
_AMOUNT = r"(\d{1,6}(?:[.,]\d{1,2})?)\s*(?:€|euros?\b)"
_BUDGET_RES = [
    re.compile(r"\bpresupuesto\b[^\d]{0,30}?" + _AMOUNT, re.I),
    re.compile(r"\b(?:tengo|dispongo de|cuento con|llevo)\s+(?:como\s+)?(?:máximo|maximo|un "
               r"máximo de|hasta|solo|sólo|unos)?\s*(?:de\s+)?" + _AMOUNT, re.I),
    re.compile(r"\bno\s+(?:quiero|puedo|deseo)\s+gastar\s+más\s+de\s+" + _AMOUNT, re.I),
    re.compile(r"\bgastar\s+(?:como\s+)?(?:máximo|maximo|hasta)\s+(?:de\s+)?" + _AMOUNT, re.I),
    re.compile(r"\ben\s+total\s+(?:como\s+)?(?:máximo|maximo|hasta)\s+" + _AMOUNT, re.I),
]


def extract_budget_cents(text: str) -> int | None:
    """Presupuesto total en céntimos si el usuario lo expresa explícitamente; si no, ``None``."""
    for rx in _BUDGET_RES:
        if m := rx.search(text):
            return round(float(m.group(1).replace(",", ".")) * 100)
    return None


def check_response(text: str, calls: list[dict], state: ConversationState) -> str | None:
    """Detectar respuestas que anuncian una acción sin hacerla o afirman una que no ocurrió.

    Devuelve una instrucción de corrección para el LLM, o ``None`` si la respuesta es válida.
    """
    ok = {c["name"] for c in calls if c["ok"]}
    if _ANNOUNCE_RE.search(text) and not {c["name"] for c in calls} & _ACTION_TOOLS:
        return ("Anunciaste una acción pero no llamaste a ninguna herramienta. Llama ahora a la "
                "herramienta necesaria; no respondas al cliente hasta tener el resultado.")
    if _CLAIM_CART_RE.search(text) and not ok & {"actualizar_carrito", "fijar_presupuesto",
                                                 "anotar_preferencia"}:
        return ("Tu respuesta afirma que modificaste el carrito, pero en este turno no hubo "
                "ninguna llamada exitosa a actualizar_carrito. Llama a la herramienta o corrige "
                "la respuesta sin afirmar cambios que no hiciste.")
    if _CLAIM_ORDER_RE.search(text) and state.order is None:
        return ("Tu respuesta afirma que el pedido está registrado, pero no existe ningún pedido. "
                "No lo afirmes: presenta la cotización o pide la confirmación del cliente.")
    return None


# --------------------------------------------------------------------------------------
# Bucle del agente
# --------------------------------------------------------------------------------------
@dataclass
class TurnResult:
    user: str
    response: str
    tool_calls: list[dict] = field(default_factory=list)
    llm_seconds: list[float] = field(default_factory=list)
    tool_seconds: float = 0.0
    total_seconds: float = 0.0
    steps: int = 0
    guard_events: list[str] = field(default_factory=list)  # correcciones pedidas al LLM
    auto_slots: dict[str, Any] = field(default_factory=dict)  # registrado por reglas de código
    unsupported_claim: bool = False  # la respuesta final siguió sin respaldo


class ShoppingAgent:
    """Un único agente con bucle ReAct acotado: LLM → herramientas → LLM … → respuesta."""

    FALLBACK = "Perdona, no pude completar la operación. ¿Me lo repites de otra forma?"

    def __init__(self, llm: LLM, store: Store, retriever, catalog: pd.DataFrame,
                 cart_id: str = "cliente", max_steps: int = 8, max_corrections: int = 2):
        self.llm, self.store, self.max_steps = llm, store, max_steps
        self.max_corrections = max_corrections
        self.state = ConversationState(cart_id=cart_id)
        self.tools = ShoppingTools(store, retriever, catalog, self.state)
        self.titles = catalog.set_index("product_id").title
        self.messages: list[dict] = []
        self.history: list[TurnResult] = []

    def _system(self) -> dict:
        return {"role": "system",
                "content": f"{SYSTEM_PROMPT}\n\n{self.state.render(self.store, self.titles)}"}

    _CART_TOOLS = {"actualizar_carrito", "fijar_presupuesto"}
    _ERR_TEXT = {
        "over_budget": "Ojo: el total supera tu presupuesto.",
        "insufficient_stock": "Ojo: no hay stock suficiente para alguna cantidad del carrito.",
        "out_of_stock": "Ojo: hay un producto agotado en el carrito.",
    }

    def _auto_quote(self, res: TurnResult) -> None:
        """Cotización automática (regla en código).

        Si en el turno cambió el carrito y no hay cotización vigente, el sistema cotiza y
        garantiza que la respuesta muestre el total y pida confirmación. Cotizar no implica
        ninguna decisión, así que no se deja a criterio del LLM. Además asegura que el cliente
        vea la cotización antes del turno en que confirma.
        """
        changed = any(c["name"] in self._CART_TOOLS and c["ok"] for c in res.tool_calls)
        # Exigir re-cotizar también tras una confirmación fallida por cambio de precio/stock, para
        # que el cliente escuche las condiciones nuevas antes de volver a confirmar.
        changed |= any(c["name"] == "confirmar_pedido" and not c["ok"] and
                       {e.get("code") for e in c["result"].get("errors", [])}
                       & {"price_changed", "stock_changed"} for c in res.tool_calls)
        cart = self.store.view_cart(self.state.cart_id)
        if not changed or self.state.quote is not None or not cart["lines"]:
            return
        out = self.tools.cotizar()
        text = res.response.strip()
        if out["ok"]:
            res.auto_slots["cotizacion"] = out["cotizacion"]
            total = out["total"].replace(" €", "")
            # Quitar la última pregunta si no es de confirmación (p. ej. "¿algo más?").
            if "¿" in text and "confirm" not in text[text.rfind("¿"):].lower():
                text = text[: text.rfind("¿")].rstrip()
            if res.unsupported_claim:  # LLM descartado: usar un mensaje fijo
                text = "Las condiciones cambiaron o se actualizó el carrito."
                res.unsupported_claim = False
            extra = [] if total in text else [f"El total es {total} euros."]
            if "confirm" not in text.lower():
                extra.append("¿Confirmas la compra?")
        else:
            codes = [e.get("code") for e in out["errors"]]
            res.auto_slots["cotizacion_fallida"] = codes
            extra = [self._ERR_TEXT[c] for c in codes if c in self._ERR_TEXT and
                     self._ERR_TEXT[c] not in text]
        self.store.log_event("auto_cotizacion", {"resultado": res.auto_slots.get(
            "cotizacion", res.auto_slots.get("cotizacion_fallida"))})
        res.response = " ".join([text, *extra]).strip()
        self.messages[-1] = {"role": "assistant", "content": res.response}

    def turn(self, user_text: str) -> TurnResult:
        t_start = time.perf_counter()
        self.state.turn += 1
        self.state.last_user = user_text
        self.messages.append({"role": "user", "content": user_text})
        res = TurnResult(user=user_text, response=self.FALLBACK)
        # Respaldo en código: si el usuario expresa un presupuesto total, registrarlo antes de
        # llamar al LLM (queda en "ESTADO ACTUAL" y en la bitácora como regla de código).
        cents = extract_budget_cents(user_text)
        if cents and cents != self.state.budget_cents:
            self.tools.fijar_presupuesto(cents / 100)
            res.auto_slots["presupuesto_cents"] = cents
            self.store.log_event("auto_presupuesto", {"budget_cents": cents,
                                                      "origen": "regla en código"})
        retry: list[dict] = []  # respuesta rechazada + instrucción; no queda en el historial
        for step in range(1, self.max_steps + 1):
            reply = self.llm.chat([self._system(), *self.messages, *retry], TOOL_SCHEMAS)
            res.llm_seconds.append(reply.seconds)
            res.steps = step
            retry = []
            if not reply.tool_calls:
                problem = check_response(reply.content, res.tool_calls, self.state)
                if problem and len(res.guard_events) < self.max_corrections:
                    res.guard_events.append(problem)
                    retry = [reply.as_message(), {"role": "system", "content": problem}]
                    continue
                res.unsupported_claim = problem is not None
                res.response = (self.FALLBACK if problem or not reply.content
                                else reply.content)
                self.messages.append({"role": "assistant", "content": res.response})
                break
            self.messages.append(reply.as_message())
            for call in reply.tool_calls:
                name, args = call["function"]["name"], call["function"].get("arguments") or {}
                t0 = time.perf_counter()
                out = self.tools.dispatch(name, args)
                res.tool_seconds += time.perf_counter() - t0
                res.tool_calls.append({"step": step, "name": name, "args": args,
                                       "ok": out.get("ok", True), "result": out})
                self.messages.append({"role": "tool", "tool_name": name,
                                      "content": json.dumps(out, ensure_ascii=False)})
        else:
            self.messages.append({"role": "assistant", "content": res.response})
        self._auto_quote(res)
        res.total_seconds = time.perf_counter() - t_start
        self.history.append(res)
        return res


def format_turn(r: TurnResult) -> str:
    """Traza legible de un turno: herramientas, correcciones, respuesta y tiempos."""
    out = [f"👤 {r.user}"]
    if "presupuesto_cents" in r.auto_slots:
        monto = fmt_eur(r.auto_slots["presupuesto_cents"])
        out.append(f"   ⚙ presupuesto registrado por código: {monto}")
    if "cotizacion" in r.auto_slots or "cotizacion_fallida" in r.auto_slots:
        q = r.auto_slots.get("cotizacion", r.auto_slots.get("cotizacion_fallida"))
        out.append(f"   ⚙ cotización automática por código: {q}")
    for c in r.tool_calls:
        args = json.dumps(c["args"], ensure_ascii=False)
        err = "" if c["ok"] else f"  ✗ {[e.get('code') for e in c['result'].get('errors', [])]}"
        out.append(f"   🔧 {c['name']}({args}){err}")
    for g in r.guard_events:
        out.append(f"   🛡 corrección pedida al LLM: {g[:90]}…")
    out.append(f"🤖 {r.response}")
    out.append(f"   ⏱ {r.total_seconds:.1f} s · LLM {[round(s, 1) for s in r.llm_seconds]} s · "
               f"herramientas {r.tool_seconds:.1f} s")
    return "\n".join(out)
