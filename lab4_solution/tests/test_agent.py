"""Pruebas del agente con un LLM guionado (sin modelo real): validan las reglas del código."""

import pandas as pd
import pytest

from asistente_compras.agent import ConversationState, ShoppingAgent, ShoppingTools, _as_int
from asistente_compras.llm import LLMReply, parse_tool_calls
from asistente_compras.store import Store, restore_store

CATALOG = pd.DataFrame({
    "product_id": ["A", "B", "C"],
    "title": ["Funda iPhone 13 negra", "Funda iPhone 13 roja", "Cargador USB-C 20W"],
    "brand": ["Spigen", "", "Anker"],
    "color": ["negro", "rojo", ""],
    "bullet_points": ["Silicona", "", "Carga rápida"],
    "description": ["", "", ""],
})


class FakeRetriever:
    def search(self, query, k=10):
        return [(p, 1.0) for p in ("A", "B", "C")][:k]


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


class ScriptedLLM:
    """Devolver respuestas predefinidas en orden, como si fuera el modelo."""

    name = "guion"

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, tools):
        return self.replies.pop(0)


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "t.db"
    restore_store(db, CATALOG.product_id)
    s = Store(db)
    for pid, price, stock in (("A", 1000, 5), ("B", 1200, 0), ("C", 2000, 3)):
        s.con.execute("UPDATE products SET price_cents=?, stock=? WHERE product_id=?",
                      (price, stock, pid))
    yield s
    s.close()


@pytest.fixture
def tools(store):
    return ShoppingTools(store, FakeRetriever(), CATALOG, ConversationState())


def test_busqueda_asigna_refs_y_trae_precio_de_la_tienda(tools):
    r = tools.buscar_productos("funda")
    assert [i["ref"] for i in r["resultados"]] == ["R1", "R2", "R3"]
    assert r["resultados"][0]["precio"] == "10,00 €" and r["resultados"][1]["disponible"] is False


def test_filtros_de_busqueda(tools):
    r = tools.buscar_productos("funda", precio_max_eur=15, solo_disponibles=True)
    assert [i["titulo"] for i in r["resultados"]] == ["Funda iPhone 13 negra"]


def test_atributos_ausentes_son_desconocidos(tools):
    tools.buscar_productos("funda")
    d = tools.ver_detalle("r2")  # resolver refs sin distinguir mayúsculas
    assert d["marca"] == "desconocido" and d["caracteristicas"] == "desconocido"


@pytest.mark.parametrize("v,esperado", [(3, 3), (3.0, 3), ("3", 3), (2.5, None),
                                        ("tres", None), (True, None)])
def test_conversion_de_cantidad(v, esperado):
    assert _as_int(v) == esperado


def test_ref_inexistente_y_cantidad_invalida(tools):
    assert tools.actualizar_carrito("R9", 1)["errors"][0]["code"] == "ref_desconocida"
    tools.buscar_productos("x")
    assert tools.actualizar_carrito("R1", 2.5)["errors"][0]["code"] == "cantidad_invalida"


def test_aviso_de_presupuesto_en_carrito(tools):
    tools.buscar_productos("x")
    tools.fijar_presupuesto("25")
    r = tools.actualizar_carrito("R1", 3)
    assert "supera el presupuesto" in r["carrito"]["aviso"]
    assert tools.cotizar()["errors"][0]["code"] == "over_budget"


def test_herramienta_y_argumentos_desconocidos(tools):
    assert tools.dispatch("borrar_todo", {})["errors"][0]["code"] == "herramienta_desconocida"
    assert tools.dispatch("ver_carrito", {"x": 1})["errors"][0]["code"] == "argumentos_invalidos"


def test_no_confirma_en_el_mismo_turno_que_cotiza(store):
    llm = ScriptedLLM([
        LLMReply("", [call("buscar_productos", consulta="funda")]),
        LLMReply("", [call("actualizar_carrito", ref="R1", cantidad=2)]),
        LLMReply("", [call("cotizar")]),
        LLMReply("", [call("confirmar_pedido")]),  # el LLM se apura: el código lo impide
        LLMReply("Son 20,00 euros, ¿confirmas?"),
    ])
    agent = ShoppingAgent(llm, store, FakeRetriever(), CATALOG)
    r = agent.turn("quiero dos fundas negras")
    confirm = [c for c in r.tool_calls if c["name"] == "confirmar_pedido"][0]
    assert confirm["result"]["errors"][0]["code"] == "falta_confirmacion_del_cliente"
    assert store.summary()["orders"] == 0


def test_flujo_completo_y_confirmacion_repetida(store):
    llm = ScriptedLLM([
        LLMReply("", [call("buscar_productos", consulta="funda")]),
        LLMReply("", [call("actualizar_carrito", ref="R1", cantidad=2), call("cotizar")]),
        LLMReply("Son 20,00 euros, ¿confirmas?"),
        LLMReply("", [call("confirmar_pedido")]),
        LLMReply("Listo, pedido O0001."),
        LLMReply("", [call("confirmar_pedido")]),
        LLMReply("Ya estaba registrado."),
    ])
    agent = ShoppingAgent(llm, store, FakeRetriever(), CATALOG)
    agent.turn("dos fundas negras")
    r2 = agent.turn("sí, confirmo")
    r3 = agent.turn("confirmo otra vez")
    assert r2.tool_calls[0]["result"]["pedido"] == "O0001"
    assert r3.tool_calls[0]["result"]["ya_estaba_registrado"] is True
    assert store.summary()["orders"] == 1


def test_estado_se_inyecta_en_el_prompt(store):
    llm = ScriptedLLM([LLMReply("", [call("fijar_presupuesto", presupuesto_eur=50)]),
                       LLMReply("Anotado.")])
    agent = ShoppingAgent(llm, store, FakeRetriever(), CATALOG)
    agent.turn("tengo 50 euros")
    assert "Presupuesto: 50,00 €" in agent._system()["content"]


def test_limite_de_pasos_devuelve_respuesta_de_respaldo(store):
    llm = ScriptedLLM([LLMReply("", [call("ver_carrito")])] * 10)
    agent = ShoppingAgent(llm, store, FakeRetriever(), CATALOG, max_steps=3)
    assert agent.turn("hola").response == ShoppingAgent.FALLBACK


def test_parseo_de_tool_calls_en_texto():
    txt = 'Busco.\n<tool_call>\n{"name": "cotizar", "arguments": {}}\n</tool_call>'
    content, calls = parse_tool_calls(txt)
    assert content == "Busco." and calls[0]["function"]["name"] == "cotizar"
    raw = '<tool_call>{"name":"ver_detalle","arguments":"{\\"ref\\":\\"R1\\"}"}</tool_call>'
    _, calls = parse_tool_calls(raw)
    assert calls[0]["function"]["arguments"] == {"ref": "R1"}


def test_control_detecta_anuncio_sin_accion_y_reintenta(store):
    llm = ScriptedLLM([
        LLMReply("Un momento, voy a buscar."),                         # rechazada
        LLMReply("", [call("buscar_productos", consulta="funda")]),
        LLMReply("Tengo tres fundas: la primera a 10 euros."),
    ])
    agent = ShoppingAgent(llm, store, FakeRetriever(), CATALOG)
    r = agent.turn("busco una funda")
    assert len(r.guard_events) == 1 and r.response.startswith("Tengo tres fundas")
    assert all("Un momento" not in m.get("content", "") for m in agent.messages)


def test_control_bloquea_afirmacion_de_carrito_sin_herramienta(store):
    llm = ScriptedLLM([LLMReply("He actualizado el carrito con dos fundas.")] * 3)
    agent = ShoppingAgent(llm, store, FakeRetriever(), CATALOG)
    r = agent.turn("ponme dos")
    assert r.unsupported_claim and r.response == ShoppingAgent.FALLBACK
    assert len(r.guard_events) == 2


def test_parseo_ignora_bloque_think_vacio():
    content, calls = parse_tool_calls('<think>\n\n</think>\n\nHola')
    assert content == "Hola" and calls == []


@pytest.mark.parametrize("texto,cents", [
    ("Busco auriculares. Tengo como máximo 60 euros.", 6000),
    ("mi presupuesto es de 45,50 €", 4550),
    ("El presupuesto total: 120 euros", 12000),
    ("no quiero gastar más de 30 euros", 3000),
    ("puedo gastar como máximo 75 euros", 7500),
    ("tengo 50€", 5000),
    ("en total como máximo 200 euros", 20000),
    # No son presupuesto total:
    ("quiero auriculares de máximo 20 euros", None),
    ("¿cuánto cuesta? ¿unos 15 euros?", None),
    ("quiero 3 fundas", None),
])
def test_extraccion_de_presupuesto(texto, cents):
    from asistente_compras.agent import extract_budget_cents
    assert extract_budget_cents(texto) == cents


def test_presupuesto_se_registra_antes_del_llm(store):
    vistos = []

    class Espia:
        name = "espía"

        def chat(self, messages, tools):
            vistos.append(messages[0]["content"])
            return LLMReply("Anotado, ¿qué buscas?")

    agent = ShoppingAgent(Espia(), store, FakeRetriever(), CATALOG)
    r = agent.turn("Hola, tengo como máximo 60 euros")
    assert r.auto_slots == {"presupuesto_cents": 6000}
    assert "Presupuesto: 60,00 €" in vistos[0]          # el LLM ya lo ve en su primera llamada
    assert "auto_presupuesto" in set(store.ops().op)
    assert agent.turn("busco fundas").auto_slots == {}  # no se repite si no cambia


def test_relevancia_baja_genera_nota(store):
    class Calibrado:
        calibrated = True

        def search(self, query, k=10):
            return [("A", 0.02), ("C", 0.01)]

    r = ShoppingTools(store, Calibrado(), CATALOG, ConversationState()).buscar_productos("avión")
    assert {i["relevancia"] for i in r["resultados"]} == {"baja"}
    assert r["nota"].startswith("ningún resultado")


def test_cotizacion_automatica_si_el_llm_no_cotiza(store):
    llm = ScriptedLLM([
        LLMReply("", [call("buscar_productos", consulta="funda")]),
        LLMReply("", [call("actualizar_carrito", ref="R1", cantidad=2)]),
        LLMReply("Añadí dos fundas. ¿Quieres algo más?"),     # no cotiza
        LLMReply("", [call("confirmar_pedido")]),
        LLMReply("Pedido registrado."),
    ])
    agent = ShoppingAgent(llm, store, FakeRetriever(), CATALOG)
    r1 = agent.turn("dos fundas negras")
    assert r1.auto_slots["cotizacion"] == "Q0001"
    assert r1.response == "Añadí dos fundas. El total es 20,00 euros. ¿Confirmas la compra?"
    assert agent.messages[-1]["content"] == r1.response   # el LLM ve lo que oyó el cliente
    r2 = agent.turn("sí, confirmo")
    assert r2.tool_calls[0]["ok"] and store.summary()["orders"] == 1


def test_cotizacion_automatica_avisa_si_supera_presupuesto(store):
    llm = ScriptedLLM([
        LLMReply("", [call("buscar_productos", consulta="funda")]),
        LLMReply("", [call("actualizar_carrito", ref="R1", cantidad=3)]),
        LLMReply("Listo, tres fundas."),
    ])
    agent = ShoppingAgent(llm, store, FakeRetriever(), CATALOG)
    r = agent.turn("tres fundas, tengo 25 euros")
    assert r.auto_slots["cotizacion_fallida"] == ["over_budget"]
    assert r.response.endswith("supera tu presupuesto.")


def test_cotizar_es_idempotente_y_permite_confirmar(store):
    llm = ScriptedLLM([
        LLMReply("", [call("buscar_productos", consulta="funda")]),
        LLMReply("", [call("actualizar_carrito", ref="R1", cantidad=1), call("cotizar")]),
        LLMReply("Son 10 euros, ¿confirmas?"),
        LLMReply("", [call("cotizar")]),            # el LLM re-cotiza al oír "confirmo"
        LLMReply("", [call("confirmar_pedido")]),   # ...y luego confirma: debe funcionar
        LLMReply("Pedido registrado."),
    ])
    agent = ShoppingAgent(llm, store, FakeRetriever(), CATALOG)
    agent.turn("una funda")
    r = agent.turn("sí, confirmo")
    assert r.tool_calls[0]["result"]["cotizacion"] == "Q0001"
    assert r.tool_calls[0]["result"]["ya_presentada_al_cliente"] is True
    assert r.tool_calls[1]["ok"] and store.summary() | {} and store.summary()["quotes"] == 1


def test_anuncio_busco_sin_herramienta_se_corrige(store):
    from asistente_compras.agent import ConversationState, check_response
    assert check_response("Busco mochilas para ti. ¿Algún color?", [], ConversationState())
    assert check_response("Te muestro tres opciones.", [], ConversationState()) is None


def test_recotiza_tras_cambio_de_precio(store):
    llm = ScriptedLLM([
        LLMReply("", [call("buscar_productos", consulta="funda")]),
        LLMReply("", [call("actualizar_carrito", ref="R1", cantidad=1)]),
        LLMReply("Añadida."),
        LLMReply("", [call("confirmar_pedido")]),
        LLMReply("El precio cambió."),
        LLMReply("", [call("confirmar_pedido")]),
        LLMReply("Listo."),
    ])
    agent = ShoppingAgent(llm, store, FakeRetriever(), CATALOG)
    agent.turn("una funda")
    store.set_price("A", 1500)
    r = agent.turn("confirmo")
    assert r.auto_slots["cotizacion"] == "Q0002" and "15,00" in r.response
    assert agent.turn("confirmo").tool_calls[0]["ok"] and store.summary()["orders"] == 1


def test_control_falsos_positivos_y_anuncio_con_otra_herramienta():
    from asistente_compras.agent import ConversationState, check_response
    st = ConversationState()
    # "actualizado el precio" no es afirmar un cambio de carrito
    assert check_response("El precio se ha actualizado a 17,99 euros.", [], st) is None
    # anunciar una búsqueda habiendo llamado solo a fijar_presupuesto: se corrige
    calls = [{"name": "fijar_presupuesto", "ok": True}]
    assert check_response("Busco mochilas para ti.", calls, st) is not None


def test_parse_tool_calls_formato_xml_qwen35():
    from asistente_compras.llm import parse_tool_calls

    text = ("Busco opciones.\n<tool_call>\n<function=buscar_productos>\n<parameter=consulta>\n"
            "cables USB tipo C\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call><function=actualizar_carrito><parameter=ref>R1</parameter>"
            "<parameter=cantidad>2</parameter></function></tool_call>")
    content, calls = parse_tool_calls(text)
    assert content == "Busco opciones."
    assert calls[0] == {"function": {"name": "buscar_productos",
                                     "arguments": {"consulta": "cables USB tipo C"}}}
    assert calls[1]["function"]["arguments"] == {"ref": "R1", "cantidad": 2}


def test_parse_tool_calls_formato_json_sigue_igual():
    from asistente_compras.llm import parse_tool_calls

    content, calls = parse_tool_calls(
        '<think></think>Hola <tool_call>{"name": "ver_carrito", "arguments": {}}</tool_call>')
    assert content == "Hola" and calls == [{"function": {"name": "ver_carrito", "arguments": {}}}]


def test_late_system_as_user_solo_cambia_los_sistemas_tardios():
    from asistente_compras.llm import late_system_as_user

    msgs = [{"role": "system", "content": "A"}, {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "x"}, {"role": "system", "content": "corrige"}]
    out = late_system_as_user(msgs)
    assert out[:3] == msgs[:3]
    assert out[3]["role"] == "user" and "corrige" in out[3]["content"]


def test_user_confirms():
    from asistente_compras.agent import user_confirms

    for t in ["Sí, confirmo la compra.", "Confirmo otra vez, por si acaso.", "si", "Adelante",
              "OK, cómpralo"]:
        assert user_confirms(t), t
    for t in ["Me quedo con la primera opción.", "Ponme una de la primera opción.",
              "Quiero comprar una funda.",
              "De acuerdo, ponme la primera alternativa que tenga stock.",
              "Vale, busca otra mochila y ponme dos.", "No, mejor no lo compro. Cancela.",
              "No confirmes todavía", ""]:
        assert not user_confirms(t), t


def test_cut_at_turn_end():
    from asistente_compras.llm import cut_at_turn_end

    raw = "Hola<tool_call>{}</tool_call><|im_end|>\n<|im_start|>user\nOk, quiero dos"
    assert cut_at_turn_end(raw) == "Hola<tool_call>{}</tool_call>"
    assert cut_at_turn_end("sin marcas") == "sin marcas"
