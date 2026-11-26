"""Pruebas del runner de escenarios con LLM guionado (sin modelo real)."""

from test_agent import CATALOG, FakeRetriever, ScriptedLLM, call

from asistente_compras.llm import LLMReply
from asistente_compras.scenarios import (
    SCENARIOS,
    Event,
    Scenario,
    Setup,
    error_seen,
    evaluate,
    n_orders,
    order_qty,
    run_scenario,
)


def test_hay_20_escenarios_y_6_de_voz_con_las_situaciones_exigidas():
    assert len(SCENARIOS) >= 20
    voz = [s for s in SCENARIOS if s.voice]
    assert len(voz) == 6
    assert {s.category for s in voz} == {"compra_ok", "ambigua", "cambio", "agotado",
                                         "correccion", "stock_cambia"}
    assert sum(len(s.turns) > 2 for s in voz) >= 3  # al menos 3 con varias intervenciones
    assert len({s.id for s in SCENARIOS}) == len(SCENARIOS)


def test_runner_evento_de_stock_y_evaluacion(tmp_path):
    sc = Scenario(
        "X", "prueba", "stock_cambia",
        turns=["dos fundas", "sí, confirmo", "confirmo otra vez"],
        setup=[Setup("funda", k=3, stock=5, prices_eur=(10.0,))],
        events=[Event(before_turn=2, kind="reduce_cart_stock")],
        expected_qty=2,
        checks=[error_seen("stock_changed"), n_orders(0), order_qty(0)],
    )
    llm = ScriptedLLM([
        LLMReply("", [call("buscar_productos", consulta="funda")]),
        LLMReply("", [call("actualizar_carrito", ref="R1", cantidad=2), call("cotizar")]),
        LLMReply("Son 20 euros, ¿confirmas?"),
        LLMReply("", [call("confirmar_pedido")]),   # falla: el stock bajó a 1
        LLMReply("Ya no hay stock suficiente."),
        LLMReply("", [call("confirmar_pedido")]),   # sin cotización vigente
        LLMReply("Primero hay que cotizar."),
    ])
    run = run_scenario(sc, llm, FakeRetriever(), CATALOG, tmp_path / "e.db")
    ev = evaluate(run)
    assert ev["passed"], ev["failed_checks"]
    assert ev["violations"] == 0 and run.snapshots[-1]["orders"] == 0
    # estado inicial reproducible
    run2 = run_scenario(sc, ScriptedLLM([LLMReply("ok")] * 3), FakeRetriever(), CATALOG,
                        tmp_path / "e.db")
    assert run2.initial_fingerprint == run.initial_fingerprint


def test_modo_voz_registra_transcripcion_y_errores(tmp_path):
    from asistente_compras.voice import Transcription

    class ASRFalso:  # devuelve una transcripción con error en la cantidad
        def transcribe(self, path):
            return Transcription("quiero cincuenta fundas", 0.5, 2.0)

    sc = Scenario("Y", "voz", "compra_ok", turns=["quiero quince fundas"], checks=[],
                  voice=True, key_terms=("fundas",))
    run = run_scenario(sc, ScriptedLLM([LLMReply("¿Cuál?")]), FakeRetriever(), CATALOG,
                       tmp_path / "v.db", audio=[tmp_path / "a.wav"], asr=ASRFalso())
    v = run.voice[0]
    assert v["transcript"] == "quiero cincuenta fundas" and v["err_cantidades"] == 1
    assert v["err_productos"] == 0 and v["wer"] > 0 and v["asr_s"] == 0.5
    assert run.turns[0].user == "quiero cincuenta fundas"  # el agente recibió lo del ASR
