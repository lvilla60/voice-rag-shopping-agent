"""Pruebas de normalización de texto y métricas de voz (sin modelos)."""

import pytest

from asistente_compras.voice import entity_errors, extract_entities, speech_text, wer


@pytest.mark.parametrize("entrada,salida", [
    ("La primera cuesta 8,99 €.", "La primera cuesta ocho euros con noventa y nueve."),
    ("Total: 21,00 euros", "Total: veintiún euros"),
    ("Cuesta 1 €", "Cuesta un euro"),
    ("La **primera** opción (R1), Barbato.", "La primera opción, Barbato."),
    ("Quedan 7 unidades 😊", "Quedan siete unidades"),
])
def test_texto_para_voz(entrada, salida):
    assert speech_text(entrada) == salida


def test_wer_ignora_forma_de_escribir_numeros():
    assert wer("La primera cuesta ocho euros con noventa y nueve.",
               "La primera cuesta 8,99€.") == 0.0
    assert wer("quiero quince lápices", "quiero cincuenta lápices") > 0


def test_entidades_y_errores_de_asr():
    e = extract_entities("Quiero quince lápices, tengo cuarenta euros")
    assert e == {"cantidades": ["15"], "importes": ["40"]}
    err = entity_errors("Necesito dos mochilas para portátil, presupuesto de cien euros",
                        "Necesito dos mochilas para portal, presupuesto de 10 euros",
                        key_terms=["mochilas", "portátil"])
    assert err == {"err_cantidades": 0, "err_importes": 1, "err_productos": 1}


def test_division_para_tts():
    from asistente_compras.voice import split_for_tts
    txt = "Hola. " + "palabra " * 60 + "fin. ¿Confirmas?"
    chunks = split_for_tts(txt, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == txt.split()


def test_ids_de_pedido_se_leen_como_numero():
    esperado = "Tu pedido número uno está confirmado."
    assert speech_text("Tu pedido O0001 está confirmado.") == esperado


def test_speech_text_omite_parentesis_y_codigos_de_modelo():
    from asistente_compras.voice import speech_text

    t = speech_text("La primera (ZLZGZ) es el Rii RM100+ por 14,99 €, cable de 1,5 m.")
    assert "ZLZGZ" not in t and "(" not in t
    assert "RM100" in t
    assert "catorce euros con noventa y nueve" in t
    assert "uno coma cinco" in t


def test_split_for_tts_respeta_el_maximo():
    from asistente_compras.voice import split_for_tts

    texto = "Una frase corta. " + "palabra, " * 60 + "fin."
    assert all(len(c) <= 160 for c in split_for_tts(texto))


def test_load_audio_16k_remuestrea(tmp_path):
    import numpy as np
    import soundfile as sf

    from asistente_compras.voice import load_audio_16k

    sf.write(str(tmp_path / "a.wav"), np.zeros((24000, 2), dtype=np.float32), 24000)  # 1 s estéreo
    a = load_audio_16k(tmp_path / "a.wav")
    assert a.dtype == np.float32 and a.ndim == 1 and len(a) == 16000
