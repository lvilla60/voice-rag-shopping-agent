"""Conversación interactiva por voz en la Mac (prueba manual de punta a punta).

Uso (en una ventana de Terminal, no dentro de un notebook)::

    cd lab4_solution
    ollama serve &          # si no está corriendo
    uv run lab4-conversar   # opciones: --modelo qwen3.5:9b  --sin-voz  --nueva-referencia

Controles en cada turno: Enter para empezar a grabar, Enter para terminar. Escribe ``t`` para
teclear el mensaje en vez de hablar, y ``q`` para salir.
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import time
from pathlib import Path

REFERENCE_TEXT = (
    "Esta grabación sirve como referencia de mi voz. La leo con calma, en un lugar sin ruido, "
    "y con el tono que uso normalmente al conversar. Los días de semana suelo caminar por el "
    "parque antes de empezar a trabajar, y los fines de semana prefiero cocinar algo sencillo "
    "y leer un buen libro."
)


class _Mic:
    """Micrófono abierto **una sola vez** por proceso y frecuencia de muestreo.

    En macOS, detener y cerrar el stream de CoreAudio en cada grabación a veces deja el
    proceso bloqueado (deadlock dentro de ``AudioOutputUnitStop``). Por eso el stream queda
    abierto todo el tiempo y solo se guardan los bloques mientras ``recording`` es True.
    """

    _open: dict[int, _Mic] = {}

    def __init__(self, samplerate: int):
        import sounddevice as sd

        self.q: queue.Queue = queue.Queue()
        self.recording = False
        self.stream = sd.InputStream(samplerate=samplerate, channels=1, dtype="float32",
                                     callback=self._cb)
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        if self.recording:
            self.q.put(indata.copy())

    @classmethod
    def get(cls, samplerate: int) -> _Mic:
        if samplerate not in cls._open:
            cls._open[samplerate] = cls(samplerate)
        return cls._open[samplerate]


def _exit_now(code: int = 0) -> None:
    """Salir sin cerrar los streams de audio (cerrarlos es lo que puede bloquear en macOS)."""
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def record_until_enter(path: Path, samplerate: int = 16000, max_seconds: float = 20.0,
                       warn_silence: bool = True) -> Path | None:
    """Grabar desde un Enter hasta el siguiente Enter, o hasta ``max_seconds``.

    Muestra un contador con el nivel de audio para confirmar que el micrófono capta la voz.
    """
    import select

    import numpy as np
    import soundfile as sf

    mic = _Mic.get(samplerate)
    input("   [Enter] para empezar a grabar… ")
    print(f"   🔴 Grabando (máx. {max_seconds:.0f} s)… [Enter] para terminar")
    while not mic.q.empty():  # descartar restos de una grabación anterior
        mic.q.get()
    chunks, t0, level = [], time.perf_counter(), 0.0
    mic.recording = True
    try:
        while (el := time.perf_counter() - t0) < max_seconds:
            while not mic.q.empty():
                c = mic.q.get()
                chunks.append(c)
                level = max(level * 0.8, float(np.abs(c).max()))
            db = 20 * np.log10(max(level, 1e-5))  # −60 dB (silencio) … 0 dB (máximo)
            bar = "█" * int(np.clip((db + 60) / 60, 0, 1) * 30)
            sys.stdout.write(f"\r   {el:5.1f} s  nivel |{bar:<30}|")
            sys.stdout.flush()
            if select.select([sys.stdin], [], [], 0.1)[0]:  # detectar si se presionó Enter
                sys.stdin.readline()
                break
    finally:
        mic.recording = False
    time.sleep(0.05)
    while not mic.q.empty():
        chunks.append(mic.q.get())
    print()
    if not chunks:
        print("   ⚠ No llegó audio del micrófono. Revisa el permiso de micrófono de la terminal.")
        return None
    audio = np.concatenate(chunks)
    if warn_silence and float(np.abs(audio).max()) < 0.01:
        print("   ⚠ La grabación está casi en silencio: ¿la terminal tiene permiso de micrófono?")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, samplerate)
    print(f"   ✔ {len(audio) / samplerate:.1f} s grabados")
    return path


def play(path: Path) -> None:
    subprocess.run(["afplay", str(path)], check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Conversación por voz con el asistente de compras")
    ap.add_argument("--modelo", default="qwen3.5:9b")
    ap.add_argument("--sin-voz", action="store_true", help="no sintetizar la respuesta (rápido)")
    ap.add_argument("--nueva-referencia", action="store_true", help="volver a grabar tu voz")
    args = ap.parse_args()

    import pandas as pd

    from asistente_compras.agent import ShoppingAgent, format_turn
    from asistente_compras.config import get_paths
    from asistente_compras.llm import make_llm
    from asistente_compras.retrieval import build_system_retriever
    from asistente_compras.store import Store, restore_store
    from asistente_compras.voice import MLXWhisperASR, XTTSVoice

    paths = get_paths().ensure()
    voz_dir = paths.root / "voz"
    ref = voz_dir / "referencia.wav"
    out_dir = voz_dir / "conversacion" / time.strftime("%Y%m%d_%H%M%S")

    if not args.sin_voz and (args.nueva_referencia or not ref.exists()):
        print("\n=== Grabación de tu voz de referencia (20–30 s) ===")
        print("Lee en voz alta, con calma:\n")
        print(f"  «{REFERENCE_TEXT}»\n")
        record_until_enter(ref, samplerate=22050, max_seconds=40)

    print("\nCargando modelos (solo la primera vez tarda)…")
    t = time.perf_counter()
    catalog = pd.read_parquet(paths.processed / "catalog.parquet")
    retriever = build_system_retriever(catalog, paths.index)
    t_ret = time.perf_counter() - t
    t = time.perf_counter()
    llm = make_llm(args.modelo, backend="ollama")
    llm.chat([{"role": "user", "content": "hola"}], [])  # precargar el modelo en Ollama
    t_llm = time.perf_counter() - t
    t = time.perf_counter()
    asr = MLXWhisperASR()
    t_asr = time.perf_counter() - t
    t = time.perf_counter()
    tts = None if args.sin_voz else XTTSVoice(ref)
    t_tts = time.perf_counter() - t
    print(f"Carga inicial: recuperador {t_ret:.1f} s · LLM {t_llm:.1f} s · ASR {t_asr:.1f} s"
          f" · TTS {t_tts:.1f} s")

    db = paths.root / "store" / "conversacion.db"
    restore_store(db, catalog.product_id)
    store = Store(db)
    agent = ShoppingAgent(llm, store, retriever, catalog)
    print("\nTienda restaurada. Ejemplo: «Necesito dos cables USB tipo C, mi presupuesto es de "
          "40 euros».")
    print("Controles: [Enter] hablar · t = escribir · q = salir\n")

    n = 0
    while True:
        cmd = input("¿Hablar [Enter], escribir [t] o salir [q]? ").strip().lower()
        if cmd == "q":
            break
        n += 1
        t_asr_turn = 0.0
        if cmd == "t":
            text = input("   ✍  ").strip()
        else:
            wav = record_until_enter(out_dir / f"usuario_{n:02d}.wav")
            if wav is None:
                continue
            t_end = time.perf_counter()
            tr = asr.transcribe(wav)
            text, t_asr_turn = tr.text, time.perf_counter() - t_end
            print(f"   🎙 Entendí: «{text}»")
        if not text:
            continue
        r = agent.turn(text)
        print(format_turn(r))
        if tts is not None:
            syn = tts.synthesize(r.response, out_dir / f"respuesta_{n:02d}.wav")
            print(f"   ⏱ ASR {t_asr_turn:.1f} s · agente {r.total_seconds:.1f} s · "
                  f"TTS {syn['seconds']:.1f} s → total "
                  f"{t_asr_turn + r.total_seconds + syn['seconds']:.1f} s")
            play(syn["path"])
        print()

    print("\nEstado final:", store.summary())
    print(agent.state.render(store, catalog.set_index("product_id").title))
    print(f"Audios guardados en: {out_dir}")
    store.close()
    _exit_now()


if __name__ == "__main__":
    sys.exit(main())


def grabar_escenarios() -> None:
    """Grabar los turnos de los 6 escenarios de voz: ``uv run lab4-grabar``.

    Guarda ``data/voz/escenarios/<ID>/turno_NN.wav``. Si un archivo ya existe se omite (salvo
    ``--regrabar``), así se puede interrumpir y continuar después.
    """
    ap = argparse.ArgumentParser(description="Grabar los turnos de los escenarios de voz")
    ap.add_argument("--solo", help="IDs separados por coma, p. ej. V1,V3")
    ap.add_argument("--regrabar", action="store_true", help="volver a grabar los existentes")
    args = ap.parse_args()

    from asistente_compras.config import get_paths
    from asistente_compras.scenarios import SCENARIOS

    base = get_paths().ensure().root / "voz" / "escenarios"
    ids = set(args.solo.split(",")) if args.solo else None
    voz = [s for s in SCENARIOS if s.voice and (ids is None or s.id in ids)]
    total = sum(len(s.turns) for s in voz)
    print(f"\n{len(voz)} escenarios · {total} frases. Lee cada frase tal cual aparece, con tono "
          "natural, como si hablaras con el asistente.")
    print("Controles: [Enter] grabar · [Enter] terminar · después: [Enter] seguir, "
          "r = repetir, q = salir\n")
    n = 0
    for sc in voz:
        print(f"━━━ {sc.id} · {sc.title}")
        if sc.id == "V5":
            print("    (Nota: en el turno 1 di «cincuenta» a propósito: simula el error de "
                  "transcripción que luego corriges.)")
        for i, text in enumerate(sc.turns, start=1):
            n += 1
            path = base / sc.id / f"turno_{i:02d}.wav"
            if path.exists() and not args.regrabar:
                print(f"  [{n}/{total}] ✓ ya grabado: «{text}»")
                continue
            while True:
                print(f"\n  [{n}/{total}] Lee:  «{text}»")
                if record_until_enter(path, samplerate=16000, max_seconds=20) is None:
                    continue
                cmd = input("   [Enter] seguir · r repetir · q salir: ").strip().lower()
                if cmd == "q":
                    print(f"\nPausado. Para continuar: uv run lab4-grabar  ({base})")
                    _exit_now()
                if cmd != "r":
                    break
    print(f"\n✔ Listo. Audios en {base}")
    _exit_now()
