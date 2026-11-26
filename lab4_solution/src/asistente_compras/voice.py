"""Voz: ASR, TTS con clonación, normalización de texto, métricas y pipeline de punta a punta.

Pipeline de un turno hablado (todo local, sin APIs externas)::

    audio usuario ─► ASR (Whisper) ─► texto ─► agente (LLM + herramientas) ─► respuesta
                                                                                │
                   audio respuesta ◄─ TTS XTTS-v2 (voz clonada) ◄─ texto para voz ┘

La latencia se mide **desde que termina la grabación** hasta que el audio de respuesta está
listo, desglosada por etapa, y se separa la carga inicial de los modelos (una vez) de los
turnos posteriores.

Backends: en la Mac, ``mlx-whisper`` (Apple Silicon) y XTTS en CPU; en Colab,
``faster-whisper`` y XTTS en GPU. Mismos modelos de Whisper (large-v3-turbo) en ambos.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------
# Texto para voz y normalización para métricas
# --------------------------------------------------------------------------------------
_MD_RE = re.compile(r"[*_#`>]+")
_REF_RE = re.compile(r"\s*\(?\bR\d{1,3}\b\)?")
_PRICE_RE = re.compile(r"(\d{1,6})(?:[.,](\d{1,2}))?\s*(?:€|euros?\b|EUR\b)", re.I)
_NUM_RE = re.compile(r"(?<![A-Za-z\d])\d+(?:[.,]\d+)?(?![A-Za-z\d])")  # no partir "RM100" ni "USB3"
_ID_RE = re.compile(r"\b[OQ](\d{3,6})\b")  # ids de pedido / cotización
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")


def _words(n: int) -> str:
    from num2words import num2words

    return num2words(n, lang="es")


def _euros(m: re.Match) -> str:
    ent, dec = int(m.group(1)), m.group(2)
    ent_w = "un" if ent == 1 else re.sub(r"uno$", "ún", _words(ent))
    out = f"{ent_w} {'euro' if ent == 1 else 'euros'}"
    if dec and int(dec.ljust(2, "0")):
        out += f" con {_words(int(dec.ljust(2, '0')))}"
    return out


def _number(m: re.Match) -> str:
    """"3" → "tres"; "1,5" → "uno coma cinco"; los miles con punto ("1.000") quedan enteros."""
    g = m.group()
    ip, _, dec = g.replace(".", ",").partition(",")
    if dec and len(dec) == 3 and "." in g:  # leer 1.000 como mil
        return _words(int(ip + dec))
    return _words(int(ip)) + (f" coma {_words(int(dec))}" if dec else "")


def speech_text(text: str) -> str:
    """Preparar la respuesta del agente para el TTS.

    Quita markdown, emojis y refs internas ("R1"), y escribe precios y números con palabras
    ("8,99 €" → "ocho euros con noventa y nueve"). Así la síntesis no deletrea símbolos.
    """
    t = _EMOJI_RE.sub("", _MD_RE.sub("", text))
    # No leer lo que va entre paréntesis (marcas como "ZLZGZ", aclaraciones): son nombres
    # impronunciables que además hacían que XTTS se descontrolara ("No. No. No…").
    t = re.sub(r"\s*\([^)]*\)", "", t)
    t = _REF_RE.sub("", t)
    t = _ID_RE.sub(lambda m: f"número {int(m.group(1))}", t)  # "O0001" → "número 1"
    t = _PRICE_RE.sub(_euros, t)
    t = _NUM_RE.sub(_number, t)
    t = re.sub(r"\s+([,.;:?!])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_for_wer(text: str) -> str:
    """Forma canónica para comparar textos: números en palabras, sin puntuación ni mayúsculas.

    Evita contar como error "8,99 €" frente a "ocho euros con noventa y nueve".
    """
    t = speech_text(text).lower()
    t = re.sub(r"[^\w\sáéíóúüñ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def wer(reference: str, hypothesis: str) -> float:
    import jiwer

    ref, hyp = normalize_for_wer(reference), normalize_for_wer(hypothesis)
    return float(jiwer.wer(ref, hyp)) if ref else float(bool(hyp))


def _to_digits(text: str) -> str:
    from text_to_num import alpha2digit

    t = alpha2digit(text.lower(), "es", threshold=0)
    return re.sub(r"(\d)[.,](\d)", r"\1.\2", t)


def extract_entities(text: str) -> dict[str, list[str]]:
    """Cantidades e importes mencionados, en dígitos (para comparar ASR con la referencia)."""
    t = _to_digits(text)
    amounts = re.findall(r"(\d+(?:\.\d+)?)\s*(?:€|euros?\b)", t)
    t_wo = re.sub(r"\d+(?:\.\d+)?\s*(?:€|euros?\b)", " ", t)
    qty = re.findall(r"\b\d+\b", t_wo)
    return {"cantidades": sorted(qty), "importes": sorted(amounts)}


def entity_errors(reference: str, hypothesis: str, key_terms: list[str] = ()) -> dict[str, int]:
    """Errores del ASR en cantidades, importes y productos (lo que exige el enunciado).

    Se cuenta cada elemento de la referencia que falta o cambió en la transcripción.
    ``key_terms``: palabras de producto del escenario (p. ej. "mochila", "portátil").
    """
    r, h = extract_entities(reference), extract_entities(hypothesis)
    hyp_norm = normalize_for_wer(hypothesis)

    def missing(a: list[str], b: list[str]) -> int:
        b = list(b)
        n = 0
        for x in a:
            if x in b:
                b.remove(x)
            else:
                n += 1
        return n

    prod = sum(normalize_for_wer(k) not in hyp_norm for k in key_terms
               if normalize_for_wer(k) in normalize_for_wer(reference))
    return {"err_cantidades": missing(r["cantidades"], h["cantidades"]),
            "err_importes": missing(r["importes"], h["importes"]),
            "err_productos": prod}


# --------------------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------------------
@dataclass
class Transcription:
    text: str
    seconds: float
    audio_seconds: float


def _duration(path: Path) -> float:
    import soundfile as sf

    return float(sf.info(str(path)).duration)


class MLXWhisperASR:
    """Whisper large-v3-turbo en Apple Silicon (Mac)."""

    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo"):
        import mlx_whisper

        self._mlx, self.model, self.name = mlx_whisper, model, f"mlx:{model.split('/')[-1]}"

    def transcribe(self, path: Path) -> Transcription:
        t0 = time.perf_counter()
        r = self._mlx.transcribe(str(path), path_or_hf_repo=self.model, language="es",
                                 temperature=0.0)
        return Transcription(r["text"].strip(), time.perf_counter() - t0, _duration(path))


class FasterWhisperASR:
    """Whisper large-v3-turbo con CTranslate2 (Colab, GPU float16)."""

    def __init__(self, model: str = "large-v3-turbo", device: str = "cuda",
                 compute_type: str = "int8_float16"):  # usar pesos int8: ~0,8 GB menos de GPU
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.name = f"faster-whisper:{model}"

    def transcribe(self, path: Path) -> Transcription:
        t0 = time.perf_counter()
        # Entregar el audio ya decodificado: el lector interno de faster-whisper (PyAV) falla
        # con versiones recientes de `av` (p. ej. la de Colab: "unexpected keyword
        # 'metadata_errors'").
        segs, _ = self.model.transcribe(load_audio_16k(path), language="es", beam_size=5,
                                        vad_filter=True, temperature=0.0)
        text = " ".join(s.text.strip() for s in segs)  # consumir el generador aquí
        return Transcription(text.strip(), time.perf_counter() - t0, _duration(path))


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    """WAV → float32 mono a ``sr`` Hz con ``soundfile``.

    Ni PyAV ni torchaudio: sus versiones recientes cambian de API o dependen de ``torchcodec`` y
    FFmpeg, y fallan en Colab. Lo usan Whisper, XTTS (latentes de la referencia) y ECAPA.
    """
    import soundfile as sf

    audio, in_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if in_sr != sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(in_sr, sr)
        audio = resample_poly(audio, sr // g, in_sr // g).astype(np.float32)
    return audio


def load_audio_16k(path: Path) -> np.ndarray:
    """Lo que espera Whisper: mono, 16 kHz."""
    return load_audio(path, 16000)


def _xtts_load_audio(audiopath, sampling_rate):
    """Reemplazar ``TTS.tts.models.xtts.load_audio``: mismo contrato (tensor [1, N] en [-1, 1])."""
    import torch

    return torch.from_numpy(load_audio(Path(audiopath), sampling_rate)).unsqueeze(0).clip_(-1, 1)


def make_asr(backend: str | None = None):
    if backend is None:
        import torch

        backend = "faster" if torch.cuda.is_available() else "mlx"
    return FasterWhisperASR() if backend == "faster" else MLXWhisperASR()


# --------------------------------------------------------------------------------------
# TTS con clonación
# --------------------------------------------------------------------------------------
class XTTSVoice:
    """XTTS-v2 (Coqui) con la voz de una grabación de referencia.

    Las características de la voz (latentes de condicionamiento) se calculan **una vez** al
    cargar, no en cada respuesta. Licencia del modelo: CPML (uso no comercial).
    """

    SAMPLE_RATE = 24000
    # El GPT de XTTS emite ~21,5 códigos de audio por segundo. Tope por trozo: 0,13 s por
    # carácter + 1,5 s (el habla normal medida ronda 0,08 s/carácter). Sirve para cortar
    # las "colas" de repetición sin tocar una frase normal.
    TOKENS_PER_SECOND = 21.5
    SECONDS_PER_CHAR_MAX = 0.13

    def __init__(self, reference_wav: Path | list[Path], device: str | None = None):
        import os

        import torch
        from TTS.api import TTS

        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # silenciar el aviso "max_new_tokens y max_length"
        # En la Mac, CPU es ~2× más rápido que MPS para XTTS (medido); en Colab, CUDA.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.model = self.tts.synthesizer.tts_model
        import TTS.tts.models.xtts as xtts_module

        xtts_module.load_audio = _xtts_load_audio  # leer con soundfile (ver load_audio)
        refs = reference_wav if isinstance(reference_wav, list) else [reference_wav]
        self.gpt_latent, self.spk_emb = self.model.get_conditioning_latents(
            audio_path=[str(p) for p in refs])
        self.name = f"xtts_v2@{self.device}"

    def synthesize(self, text: str, out_path: Path) -> dict[str, Any]:
        import soundfile as sf
        import torch

        spoken = speech_text(text)
        t0 = time.perf_counter()
        parts, truncated = [], 0
        with torch.inference_mode():
            for chunk in split_for_tts(spoken):
                cap_s = self.SECONDS_PER_CHAR_MAX * len(chunk) + 1.5
                # Usar temperatura baja y penalización de repetición: menos "alucinaciones" de audio
                # (frases repetidas o colas largas); el tope de tokens corta las que queden.
                out = self.model.inference(chunk, "es", self.gpt_latent, self.spk_emb,
                                           temperature=0.3, top_p=0.8, repetition_penalty=10.0,
                                           enable_text_splitting=False,
                                           max_new_tokens=int(cap_s * self.TOKENS_PER_SECOND))
                w = np.asarray(out["wav"], dtype=np.float32)
                truncated += len(w) / self.SAMPLE_RATE >= cap_s - 0.5
                parts.append(w)
                parts.append(np.zeros(int(0.15 * self.SAMPLE_RATE), dtype=np.float32))
        wav = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
        secs = time.perf_counter() - t0
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), wav, self.SAMPLE_RATE)
        return {"path": out_path, "seconds": secs, "audio_seconds": len(wav) / self.SAMPLE_RATE,
                "spoken_text": spoken, "truncated_chunks": int(truncated)}


def split_for_tts(text: str, max_chars: int = 160) -> list[str]:
    """Dividir en oraciones y agruparlas hasta ``max_chars`` (XTTS admite ~239 en español)."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    chunks, cur = [], ""
    for sent in sents:
        if len(sent) > max_chars and cur:  # cerrar lo acumulado para conservar el orden
            chunks.append(cur)
            cur = ""
        while len(sent) > max_chars:  # cortar una oración muy larga en la última coma/espacio
            cut = max(sent.rfind(",", 0, max_chars), sent.rfind(" ", 0, max_chars))
            cut = cut if cut > 0 else max_chars
            chunks.append(sent[:cut].strip(" ,"))
            sent = sent[cut:].strip(" ,")
        if cur and len(cur) + 1 + len(sent) > max_chars:
            chunks.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------------------
# Semejanza de voz (ECAPA-TDNN)
# --------------------------------------------------------------------------------------
class SpeakerSimilarity:
    """Similitud coseno entre embeddings de hablante (ECAPA-TDNN, VoxCeleb).

    Referencia: dos grabaciones de la misma persona suelen dar > 0,6–0,7; de personas
    distintas, < 0,3.
    """

    def __init__(self, device: str = "cpu"):
        from speechbrain.inference.speaker import EncoderClassifier

        if device == "cuda":  # SpeechBrain exige el índice ("cuda:0")
            device = "cuda:0"
        self.enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device},
            savedir=str(Path.home() / ".cache" / "speechbrain" / "ecapa"))

    def embed(self, path: Path) -> np.ndarray:
        import torch

        wav = torch.from_numpy(load_audio_16k(path)).unsqueeze(0)
        with torch.inference_mode():
            e = self.enc.encode_batch(wav).squeeze().cpu().numpy()
        return e / np.linalg.norm(e)

    def similarity(self, a: Path, b: Path) -> float:
        return float(self.embed(a) @ self.embed(b))


# --------------------------------------------------------------------------------------
# Grabación (Mac) y reproducción en notebook
# --------------------------------------------------------------------------------------
def record(path: Path, seconds: float, samplerate: int = 16000) -> Path:
    """Grabar del micrófono (Mac). En Colab se suben archivos .wav ya grabados."""
    import sounddevice as sd
    import soundfile as sf

    print(f"🎙  Grabando {seconds:.0f} s… habla ahora.")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1,
                   dtype="float32")
    sd.wait()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, samplerate)
    print("✔  Grabación terminada:", path)
    return path


# --------------------------------------------------------------------------------------
# Pipeline de voz de punta a punta
# --------------------------------------------------------------------------------------
@dataclass
class VoiceTurn:
    audio_in: Path
    transcript: str
    response: str
    audio_out: Path
    timings: dict[str, float] = field(default_factory=dict)
    agent_turn: Any = None


class VoiceAssistant:
    """Unir ASR → agente → TTS y medir la latencia por etapa."""

    def __init__(self, agent, asr, tts: XTTSVoice, out_dir: Path):
        self.agent, self.asr, self.tts, self.out_dir = agent, asr, tts, Path(out_dir)
        self.n = 0

    def turn(self, audio_path: Path, transcript_override: str | None = None) -> VoiceTurn:
        """Un turno hablado. ``transcript_override`` usa la transcripción correcta en vez del
        ASR (para aislar errores del ASR, como pide el enunciado)."""
        self.n += 1
        t_end_recording = time.perf_counter()  # el audio ya está grabado en este punto
        if transcript_override is None:
            tr = self.asr.transcribe(audio_path)
            text, t_asr = tr.text, tr.seconds
        else:
            text, t_asr = transcript_override, 0.0
        r = self.agent.turn(text)
        syn = self.tts.synthesize(r.response, self.out_dir / f"respuesta_{self.n:02d}.wav")
        total = time.perf_counter() - t_end_recording
        timings = {"asr": t_asr, "llm": sum(r.llm_seconds),
                   "herramientas_y_recuperacion": r.tool_seconds,
                   "agente_otros": max(0.0, r.total_seconds - sum(r.llm_seconds)
                                       - r.tool_seconds),
                   "tts": syn["seconds"], "total": total,
                   "audio_respuesta_s": syn["audio_seconds"]}
        return VoiceTurn(Path(audio_path), text, r.response, syn["path"], timings, r)
