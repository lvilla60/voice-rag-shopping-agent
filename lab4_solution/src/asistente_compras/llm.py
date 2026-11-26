"""Capa del LLM: la misma interfaz en la Mac (Ollama) y en Colab (transformers, 4-bit).

El agente no sabe dónde corre el modelo. Ambos backends usan **la plantilla nativa de tool
calling de Qwen2.5** (las herramientas van en el prompt del sistema y el modelo responde con
bloques ``<tool_call>{json}</tool_call>``), así el comportamiento es comparable.

Formato interno de mensajes (compatible con ambos)::

    {"role": "system" | "user" | "assistant" | "tool", "content": str,
     "tool_calls": [{"function": {"name": str, "arguments": dict}}]}   # solo assistant

Decodificación greedy (temperatura 0) y semilla fija: el mismo escenario produce la misma
conversación, lo que hace reproducibles los 20 escenarios de evaluación.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMReply:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply: ...


# --------------------------------------------------------------------------------------
# Parsear tool calls en texto (usado por el backend HF y como respaldo en Ollama)
# --------------------------------------------------------------------------------------
_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)  # desactivar el razonamiento de Qwen3
# Formato XML de Qwen3.5 (y Qwen3-Coder):
#   <tool_call><function=nombre><parameter=clave>valor</parameter>…</function></tool_call>
_XML_FUNC_RE = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
_XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.S)


def _xml_value(raw: str) -> Any:
    """"2" → 2, "[1, 2]" → lista, "cables USB" → texto."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_one(body: str) -> dict[str, Any] | None:
    if body.startswith("{"):  # leer el formato JSON (Qwen2.5 / Qwen3)
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return None
        args = obj.get("arguments", {})
        if isinstance(args, str):  # decodificar los argumentos cuando llegan como string JSON
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return {"function": {"name": obj.get("name", ""), "arguments": args}}
    m = _XML_FUNC_RE.search(body)  # leer el formato XML (Qwen3.5)
    if not m:
        return None
    args = {k: _xml_value(v) for k, v in _XML_PARAM_RE.findall(m.group(2))}
    return {"function": {"name": m.group(1), "arguments": args}}


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extraer los bloques ``<tool_call>`` (JSON o XML) y devolver (texto restante, llamadas)."""
    text = _THINK_RE.sub("", text)
    calls = [c for body in _TOOL_RE.findall(text) if (c := _parse_one(body.strip()))]
    return _TOOL_RE.sub("", text).strip(), calls


# --------------------------------------------------------------------------------------
# Mac: Ollama (GGUF Q4_K_M)
# --------------------------------------------------------------------------------------
class OllamaLLM:
    def __init__(self, model: str = "qwen2.5:7b", seed: int = 42, num_ctx: int = 8192,
                 max_tokens: int = 400, think: bool | None = None):
        import ollama

        self.client = ollama.Client()
        self.name = f"ollama:{model}"
        self.model = model
        self.options = {"temperature": 0, "seed": seed, "num_ctx": num_ctx,
                        "num_predict": max_tokens}
        # Modelos con "razonamiento" (p. ej. Qwen3.x): pasar False para desactivarlo y bajar
        # la latencia.
        self.extra = {"think": think} if think is not None else {}

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        t0 = time.perf_counter()
        r = self.client.chat(model=self.model, messages=messages, tools=tools,
                             options=self.options, keep_alive="30m", **self.extra)
        content = r.message.content or ""
        calls = [{"function": {"name": c.function.name, "arguments": dict(c.function.arguments)}}
                 for c in (r.message.tool_calls or [])]
        if not calls:  # respaldo: recuperar el bloque cuando queda en el texto sin parsear
            content, calls = parse_tool_calls(content)
        return LLMReply(content=content.strip(), tool_calls=calls,
                        seconds=time.perf_counter() - t0,
                        prompt_tokens=r.prompt_eval_count or 0,
                        completion_tokens=r.eval_count or 0)


# --------------------------------------------------------------------------------------
# Colab: transformers + bitsandbytes 4-bit (NF4)
# --------------------------------------------------------------------------------------
class HFLLM:
    """Qwen2.5-7B-Instruct en 4-bit NF4 (~5,5 GB de VRAM en una T4).

    Se usa ``apply_chat_template(..., tools=...)``: es la misma plantilla que usa Ollama para
    Qwen2.5, así el modelo ve el mismo formato de herramientas en ambos entornos.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct", load_in_4bit: bool = True,
                 max_tokens: int = 400, seed: int = 42,
                 chat_template_kwargs: dict | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        self.name = f"hf:{model_id}{'-4bit' if load_in_4bit else ''}"
        self.max_tokens = max_tokens
        # Pasar p. ej. {"enable_thinking": False} para Qwen3: misma conducta que think=False
        # en Ollama.
        self.template_kwargs = chat_template_kwargs or {}
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            # Usar bfloat16 si la GPU lo soporta (L4, A100): es el formato nativo de Qwen; float16
            # puede desbordarse y degradar las respuestas. La T4 solo admite float16.
            compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute, bnb_4bit_use_double_quant=True,
            )
            # dejar las capas no cuantizadas (embeddings, normas) en el mismo tipo
            kwargs["dtype"] = compute
        else:
            kwargs["torch_dtype"] = "auto"
        self.model = _load_model(model_id, AutoModelForCausalLM, kwargs).eval()
        # Declarar los tokens de fin de turno. Qwen3.5-9B no publica generation_config.json y su
        # config solo declara <|endoftext|>: sin <|im_end|> el modelo no se detiene, escribe turnos
        # falsos del cliente y agota max_tokens (lento y con respuestas contaminadas).
        stop = {self.tok.eos_token_id, *(self.tok.convert_tokens_to_ids(t)
                                         for t in ("<|im_end|>", "<|endoftext|>"))}
        self.stop_ids = sorted(i for i in stop if isinstance(i, int) and i >= 0
                               and i != self.tok.unk_token_id)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None \
            else self.stop_ids[0]

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        import torch

        t0 = time.perf_counter()
        ids = self.tok.apply_chat_template(late_system_as_user(messages), tools=tools,
                                           add_generation_prompt=True,
                                           return_tensors="pt", **self.template_kwargs)
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=self.max_tokens, do_sample=False,
                                      eos_token_id=self.stop_ids, pad_token_id=self.pad_id)
        new = out[0, ids.shape[1]:]
        text = cut_at_turn_end(self.tok.decode(new, skip_special_tokens=False))
        content, calls = parse_tool_calls(text)
        return LLMReply(content=content, tool_calls=calls, seconds=time.perf_counter() - t0,
                        prompt_tokens=int(ids.shape[1]), completion_tokens=int(new.shape[0]))


def cut_at_turn_end(text: str) -> str:
    """Quedarse con el turno del asistente: cortar en el primer fin de turno y quitar especiales."""
    for marker in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.split(marker, 1)[0]
    return text


def late_system_as_user(messages: list[dict]) -> list[dict]:
    """Algunas plantillas (Qwen3.5) solo aceptan el mensaje de sistema al inicio.

    Las correcciones del control de respuestas llegan como mensajes de sistema a mitad de la
    conversación; aquí se entregan como una nota en un mensaje de usuario. Ollama (Mac) acepta
    el formato original, por eso este ajuste solo se aplica en el backend de transformers.
    """
    return [m if m.get("role") != "system" or i == 0 else
            {"role": "user", "content": f"[Nota del sistema, no del cliente] {m['content']}"}
            for i, m in enumerate(messages)]


def _load_model(model_id: str, causal_cls, kwargs: dict[str, Any]):
    """Cargar el modelo de texto; si la arquitectura es multimodal, usar la clase imagen-texto.

    Qwen3.5 se publica como ``Qwen3_5ForConditionalGeneration`` (texto + imagen). Según la
    versión de transformers, ``AutoModelForCausalLM`` lo acepta o no; en ese caso se carga con
    ``AutoModelForImageTextToText`` y se usa solo con texto (la parte visual queda sin usar).
    """
    try:
        return causal_cls.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError, TypeError) as err:
        from transformers import AutoModelForImageTextToText

        print(f"[llm] {model_id}: AutoModelForCausalLM no aplica ({type(err).__name__}); "
              "se carga como modelo imagen-texto.")
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


# --------------------------------------------------------------------------------------
# Fábrica: elegir el backend según el entorno
# --------------------------------------------------------------------------------------
# Usar el mismo modelo en ambos entornos: clave de Ollama (Mac) → id de Hugging Face (Colab).
MODELS: dict[str, dict[str, Any]] = {
    "qwen2.5:7b": {"hf": "Qwen/Qwen2.5-7B-Instruct", "think": None},
    "qwen3:8b": {"hf": "Qwen/Qwen3-8B", "think": False},
    "qwen3.5:9b": {"hf": "Qwen/Qwen3.5-9B", "think": False},  # multimodal: ver _load_model
}


def make_llm(key: str = "qwen2.5:7b", backend: str | None = None, **kwargs) -> LLM:
    """``backend``: "ollama" (Mac), "hf" (Colab) o None = automático (CUDA ⇒ hf)."""
    spec = MODELS[key]
    if backend is None:
        try:
            import torch

            backend = "hf" if torch.cuda.is_available() else "ollama"
        except ImportError:
            backend = "ollama"
    if backend == "ollama":
        return OllamaLLM(key, think=spec["think"], **kwargs)
    tmpl = {"enable_thinking": False} if spec["think"] is False else None
    return HFLLM(spec["hf"], chat_template_kwargs=tmpl, **kwargs)


def free_gpu() -> None:
    """Liberar la memoria de la GPU después de ``del llm`` (antes de cargar otro modelo)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
