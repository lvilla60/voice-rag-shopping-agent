"""Asistente de compras por voz en español (Lab 4, MCD8016 Deep Learning, UTEC).

Módulos:

- ``config``   rutas, semillas y parámetros.
- ``data``     ESCI → catálogo, consultas y etiquetas (qrels).
- ``metrics``  nDCG@k con las ganancias del enunciado.

Próximos pasos del plan: ``store`` (tienda SQLite), ``retrieval`` (BM25, denso, re-ranker),
``agent`` (estado, herramientas, validaciones), ``voice`` (ASR y TTS) y ``evaluation``.
"""

import os as _os

# Colab trae JAX instalado y `bm25s` lo usa al importarse (top-k). JAX reserva de entrada el 75 %
# de la memoria de la GPU (~11 GB en una T4) y deja sin espacio al LLM, al re-ranker y a la voz.
# Aquí JAX solo haría un top-k sobre un vector: forzarlo a la CPU y sin reserva previa.
# Ejecutar esto antes de importar bm25s (por eso está en el __init__ del paquete).
_os.environ.setdefault("JAX_PLATFORMS", "cpu")
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Reducir la fragmentación de la memoria de GPU cuando conviven LLM, ASR, TTS y recuperador.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"
