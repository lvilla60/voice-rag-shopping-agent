# Implementación del asistente de compras por voz

**UTEC Posgrado · MCD8016 Deep Learning · Laboratorio 4 · Grupo 8**

El paquete `asistente_compras` integra preparación de ESCI en español, tienda SQLite simulada, recuperación híbrida, un agente conversacional con herramientas y un flujo ASR–TTS. Los modelos se ejecutan en el entorno del laboratorio. La instalación puede descargar dependencias y checkpoints.

[Descripción del proyecto](../README.md) · [Arquitectura](../docs/ARCHITECTURE.md) · [Resultados](../docs/EXPERIMENTS.md) · [Guía de reproducción](../docs/REPRODUCIBILITY.md)

## Google Colab

1. Crear una carpeta escribible `MyDrive/lab4/` con `data/` y `lab4_solution/` al mismo nivel. Las celdas fijan `/content/drive/MyDrive/lab4` y no dependen del nombre del directorio de descarga.
2. Colocar la implementación en `lab4_solution/` junto con el `pyproject.toml` incluido para la instalación editable.
3. Abrir la copia correspondiente en `notebooks_public/`. Con un conjunto nuevo, empezar en 01 para preparar el catálogo. El 00 es una prueba de integración opcional con catálogo preparado y audios de V1. Seguir con 02 para la tienda, 03 para recuperación, 04 para el agente, 05 para comparar modelos y 06 para voz.
4. Para reproducir las condiciones de GPU, seleccionar NVIDIA L4 y usar un kernel limpio. Las primeras celdas montan Drive, instalan los extras correspondientes y fijan las rutas. Los notebooks pueden escribir sobre resultados generados en tu propia copia.

Las copias para publicación muestran las salidas revisadas de Colab en los notebooks 00, 03, 05 y 06. Los notebooks originales 00, 03, 05 y 06 conservan las salidas oficiales de Colab en la copia local. Los registros de desarrollo de Mac se conservan por separado.

## Dependencias por etapa

| Extra | Función |
|---|---|
| Núcleo | Preparación de datos, Parquet, tienda y métricas. |
| `rag` | `bm25s`, stemming en español, Sentence Transformers y evaluación estadística. |
| `llm` | Transformers, Accelerate, bitsandbytes para CUDA en Linux y kernels de atención lineal en Linux. |
| `voz` | faster-whisper, Coqui TTS, SpeechBrain, audio y normalización de texto. |
| `mac` | Ollama, MLX Whisper y captura de audio en Apple Silicon. |

El proyecto declara Python 3.11 o posterior. `uv.lock` registra las resoluciones de uv. Colab instala con pip desde el manifiesto y puede resolver otras versiones.

## Desarrollo local en Mac

Desde la raíz del proyecto, configurar `LAB4_DATA_DIR` antes de entrar en `lab4_solution/`. El valor por defecto del paquete apunta a `lab4_solution/data/`.

~~~bash
export LAB4_DATA_DIR="$(pwd)/data"
cd lab4_solution
uv sync --extra rag --extra mac --extra llm --extra voz
~~~

Iniciar `ollama serve` en otra terminal para las tareas que usan el LLM local. Los comandos conservados por el proyecto son:

~~~bash
ollama pull qwen3.5:9b
uv run pytest
uv run lab4-preparar-datos
uv run lab4-conversar
uv run lab4-grabar
~~~

`lab4-preparar-datos` implementa la preparación sin el análisis exploratorio del notebook 01. El comando de conversación usa una voz de referencia autorizada y el de grabación captura los turnos de los seis escenarios. La ejecución de voz requiere los recursos indicados en la [guía de datos](../docs/DATA_AVAILABILITY.md).

Si el entorno editable falla en una carpeta sincronizada por iCloud, el problema histórico se registró con archivos `.pth`. Mantén el entorno virtual en una ubicación que permita a Python leerlos.

## Módulos

- `data.py` y `config.py` preparan el catálogo, los splits y la configuración. Las etiquetas se conservan separadas para `evaluation.py`.
- `retrieval.py` implementa BM25, E5, fusión RRF y re-ranking.
- `llm.py` selecciona Ollama o Transformers. `agent.py` controla estado, herramientas y validaciones de respuestas.
- `store.py` valida precios, cantidades, presupuesto, stock y confirmaciones en SQLite.
- `scenarios.py` define los 20 casos reproducibles. `voice.py` y `cli.py` cubren la transcripción, síntesis, evaluación y grabación.

Los scripts de Bash en `scripts/` preparan un ZIP público o la carpeta de Colab con `data/` y `lab4_solution/` al mismo nivel. Ambos admiten `--dry-run`. La [guía de reproducción](../docs/REPRODUCIBILITY.md) describe los destinos y las opciones para datos y audio privados.
