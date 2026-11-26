# Reproducing the workflow

## Colab workspace

The original setup cells expect the following Google Drive layout. The proposed publication set does not include the data directory. Make the layout in your own writable Drive workspace.

~~~text
MyDrive/lab4/
├── data/
└── lab4_solution/
    ├── README.md
    ├── pyproject.toml
    ├── src/asistente_compras/
    └── notebooks_public/
~~~

All seven notebooks mount Drive at `/content/drive` when running in Colab, install the project in editable mode with stage-specific extras, and set `LAB4_DATA_DIR=/content/drive/MyDrive/lab4/data`. The `lab4` directory name and sibling layout are required by the original cells. You can open the copies from `lab4_solution/notebooks_public/`. They can also be placed in `lab4_solution/notebooks/` in your own reproduction workspace if you use the original folder convention.

The setup cells use the included `lab4_solution/pyproject.toml` for editable installation. The checked-in lockfile records versions for the uv workflow. The notebook installation uses pip and can resolve different versions on a fresh runtime.

The reported GPU runs used an NVIDIA L4. Notebook 00 includes an older T4 suggestion, and the original process documents later observed memory problems on T4 with the combined models. Use an L4 to match the reported environment. Use a fresh kernel when changing extras or when a runtime restart is requested. When models and packages are fetched, your runtime needs network access and enough temporary storage.

## Notebook order

| Notebook | Role | Inputs and outputs |
|---|---|---|
| [01 · Data preparation](../lab4_solution/notebooks_public/01_preparacion_datos.ipynb) | First step for a fresh data directory | Acquires upstream ESCI Parquet, selects the Spanish subset and writes the prepared catalog, queries, qrels and manifest. Skip this only if an authorized prepared dataset is already available. |
| [00 · Integration check](../lab4_solution/notebooks_public/00_prueba_colab.ipynb) | Optional, after data preparation | Checks the GPU, retrieval and LLM backends, ASR and TTS. Its audio smoke test needs a reference voice and the V1 user recordings. |
| [02 · Simulated shop](../lab4_solution/notebooks_public/02_tienda_simulada.ipynb) | Validate and restore shop state | Uses the prepared catalog to generate deterministic prices and stock and exercises quotation and order validation. |
| [03 · Retrieval](../lab4_solution/notebooks_public/03_recuperacion.ipynb) | Tune on development and evaluate once on test | Uses the catalog, queries and qrels. Generates or reuses E5 embeddings and writes benchmark results. |
| [04 · Agent](../lab4_solution/notebooks_public/04_agente.ipynb) | Demonstrate the tool loop and guards | Uses prepared catalog and the retrieval and LLM extras. Exercises state, tools and validation. |
| [05 · Operational scenarios](../lab4_solution/notebooks_public/05_escenarios.ipynb) | Compare three LLMs on the 20 text-input scenarios | Restores the store per case and writes scenario summaries and local traces to model-specific output directories. |
| [06 · Voice](../lab4_solution/notebooks_public/06_voz.ipynb) | Evaluate Qwen3.5-9B with audio and correct transcriptions | Requires a reference voice and the 25 user-turn recordings. Produces local audio, traces and timing and quality tables. |

Run scenarios in a disposable copy of your own generated dataset and results when you need to retain earlier evidence. The notebooks write to fixed output paths and can overwrite prior results there. The clean copies have valid notebook structure and matching executable-cell sources. This preparation did not execute them.

## Local development and path assumptions

Outside Colab, notebook setup uses the parent of the current working directory as the project directory. Start the Jupyter kernel with its working directory inside `lab4_solution/notebooks_public/` or `lab4_solution/notebooks/`. Set `LAB4_DATA_DIR` to the absolute path of the sibling `data/` directory before starting the kernel. The default calculated by `config.py` is `lab4_solution/data/`.

The local implementation README documents the existing uv, test and Apple Silicon commands. Its default dependency workflow requires the original dependency manifest. The packaging scripts remain local. In particular, `lab4_solution/scripts/empaquetar_drive.sh` reads from `lab4_solution/data/` because it sets `D="$SRC/data"`, while the delivered project has `data/` beside `lab4_solution/`. Assemble the Drive layout directly. The sharing script also expects separate institutional and run artifacts and copies materials that are excluded from publication.

## Voice inputs and repeatability

For voice reproduction, supply your own authorized reference WAV at `data/voz/referencia.wav`. Record the scenario prompts under `data/voz/escenarios/V*/turno_NN.wav`, using the scenario definitions from `scenarios.py`. The official voice run used one speaker and 25 turns. New recordings, runtime versions and model outputs can change recognition, synthesis, latency and scenario outcomes.

The public materials describe how to reconstruct the system and the experimental protocol. Repeating the original voice measurement exactly would require the original recordings and runtime context. [Data availability](DATA_AVAILABILITY.md)
