# Asistente de compras por voz con RAG

**UTEC Posgrado · MCD8016 Deep Learning · Laboratorio 4 · Grupo 8**

Prototipo experimental en español para buscar productos, mantener un carrito y confirmar pedidos en una tienda simulada. Combina recuperación léxica y semántica, un agente con herramientas, reconocimiento de voz y síntesis condicionada por una voz de referencia autorizada.

| Para consultar | Documento |
|---|---|
| Vista general y resultados principales en inglés | [README](README.md) |
| Componentes, estado conversacional y validación transaccional | [Arquitectura](docs/ARCHITECTURE.md) |
| Métricas, ablación y fuentes de evidencia | [Experimentos](docs/EXPERIMENTS.md) |
| Flujo en Colab y requisitos de datos | [Reproducibilidad](docs/REPRODUCIBILITY.md) |
| Casos de compra, aclaración y fallos de recuperación | [Demostración](docs/DEMONSTRATION.md) |
| Dependencias y configuración | [Guía de implementación](lab4_solution/README.md) |

En las corridas oficiales de Colab L4, el recuperador completo alcanzó **0,7884 nDCG@10**. Qwen3.5-9B completó **16 de 20** escenarios operativos, con cero violaciones registradas. La evaluación por voz y la repetición con transcripción correcta coincidieron en **3 de 6** escenarios. La latencia media de voz fue de **21,34 s por turno**.

Las copias para publicación conservan el código y muestran las salidas revisadas de Colab en los notebooks 00, 03, 05 y 06. Los audios personales, las trazas y los resultados originales se conservan localmente.

El documento original atribuye el proyecto a **Carlos Irazabal**. La lista completa de integrantes del Grupo 8 queda pendiente de confirmación.
