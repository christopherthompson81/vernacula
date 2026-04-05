---
title: "Cargar trabajos completados"
description: "Cómo abrir los resultados de una transcripción completada anteriormente."
topic_id: operations_loading_completed_jobs
---

# Cargar trabajos completados

Todos los trabajos de transcripción completados se guardan en la base de datos local y permanecen accesibles en la tabla **Historial de transcripciones** de la pantalla de inicio.

## Cómo cargar un trabajo completado

1. En la pantalla de inicio, localiza el trabajo en la tabla **Historial de transcripciones**. Los trabajos completados muestran un indicador de estado `complete`.
2. Haz clic en `Load` en la columna **Acciones** del trabajo.
3. La aplicación cambia a la vista **Resultados**, donde se muestran todos los segmentos transcritos de ese trabajo.

## Vista de resultados

La vista de resultados muestra:

- El nombre del archivo de audio como encabezado de la página.
- Un subtítulo con el recuento de segmentos (por ejemplo, `42 segment(s)`).
- Una tabla de segmentos con las columnas **Speaker**, **Start**, **End** y **Content**.

Desde la vista de resultados puedes:

- [Editar la transcripción](editing_transcripts.md) — revisar y corregir el texto, ajustar los tiempos, combinar o dividir segmentos, y verificarlos mientras escuchas el audio.
- [Editar los nombres de los hablantes](editing_speaker_names.md) — reemplazar los identificadores genéricos como `speaker_0` por nombres reales.
- [Exportar la transcripción](exporting_results.md) — guardar la transcripción en Excel, CSV, JSON, SRT, Markdown, Word o SQLite.

Para volver a la lista del historial, haz clic en `← Back to History`.

---