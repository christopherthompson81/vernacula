---
title: "Pausar, reanudar o eliminar trabajos"
description: "Cómo pausar un trabajo en ejecución, reanudar uno detenido o eliminar un trabajo del historial."
topic_id: operations_pausing_resuming_removing
---

# Pausar, reanudar o eliminar trabajos

## Pausar un trabajo

Puede pausar un trabajo en ejecución o en cola desde dos lugares:

- **Vista de progreso** — haga clic en `Pause` en la esquina inferior derecha mientras observa el trabajo activo.
- **Tabla de historial de transcripciones** — haga clic en `Pause` en la columna **Actions** de cualquier fila cuyo estado sea `running` o `queued`.

Después de hacer clic en `Pause`, la línea de estado muestra `Pausing…` mientras la aplicación finaliza la unidad de procesamiento actual. A continuación, el estado del trabajo cambia a `cancelled` en la tabla del historial.

> Al pausar se guardan todos los segmentos transcritos hasta ese momento. Puede reanudar el trabajo más adelante sin perder ese progreso.

## Reanudar un trabajo

Para reanudar un trabajo pausado o fallido:

1. En la pantalla de inicio, busque el trabajo en la tabla **Transcription History**. Su estado será `cancelled` o `failed`.
2. Haga clic en `Resume` en la columna **Actions**.
3. La aplicación regresa a la vista **Progress** y continúa desde donde se detuvo el procesamiento.

La línea de estado muestra `Resuming…` brevemente mientras el trabajo se reinicializa.

## Eliminar un trabajo

Para eliminar de forma permanente un trabajo y su transcripción del historial:

1. En la tabla **Transcription History**, haga clic en `Remove` en la columna **Actions** del trabajo que desea eliminar.

El trabajo se elimina de la lista y sus datos se borran de la base de datos local. Esta acción no se puede deshacer. Los archivos exportados guardados en el disco no se ven afectados.

---