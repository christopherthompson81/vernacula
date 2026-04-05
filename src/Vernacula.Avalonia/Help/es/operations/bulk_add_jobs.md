---
title: "Añadir Varios Archivos de Audio a la Cola"
description: "Cómo agregar varios archivos de audio a la cola de trabajos a la vez."
topic_id: operations_bulk_add_jobs
---

# Añadir Varios Archivos de Audio a la Cola

Usa **Bulk Add Jobs** para poner en cola varios archivos de audio o vídeo para transcripción en un solo paso. La aplicación los procesa de uno en uno en el orden en que fueron añadidos.

## Requisitos Previos

- Todos los archivos de modelo deben estar descargados. La tarjeta **Model Status** debe mostrar `All N model file(s) present ✓`. Consulta [Descargar Modelos](../first_steps/downloading_models.md).

## Cómo Añadir Trabajos en Lote

1. En la pantalla de inicio, haz clic en `Bulk Add Jobs`.
2. Se abre un selector de archivos. Selecciona uno o más archivos de audio o vídeo — mantén pulsado `Ctrl` o `Shift` para seleccionar varios archivos.
3. Haz clic en **Open**. Cada archivo seleccionado se añade a la tabla **Transcription History** como un trabajo independiente.

> **Archivos de vídeo con múltiples pistas de audio:** Si un archivo de vídeo contiene más de una pista de audio (por ejemplo, varios idiomas o una pista de comentarios del director), la aplicación crea automáticamente un trabajo por cada pista.

## Nombres de los Trabajos

Cada trabajo recibe automáticamente el nombre de su archivo de audio. Puedes renombrar un trabajo en cualquier momento haciendo clic en su nombre en la columna **Title** de la tabla Transcription History, editando el texto y pulsando `Enter` o haciendo clic fuera del campo.

## Comportamiento de la Cola

- Si no hay ningún trabajo en ejecución en ese momento, el primer archivo comienza inmediatamente y el resto se muestran como `queued`.
- Si ya hay un trabajo en ejecución, todos los archivos recién añadidos se muestran como `queued` y comenzarán automáticamente en secuencia.
- Para supervisar el trabajo activo, haz clic en `Monitor` en su columna **Actions**. Consulta [Supervisar Trabajos](monitoring_jobs.md).
- Para pausar o eliminar un trabajo en cola antes de que comience, usa los botones `Pause` o `Remove` en su columna **Actions**. Consulta [Pausar, Reanudar o Eliminar Trabajos](pausing_resuming_removing.md).

---