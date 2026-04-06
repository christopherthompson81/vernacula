---
title: "Descarga de Modelos"
description: "Cómo descargar los archivos de modelo de IA necesarios para la transcripción."
topic_id: first_steps_downloading_models
---

# Descarga de Modelos

Vernacula-Desktop requiere archivos de modelo de IA para funcionar. Estos no se incluyen con la aplicación y deben descargarse antes de realizar la primera transcripción.

## Estado de los Modelos (Pantalla de Inicio)

Una barra de estado delgada en la parte superior de la pantalla de inicio indica si los modelos están listos. Cuando faltan archivos, también muestra un botón `Open Settings` que le lleva directamente a la administración de modelos.

| Estado | Significado |
|---|---|
| `All N model file(s) present ✓` | Todos los archivos necesarios están descargados y listos. |
| `N model file(s) missing: …` | Uno o más archivos están ausentes; abra Configuración para descargarlos. |

Cuando los modelos están listos, los botones `New Transcription` y `Bulk Add Jobs` se activan.

## Cómo Descargar los Modelos

1. En la pantalla de inicio, haga clic en `Open Settings` (o vaya a `Settings… > Models`).
2. En la sección **Models**, haga clic en `Download Missing Models`.
3. Aparecen una barra de progreso y una línea de estado que muestran el archivo actual, su posición en la cola y el tamaño de la descarga; por ejemplo: `[1/3] encoder-model.onnx — 42 MB`.
4. Espere a que el estado muestre `Download complete.`

## Cancelar una Descarga

Para detener una descarga en curso, haga clic en `Cancel`. La línea de estado mostrará `Download cancelled.` Los archivos descargados parcialmente se conservan, por lo que la descarga se reanuda desde donde se dejó la próxima vez que haga clic en `Download Missing Models`.

## Errores de Descarga

Si una descarga falla, la línea de estado muestra `Download failed: <reason>`. Compruebe su conexión a Internet y haga clic en `Download Missing Models` de nuevo para volver a intentarlo. La aplicación reanuda la descarga desde el último archivo completado correctamente.
