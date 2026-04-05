---
title: "Edición de transcripciones"
description: "Cómo revisar, corregir y verificar segmentos transcritos en el editor de transcripciones."
topic_id: operations_editing_transcripts
---

# Edición de transcripciones

El **Editor de transcripciones** le permite revisar la salida de ASR, corregir texto, renombrar hablantes de forma directa, ajustar los tiempos de los segmentos y marcarlos como verificados, todo mientras escucha el audio original.

## Abrir el editor

1. Cargue un trabajo completado (consulte [Cargar trabajos completados](loading_completed_jobs.md)).
2. En la vista **Resultados**, haga clic en `Edit Transcript`.

El editor se abre como una ventana independiente y puede permanecer abierto junto a la aplicación principal.

## Diseño

Cada segmento se muestra como una tarjeta con dos paneles en paralelo:

- **Panel izquierdo** — la salida original de ASR con coloración de confianza por palabra. Las palabras sobre las que el modelo tenía menos certeza aparecen en rojo; las palabras de alta confianza aparecen con el color de texto normal.
- **Panel derecho** — un cuadro de texto editable. Realice las correcciones aquí; las diferencias respecto al original se resaltan mientras escribe.

La etiqueta del hablante y el intervalo de tiempo aparecen sobre cada tarjeta. Haga clic en una tarjeta para enfocarla y mostrar sus iconos de acción. Pase el cursor sobre cualquier icono para ver una información emergente que describe su función.

## Leyenda de iconos

### Barra de reproducción

| Icono | Acción |
|-------|--------|
| ▶ | Reproducir |
| ⏸ | Pausar |
| ⏮ | Ir al segmento anterior |
| ⏭ | Ir al segmento siguiente |

### Acciones de la tarjeta de segmento

| Icono | Acción |
|-------|--------|
| <mdl2 ch="E77B"/> | Reasignar el segmento a un hablante diferente |
| <mdl2 ch="E916"/> | Ajustar los tiempos de inicio y fin del segmento |
| <mdl2 ch="EA39"/> | Suprimir o anular la supresión del segmento |
| <mdl2 ch="E72B"/> | Combinar con el segmento anterior |
| <mdl2 ch="E72A"/> | Combinar con el segmento siguiente |
| <mdl2 ch="E8C6"/> | Dividir el segmento |
| <mdl2 ch="E72C"/> | Volver a ejecutar ASR en este segmento |

## Reproducción de audio

Una barra de reproducción se extiende en la parte superior de la ventana del editor:

| Control | Acción |
|---------|--------|
| Icono de reproducción / pausa | Iniciar o pausar la reproducción |
| Barra de búsqueda | Arrastre para saltar a cualquier posición en el audio |
| Control deslizante de velocidad | Ajustar la velocidad de reproducción (0,5× – 2×) |
| Iconos anterior / siguiente | Saltar al segmento anterior o siguiente |
| Menú desplegable de modo de reproducción | Seleccionar uno de los tres modos de reproducción (ver más abajo) |
| Control deslizante de volumen | Ajustar el volumen de reproducción |

Durante la reproducción, la palabra que se está pronunciando en ese momento aparece resaltada en el panel izquierdo. Al pausar después de una búsqueda, el resaltado se actualiza a la palabra en la posición de búsqueda.

### Modos de reproducción

| Modo | Comportamiento |
|------|----------------|
| `Single` | Reproduce el segmento actual una vez y se detiene. |
| `Auto-advance` | Reproduce el segmento actual; al finalizar, lo marca como verificado y avanza al siguiente. |
| `Continuous` | Reproduce todos los segmentos en secuencia sin marcar ninguno como verificado. |

Seleccione el modo activo en el menú desplegable de la barra de reproducción.

## Editar un segmento

1. Haga clic en una tarjeta para enfocarla.
2. Edite el texto en el panel derecho. Los cambios se guardan automáticamente cuando mueve el foco a otra tarjeta.

## Renombrar un hablante

Haga clic en la etiqueta del hablante dentro de la tarjeta enfocada y escriba un nuevo nombre. Presione `Enter` o haga clic fuera para guardar. El nuevo nombre se aplica únicamente a esa tarjeta; para renombrar un hablante de forma global, use [Editar nombres de hablantes](editing_speaker_names.md) desde la vista Resultados.

## Verificar un segmento

Haga clic en la casilla `Verified` de una tarjeta enfocada para marcarla como revisada. El estado de verificación se guarda en la base de datos y es visible en el editor en cargas futuras.

## Suprimir un segmento

Haga clic en `Suppress` en una tarjeta enfocada para ocultar el segmento de las exportaciones (útil para ruido, música u otras secciones que no corresponden a voz). Haga clic en `Unsuppress` para restaurarlo.

## Ajustar los tiempos del segmento

Haga clic en `Adjust Times` en una tarjeta enfocada para abrir el cuadro de diálogo de ajuste de tiempos. Use la rueda del ratón sobre el campo **Start** o **End** para modificar el valor en incrementos de 0,1 segundos, o escriba un valor directamente. Haga clic en `Save` para aplicar.

## Combinar segmentos

- Haga clic en `⟵ Merge` para combinar el segmento enfocado con el segmento inmediatamente anterior.
- Haga clic en `Merge ⟶` para combinar el segmento enfocado con el segmento inmediatamente siguiente.

El texto combinado y el intervalo de tiempo de ambas tarjetas se unen. Esto es útil cuando una única expresión hablada quedó dividida en dos segmentos.

## Dividir un segmento

Haga clic en `Split…` en una tarjeta enfocada para abrir el cuadro de diálogo de división. Coloque el punto de división dentro del texto y confirme. Se crean dos nuevos segmentos que cubren el intervalo de tiempo original. Esto es útil cuando dos expresiones distintas quedaron unidas en un solo segmento.

## Volver a ejecutar ASR

Haga clic en `Redo ASR` en una tarjeta enfocada para volver a ejecutar el reconocimiento de voz en el audio de ese segmento. El modelo procesa únicamente el fragmento de audio correspondiente a ese segmento y genera una transcripción nueva de fuente única.

Use esta función cuando:

- Un segmento proviene de una combinación y no puede dividirse (los segmentos combinados abarcan múltiples fuentes de ASR; Volver a ejecutar ASR los consolida en uno, tras lo cual `Split…` queda disponible).
- La transcripción original es deficiente y desea obtener un segundo resultado limpio sin editar manualmente.

**Nota:** El texto que ya haya escrito en el panel derecho se descarta y se reemplaza con la nueva salida de ASR. La operación requiere que el archivo de audio esté cargado; el botón se deshabilita si el audio no está disponible.