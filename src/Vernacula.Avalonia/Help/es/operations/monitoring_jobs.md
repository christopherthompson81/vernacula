---
title: "Supervisión de trabajos"
description: "Cómo ver el progreso de un trabajo en ejecución o en cola."
topic_id: operations_monitoring_jobs
---

# Supervisión de trabajos

La vista **Progreso** muestra en tiempo real el estado de un trabajo de transcripción en ejecución.

## Abrir la vista Progreso

- Al iniciar una nueva transcripción, la aplicación accede automáticamente a la vista Progreso.
- Para un trabajo que ya esté en ejecución o en cola, encuéntralo en la tabla **Historial de transcripciones** y haz clic en `Monitor` en su columna **Acciones**.

## Leer la vista Progreso

| Elemento | Descripción |
|---|---|
| Barra de progreso | Porcentaje de finalización global. Aparece indeterminada (animada) mientras el trabajo se está iniciando o reanudando. |
| Etiqueta de porcentaje | Porcentaje numérico que se muestra a la derecha de la barra. |
| Mensaje de estado | Actividad actual; por ejemplo, `Audio Analysis` o `Speech Recognition`. Muestra `Waiting in queue…` si el trabajo aún no ha comenzado. |
| Tabla de segmentos | Actualización en tiempo real de los segmentos transcritos, con las columnas **Speaker**, **Start**, **End** y **Content**. Se desplaza automáticamente a medida que llegan nuevos segmentos. |

## Fases del progreso

Las fases que se muestran dependen del **Modo de segmentación** seleccionado en la configuración.

**Modo de diarización de hablantes** (predeterminado):

1. **Audio Analysis** — La diarización SortFormer recorre todo el archivo para identificar los límites entre hablantes. La barra puede permanecer cerca del 0 % hasta que esta fase finalice.
2. **Speech Recognition** — Se transcribe cada segmento por hablante. El porcentaje aumenta de forma progresiva durante esta fase.

**Modo de detección de actividad de voz**:

1. **Detecting speech segments** — Silero VAD analiza el archivo para localizar las regiones con voz. Esta fase es rápida.
2. **Speech Recognition** — Se transcribe cada región de voz detectada.

En ambos modos, la tabla de segmentos en tiempo real se va completando a medida que avanza la transcripción.

## Salir de la vista

Haz clic en `← Back to Home` para volver a la pantalla principal sin interrumpir el trabajo. El trabajo continúa ejecutándose en segundo plano y su estado se actualiza en la tabla **Historial de transcripciones**. Haz clic en `Monitor` en cualquier momento para volver a la vista Progreso.

---