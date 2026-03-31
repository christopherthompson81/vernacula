---
title: "Flujo de Trabajo de Nueva Transcripción"
description: "Guía paso a paso para transcribir un archivo de audio."
topic_id: operations_new_transcription
---

# Flujo de Trabajo de Nueva Transcripción

Utilice este flujo de trabajo para transcribir un único archivo de audio.

## Requisitos Previos

- Todos los archivos de modelos deben estar descargados. La tarjeta **Estado del Modelo** debe mostrar `All N model file(s) present ✓`. Consulte [Descarga de Modelos](../first_steps/downloading_models.md).

## Formatos Compatibles

### Audio

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Vídeo

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Los archivos de vídeo se decodifican a través de FFmpeg. Si un archivo de vídeo contiene **múltiples pistas de audio** (por ejemplo, varios idiomas o pistas de comentarios), se crea automáticamente un trabajo de transcripción por cada pista.

## Pasos

### 1. Abrir el formulario de Nueva Transcripción

Haga clic en `New Transcription` en la pantalla de inicio, o vaya a `File > New Transcription`.

### 2. Seleccionar un archivo multimedia

Haga clic en `Browse…` junto al campo **Audio File**. Se abre un selector de archivos filtrado por los formatos de audio y vídeo compatibles. Seleccione su archivo y haga clic en **Open**. La ruta del archivo aparece en el campo.

### 3. Asignar un nombre al trabajo

El campo **Job Name** se rellena automáticamente con el nombre del archivo. Edítelo si desea una etiqueta diferente — este nombre aparece en el Historial de Transcripciones en la pantalla de inicio.

### 4. Iniciar la transcripción

Haga clic en `Start Transcription`. La aplicación cambia a la vista de **Progreso**.

Para volver atrás sin iniciar el proceso, haga clic en `← Back`.

## Qué Ocurre a Continuación

El trabajo pasa por dos fases que se muestran en la barra de progreso:

1. **Análisis de Audio** — diarización de hablantes: identificación de quién habla y en qué momento.
2. **Reconocimiento de Voz** — conversión de voz a texto segmento por segmento.

Los segmentos transcritos aparecen en la tabla en tiempo real a medida que se generan. Cuando el procesamiento finaliza, la aplicación pasa automáticamente a la vista de **Resultados**.

Si agrega un trabajo mientras ya hay otro en ejecución, el nuevo trabajo mostrará el estado `queued` y comenzará cuando el trabajo actual termine. Consulte [Supervisión de Trabajos](monitoring_jobs.md).

---