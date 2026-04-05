---
title: "Configuración"
description: "Descripción general de todas las opciones de la ventana Configuración."
topic_id: first_steps_settings_window
---

# Configuración

La ventana **Configuración** le permite controlar la configuración del hardware, la gestión de modelos, el modo de segmentación, el comportamiento del editor, la apariencia y el idioma. Ábrala desde la barra de menú: `Settings…`.

## Hardware y rendimiento

Esta sección muestra el estado de su GPU NVIDIA y la pila de software CUDA, e informa el límite dinámico de lotes utilizado durante la transcripción por GPU.

| Elemento | Descripción |
|---|---|
| Nombre de la GPU y VRAM | GPU NVIDIA detectada y memoria de vídeo disponible. |
| CUDA Toolkit | Indica si las bibliotecas de tiempo de ejecución de CUDA se encontraron a través de `CUDA_PATH`. |
| cuDNN | Indica si los archivos DLL de tiempo de ejecución de cuDNN están disponibles. |
| Aceleración CUDA | Indica si ONNX Runtime cargó correctamente el proveedor de ejecución de CUDA. |

Haga clic en `Re-check` para volver a ejecutar la detección de hardware sin reiniciar la aplicación; resulta útil después de instalar CUDA o cuDNN.

Cuando no se detectan esos componentes, se muestran enlaces de descarga directa para el CUDA Toolkit y cuDNN.

El mensaje de **límite de lotes** informa cuántos segundos de audio se procesan en cada ejecución de GPU. Este valor se deriva de la VRAM libre tras cargar los modelos y se ajusta automáticamente.

Para obtener instrucciones completas sobre la configuración de CUDA, consulte [Instalar CUDA y cuDNN](cuda_installation.md).

## Modelos

Esta sección gestiona los archivos de modelos de IA necesarios para la transcripción.

- **Precisión del modelo** — elija `INT8 (smaller download)` o `FP32 (more accurate)`. Consulte [Seleccionar la precisión de los pesos del modelo](model_precision.md).
- **Descargar modelos faltantes** — descarga los archivos de modelos que aún no están presentes en el disco. Una barra de progreso y una línea de estado muestran el avance de cada archivo durante la descarga.
- **Buscar actualizaciones** — comprueba si hay pesos de modelo más recientes disponibles. Un banner de actualización también aparece automáticamente en la pantalla de inicio cuando se detectan pesos actualizados.

## Modo de segmentación

Controla cómo se divide el audio en segmentos antes del reconocimiento de voz.

| Modo | Descripción |
|---|---|
| **Diarización de hablantes** | Utiliza el modelo SortFormer para identificar hablantes individuales y etiquetar cada segmento. Ideal para entrevistas, reuniones y grabaciones con múltiples hablantes. |
| **Detección de actividad de voz** | Utiliza Silero VAD para detectar únicamente las regiones de voz, sin etiquetas de hablante. Más rápido que la diarización y adecuado para audio con un solo hablante. |

## Editor de transcripciones

**Modo de reproducción predeterminado** — establece el modo de reproducción que se utiliza al abrir el editor de transcripciones. También puede cambiarlo directamente en el editor en cualquier momento. Consulte [Editar transcripciones](../operations/editing_transcripts.md) para obtener una descripción de cada modo.

## Apariencia

Seleccione el tema **Oscuro** o **Claro**. El cambio se aplica de inmediato. Consulte [Seleccionar un tema](theme.md).

## Idioma

Seleccione el idioma de visualización para la interfaz de la aplicación. El cambio se aplica de inmediato. Consulte [Seleccionar un idioma](language.md).

---