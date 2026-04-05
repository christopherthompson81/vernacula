---
title: "Selección de la precisión de los pesos del modelo"
description: "Cómo elegir entre la precisión de modelo INT8 y FP32 y cuáles son las ventajas e inconvenientes de cada opción."
topic_id: first_steps_model_precision
---

# Selección de la precisión de los pesos del modelo

La precisión del modelo controla el formato numérico que utilizan los pesos del modelo de IA. Afecta al tamaño de descarga, al uso de memoria y a la exactitud.

## Opciones de precisión

### INT8 (descarga más pequeña)

- Archivos de modelo más pequeños: se descargan más rápido y requieren menos espacio en disco.
- Exactitud ligeramente inferior en algunos tipos de audio.
- Recomendado si dispones de espacio en disco limitado o una conexión a Internet más lenta.

### FP32 (más preciso)

- Archivos de modelo más grandes.
- Mayor exactitud, especialmente en audio difícil con acentos o ruido de fondo.
- Recomendado cuando la exactitud es la prioridad y dispones de suficiente espacio en disco.
- Obligatorio para la aceleración GPU mediante CUDA: la ruta de GPU siempre utiliza FP32 independientemente de esta configuración.

## Cómo cambiar la precisión

Abre `Settings…` desde la barra de menús, ve a la sección **Models** y selecciona `INT8 (smaller download)` o `FP32 (more accurate)`.

## Después de cambiar la precisión

Cambiar la precisión requiere un conjunto diferente de archivos de modelo. Si los modelos de la nueva precisión aún no se han descargado, haz clic en `Download Missing Models` en Configuración. Los archivos descargados anteriormente para la otra precisión se conservan en el disco y no es necesario volver a descargarlos si decides revertir el cambio.

---