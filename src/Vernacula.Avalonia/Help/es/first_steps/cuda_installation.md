---
title: "Instalación de CUDA y cuDNN para aceleración por GPU"
description: "Cómo configurar NVIDIA CUDA y cuDNN para que Parakeet Transcription pueda usar tu GPU."
topic_id: first_steps_cuda_installation
---

# Instalación de CUDA y cuDNN para aceleración por GPU

Parakeet Transcription puede usar una GPU de NVIDIA para acelerar significativamente la transcripción. La aceleración por GPU requiere que el kit de herramientas NVIDIA CUDA y las bibliotecas de tiempo de ejecución cuDNN estén instalados en tu sistema.

## Requisitos

- Una GPU de NVIDIA compatible con CUDA (se recomienda GeForce GTX serie 10 o posterior).
- Windows 10 u 11 (64 bits).
- Los archivos de modelo ya deben estar descargados. Consulta [Descarga de modelos](downloading_models.md).

## Pasos de instalación

### 1. Instalar el kit de herramientas CUDA

Descarga y ejecuta el instalador del kit de herramientas CUDA desde el sitio web para desarrolladores de NVIDIA. Durante la instalación, acepta las rutas predeterminadas. El instalador establece automáticamente la variable de entorno `CUDA_PATH`; Parakeet usa esta variable para localizar las bibliotecas CUDA.

### 2. Instalar cuDNN

Descarga el archivo ZIP de cuDNN correspondiente a tu versión de CUDA instalada desde el sitio web para desarrolladores de NVIDIA. Extrae el archivo y copia el contenido de sus carpetas `bin`, `include` y `lib` en las carpetas correspondientes dentro del directorio de instalación del kit de herramientas CUDA (la ruta que indica `CUDA_PATH`).

Alternativamente, instala cuDNN usando el instalador de NVIDIA cuDNN si hay uno disponible para tu versión de CUDA.

### 3. Reiniciar la aplicación

Cierra y vuelve a abrir Parakeet Transcription tras la instalación. La aplicación comprueba la presencia de CUDA al iniciarse.

## Estado de la GPU en Configuración

Abre `Settings…` desde la barra de menús y consulta la sección **Hardware & Performance**. Cada componente muestra una marca de verificación (✓) cuando es detectado:

| Elemento | Significado |
|---|---|
| Nombre de la GPU y VRAM | Se encontró tu GPU de NVIDIA |
| CUDA Toolkit ✓ | Bibliotecas CUDA localizadas mediante `CUDA_PATH` |
| cuDNN ✓ | DLLs de tiempo de ejecución de cuDNN encontradas |
| CUDA Acceleration ✓ | ONNX Runtime cargó el proveedor de ejecución CUDA |

Si algún elemento no aparece tras la instalación, haz clic en `Re-check` para volver a ejecutar la detección de hardware sin reiniciar la aplicación.

La ventana de Configuración también ofrece enlaces de descarga directa para el kit de herramientas CUDA y cuDNN si aún no están instalados.

### Solución de problemas

Si `CUDA Acceleration` no muestra una marca de verificación, verifica lo siguiente:

- La variable de entorno `CUDA_PATH` está establecida (comprueba `System > Advanced system settings > Environment Variables`).
- Los archivos DLL de cuDNN se encuentran en un directorio incluido en la variable `PATH` del sistema o dentro de la carpeta `bin` de CUDA.
- El controlador de tu GPU está actualizado.

### Tamaño de lote

Cuando CUDA está activo, la sección **Hardware & Performance** también muestra el límite de lote dinámico actual: la cantidad máxima de segundos de audio procesados en una sola ejecución en la GPU. Este valor se calcula a partir de la VRAM libre una vez cargados los modelos y se ajusta automáticamente si cambia la memoria disponible.

## Uso sin GPU

Si CUDA no está disponible, Parakeet recurre automáticamente al procesamiento por CPU. La transcripción sigue funcionando, pero será más lenta, especialmente con archivos de audio largos.

---