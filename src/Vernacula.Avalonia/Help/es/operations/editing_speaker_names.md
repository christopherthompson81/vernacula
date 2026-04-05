---
title: "Edición de Nombres de Hablantes"
description: "Cómo reemplazar los ID de hablantes genéricos con nombres reales en una transcripción."
topic_id: operations_editing_speaker_names
---

# Edición de Nombres de Hablantes

El motor de transcripción etiqueta automáticamente a cada hablante con un ID genérico (por ejemplo, `speaker_0`, `speaker_1`). Puede reemplazarlos con nombres reales que aparecerán en toda la transcripción y en los archivos exportados.

## Cómo Editar los Nombres de Hablantes

1. Abra un trabajo completado. Consulte [Cargar Trabajos Completados](loading_completed_jobs.md).
2. En la vista **Resultados**, haga clic en `Edit Speaker Names`.
3. Se abre el cuadro de diálogo **Edit Speaker Names** con dos columnas:
   - **Speaker ID** — la etiqueta original asignada por el modelo (de solo lectura).
   - **Display Name** — el nombre que se muestra en la transcripción (editable).
4. Haga clic en una celda de la columna **Display Name** y escriba el nombre del hablante.
5. Presione `Tab` o haga clic en otra fila para pasar al siguiente hablante.
6. Haga clic en `Save` para aplicar los cambios, o en `Cancel` para descartarlos.

## Dónde Aparecen los Nombres

Los nombres para mostrar actualizados reemplazan los ID genéricos en:

- La tabla de segmentos en la vista de Resultados.
- Todos los archivos exportados (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Editar los Nombres Nuevamente

Puede volver a abrir el cuadro de diálogo Edit Speaker Names en cualquier momento mientras el trabajo esté cargado en la vista de Resultados. Los cambios se guardan en la base de datos local y persisten entre sesiones.

---