---
title: "Scelta della Precisione dei Pesi del Modello"
description: "Come scegliere tra la precisione del modello INT8 e FP32 e quali sono i compromessi."
topic_id: first_steps_model_precision
---

# Scelta della Precisione dei Pesi del Modello

La precisione del modello controlla il formato numerico utilizzato dai pesi del modello AI. Influisce sulle dimensioni del download, sull'utilizzo della memoria e sull'accuratezza.

## Opzioni di Precisione

### INT8 (download più piccolo)

- File del modello più piccoli — download più veloce e minor spazio su disco richiesto.
- Accuratezza leggermente inferiore su alcuni audio.
- Consigliato se si dispone di spazio su disco limitato o di una connessione Internet più lenta.

### FP32 (più accurato)

- File del modello più grandi.
- Maggiore accuratezza, specialmente su audio difficile con accenti o rumore di fondo.
- Consigliato quando l'accuratezza è la priorità e si dispone di spazio su disco sufficiente.
- Richiesto per l'accelerazione GPU CUDA — il percorso GPU utilizza sempre FP32 indipendentemente da questa impostazione.

## Come Cambiare la Precisione

Apri `Settings…` dalla barra dei menu, quindi vai alla sezione **Models** e seleziona `INT8 (smaller download)` oppure `FP32 (more accurate)`.

## Dopo Aver Cambiato la Precisione

La modifica della precisione richiede un diverso set di file del modello. Se i modelli per la nuova precisione non sono ancora stati scaricati, fai clic su `Download Missing Models` nelle Impostazioni. I file precedentemente scaricati per l'altra precisione vengono mantenuti sul disco e non devono essere riscaricati in caso di ripristino.

---