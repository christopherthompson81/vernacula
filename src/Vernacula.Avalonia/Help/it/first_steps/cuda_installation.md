---
title: "Installazione di CUDA e cuDNN per l'accelerazione GPU"
description: "Come configurare NVIDIA CUDA e cuDNN affinché Vernacula-Desktop possa utilizzare la GPU."
topic_id: first_steps_cuda_installation
---

# Installazione di CUDA e cuDNN per l'accelerazione GPU

Vernacula-Desktop può utilizzare una GPU NVIDIA per accelerare significativamente la trascrizione. L'accelerazione GPU richiede l'installazione sul sistema di NVIDIA CUDA Toolkit e delle librerie runtime cuDNN.

## Requisiti

- Una GPU NVIDIA compatibile con CUDA (si consiglia una GeForce GTX serie 10 o successiva).
- Windows 10 o 11 (64 bit).
- I file del modello devono essere già stati scaricati. Vedere [Scaricamento dei modelli](downloading_models.md).

## Procedura di installazione

### 1. Installare CUDA Toolkit

Scaricare ed eseguire il programma di installazione di CUDA Toolkit dal sito per sviluppatori NVIDIA. Durante l'installazione, accettare i percorsi predefiniti. Il programma di installazione imposta automaticamente la variabile d'ambiente `CUDA_PATH` — Vernacula-Desktop utilizza questa variabile per individuare le librerie CUDA.

### 2. Installare cuDNN

Scaricare l'archivio ZIP di cuDNN per la versione di CUDA installata dal sito per sviluppatori NVIDIA. Estrarre l'archivio e copiare il contenuto delle cartelle `bin`, `include` e `lib` nelle cartelle corrispondenti all'interno della directory di installazione di CUDA Toolkit (il percorso indicato da `CUDA_PATH`).

In alternativa, installare cuDNN utilizzando il programma di installazione NVIDIA cuDNN, se disponibile per la propria versione di CUDA.

### 3. Riavviare l'applicazione

Chiudere e riaprire Vernacula-Desktop al termine dell'installazione. L'applicazione verifica la presenza di CUDA all'avvio.

## Stato della GPU nelle impostazioni

Aprire `Settings…` dalla barra dei menu e consultare la sezione **Hardware & Performance**. Ogni componente mostra un segno di spunta (✓) quando viene rilevato:

| Elemento | Significato |
|---|---|
| Nome GPU e VRAM | La GPU NVIDIA è stata trovata |
| CUDA Toolkit ✓ | Librerie CUDA individuate tramite `CUDA_PATH` |
| cuDNN ✓ | DLL runtime cuDNN trovate |
| CUDA Acceleration ✓ | ONNX Runtime ha caricato il provider di esecuzione CUDA |

Se un elemento risulta mancante dopo l'installazione, fare clic su `Re-check` per ripetere il rilevamento dell'hardware senza riavviare l'applicazione.

La finestra Impostazioni fornisce anche collegamenti diretti per il download di CUDA Toolkit e cuDNN, qualora non siano ancora installati.

### Risoluzione dei problemi

Se `CUDA Acceleration` non mostra il segno di spunta, verificare che:

- La variabile d'ambiente `CUDA_PATH` sia impostata (controllare in `System > Advanced system settings > Environment Variables`).
- Le DLL di cuDNN si trovino in una directory inclusa nel `PATH` di sistema oppure all'interno della cartella `bin` di CUDA.
- Il driver della GPU sia aggiornato.

### Dimensionamento dei batch

Quando CUDA è attivo, la sezione **Hardware & Performance** mostra anche il limite dinamico corrente dei batch, ovvero il numero massimo di secondi di audio elaborati in una singola esecuzione sulla GPU. Questo valore viene calcolato in base alla VRAM libera dopo il caricamento dei modelli e si adegua automaticamente in caso di variazioni della memoria disponibile.

## Utilizzo senza GPU

Se CUDA non è disponibile, Vernacula-Desktop passa automaticamente all'elaborazione tramite CPU. La trascrizione continua a funzionare, ma risulterà più lenta, in particolare per file audio di lunga durata.

---