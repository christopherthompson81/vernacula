---
title: "Impostazioni"
description: "Panoramica di tutte le opzioni nella finestra Impostazioni."
topic_id: first_steps_settings_window
---

# Impostazioni

La finestra **Impostazioni** consente di controllare la configurazione hardware, la gestione dei modelli, la modalità di segmentazione, il comportamento dell'editor, l'aspetto e la lingua. Aprila dalla barra dei menu: `Settings…`.

## Hardware e prestazioni

Questa sezione mostra lo stato della GPU NVIDIA e dello stack software CUDA, e riporta il limite dinamico del batch utilizzato durante la trascrizione su GPU.

| Elemento | Descrizione |
|---|---|
| Nome GPU e VRAM | GPU NVIDIA rilevata e memoria video disponibile. |
| CUDA Toolkit | Indica se le librerie di runtime CUDA sono state trovate tramite `CUDA_PATH`. |
| cuDNN | Indica se le DLL di runtime cuDNN sono disponibili. |
| Accelerazione CUDA | Indica se ONNX Runtime ha caricato correttamente il provider di esecuzione CUDA. |

Fai clic su `Re-check` per rieseguire il rilevamento hardware senza riavviare l'applicazione — utile dopo aver installato CUDA o cuDNN.

Quando tali componenti non vengono rilevati, vengono mostrati i link per il download diretto di CUDA Toolkit e cuDNN.

Il messaggio relativo al **limite del batch** riporta quanti secondi di audio vengono elaborati in ogni esecuzione sulla GPU. Questo valore viene derivato dalla VRAM libera dopo il caricamento dei modelli e si adatta automaticamente.

Per le istruzioni complete sulla configurazione di CUDA, consulta [Installazione di CUDA e cuDNN](cuda_installation.md).

## Modelli

Questa sezione gestisce i file dei modelli AI necessari per la trascrizione.

- **Precisione del modello** — scegli `INT8 (smaller download)` o `FP32 (more accurate)`. Consulta [Scegliere la precisione dei pesi del modello](model_precision.md).
- **Download Missing Models** — scarica i file dei modelli non ancora presenti sul disco. Una barra di avanzamento e una riga di stato tengono traccia di ogni file durante il download.
- **Check for Updates** — verifica se sono disponibili pesi del modello più recenti. Un banner di aggiornamento appare automaticamente anche nella schermata principale quando vengono rilevati pesi aggiornati.

## Modalità di segmentazione

Controlla come l'audio viene suddiviso in segmenti prima del riconoscimento vocale.

| Modalità | Descrizione |
|---|---|
| **Diarizzazione del parlante** | Utilizza il modello SortFormer per identificare i singoli parlanti ed etichettare ogni segmento. Ideale per interviste, riunioni e registrazioni con più parlanti. |
| **Rilevamento dell'attività vocale** | Utilizza Silero VAD per rilevare solo le regioni con parlato — senza etichette del parlante. Più veloce della diarizzazione e adatto all'audio con un solo parlante. |

## Editor delle trascrizioni

**Modalità di riproduzione predefinita** — imposta la modalità di riproduzione utilizzata all'apertura dell'editor delle trascrizioni. Puoi modificarla direttamente nell'editor in qualsiasi momento. Consulta [Modifica delle trascrizioni](../operations/editing_transcripts.md) per una descrizione di ogni modalità.

## Aspetto

Seleziona il tema **Scuro** o **Chiaro**. La modifica viene applicata immediatamente. Consulta [Scegliere un tema](theme.md).

## Lingua

Seleziona la lingua di visualizzazione per l'interfaccia dell'applicazione. La modifica viene applicata immediatamente. Consulta [Scegliere una lingua](language.md).

---