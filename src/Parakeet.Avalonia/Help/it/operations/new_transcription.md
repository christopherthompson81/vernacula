---
title: "Nuovo Flusso di Lavoro per la Trascrizione"
description: "Guida dettagliata per trascrivere un file audio."
topic_id: operations_new_transcription
---

# Nuovo Flusso di Lavoro per la Trascrizione

Usa questo flusso di lavoro per trascrivere un singolo file audio.

## Prerequisiti

- Tutti i file del modello devono essere scaricati. La scheda **Stato del Modello** deve mostrare `All N model file(s) present ✓`. Consulta [Scaricamento dei Modelli](../first_steps/downloading_models.md).

## Formati Supportati

### Audio

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

I file video vengono decodificati tramite FFmpeg. Se un file video contiene **più flussi audio** (ad esempio più lingue o tracce di commento), viene creato automaticamente un processo di trascrizione per ciascun flusso.

## Passaggi

### 1. Apri il modulo Nuova Trascrizione

Fai clic su `New Transcription` nella schermata principale, oppure vai su `File > New Transcription`.

### 2. Seleziona un file multimediale

Fai clic su `Browse…` accanto al campo **Audio File**. Si apre un selettore di file filtrato per i formati audio e video supportati. Seleziona il file e fai clic su **Open**. Il percorso del file viene visualizzato nel campo.

### 3. Assegna un nome al processo

Il campo **Job Name** viene precompilato con il nome del file. Modificalo se desideri un'etichetta diversa — questo nome viene visualizzato nella Cronologia delle Trascrizioni nella schermata principale.

### 4. Avvia la trascrizione

Fai clic su `Start Transcription`. L'applicazione passa alla vista **Progress**.

Per tornare indietro senza avviare, fai clic su `← Back`.

## Cosa Succede Dopo

Il processo si svolge in due fasi mostrate nella barra di avanzamento:

1. **Analisi Audio** — diarizzazione degli interlocutori: identificazione di chi sta parlando e quando.
2. **Riconoscimento Vocale** — conversione del parlato in testo segmento per segmento.

I segmenti trascritti vengono visualizzati nella tabella in tempo reale man mano che vengono prodotti. Al termine dell'elaborazione, l'applicazione passa automaticamente alla vista **Results**.

Se aggiungi un processo mentre un altro è già in esecuzione, il nuovo processo mostrerà lo stato `queued` e verrà avviato al termine di quello corrente. Consulta [Monitoraggio dei Processi](monitoring_jobs.md).

---