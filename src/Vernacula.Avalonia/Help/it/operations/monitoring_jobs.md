---
title: "Monitoraggio dei lavori"
description: "Come visualizzare l'avanzamento di un lavoro in esecuzione o in coda."
topic_id: operations_monitoring_jobs
---

# Monitoraggio dei lavori

La vista **Avanzamento** offre una visualizzazione in tempo reale di un lavoro di trascrizione in corso.

## Apertura della vista Avanzamento

- Quando si avvia una nuova trascrizione, l'applicazione passa automaticamente alla vista Avanzamento.
- Per un lavoro già in esecuzione o in coda, individuarlo nella tabella **Cronologia trascrizioni** e fare clic su `Monitor` nella colonna **Azioni**.

## Lettura della vista Avanzamento

| Elemento | Descrizione |
|---|---|
| Barra di avanzamento | Percentuale di completamento complessiva. Indeterminata (animata) mentre il lavoro è in fase di avvio o ripresa. |
| Etichetta percentuale | Percentuale numerica mostrata a destra della barra. |
| Messaggio di stato | Attività corrente — ad esempio `Audio Analysis` o `Speech Recognition`. Mostra `Waiting in queue…` se il lavoro non è ancora iniziato. |
| Tabella segmenti | Feed in tempo reale dei segmenti trascritti con le colonne **Speaker**, **Start**, **End** e **Content**. Scorre automaticamente man mano che arrivano nuovi segmenti. |

## Fasi di avanzamento

Le fasi visualizzate dipendono dalla **Modalità di segmentazione** selezionata nelle Impostazioni.

**Modalità Speaker Diarization** (predefinita):

1. **Audio Analysis** — la diarizzazione SortFormer viene eseguita sull'intero file per identificare i confini tra i parlanti. La barra potrebbe restare vicino allo 0% fino al completamento di questa fase.
2. **Speech Recognition** — ogni segmento del parlante viene trascritto. La percentuale aumenta costantemente durante questa fase.

**Modalità Voice Activity Detection**:

1. **Detecting speech segments** — Silero VAD analizza il file per individuare le regioni di parlato. Questa fase è rapida.
2. **Speech Recognition** — ogni regione di parlato rilevata viene trascritta.

In entrambe le modalità, la tabella dei segmenti in tempo reale si popola man mano che la trascrizione procede.

## Navigazione verso altre schermate

Fare clic su `← Back to Home` per tornare alla schermata principale senza interrompere il lavoro. Il lavoro continua a essere eseguito in background e il suo stato si aggiorna nella tabella **Cronologia trascrizioni**. Fare clic su `Monitor` in qualsiasi momento per tornare alla vista Avanzamento.

---