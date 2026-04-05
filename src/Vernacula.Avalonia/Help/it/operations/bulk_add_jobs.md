---
title: "Aggiungere più file audio in coda"
description: "Come aggiungere più file audio alla coda dei lavori in una sola operazione."
topic_id: operations_bulk_add_jobs
---

# Aggiungere più file audio in coda

Usa **Aggiunta multipla di lavori** per accodare più file audio o video per la trascrizione in un unico passaggio. L'applicazione li elabora uno alla volta nell'ordine in cui sono stati aggiunti.

## Prerequisiti

- Tutti i file del modello devono essere scaricati. La scheda **Stato del modello** deve mostrare `All N model file(s) present ✓`. Vedere [Scaricare i modelli](../first_steps/downloading_models.md).

## Come aggiungere più lavori in coda

1. Nella schermata principale, fai clic su `Bulk Add Jobs`.
2. Si apre una finestra di selezione file. Seleziona uno o più file audio o video — tieni premuto `Ctrl` o `Shift` per selezionare più file contemporaneamente.
3. Fai clic su **Apri**. Ogni file selezionato viene aggiunto alla tabella **Cronologia trascrizioni** come lavoro separato.

> **File video con più flussi audio:** Se un file video contiene più di un flusso audio (ad esempio, più lingue o una traccia con il commento del regista), l'applicazione crea automaticamente un lavoro per ogni flusso.

## Nomi dei lavori

A ogni lavoro viene assegnato automaticamente il nome del file audio corrispondente. Puoi rinominare un lavoro in qualsiasi momento facendo clic sul suo nome nella colonna **Titolo** della tabella Cronologia trascrizioni, modificando il testo e premendo `Enter` oppure facendo clic altrove.

## Comportamento della coda

- Se nessun lavoro è in esecuzione, il primo file viene avviato immediatamente e gli altri vengono mostrati come `queued`.
- Se un lavoro è già in esecuzione, tutti i file appena aggiunti vengono mostrati come `queued` e verranno avviati automaticamente in sequenza.
- Per monitorare il lavoro attivo, fai clic su `Monitor` nella colonna **Azioni** corrispondente. Vedere [Monitorare i lavori](monitoring_jobs.md).
- Per mettere in pausa o rimuovere un lavoro in coda prima che venga avviato, usa i pulsanti `Pause` o `Remove` nella colonna **Azioni** corrispondente. Vedere [Mettere in pausa, riprendere o rimuovere i lavori](pausing_resuming_removing.md).

---