---
title: "Mettere in pausa, riprendere o rimuovere i lavori"
description: "Come mettere in pausa un lavoro in esecuzione, riprenderne uno interrotto o eliminare un lavoro dalla cronologia."
topic_id: operations_pausing_resuming_removing
---

# Mettere in pausa, riprendere o rimuovere i lavori

## Mettere in pausa un lavoro

È possibile mettere in pausa un lavoro in esecuzione o in coda da due posizioni:

- **Visualizzazione avanzamento** — fare clic su `Pause` nell'angolo in basso a destra mentre si monitora il lavoro attivo.
- **Tabella Cronologia trascrizioni** — fare clic su `Pause` nella colonna **Actions** di qualsiasi riga il cui stato sia `running` o `queued`.

Dopo aver fatto clic su `Pause`, la riga di stato mostra `Pausing…` mentre l'applicazione termina l'unità di elaborazione corrente. Lo stato del lavoro cambia quindi in `cancelled` nella tabella della cronologia.

> La pausa salva tutti i segmenti trascritti fino a quel momento. È possibile riprendere il lavoro in seguito senza perdere il lavoro svolto.

## Riprendere un lavoro

Per riprendere un lavoro messo in pausa o non riuscito:

1. Nella schermata principale, individuare il lavoro nella tabella **Transcription History**. Il suo stato sarà `cancelled` o `failed`.
2. Fare clic su `Resume` nella colonna **Actions**.
3. L'applicazione torna alla visualizzazione **Progress** e continua dal punto in cui l'elaborazione si era interrotta.

La riga di stato mostra brevemente `Resuming…` durante la reinizializzazione del lavoro.

## Rimuovere un lavoro

Per eliminare definitivamente un lavoro e la relativa trascrizione dalla cronologia:

1. Nella tabella **Transcription History**, fare clic su `Remove` nella colonna **Actions** del lavoro che si desidera eliminare.

Il lavoro viene rimosso dall'elenco e i relativi dati vengono eliminati dal database locale. Questa operazione non può essere annullata. I file esportati salvati su disco non vengono interessati.

---