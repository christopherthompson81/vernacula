---
title: "Download dei modelli"
description: "Come scaricare i file del modello AI necessari per la trascrizione."
topic_id: first_steps_downloading_models
---

# Download dei modelli

Parakeet Transcription richiede i file del modello AI per funzionare. Questi non sono inclusi nell'applicazione e devono essere scaricati prima della prima trascrizione.

## Stato dei modelli (schermata iniziale)

Una sottile barra di stato nella parte superiore della schermata iniziale indica se i modelli sono pronti. Quando mancano dei file, viene mostrato anche un pulsante `Open Settings` che porta direttamente alla gestione dei modelli.

| Stato | Significato |
|---|---|
| `All N model file(s) present ✓` | Tutti i file richiesti sono stati scaricati e sono pronti. |
| `N model file(s) missing: …` | Uno o più file sono assenti; aprire le Impostazioni per scaricarli. |

Quando i modelli sono pronti, i pulsanti `New Transcription` e `Bulk Add Jobs` diventano attivi.

## Come scaricare i modelli

1. Nella schermata iniziale, fare clic su `Open Settings` (oppure andare su `Settings… > Models`).
2. Nella sezione **Models**, fare clic su `Download Missing Models`.
3. Vengono visualizzati una barra di avanzamento e una riga di stato che mostrano il file corrente, la sua posizione nella coda e la dimensione del download — ad esempio: `[1/3] encoder-model.onnx — 42 MB`.
4. Attendere che lo stato mostri `Download complete.`

## Annullamento di un download

Per interrompere un download in corso, fare clic su `Cancel`. La riga di stato mostrerà `Download cancelled.` I file parzialmente scaricati vengono conservati, in modo che il download riprenda dal punto in cui era stato interrotto la volta successiva che si fa clic su `Download Missing Models`.

## Errori di download

Se un download non riesce, la riga di stato mostra `Download failed: <reason>`. Verificare la connessione Internet e fare nuovamente clic su `Download Missing Models` per riprovare. L'applicazione riprende dall'ultimo file completato con successo.

## Modifica della precisione

I file del modello da scaricare dipendono dalla **Model Precision** selezionata. Per modificarla, andare su `Settings… > Models > Model Precision`. Se si cambia la precisione dopo aver effettuato il download, il nuovo set di file deve essere scaricato separatamente. Vedere [Selezione della precisione dei pesi del modello](model_precision.md).

---