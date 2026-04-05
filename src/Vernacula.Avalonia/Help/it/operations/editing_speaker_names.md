---
title: "Modifica dei nomi dei parlanti"
description: "Come sostituire gli ID generici dei parlanti con nomi reali in una trascrizione."
topic_id: operations_editing_speaker_names
---

# Modifica dei nomi dei parlanti

Il motore di trascrizione assegna automaticamente a ciascun parlante un ID generico (ad esempio, `speaker_0`, `speaker_1`). È possibile sostituirli con nomi reali che appariranno nell'intera trascrizione e in tutti i file esportati.

## Come modificare i nomi dei parlanti

1. Apri un lavoro completato. Consulta [Caricamento dei lavori completati](loading_completed_jobs.md).
2. Nella vista **Risultati**, fai clic su `Edit Speaker Names`.
3. Si apre la finestra di dialogo **Edit Speaker Names** con due colonne:
   - **Speaker ID** — l'etichetta originale assegnata dal modello (sola lettura).
   - **Display Name** — il nome visualizzato nella trascrizione (modificabile).
4. Fai clic su una cella nella colonna **Display Name** e digita il nome del parlante.
5. Premi `Tab` o fai clic su un'altra riga per passare al parlante successivo.
6. Fai clic su `Save` per applicare le modifiche, oppure su `Cancel` per annullarle.

## Dove appaiono i nomi

I nomi visualizzati aggiornati sostituiscono gli ID generici in:

- La tabella dei segmenti nella vista Risultati.
- Tutti i file esportati (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Modificare i nomi nuovamente

È possibile riaprire la finestra di dialogo Edit Speaker Names in qualsiasi momento mentre il lavoro è caricato nella vista Risultati. Le modifiche vengono salvate nel database locale e sono mantenute tra una sessione e l'altra.

---