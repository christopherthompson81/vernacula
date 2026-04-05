---
title: "Caricamento dei lavori completati"
description: "Come aprire i risultati di una trascrizione precedentemente completata."
topic_id: operations_loading_completed_jobs
---

# Caricamento dei lavori completati

Tutti i lavori di trascrizione completati vengono salvati nel database locale e rimangono accessibili nella tabella **Cronologia trascrizioni** nella schermata principale.

## Come caricare un lavoro completato

1. Nella schermata principale, individua il lavoro nella tabella **Cronologia trascrizioni**. I lavori completati mostrano un'etichetta di stato `complete`.
2. Fai clic su `Load` nella colonna **Azioni** del lavoro.
3. L'applicazione passa alla visualizzazione **Risultati**, che mostra tutti i segmenti trascritti per quel lavoro.

## Visualizzazione Risultati

La visualizzazione Risultati mostra:

- Il nome del file audio come intestazione della pagina.
- Un sottotitolo con il conteggio dei segmenti (ad esempio, `42 segment(s)`).
- Una tabella dei segmenti con le colonne **Parlante**, **Inizio**, **Fine** e **Contenuto**.

Dalla visualizzazione Risultati è possibile:

- [Modificare la trascrizione](editing_transcripts.md) — rivedere e correggere il testo, regolare i tempi, unire o dividere segmenti e verificare i segmenti durante l'ascolto dell'audio.
- [Modificare i nomi dei parlanti](editing_speaker_names.md) — sostituire gli ID generici come `speaker_0` con nomi reali.
- [Esportare la trascrizione](exporting_results.md) — salvare la trascrizione in Excel, CSV, JSON, SRT, Markdown, Word o SQLite.

Per tornare all'elenco della cronologia, fai clic su `← Back to History`.

---