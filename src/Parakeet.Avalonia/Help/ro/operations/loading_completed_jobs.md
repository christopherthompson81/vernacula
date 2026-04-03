---
title: "Încărcarea lucrărilor finalizate"
description: "Cum să deschideți rezultatele unei transcrieri finalizate anterior."
topic_id: operations_loading_completed_jobs
---

# Încărcarea lucrărilor finalizate

Toate lucrările de transcriere finalizate sunt salvate în baza de date locală și rămân accesibile în tabelul **Istoric transcrieri** de pe ecranul principal.

## Cum să încărcați o lucrare finalizată

1. Pe ecranul principal, localizați lucrarea în tabelul **Istoric transcrieri**. Lucrările finalizate afișează un marcaj de stare `complete`.
2. Faceți clic pe `Load` în coloana **Acțiuni** a lucrării respective.
3. Aplicația trece la vizualizarea **Rezultate**, afișând toate segmentele transcrise pentru acea lucrare.

## Vizualizarea Rezultate

Vizualizarea Rezultate afișează:

- Numele fișierului audio ca titlu al paginii.
- Un subtitlu cu numărul de segmente (de exemplu, `42 segment(s)`).
- Un tabel de segmente cu coloanele **Vorbitor**, **Start**, **Sfârșit** și **Conținut**.

Din vizualizarea Rezultate puteți:

- [Editați transcrierea](editing_transcripts.md) — revizuiți și corectați textul, ajustați sincronizarea, îmbinați sau împărțiți segmente și verificați segmentele în timp ce ascultați înregistrarea audio.
- [Editați numele vorbitorilor](editing_speaker_names.md) — înlocuiți identificatorii generici, precum `speaker_0`, cu nume reale.
- [Exportați transcrierea](exporting_results.md) — salvați transcrierea în format Excel, CSV, JSON, SRT, Markdown, Word sau SQLite.

Pentru a reveni la lista de istorice, faceți clic pe `← Back to History`.

---