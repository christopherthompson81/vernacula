---
title: "Editarea numelor vorbitorilor"
description: "Cum să înlocuiți ID-urile generice ale vorbitorilor cu nume reale într-o transcriere."
topic_id: operations_editing_speaker_names
---

# Editarea numelor vorbitorilor

Motorul de transcriere etichetează automat fiecare vorbitor cu un ID generic (de exemplu, `speaker_0`, `speaker_1`). Puteți înlocui aceste ID-uri cu nume reale, care vor apărea în tot cuprinsul transcrierii și în orice fișiere exportate.

## Cum se editează numele vorbitorilor

1. Deschideți un job finalizat. Consultați [Încărcarea joburilor finalizate](loading_completed_jobs.md).
2. În vizualizarea **Results**, faceți clic pe `Edit Speaker Names`.
3. Se deschide dialogul **Edit Speaker Names** cu două coloane:
   - **Speaker ID** — eticheta originală atribuită de model (numai pentru citire).
   - **Display Name** — numele afișat în transcriere (editabil).
4. Faceți clic pe o celulă din coloana **Display Name** și introduceți numele vorbitorului.
5. Apăsați `Tab` sau faceți clic pe un alt rând pentru a trece la următorul vorbitor.
6. Faceți clic pe `Save` pentru a aplica modificările sau pe `Cancel` pentru a le anula.

## Unde apar numele

Numele de afișare actualizate înlocuiesc ID-urile generice în:

- Tabelul de segmente din vizualizarea Results.
- Toate fișierele exportate (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Editarea repetată a numelor

Puteți redeschide dialogul Edit Speaker Names oricând, atât timp cât jobul este încărcat în vizualizarea Results. Modificările sunt salvate în baza de date locală și sunt păstrate între sesiuni.

---