---
title: "Flux de lucru pentru transcriere nouă"
description: "Ghid pas cu pas pentru transcrierea unui fișier audio."
topic_id: operations_new_transcription
---

# Flux de lucru pentru transcriere nouă

Utilizați acest flux de lucru pentru a transcrie un singur fișier audio.

## Cerințe prealabile

- Toate fișierele de model trebuie să fie descărcate. Cardul **Stare modele** trebuie să afișeze `All N model file(s) present ✓`. Consultați [Descărcarea modelelor](../first_steps/downloading_models.md).

## Formate acceptate

### Audio

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Fișierele video sunt decodate prin FFmpeg. Dacă un fișier video conține **mai multe fluxuri audio** (de ex. mai multe limbi sau piste de comentarii), se creează automat câte un job de transcriere pentru fiecare flux.

## Pași

### 1. Deschideți formularul de transcriere nouă

Faceți clic pe `New Transcription` pe ecranul principal sau accesați `File > New Transcription`.

### 2. Selectați un fișier media

Faceți clic pe `Browse…` de lângă câmpul **Fișier audio**. Se deschide un selector de fișiere filtrat după formatele audio și video acceptate. Selectați fișierul și faceți clic pe **Open**. Calea fișierului apare în câmp.

### 3. Denumiți jobul

Câmpul **Nume job** este completat automat pe baza numelui fișierului. Modificați-l dacă doriți o altă etichetă — acest nume apare în Istoricul transcrierilor de pe ecranul principal.

### 4. Porniți transcrierea

Faceți clic pe `Start Transcription`. Aplicația comută la vizualizarea **Progres**.

Pentru a reveni fără a porni transcrierea, faceți clic pe `← Back`.

## Ce se întâmplă în continuare

Jobul parcurge două faze afișate în bara de progres:

1. **Analiza audio** — diarizarea vorbitorilor: identificarea cine vorbește și când.
2. **Recunoașterea vorbirii** — conversia vorbirii în text, segment cu segment.

Segmentele transcrise apar în tabelul live pe măsură ce sunt generate. Când procesarea este completă, aplicația trece automat la vizualizarea **Rezultate**.

Dacă adăugați un job în timp ce un altul este deja în execuție, noul job va afișa starea `queued` și va porni când jobul curent se finalizează. Consultați [Monitorizarea joburilor](monitoring_jobs.md).

---