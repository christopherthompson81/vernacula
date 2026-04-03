---
title: "Pausarea, Reluarea sau Eliminarea Lucrărilor"
description: "Cum să pausați o lucrare în desfășurare, să reluați una oprită sau să ștergeți o lucrare din istoric."
topic_id: operations_pausing_resuming_removing
---

# Pausarea, Reluarea sau Eliminarea Lucrărilor

## Pausarea unei Lucrări

Puteți pausa o lucrare în desfășurare sau în așteptare din două locuri:

- **Vizualizarea progresului** — faceți clic pe `Pause` în colțul din dreapta jos în timp ce urmăriți lucrarea activă.
- **Tabelul Istoricului Transcrierilor** — faceți clic pe `Pause` în coloana **Actions** a oricărui rând al cărui status este `running` sau `queued`.

După ce faceți clic pe `Pause`, linia de status afișează `Pausing…` în timp ce aplicația finalizează unitatea de procesare curentă. Statusul lucrării se schimbă apoi în `cancelled` în tabelul de istoric.

> Pausarea salvează toate segmentele transcrise până în acel moment. Puteți relua lucrarea mai târziu fără a pierde acea muncă.

## Reluarea unei Lucrări

Pentru a relua o lucrare pausată sau eșuată:

1. Pe ecranul principal, găsiți lucrarea în tabelul **Transcription History**. Statusul acesteia va fi `cancelled` sau `failed`.
2. Faceți clic pe `Resume` în coloana **Actions**.
3. Aplicația revine la vizualizarea **Progress** și continuă de acolo unde s-a oprit procesarea.

Linia de status afișează `Resuming…` pentru scurt timp în timp ce lucrarea se reinițializează.

## Eliminarea unei Lucrări

Pentru a șterge definitiv o lucrare și transcrierea sa din istoric:

1. În tabelul **Transcription History**, faceți clic pe `Remove` în coloana **Actions** a lucrării pe care doriți să o ștergeți.

Lucrarea este eliminată din listă, iar datele sale sunt șterse din baza de date locală. Această acțiune nu poate fi anulată. Fișierele exportate salvate pe disc nu sunt afectate.

---