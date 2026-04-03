---
title: "Monitorizarea lucrărilor"
description: "Cum să urmăriți progresul unei lucrări în execuție sau în așteptare."
topic_id: operations_monitoring_jobs
---

# Monitorizarea lucrărilor

Vizualizarea **Progres** vă oferă o vedere în timp real a unei lucrări de transcriere în execuție.

## Deschiderea vizualizării Progres

- Când porniți o nouă transcriere, aplicația trece automat la vizualizarea Progres.
- Pentru o lucrare deja în execuție sau în așteptare, găsiți-o în tabelul **Istoricul transcrierilor** și faceți clic pe `Monitor` din coloana **Acțiuni**.

## Citirea vizualizării Progres

| Element | Descriere |
|---|---|
| Bara de progres | Procentul general de finalizare. Indeterminată (animată) în timp ce lucrarea pornește sau se reia. |
| Eticheta procentuală | Procentul numeric afișat în dreapta barei. |
| Mesajul de stare | Activitatea curentă — de exemplu `Audio Analysis` sau `Speech Recognition`. Afișează `Waiting in queue…` dacă lucrarea nu a început încă. |
| Tabelul de segmente | Flux în timp real al segmentelor transcrise, cu coloanele **Speaker**, **Start**, **End** și **Content**. Derulează automat pe măsură ce sosesc segmente noi. |

## Fazele progresului

Fazele afișate depind de **Modul de segmentare** selectat în Setări.

**Modul Diarizare vorbitori** (implicit):

1. **Analiza audio** — diarizarea SortFormer rulează pe întregul fișier pentru a identifica limitele dintre vorbitori. Bara poate rămâne aproape de 0% până la finalizarea acestei faze.
2. **Recunoaștere vocală** — fiecare segment de vorbitor este transcris. Procentul crește constant în această fază.

**Modul Detectare activitate vocală**:

1. **Detectarea segmentelor de vorbire** — Silero VAD scanează fișierul pentru a găsi regiunile de vorbire. Această fază este rapidă.
2. **Recunoaștere vocală** — fiecare regiune de vorbire detectată este transcrisă.

În ambele moduri, tabelul de segmente în timp real se completează pe măsură ce transcrierea avansează.

## Navigarea în altă parte

Faceți clic pe `← Back to Home` pentru a reveni la ecranul principal fără a întrerupe lucrarea. Lucrarea continuă să ruleze în fundal, iar starea sa se actualizează în tabelul **Istoricul transcrierilor**. Faceți clic pe `Monitor` din nou oricând pentru a reveni la vizualizarea Progres.

---