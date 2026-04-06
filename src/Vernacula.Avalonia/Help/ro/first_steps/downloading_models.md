---
title: "Descărcarea Modelelor"
description: "Cum să descărcați fișierele modelului AI necesare pentru transcriere."
topic_id: first_steps_downloading_models
---

# Descărcarea Modelelor

Vernacula-Desktop necesită fișiere de model AI pentru a funcționa. Acestea nu sunt incluse în aplicație și trebuie descărcate înainte de prima transcriere.

## Starea Modelelor (Ecranul Principal)

O bară de stare îngustă în partea de sus a ecranului principal indică dacă modelele sunt pregătite. Când lipsesc fișiere, afișează și un buton `Open Settings` care vă duce direct la gestionarea modelelor.

| Stare | Semnificație |
|---|---|
| `All N model file(s) present ✓` | Toate fișierele necesare sunt descărcate și pregătite. |
| `N model file(s) missing: …` | Unul sau mai multe fișiere lipsesc; deschideți Setările pentru a le descărca. |

Când modelele sunt pregătite, butoanele `New Transcription` și `Bulk Add Jobs` devin active.

## Cum să Descărcați Modelele

1. Pe ecranul principal, faceți clic pe `Open Settings` (sau mergeți la `Settings… > Models`).
2. În secțiunea **Models**, faceți clic pe `Download Missing Models`.
3. O bară de progres și o linie de stare apar, afișând fișierul curent, poziția sa în coadă și dimensiunea descărcării — de exemplu: `[1/3] encoder-model.onnx — 42 MB`.
4. Așteptați până când starea afișează `Download complete.`

## Anularea unei Descărcări

Pentru a opri o descărcare în curs, faceți clic pe `Cancel`. Linia de stare va afișa `Download cancelled.` Fișierele descărcate parțial sunt păstrate, astfel încât descărcarea va continua de unde a rămas data viitoare când faceți clic pe `Download Missing Models`.

## Erori de Descărcare

Dacă o descărcare eșuează, linia de stare afișează `Download failed: <reason>`. Verificați conexiunea la internet și faceți clic din nou pe `Download Missing Models` pentru a reîncerca. Aplicația reia descărcarea de la ultimul fișier finalizat cu succes.

## Schimbarea Preciziei

Fișierele de model care trebuie descărcate depind de **Model Precision** selectată. Pentru a o modifica, mergeți la `Settings… > Models > Model Precision`. Dacă schimbați precizia după descărcare, noul set de fișiere trebuie descărcat separat. Consultați [Alegerea Preciziei Greutăților Modelului](model_precision.md).

---