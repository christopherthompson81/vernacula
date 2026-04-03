---
title: "Adăugarea mai multor fișiere audio în coadă"
description: "Cum să adăugați mai multe fișiere audio în coada de joburi dintr-o singură operațiune."
topic_id: operations_bulk_add_jobs
---

# Adăugarea mai multor fișiere audio în coadă

Utilizați **Bulk Add Jobs** pentru a pune în coadă mai multe fișiere audio sau video pentru transcriere într-un singur pas. Aplicația le procesează pe rând, în ordinea în care au fost adăugate.

## Cerințe preliminare

- Toate fișierele de model trebuie să fie descărcate. Cardul **Model Status** trebuie să afișeze `All N model file(s) present ✓`. Consultați [Descărcarea modelelor](../first_steps/downloading_models.md).

## Cum să adăugați joburi în bloc

1. Pe ecranul principal, faceți clic pe `Bulk Add Jobs`.
2. Se deschide un selector de fișiere. Selectați unul sau mai multe fișiere audio sau video — țineți apăsat `Ctrl` sau `Shift` pentru a selecta mai multe fișiere.
3. Faceți clic pe **Open**. Fiecare fișier selectat este adăugat în tabelul **Transcription History** ca un job separat.

> **Fișiere video cu mai multe fluxuri audio:** Dacă un fișier video conține mai mult de un flux audio (de exemplu, mai multe limbi sau un comentariu al regizorului), aplicația creează automat câte un job pentru fiecare flux.

## Denumirea joburilor

Fiecare job este denumit automat după numele fișierului audio corespunzător. Puteți redenumi oricând un job făcând clic pe numele său din coloana **Title** a tabelului Transcription History, editând textul și apăsând `Enter` sau făcând clic în altă parte.

## Comportamentul cozii

- Dacă niciun job nu rulează în acel moment, primul fișier începe imediat, iar celelalte sunt afișate ca `queued`.
- Dacă un job rulează deja, toate fișierele nou adăugate sunt afișate ca `queued` și vor porni automat, pe rând.
- Pentru a monitoriza jobul activ, faceți clic pe `Monitor` din coloana **Actions** a acestuia. Consultați [Monitorizarea joburilor](monitoring_jobs.md).
- Pentru a întrerupe sau elimina un job din coadă înainte ca acesta să înceapă, utilizați butoanele `Pause` sau `Remove` din coloana **Actions** a acestuia. Consultați [Întreruperea, reluarea sau eliminarea joburilor](pausing_resuming_removing.md).

---