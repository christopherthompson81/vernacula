---
title: "Urejanje imen govorcev"
description: "Kako zamenjati splošne ID-je govorcev z resničnimi imeni v prepisu."
topic_id: operations_editing_speaker_names
---

# Urejanje imen govorcev

Mehanizem za prepisovanje vsakemu govorcu samodejno dodeli splošni ID (na primer `speaker_0`, `speaker_1`). Te ID-je lahko zamenjate z resničnimi imeni, ki se bodo prikazovala v celotnem prepisu in v vseh izvoženih datotekah.

## Kako urediti imena govorcev

1. Odprite dokončano opravilo. Glejte [Nalaganje dokončanih opravil](loading_completed_jobs.md).
2. V pogledu **Rezultati** kliknite `Edit Speaker Names`.
3. Odpre se pogovorno okno **Edit Speaker Names** z dvema stolpcema:
   - **Speaker ID** — izvirna oznaka, ki jo je dodelil model (samo za branje).
   - **Display Name** — ime, prikazano v prepisu (možnost urejanja).
4. Kliknite celico v stolpcu **Display Name** in vnesite ime govorca.
5. Pritisnite `Tab` ali kliknite drugo vrstico, da se premaknete na naslednjega govorca.
6. Kliknite `Save`, da uveljavite spremembe, ali `Cancel`, da jih zavržete.

## Kje se imena prikažejo

Posodobljena prikazna imena nadomestijo splošne ID-je v:

- tabeli segmentov v pogledu Rezultati,
- vseh izvoženih datotekah (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Ponovno urejanje imen

Pogovorno okno za urejanje imen govorcev lahko znova odprete kadarkoli, ko je opravilo naloženo v pogledu Rezultati. Spremembe se shranijo v lokalno podatkovno bazo in se ohranijo med sejami.

---