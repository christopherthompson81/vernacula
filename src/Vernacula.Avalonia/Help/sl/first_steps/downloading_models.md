---
title: "Prenos modelov"
description: "Kako prenesti datoteke modelov umetne inteligence, potrebne za transkripcijo."
topic_id: first_steps_downloading_models
---

# Prenos modelov

Vernacula-Desktop za delovanje potrebuje datoteke modelov umetne inteligence. Te niso vključene v aplikacijo in jih je treba prenesti pred prvo transkripcijo.

## Stanje modelov (domači zaslon)

Ozka vrstica stanja na vrhu domačega zaslona prikazuje, ali so vaši modeli pripravljeni. Ko datoteke manjkajo, se prikaže tudi gumb `Open Settings`, ki vas neposredno pripelje do upravljanja modelov.

| Stanje | Pomen |
|---|---|
| `All N model file(s) present ✓` | Vse zahtevane datoteke so prenesene in pripravljene. |
| `N model file(s) missing: …` | Ena ali več datotek manjka; odprite Nastavitve za prenos. |

Ko so modeli pripravljeni, postaneta gumba `New Transcription` in `Bulk Add Jobs` aktivna.

## Kako prenesti modele

1. Na domačem zaslonu kliknite `Open Settings` (ali pojdite na `Settings… > Models`).
2. V razdelku **Models** kliknite `Download Missing Models`.
3. Prikaže se vrstica napredka in vrstica stanja, ki prikazuje trenutno datoteko, njeno mesto v čakalni vrsti in velikost prenosa — na primer: `[1/3] encoder-model.onnx — 42 MB`.
4. Počakajte, da stanje prikaže `Download complete.`

## Preklic prenosa

Če želite ustaviti prenos v teku, kliknite `Cancel`. Vrstica stanja bo prikazala `Download cancelled.` Delno prenesene datoteke so ohranjene, zato se prenos ob naslednjem kliku `Download Missing Models` nadaljuje od mesta, kjer je bil prekinjen.

## Napake pri prenosu

Če prenos ne uspe, vrstica stanja prikaže `Download failed: <reason>`. Preverite internetno povezavo in znova kliknite `Download Missing Models` za ponovni poskus. Aplikacija se nadaljuje od zadnje uspešno prenesene datoteke.
