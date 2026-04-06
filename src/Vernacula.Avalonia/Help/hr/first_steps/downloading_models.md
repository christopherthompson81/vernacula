---
title: "Preuzimanje modela"
description: "Kako preuzeti datoteke AI modela potrebne za transkripciju."
topic_id: first_steps_downloading_models
---

# Preuzimanje modela

Vernacula-Desktop zahtijeva datoteke AI modela za rad. Te datoteke nisu uključene u aplikaciju i moraju se preuzeti prije prve transkripcije.

## Status modela (početni zaslon)

Tanak statusni redak na vrhu početnog zaslona prikazuje jesu li vaši modeli spremni. Kada nedostaju datoteke, prikazuje se i gumb `Open Settings` koji vas vodi izravno na upravljanje modelima.

| Status | Značenje |
|---|---|
| `All N model file(s) present ✓` | Sve potrebne datoteke su preuzete i spremne. |
| `N model file(s) missing: …` | Jedna ili više datoteka nedostaje; otvorite Postavke za preuzimanje. |

Kada su modeli spremni, gumbi `New Transcription` i `Bulk Add Jobs` postaju aktivni.

## Kako preuzeti modele

1. Na početnom zaslonu kliknite `Open Settings` (ili idite na `Settings… > Models`).
2. U odjeljku **Models** kliknite `Download Missing Models`.
3. Pojavljuju se traka napretka i statusni redak koji prikazuju trenutnu datoteku, njezin položaj u redu čekanja i veličinu preuzimanja — na primjer: `[1/3] encoder-model.onnx — 42 MB`.
4. Pričekajte da status prikaže `Download complete.`

## Otkazivanje preuzimanja

Za zaustavljanje preuzimanja u tijeku kliknite `Cancel`. Statusni redak prikazat će `Download cancelled.` Djelomično preuzete datoteke se čuvaju, tako da se preuzimanje nastavlja od mjesta gdje je stalo sljedeći put kada kliknete `Download Missing Models`.

## Greške pri preuzimanju

Ako preuzimanje ne uspije, statusni redak prikazuje `Download failed: <reason>`. Provjerite internetsku vezu i kliknite `Download Missing Models` ponovo za ponovni pokušaj. Aplikacija nastavlja od posljednje uspješno dovršene datoteke.

## Promjena preciznosti

Datoteke modela koje je potrebno preuzeti ovise o odabranoj **Model Precision**. Za promjenu idite na `Settings… > Models > Model Precision`. Ako promijenite preciznost nakon preuzimanja, novi skup datoteka mora se preuzeti zasebno. Pogledajte [Odabir preciznosti težina modela](model_precision.md).

---