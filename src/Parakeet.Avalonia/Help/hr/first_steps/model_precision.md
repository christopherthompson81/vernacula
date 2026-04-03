---
title: "Odabir preciznosti težina modela"
description: "Kako odabrati između INT8 i FP32 preciznosti modela i koji su kompromisi."
topic_id: first_steps_model_precision
---

# Odabir preciznosti težina modela

Preciznost modela određuje numerički format koji AI model koristi za svoje težine. Utječe na veličinu preuzimanja, korištenje memorije i točnost.

## Mogućnosti preciznosti

### INT8 (manje preuzimanje)

- Manje datoteke modela — brže preuzimanje i manje potrebnog prostora na disku.
- Neznatno niža točnost na nekim audio zapisima.
- Preporučuje se ako imate ograničen prostor na disku ili sporiju internetsku vezu.

### FP32 (veća točnost)

- Veće datoteke modela.
- Veća točnost, posebno na zahtjevnom audiju s naglaskom ili pozadinskom bukom.
- Preporučuje se kada je točnost prioritet i imate dovoljno prostora na disku.
- Obavezno za CUDA GPU ubrzanje — GPU put uvijek koristi FP32 bez obzira na ovu postavku.

## Kako promijeniti preciznost

Otvorite `Settings…` iz trake izbornika, zatim idite na odjeljak **Models** i odaberite `INT8 (smaller download)` ili `FP32 (more accurate)`.

## Nakon promjene preciznosti

Promjena preciznosti zahtijeva drugi skup datoteka modela. Ako datoteke modela za novu preciznost još nisu preuzete, kliknite `Download Missing Models` u postavkama. Prethodno preuzete datoteke za drugu preciznost ostaju na disku i ne moraju se ponovo preuzimati ako se vratite na nju.

---