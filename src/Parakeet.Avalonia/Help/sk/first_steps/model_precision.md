---
title: "Výber presnosti váh modelu"
description: "Ako si vybrať medzi presnosťou modelu INT8 a FP32 a aké sú ich výhody a nevýhody."
topic_id: first_steps_model_precision
---

# Výber presnosti váh modelu

Presnosť modelu určuje numerický formát používaný váhami modelu AI. Ovplyvňuje veľkosť sťahovania, využitie pamäte a presnosť výsledkov.

## Možnosti presnosti

### INT8 (menšie sťahovanie)

- Menšie súbory modelu — rýchlejšie sťahovanie a menšie nároky na diskový priestor.
- Mierne nižšia presnosť pri niektorých zvukových nahrávkach.
- Odporúča sa, ak máte obmedzený diskový priestor alebo pomalšie internetové pripojenie.

### FP32 (vyššia presnosť)

- Väčšie súbory modelu.
- Vyššia presnosť, najmä pri náročných nahrávkach s prízvukmi alebo šumom na pozadí.
- Odporúča sa, keď je prioritou presnosť a máte dostatočný diskový priestor.
- Vyžaduje sa pre GPU akceleráciu CUDA — cesta cez GPU vždy používa FP32 bez ohľadu na toto nastavenie.

## Ako zmeniť presnosť

Otvorte `Settings…` v paneli ponuky, prejdite do sekcie **Models** a vyberte možnosť `INT8 (smaller download)` alebo `FP32 (more accurate)`.

## Po zmene presnosti

Zmena presnosti vyžaduje iný súbor súborov modelu. Ak súbory modelu pre novú presnosť ešte neboli stiahnuté, kliknite na `Download Missing Models` v nastaveniach. Predtým stiahnuté súbory pre druhú presnosť zostávajú na disku a v prípade prepnutia späť ich nie je potrebné znova sťahovať.

---