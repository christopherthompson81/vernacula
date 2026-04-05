---
title: "Výběr přesnosti vah modelu"
description: "Jak si vybrat mezi přesností modelu INT8 a FP32 a jaké jsou jejich vzájemné kompromisy."
topic_id: first_steps_model_precision
---

# Výběr přesnosti vah modelu

Přesnost modelu určuje číselný formát používaný vahami modelu AI. Ovlivňuje velikost stahovaných souborů, využití paměti a přesnost rozpoznávání.

## Možnosti přesnosti

### INT8 (menší stahování)

- Menší soubory modelu — rychlejší stahování a nižší nároky na místo na disku.
- Mírně nižší přesnost u některých zvukových nahrávek.
- Doporučeno, pokud máte omezený prostor na disku nebo pomalejší připojení k internetu.

### FP32 (vyšší přesnost)

- Větší soubory modelu.
- Vyšší přesnost, zejména u obtížných nahrávek s přízvukem nebo šumem na pozadí.
- Doporučeno, pokud je přesnost prioritou a máte dostatek místa na disku.
- Vyžadováno pro hardwarovou akceleraci CUDA GPU — cesta přes GPU vždy používá FP32 bez ohledu na toto nastavení.

## Jak změnit přesnost

Otevřete `Settings…` v panelu nabídek, přejděte do sekce **Models** a vyberte buď `INT8 (smaller download)`, nebo `FP32 (more accurate)`.

## Po změně přesnosti

Změna přesnosti vyžaduje odlišnou sadu souborů modelu. Pokud soubory pro novou přesnost ještě nebyly staženy, klikněte v nastavení na `Download Missing Models`. Dříve stažené soubory pro druhou přesnost zůstávají na disku a není třeba je znovu stahovat, pokud se rozhodnete přepnout zpět.

---