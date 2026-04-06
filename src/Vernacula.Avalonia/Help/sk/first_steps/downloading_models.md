---
title: "Sťahovanie modelov"
description: "Ako stiahnuť súbory AI modelov potrebné na prepis."
topic_id: first_steps_downloading_models
---

# Sťahovanie modelov

Vernacula-Desktop na svoju činnosť vyžaduje súbory AI modelov. Tieto súbory nie sú súčasťou aplikácie a musia byť stiahnuté pred prvým prepisom.

## Stav modelov (domovská obrazovka)

V hornej časti domovskej obrazovky sa zobrazuje úzky stavový riadok, ktorý ukazuje, či sú vaše modely pripravené. Ak niektoré súbory chýbajú, zobrazí sa aj tlačidlo `Open Settings`, ktoré vás priamo prenesie do správy modelov.

| Stav | Význam |
|---|---|
| `All N model file(s) present ✓` | Všetky požadované súbory sú stiahnuté a pripravené. |
| `N model file(s) missing: …` | Jeden alebo viac súborov chýba; otvorte Nastavenia a stiahnite ich. |

Keď sú modely pripravené, tlačidlá `New Transcription` a `Bulk Add Jobs` sa stanú aktívnymi.

## Ako stiahnuť modely

1. Na domovskej obrazovke kliknite na `Open Settings` (alebo prejdite do `Settings… > Models`).
2. V sekcii **Models** kliknite na `Download Missing Models`.
3. Zobrazí sa ukazovateľ priebehu a stavový riadok s názvom aktuálneho súboru, jeho poradím vo fronte a veľkosťou sťahovania — napríklad: `[1/3] encoder-model.onnx — 42 MB`.
4. Počkajte, kým sa stav nezmení na `Download complete.`

## Zrušenie sťahovania

Ak chcete zastaviť prebiehajúce sťahovanie, kliknite na `Cancel`. Stavový riadok zobrazí správu `Download cancelled.` Čiastočne stiahnuté súbory sú zachované, takže sťahovanie pri ďalšom kliknutí na `Download Missing Models` pokračuje od miesta, kde bolo prerušené.

## Chyby pri sťahovaní

Ak sťahovanie zlyhá, stavový riadok zobrazí správu `Download failed: <reason>`. Skontrolujte internetové pripojenie a opätovným kliknutím na `Download Missing Models` skúste sťahovanie zopakovať. Aplikácia pokračuje od posledného úspešne stiahnutého súboru.
