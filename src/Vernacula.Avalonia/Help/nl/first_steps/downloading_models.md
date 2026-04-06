---
title: "Modellen Downloaden"
description: "Hoe u de AI-modelbestanden downloadt die nodig zijn voor transcriptie."
topic_id: first_steps_downloading_models
---

# Modellen Downloaden

Vernacula-Desktop vereist AI-modelbestanden om te kunnen werken. Deze worden niet meegeleverd met de applicatie en moeten worden gedownload vóór uw eerste transcriptie.

## Modelstatus (Startscherm)

Een smalle statusbalk bovenaan het startscherm geeft aan of uw modellen gereed zijn. Als er bestanden ontbreken, wordt ook een knop `Open Settings` weergegeven waarmee u rechtstreeks naar het modelbeheer gaat.

| Status | Betekenis |
|---|---|
| `All N model file(s) present ✓` | Alle vereiste bestanden zijn gedownload en gereed. |
| `N model file(s) missing: …` | Een of meer bestanden ontbreken; open Instellingen om te downloaden. |

Wanneer de modellen gereed zijn, worden de knoppen `New Transcription` en `Bulk Add Jobs` actief.

## Modellen Downloaden

1. Klik op het startscherm op `Open Settings` (of ga naar `Settings… > Models`).
2. Klik in het gedeelte **Models** op `Download Missing Models`.
3. Een voortgangsbalk en een statusregel verschijnen met de naam van het huidige bestand, de positie in de wachtrij en de downloadgrootte — bijvoorbeeld: `[1/3] encoder-model.onnx — 42 MB`.
4. Wacht tot de status `Download complete.` weergeeft.

## Een Download Annuleren

Om een lopende download te stoppen, klikt u op `Cancel`. De statusregel toont dan `Download cancelled.` Gedeeltelijk gedownloade bestanden worden bewaard, zodat de download de volgende keer dat u op `Download Missing Models` klikt, verdergaat waar deze was gestopt.

## Downloadfouten

Als een download mislukt, toont de statusregel `Download failed: <reason>`. Controleer uw internetverbinding en klik opnieuw op `Download Missing Models` om het opnieuw te proberen. De applicatie hervat vanaf het laatste succesvol voltooide bestand.
