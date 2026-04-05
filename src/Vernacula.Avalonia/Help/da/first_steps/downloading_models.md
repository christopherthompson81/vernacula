---
title: "Hentning af modeller"
description: "Sådan henter du de AI-modelfiler, der kræves til transskription."
topic_id: first_steps_downloading_models
---

# Hentning af modeller

Parakeet Transskription kræver AI-modelfiler for at kunne køre. Disse følger ikke med applikationen og skal hentes, inden du foretager din første transskription.

## Modelstatus (startskærm)

En smal statuslinje øverst på startskærmen viser, om dine modeller er klar. Hvis der mangler filer, vises der også en `Open Settings`-knap, som fører dig direkte til modelstyring.

| Status | Betydning |
|---|---|
| `All N model file(s) present ✓` | Alle nødvendige filer er hentet og klar. |
| `N model file(s) missing: …` | En eller flere filer mangler. Åbn Indstillinger for at hente dem. |

Når modellerne er klar, bliver knapperne `New Transcription` og `Bulk Add Jobs` aktive.

## Sådan henter du modeller

1. På startskærmen klikker du på `Open Settings` (eller gå til `Settings… > Models`).
2. I afsnittet **Models** klikker du på `Download Missing Models`.
3. En statuslinje og en fremdriftslinje viser den aktuelle fil, dens position i køen og filstørrelsen — for eksempel: `[1/3] encoder-model.onnx — 42 MB`.
4. Vent, til statuslinjen viser `Download complete.`

## Annullering af en hentning

Hvis du vil stoppe en igangværende hentning, skal du klikke på `Cancel`. Statuslinjen viser herefter `Download cancelled.` Delvist hentede filer bevares, så hentningen genoptages fra det sted, den slap, næste gang du klikker på `Download Missing Models`.

## Hentningsfejl

Hvis en hentning mislykkes, viser statuslinjen `Download failed: <reason>`. Kontrollér din internetforbindelse, og klik på `Download Missing Models` igen for at prøve på ny. Applikationen genoptager hentningen fra den sidst fuldført hentede fil.

## Ændring af præcision

Hvilke modelfiler der skal hentes, afhænger af den valgte **Model Precision**. For at ændre den skal du gå til `Settings… > Models > Model Precision`. Hvis du skifter præcision efter at have hentet filer, skal det nye sæt filer hentes separat. Se [Valg af modelvægtpræcision](model_precision.md).

---