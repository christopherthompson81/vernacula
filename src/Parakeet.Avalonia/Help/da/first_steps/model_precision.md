---
title: "Valg af modelvægtpræcision"
description: "Sådan vælger du mellem INT8 og FP32 modelpræcision, og hvad afvejningerne er."
topic_id: first_steps_model_precision
---

# Valg af modelvægtpræcision

Modelpræcision styrer det numeriske format, der bruges af AI-modellens vægte. Det påvirker downloadstørrelse, hukommelsesforbrug og nøjagtighed.

## Præcisionsindstillinger

### INT8 (mindre download)

- Mindre modelfiler — hurtigere at downloade og kræver mindre diskplads.
- Lidt lavere nøjagtighed på noget lyd.
- Anbefales, hvis du har begrænset diskplads eller en langsommere internetforbindelse.

### FP32 (mere nøjagtig)

- Større modelfiler.
- Højere nøjagtighed, særligt ved vanskelig lyd med accenter eller baggrundsstøj.
- Anbefales, når nøjagtighed er prioriteten, og du har tilstrækkelig diskplads.
- Påkrævet for CUDA GPU-acceleration — GPU-stien bruger altid FP32 uanset denne indstilling.

## Sådan ændrer du præcision

Åbn `Settings…` fra menulinjen, gå derefter til afsnittet **Models**, og vælg enten `INT8 (smaller download)` eller `FP32 (more accurate)`.

## Efter ændring af præcision

Ændring af præcision kræver et andet sæt modelfiler. Hvis modellerne for den nye præcision endnu ikke er blevet downloadet, skal du klikke på `Download Missing Models` i Indstillinger. Tidligere downloadede filer for den anden præcision bevares på disken og behøver ikke at blive downloadet igen, hvis du skifter tilbage.

---