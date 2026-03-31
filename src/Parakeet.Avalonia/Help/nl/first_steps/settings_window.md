---
title: "Instellingen"
description: "Overzicht van alle opties in het venster Instellingen."
topic_id: first_steps_settings_window
---

# Instellingen

Het venster **Instellingen** geeft u controle over de hardwareconfiguratie, het modelbeheer, de segmentatiemodus, het gedrag van de editor, de weergave en de taal. Open het via de menubalk: `Settings…`.

## Hardware en prestaties

Dit gedeelte toont de status van uw NVIDIA GPU en de CUDA-softwarestack, en meldt het dynamische batchplafond dat wordt gebruikt tijdens GPU-transcriptie.

| Item | Beschrijving |
|---|---|
| GPU-naam en VRAM | Gedetecteerde NVIDIA GPU en beschikbaar videogeheugen. |
| CUDA Toolkit | Of de CUDA-runtimebibliotheken zijn gevonden via `CUDA_PATH`. |
| cuDNN | Of de cuDNN-runtime-DLL's beschikbaar zijn. |
| CUDA-versnelling | Of ONNX Runtime de CUDA-uitvoeringsaanbieder met succes heeft geladen. |

Klik op `Re-check` om de hardwaredetectie opnieuw uit te voeren zonder de applicatie opnieuw te starten — handig na het installeren van CUDA of cuDNN.

Directe downloadkoppelingen voor de CUDA Toolkit en cuDNN worden weergegeven wanneer deze componenten niet worden gedetecteerd.

Het bericht over het **batchplafond** geeft aan hoeveel seconden audio er in elke GPU-run worden verwerkt. Deze waarde wordt afgeleid van het vrije VRAM nadat de modellen zijn geladen en past zich automatisch aan.

Zie [CUDA en cuDNN installeren](cuda_installation.md) voor volledige installatie-instructies voor CUDA.

## Modellen

Dit gedeelte beheert de AI-modelbestanden die nodig zijn voor transcriptie.

- **Modelnauwkeurigheid** — kies `INT8 (smaller download)` of `FP32 (more accurate)`. Zie [De precisie van modelgewichten kiezen](model_precision.md).
- **Ontbrekende modellen downloaden** — downloadt modelbestanden die nog niet aanwezig zijn op de schijf. Een voortgangsbalk en een statusregel volgen elk bestand tijdens het downloaden.
- **Controleren op updates** — controleert of er nieuwere modelgewichten beschikbaar zijn. Er verschijnt ook automatisch een updatebanner op het startscherm wanneer bijgewerkte gewichten worden gedetecteerd.

## Segmentatiemodus

Bepaalt hoe de audio wordt opgedeeld in segmenten vóór spraakherkenning.

| Modus | Beschrijving |
|---|---|
| **Sprekersegmentatie** | Gebruikt het SortFormer-model om individuele sprekers te identificeren en elk segment te labelen. Het meest geschikt voor interviews, vergaderingen en opnamen met meerdere sprekers. |
| **Spraakactiviteitsdetectie** | Gebruikt Silero VAD om alleen spraakgedeelten te detecteren — zonder sprekerlabels. Sneller dan sprekersegmentatie en goed geschikt voor audio met één spreker. |

## Transcriptie-editor

**Standaard afspeelmodus** — stelt de afspeelmodus in die wordt gebruikt wanneer u de transcriptie-editor opent. U kunt deze ook op elk moment rechtstreeks in de editor wijzigen. Zie [Transcripties bewerken](../operations/editing_transcripts.md) voor een beschrijving van elke modus.

## Weergave

Selecteer het thema **Donker** of **Licht**. De wijziging wordt onmiddellijk toegepast. Zie [Een thema kiezen](theme.md).

## Taal

Selecteer de weergavetaal voor de applicatie-interface. De wijziging wordt onmiddellijk toegepast. Zie [Een taal kiezen](language.md).

---