---
title: "CUDA en cuDNN installeren voor GPU-versnelling"
description: "Hoe u NVIDIA CUDA en cuDNN instelt zodat Parakeet Transcription uw GPU kan gebruiken."
topic_id: first_steps_cuda_installation
---

# CUDA en cuDNN installeren voor GPU-versnelling

Parakeet Transcription kan een NVIDIA GPU gebruiken om de transcriptie aanzienlijk te versnellen. GPU-versnelling vereist dat de NVIDIA CUDA Toolkit en de cuDNN-runtimebibliotheken op uw systeem zijn geïnstalleerd.

## Vereisten

- Een NVIDIA GPU met CUDA-ondersteuning (GeForce GTX 10-serie of nieuwer wordt aanbevolen).
- Windows 10 of 11 (64-bit).
- De modelbestanden moeten al zijn gedownload. Zie [Modellen downloaden](downloading_models.md).

## Installatiestappen

### 1. Installeer de CUDA Toolkit

Download en voer het installatieprogramma van de CUDA Toolkit uit vanaf de NVIDIA-developerwebsite. Accepteer tijdens de installatie de standaardpaden. Het installatieprogramma stelt de omgevingsvariabele `CUDA_PATH` automatisch in — Parakeet gebruikt deze variabele om de CUDA-bibliotheken te vinden.

### 2. Installeer cuDNN

Download het cuDNN ZIP-archief voor uw geïnstalleerde CUDA-versie van de NVIDIA-developerwebsite. Pak het archief uit en kopieer de inhoud van de mappen `bin`, `include` en `lib` naar de overeenkomstige mappen in de installatiemap van uw CUDA Toolkit (het pad dat wordt weergegeven door `CUDA_PATH`).

U kunt cuDNN ook installeren via het NVIDIA cuDNN-installatieprogramma, als dat beschikbaar is voor uw CUDA-versie.

### 3. Herstart de applicatie

Sluit Parakeet Transcription en open het opnieuw na de installatie. De applicatie controleert bij het opstarten op de aanwezigheid van CUDA.

## GPU-status in Instellingen

Open `Settings…` via de menubalk en bekijk het gedeelte **Hardware & Performance**. Elk onderdeel toont een vinkje (✓) wanneer het is gedetecteerd:

| Item | Betekenis |
|---|---|
| GPU-naam en VRAM | Uw NVIDIA GPU is gevonden |
| CUDA Toolkit ✓ | CUDA-bibliotheken gevonden via `CUDA_PATH` |
| cuDNN ✓ | cuDNN-runtime-DLL's gevonden |
| CUDA Acceleration ✓ | ONNX Runtime heeft de CUDA-uitvoeringsomgeving geladen |

Als een item ontbreekt na de installatie, klik dan op `Re-check` om de hardwaredetectie opnieuw uit te voeren zonder de applicatie te herstarten.

Het venster Instellingen biedt ook directe downloadkoppelingen voor de CUDA Toolkit en cuDNN als deze nog niet zijn geïnstalleerd.

### Probleemoplossing

Als `CUDA Acceleration` geen vinkje toont, controleer dan het volgende:

- De omgevingsvariabele `CUDA_PATH` is ingesteld (controleer via `System > Advanced system settings > Environment Variables`).
- De cuDNN-DLL's bevinden zich in een map die is opgenomen in uw systeem-`PATH`, of in de CUDA-map `bin`.
- Uw GPU-stuurprogramma is up-to-date.

### Batchgrootte

Wanneer CUDA actief is, toont het gedeelte **Hardware & Performance** ook het huidige dynamische batchmaximum — het maximale aantal seconden audio dat in één GPU-run wordt verwerkt. Deze waarde wordt berekend op basis van het vrije VRAM nadat de modellen zijn geladen en past zich automatisch aan als uw beschikbare geheugen verandert.

## Uitvoeren zonder GPU

Als CUDA niet beschikbaar is, schakelt Parakeet automatisch over op CPU-verwerking. Transcriptie werkt dan nog steeds, maar zal trager zijn, vooral bij lange audiobestanden.

---