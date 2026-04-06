---
title: "Installation af CUDA og cuDNN til GPU-acceleration"
description: "Sådan opsætter du NVIDIA CUDA og cuDNN, så Vernacula-Desktop kan bruge din GPU."
topic_id: first_steps_cuda_installation
---

# Installation af CUDA og cuDNN til GPU-acceleration

Vernacula-Desktop kan bruge en NVIDIA GPU til at accelerere transskription betydeligt. GPU-acceleration kræver, at NVIDIA CUDA Toolkit og cuDNN-runtimebiblioteker er installeret på dit system.

## Krav

- En NVIDIA GPU med understøttelse af CUDA (GeForce GTX 10-serien eller nyere anbefales).
- Windows 10 eller 11 (64-bit).
- Modelfiler skal allerede være downloadet. Se [Download af modeller](downloading_models.md).

## Installationstrin

### 1. Installer CUDA Toolkit

Download og kør CUDA Toolkit-installationsprogrammet fra NVIDIA's udviklerwebsted. Accepter standardstierne under installationen. Installationsprogrammet indstiller automatisk miljøvariablen `CUDA_PATH` — Vernacula-Desktop bruger denne variabel til at finde CUDA-bibliotekerne.

### 2. Installer cuDNN

Download cuDNN ZIP-arkivet til din installerede CUDA-version fra NVIDIA's udviklerwebsted. Udpak arkivet, og kopiér indholdet af mapperne `bin`, `include` og `lib` til de tilsvarende mapper i din CUDA Toolkit-installationsmappe (den sti, der vises af `CUDA_PATH`).

Du kan også installere cuDNN ved hjælp af NVIDIA cuDNN-installationsprogrammet, hvis et sådant er tilgængeligt til din CUDA-version.

### 3. Genstart programmet

Luk og åbn Vernacula-Desktop igen efter installationen. Programmet kontrollerer for CUDA ved opstart.

## GPU-status i indstillinger

Åbn `Settings…` fra menulinjen, og se på afsnittet **Hardware & Performance**. Hvert element viser et flueben (✓), når det registreres:

| Element | Hvad det betyder |
|---|---|
| GPU-navn og VRAM | Din NVIDIA GPU blev fundet |
| CUDA Toolkit ✓ | CUDA-biblioteker fundet via `CUDA_PATH` |
| cuDNN ✓ | cuDNN-runtime-DLL'er fundet |
| CUDA Acceleration ✓ | ONNX Runtime indlæste CUDA-udførelsesudbyder |

Hvis et element mangler efter installationen, skal du klikke på `Re-check` for at køre hardwareregistrering igen uden at genstarte programmet.

Indstillingsvinduet indeholder også direkte downloadlinks til CUDA Toolkit og cuDNN, hvis de endnu ikke er installeret.

### Fejlfinding

Hvis `CUDA Acceleration` ikke viser et flueben, skal du kontrollere, at:

- Miljøvariablen `CUDA_PATH` er indstillet (tjek `System > Advanced system settings > Environment Variables`).
- cuDNN-DLL'erne befinder sig i en mappe på dit systems `PATH` eller i CUDA's `bin`-mappe.
- Din GPU-driver er opdateret.

### Batchstørrelse

Når CUDA er aktivt, viser afsnittet **Hardware & Performance** også det aktuelle dynamiske batchloft — det maksimale antal sekunders lyd, der behandles i én GPU-kørsel. Denne værdi beregnes ud fra ledig VRAM, efter at modeller er indlæst, og justeres automatisk, hvis din tilgængelige hukommelse ændres.

## Kørsel uden GPU

Hvis CUDA ikke er tilgængeligt, falder Vernacula-Desktop automatisk tilbage til CPU-behandling. Transskription fungerer stadig, men vil være langsommere, især for lange lydfiler.

---