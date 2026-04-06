---
title: "Installera CUDA och cuDNN för GPU-acceleration"
description: "Hur du konfigurerar NVIDIA CUDA och cuDNN så att Vernacula-Desktop kan använda din GPU."
topic_id: first_steps_cuda_installation
---

# Installera CUDA och cuDNN för GPU-acceleration

Vernacula-Desktop kan använda en NVIDIA GPU för att påskynda transkriberingen avsevärt. GPU-acceleration kräver att NVIDIA CUDA Toolkit och cuDNN-körtidsbibliotek är installerade på ditt system.

## Krav

- En NVIDIA GPU med stöd för CUDA (GeForce GTX 10-serien eller senare rekommenderas).
- Windows 10 eller 11 (64-bitars).
- Modellfiler måste redan vara nedladdade. Se [Ladda ned modeller](downloading_models.md).

## Installationssteg

### 1. Installera CUDA Toolkit

Ladda ned och kör CUDA Toolkit-installationsprogrammet från NVIDIAs utvecklarwebbplats. Godkänn standardsökvägarna under installationen. Installationsprogrammet ställer automatiskt in miljövariabeln `CUDA_PATH` — Vernacula-Desktop använder den här variabeln för att hitta CUDA-biblioteken.

### 2. Installera cuDNN

Ladda ned cuDNN ZIP-arkivet för din installerade CUDA-version från NVIDIAs utvecklarwebbplats. Packa upp arkivet och kopiera innehållet i mapparna `bin`, `include` och `lib` till motsvarande mappar i installationskatalogen för CUDA Toolkit (den sökväg som visas av `CUDA_PATH`).

Du kan också installera cuDNN med NVIDIAs cuDNN-installationsprogram om ett sådant finns tillgängligt för din CUDA-version.

### 3. Starta om programmet

Stäng och öppna Vernacula-Desktop igen efter installationen. Programmet söker efter CUDA vid start.

## GPU-status i inställningarna

Öppna `Settings…` från menyraden och titta i avsnittet **Hardware & Performance**. Varje komponent visar en bockmarkering (✓) när den identifieras:

| Objekt | Vad det innebär |
|---|---|
| GPU-namn och VRAM | Din NVIDIA GPU hittades |
| CUDA Toolkit ✓ | CUDA-bibliotek hittades via `CUDA_PATH` |
| cuDNN ✓ | cuDNN-körtids-DLL:er hittades |
| CUDA Acceleration ✓ | ONNX Runtime laddade CUDA-exekveringsleverantören |

Om något objekt saknas efter installationen klickar du på `Re-check` för att köra maskinvaruidentifieringen igen utan att starta om programmet.

Inställningsfönstret innehåller även direkta nedladdningslänkar för CUDA Toolkit och cuDNN om de ännu inte är installerade.

### Felsökning

Om `CUDA Acceleration` inte visar en bockmarkering kontrollerar du att:

- Miljövariabeln `CUDA_PATH` är inställd (kontrollera `System > Advanced system settings > Environment Variables`).
- cuDNN-DLL:erna finns i en katalog som ingår i systemets `PATH` eller inuti CUDA-mappen `bin`.
- Din GPU-drivrutin är uppdaterad.

### Batchstorlek

När CUDA är aktivt visar avsnittet **Hardware & Performance** också det aktuella dynamiska batchgränsvärdet — det maximala antalet sekunders ljud som bearbetas i en GPU-körning. Det här värdet beräknas utifrån ledigt VRAM efter att modellerna har lästs in och justeras automatiskt om ditt tillgängliga minne förändras.

## Köra utan GPU

Om CUDA inte är tillgängligt faller Vernacula-Desktop automatiskt tillbaka på CPU-bearbetning. Transkriberingen fungerar fortfarande men går långsammare, särskilt för långa ljudfiler.

---