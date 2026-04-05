---
title: "CUDA ja cuDNN installimine GPU-kiirenduse jaoks"
description: "Kuidas seadistada NVIDIA CUDA ja cuDNN, et Parakeet Transcription saaks kasutada teie GPU-d."
topic_id: first_steps_cuda_installation
---

# CUDA ja cuDNN installimine GPU-kiirenduse jaoks

Parakeet Transcription saab kasutada NVIDIA GPU-d transkriptsiooni oluliseks kiirendamiseks. GPU-kiirendus nõuab, et teie süsteemi oleksid installitud NVIDIA CUDA Toolkit ja cuDNN käitusaegsed teegid.

## Nõuded

- NVIDIA GPU, mis toetab CUDA-t (soovitatav on GeForce GTX 10-seeria või uuem).
- Windows 10 või 11 (64-bitine).
- Mudelifailid peavad olema juba alla laaditud. Vt [Mudelite allalaadimine](downloading_models.md).

## Installimise sammud

### 1. Installige CUDA Toolkit

Laadige NVIDIA arendaja veebisaidilt alla CUDA Toolkiti installija ja käivitage see. Installimise ajal nõustuge vaikimisi kaustadega. Installija seab `CUDA_PATH` keskkonnamuutuja automaatselt — Parakeet kasutab seda muutujat CUDA teekide leidmiseks.

### 2. Installige cuDNN

Laadige NVIDIA arendaja veebisaidilt alla teie installitud CUDA versioonile vastav cuDNN ZIP-arhiiv. Pakkige arhiiv lahti ja kopeerige selle `bin`-, `include`- ja `lib`-kaustade sisu CUDA Toolkiti installikausta vastavatesse kaustadesse (tee, mida näitab `CUDA_PATH`).

Teise võimalusena installige cuDNN NVIDIA cuDNN installija abil, kui see on teie CUDA versiooni jaoks saadaval.

### 3. Taaskäivitage rakendus

Sulgege ja avage Parakeet Transcription pärast installimist uuesti. Rakendus kontrollib CUDA olemasolu käivitamisel.

## GPU olek seadetes

Avage menüüribalt `Settings…` ja vaadake jaotist **Hardware & Performance**. Iga komponent näitab linnukest (✓), kui see on tuvastatud:

| Üksus | Tähendus |
|---|---|
| GPU nimi ja VRAM | Teie NVIDIA GPU leiti |
| CUDA Toolkit ✓ | CUDA teegid leiti `CUDA_PATH` kaudu |
| cuDNN ✓ | cuDNN käitusaegsed DLL-failid leiti |
| CUDA Acceleration ✓ | ONNX Runtime laadis CUDA täitumisteenuse pakkuja |

Kui mõni üksus puudub pärast installimist, klõpsake `Re-check`, et käivitada riistvara tuvastamine uuesti ilma rakendust taaskäivitamata.

Seadete aken pakub ka otselingid CUDA Toolkiti ja cuDNN allalaadimiseks, kui need pole veel installitud.

### Tõrkeotsing

Kui `CUDA Acceleration` ei näita linnukest, kontrollige järgmist:

- `CUDA_PATH` keskkonnamuutuja on seatud (kontrollige `System > Advanced system settings > Environment Variables`).
- cuDNN DLL-failid asuvad teie süsteemi `PATH`-is olevas kaustas või CUDA `bin`-kaustas.
- Teie GPU draiver on ajakohane.

### Pakktöötluse suurus

Kui CUDA on aktiivne, kuvab jaotis **Hardware & Performance** ka praeguse dünaamilise pakktöötluse ülempiiri — maksimaalsed sekundid helifailist, mida töödeldakse ühes GPU-käivituses. See väärtus arvutatakse pärast mudelite laadimist vabast VRAM-ist ja kohandub automaatselt, kui saadaolev mälumaht muutub.

## Töötamine ilma GPU-ta

Kui CUDA ei ole saadaval, lülitub Parakeet automaatselt CPU-töötlusele. Transkriptsioon töötab endiselt, kuid on aeglasem, eriti pikkade helifailide puhul.

---