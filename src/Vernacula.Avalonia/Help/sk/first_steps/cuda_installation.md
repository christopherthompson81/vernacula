---
title: "Inštalácia CUDA a cuDNN pre akceleráciu GPU"
description: "Ako nastaviť NVIDIA CUDA a cuDNN, aby Vernacula-Desktop mohol využívať váš GPU."
topic_id: first_steps_cuda_installation
---

# Inštalácia CUDA a cuDNN pre akceleráciu GPU

Vernacula-Desktop môže využívať NVIDIA GPU na výrazné zrýchlenie prepisu. Akcelerácia GPU vyžaduje, aby boli vo vašom systéme nainštalované NVIDIA CUDA Toolkit a knižnice runtime cuDNN.

## Požiadavky

- NVIDIA GPU s podporou CUDA (odporúča sa GeForce GTX série 10 alebo novší).
- Windows 10 alebo 11 (64-bit).
- Súbory modelov musia byť už stiahnuté. Pozrite si [Sťahovanie modelov](downloading_models.md).

## Kroky inštalácie

### 1. Nainštalujte CUDA Toolkit

Stiahnite a spustite inštalačný program CUDA Toolkit z webovej stránky pre vývojárov NVIDIA. Počas inštalácie prijmite predvolené cesty. Inštalačný program automaticky nastaví premennú prostredia `CUDA_PATH` — Vernacula-Desktop používa túto premennú na vyhľadanie knižníc CUDA.

### 2. Nainštalujte cuDNN

Z webovej stránky pre vývojárov NVIDIA si stiahnite archív ZIP cuDNN pre vašu nainštalovanú verziu CUDA. Rozbaľte archív a skopírujte obsah jeho priečinkov `bin`, `include` a `lib` do zodpovedajúcich priečinkov v adresári inštalácie CUDA Toolkit (cesta zobrazená premennou `CUDA_PATH`).

Prípadne nainštalujte cuDNN pomocou inštalačného programu NVIDIA cuDNN, ak je dostupný pre vašu verziu CUDA.

### 3. Reštartujte aplikáciu

Po dokončení inštalácie zatvorte a znovu otvorte Vernacula-Desktop. Aplikácia kontroluje prítomnosť CUDA pri spustení.

## Stav GPU v nastaveniach

Otvorte `Settings…` z panela ponuky a pozrite si sekciu **Hardware & Performance**. Každá položka zobrazí začiarknutie (✓), keď je detekovaná:

| Položka | Čo to znamená |
|---|---|
| Názov GPU a VRAM | Bol nájdený váš NVIDIA GPU |
| CUDA Toolkit ✓ | Knižnice CUDA boli nájdené prostredníctvom `CUDA_PATH` |
| cuDNN ✓ | Boli nájdené runtime DLL súbory cuDNN |
| CUDA Acceleration ✓ | ONNX Runtime načítal poskytovateľa spustenia CUDA |

Ak niektorá položka po inštalácii chýba, kliknite na `Re-check`, čím znovu spustíte detekciu hardvéru bez reštartovania aplikácie.

Okno Nastavenia tiež poskytuje priame odkazy na stiahnutie CUDA Toolkit a cuDNN, ak ešte nie sú nainštalované.

### Riešenie problémov

Ak `CUDA Acceleration` nezobrazuje začiarknutie, overte, že:

- Premenná prostredia `CUDA_PATH` je nastavená (skontrolujte `System > Advanced system settings > Environment Variables`).
- Súbory DLL cuDNN sa nachádzajú v adresári uvedenom v systémovej premennej `PATH` alebo v priečinku `bin` inštalácie CUDA.
- Ovládač vášho GPU je aktuálny.

### Veľkosť dávky

Keď je CUDA aktívna, sekcia **Hardware & Performance** zobrazuje aj aktuálny dynamický strop dávky — maximálny počet sekúnd zvuku spracovaného v jednom behu GPU. Táto hodnota sa vypočíta z voľnej pamäte VRAM po načítaní modelov a automaticky sa prispôsobuje pri zmene dostupnej pamäte.

## Spustenie bez GPU

Ak CUDA nie je dostupná, Vernacula-Desktop automaticky prejde na spracovanie pomocou CPU. Prepis bude stále fungovať, avšak bude pomalší, najmä pri dlhých zvukových súboroch.

---