---
title: "Instalace CUDA a cuDNN pro akceleraci GPU"
description: "Jak nastavit NVIDIA CUDA a cuDNN, aby Vernacula-Desktop mohlo využívat váš GPU."
topic_id: first_steps_cuda_installation
---

# Instalace CUDA a cuDNN pro akceleraci GPU

Vernacula-Desktop může využívat NVIDIA GPU k výraznému urychlení přepisu. Akcelerace GPU vyžaduje, aby byly na vašem systému nainstalovány NVIDIA CUDA Toolkit a runtime knihovny cuDNN.

## Požadavky

- NVIDIA GPU s podporou CUDA (doporučuje se GeForce GTX 10. generace nebo novější).
- Windows 10 nebo 11 (64bitový).
- Soubory modelů musí být již staženy. Viz [Stahování modelů](downloading_models.md).

## Postup instalace

### 1. Nainstalujte CUDA Toolkit

Stáhněte a spusťte instalační program CUDA Toolkit z webu NVIDIA pro vývojáře. Během instalace přijměte výchozí cesty. Instalátor automaticky nastaví proměnnou prostředí `CUDA_PATH` — Vernacula-Desktop tuto proměnnou používá k vyhledání knihoven CUDA.

### 2. Nainstalujte cuDNN

Stáhněte archiv ZIP cuDNN pro vaši nainstalovanou verzi CUDA z webu NVIDIA pro vývojáře. Rozbalte archiv a zkopírujte obsah složek `bin`, `include` a `lib` do odpovídajících složek v instalačním adresáři CUDA Toolkit (cesta zobrazená v `CUDA_PATH`).

Případně nainstalujte cuDNN pomocí instalačního programu NVIDIA cuDNN, pokud je pro vaši verzi CUDA k dispozici.

### 3. Restartujte aplikaci

Po instalaci zavřete a znovu otevřete Vernacula-Desktop. Aplikace při spuštění zkontroluje přítomnost CUDA.

## Stav GPU v nastavení

Otevřete `Settings…` z panelu nabídek a podívejte se na část **Hardware & Performance**. U každé součásti se zobrazí zaškrtnutí (✓), pokud byla rozpoznána:

| Položka | Co znamená |
|---|---|
| Název GPU a VRAM | Váš NVIDIA GPU byl nalezen |
| CUDA Toolkit ✓ | Knihovny CUDA nalezeny prostřednictvím `CUDA_PATH` |
| cuDNN ✓ | Runtime DLL soubory cuDNN nalezeny |
| CUDA Acceleration ✓ | ONNX Runtime načetl poskytovatele spuštění CUDA |

Pokud po instalaci některá položka chybí, klikněte na `Re-check` a znovu spusťte detekci hardwaru bez restartu aplikace.

Okno Nastavení také obsahuje přímé odkazy ke stažení CUDA Toolkit a cuDNN, pokud ještě nejsou nainstalovány.

### Řešení problémů

Pokud `CUDA Acceleration` nezobrazuje zaškrtnutí, ověřte, zda:

- Je nastavena proměnná prostředí `CUDA_PATH` (zkontrolujte `System > Advanced system settings > Environment Variables`).
- Soubory DLL cuDNN jsou umístěny v adresáři na systémové proměnné `PATH` nebo ve složce `bin` CUDA.
- Ovladač vašeho GPU je aktuální.

### Velikost dávky

Pokud je CUDA aktivní, zobrazuje část **Hardware & Performance** také aktuální dynamický strop dávky — maximální počet sekund zvuku zpracovaného v jednom běhu GPU. Tato hodnota se vypočítává z volné paměti VRAM po načtení modelů a automaticky se přizpůsobuje, pokud se změní dostupná paměť.

## Provoz bez GPU

Pokud CUDA není k dispozici, Vernacula-Desktop automaticky přepne na zpracování pomocí CPU. Přepis stále funguje, ale bude pomalejší, zejména u dlouhých zvukových souborů.

---