---
title: "Nastavenia"
description: "Prehľad všetkých možností v okne Nastavenia."
topic_id: first_steps_settings_window
---

# Nastavenia

Okno **Nastavenia** vám umožňuje ovládať konfiguráciu hardvéru, správu modelov, režim segmentácie, správanie editora, vzhľad a jazyk. Otvorte ho z panela ponuky: `Settings…`.

## Hardvér a výkon

Táto časť zobrazuje stav vašej NVIDIA GPU a softvérového zásobníka CUDA a uvádza dynamický strop dávky používaný pri prepise pomocou GPU.

| Položka | Popis |
|---|---|
| Názov GPU a VRAM | Rozpoznaná NVIDIA GPU a dostupná video pamäť. |
| CUDA Toolkit | Či boli knižnice CUDA runtime nájdené prostredníctvom `CUDA_PATH`. |
| cuDNN | Či sú dostupné knižnice DLL cuDNN runtime. |
| Akcelerácia CUDA | Či ONNX Runtime úspešne načítal poskytovateľa spúšťania CUDA. |

Kliknite na `Re-check`, aby ste znovu spustili detekciu hardvéru bez reštartovania aplikácie — užitočné po inštalácii CUDA alebo cuDNN.

Priame odkazy na stiahnutie CUDA Toolkit a cuDNN sa zobrazia, keď tieto komponenty nie sú rozpoznané.

Správa o **strope dávky** udáva, koľko sekúnd zvuku sa spracuje v každom behu GPU. Táto hodnota sa odvodzuje od voľnej VRAM po načítaní modelov a upravuje sa automaticky.

Úplné pokyny na nastavenie CUDA nájdete v časti [Inštalácia CUDA a cuDNN](cuda_installation.md).

## Modely

Táto časť spravuje súbory modelov AI potrebné na prepis.

- **Stiahnuť chýbajúce modely** — stiahne všetky súbory modelov, ktoré ešte nie sú prítomné na disku. Priebeh sťahovania každého súboru sleduje panel priebehu a stavový riadok.
- **Skontrolovať aktualizácie** — skontroluje, či sú dostupné novšie váhy modelov. Banner s aktualizáciou sa tiež automaticky zobrazí na domovskej obrazovke, keď sú rozpoznané aktualizované váhy.

## Režim segmentácie

Ovláda, ako sa zvuk rozdeľuje do segmentov pred rozpoznávaním reči.

| Režim | Popis |
|---|---|
| **Diarizácia rečníkov** | Používa model SortFormer na identifikáciu jednotlivých rečníkov a označenie každého segmentu. Najvhodnejší pre rozhovory, stretnutia a nahrávky s viacerými rečníkmi. |
| **Detekcia hlasovej aktivity** | Používa Silero VAD na detekciu úsekov reči bez označenia rečníkov. Rýchlejšie ako diarizácia a vhodné pre zvuk s jedným rečníkom. |

## Editor prepisu

**Predvolený režim prehrávania** — nastaví režim prehrávania, ktorý sa použije pri otvorení editora prepisu. Môžete ho tiež kedykoľvek zmeniť priamo v editore. Popis každého režimu nájdete v časti [Úprava prepisov](../operations/editing_transcripts.md).

## Vzhľad

Vyberte **tmavý** alebo **svetlý** motív. Zmena sa prejaví okamžite. Pozrite si [Výber motívu](theme.md).

## Jazyk

Vyberte jazyk zobrazenia rozhrania aplikácie. Zmena sa prejaví okamžite. Pozrite si [Výber jazyka](language.md).

---