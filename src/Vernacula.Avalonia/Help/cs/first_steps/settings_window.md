---
title: "Nastavení"
description: "Přehled všech možností v okně Nastavení."
topic_id: first_steps_settings_window
---

# Nastavení

Okno **Nastavení** umožňuje ovládat konfiguraci hardwaru, správu modelů, režim segmentace, chování editoru, vzhled a jazyk. Otevřete jej z panelu nabídek: `Settings…`.

## Hardware a výkon

Tato část zobrazuje stav vaší NVIDIA GPU a softwarového zásobníku CUDA a hlásí dynamický strop dávky používaný při přepisu pomocí GPU.

| Položka | Popis |
|---|---|
| Název GPU a VRAM | Detekovaná NVIDIA GPU a dostupná video paměť. |
| CUDA Toolkit | Zda byly nalezeny běhové knihovny CUDA prostřednictvím `CUDA_PATH`. |
| cuDNN | Zda jsou dostupné běhové knihovny DLL cuDNN. |
| Akcelerace CUDA | Zda ONNX Runtime úspěšně načetlo poskytovatele provádění CUDA. |

Kliknutím na `Re-check` znovu spustíte detekci hardwaru bez nutnosti restartovat aplikaci — užitečné po instalaci CUDA nebo cuDNN.

Přímé odkazy ke stažení CUDA Toolkit a cuDNN se zobrazí, pokud tyto komponenty nejsou detekovány.

Zpráva o **stropu dávky** udává, kolik sekund zvuku je zpracováno v každém běhu GPU. Tato hodnota se odvozuje z volné VRAM po načtení modelů a přizpůsobuje se automaticky.

Kompletní pokyny k nastavení CUDA naleznete v části [Instalace CUDA a cuDNN](cuda_installation.md).

## Modely

Tato část spravuje soubory modelů AI potřebné pro přepis.

- **Stáhnout chybějící modely** — stáhne všechny soubory modelů, které dosud nejsou uloženy na disku. Průběh stahování každého souboru je sledován ukazatelem průběhu a stavovým řádkem.
- **Zkontrolovat aktualizace** — zkontroluje, zda jsou dostupné novější váhy modelů. Na domovské obrazovce se také automaticky zobrazí banner s upozorněním na aktualizaci, pokud jsou zjištěny aktualizované váhy.

## Režim segmentace

Určuje způsob rozdělení zvuku do segmentů před rozpoznáváním řeči.

| Režim | Popis |
|---|---|
| **Diarizace mluvčích** | Používá model SortFormer k identifikaci jednotlivých mluvčích a označení každého segmentu. Nejvhodnější pro rozhovory, schůzky a nahrávky s více mluvčími. |
| **Detekce hlasové aktivity** | Používá Silero VAD k detekci pouze řečových úseků — bez označení mluvčích. Rychlejší než diarizace a vhodná pro zvuk s jediným mluvčím. |

## Editor přepisu

**Výchozí režim přehrávání** — nastavuje režim přehrávání používaný při otevření editoru přepisu. Kdykoli jej můžete změnit také přímo v editoru. Popis jednotlivých režimů naleznete v části [Úprava přepisů](../operations/editing_transcripts.md).

## Vzhled

Vyberte **Tmavý** nebo **Světlý** motiv. Změna se projeví okamžitě. Viz [Výběr motivu](theme.md).

## Jazyk

Vyberte jazyk zobrazení rozhraní aplikace. Změna se projeví okamžitě. Viz [Výběr jazyka](language.md).

---