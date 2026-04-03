---
title: "Sledování úloh"
description: "Jak sledovat průběh spuštěné nebo čekající úlohy."
topic_id: operations_monitoring_jobs
---

# Sledování úloh

Zobrazení **Průběh** poskytuje živý náhled na právě probíhající úlohu přepisu.

## Otevření zobrazení Průběh

- Po spuštění nového přepisu přejde aplikace do zobrazení Průběh automaticky.
- Pro úlohu, která již běží nebo čeká ve frontě, ji najděte v tabulce **Historie přepisů** a klikněte na `Monitor` ve sloupci **Akce**.

## Čtení zobrazení Průběh

| Prvek | Popis |
|---|---|
| Ukazatel průběhu | Celkové procento dokončení. Zobrazuje se jako neurčitý (animovaný) při spouštění nebo obnovování úlohy. |
| Popisek procent | Číselné procento zobrazené napravo od ukazatele. |
| Stavová zpráva | Aktuální aktivita – například `Audio Analysis` nebo `Speech Recognition`. Zobrazuje `Waiting in queue…`, pokud úloha ještě nezačala. |
| Tabulka segmentů | Živý přehled přepsaných segmentů se sloupci **Mluvčí**, **Začátek**, **Konec** a **Obsah**. Automaticky se posouvá při příchodu nových segmentů. |

## Fáze průběhu

Zobrazené fáze závisí na **Režimu segmentace** zvoleném v nastavení.

**Režim diarizace mluvčích** (výchozí):

1. **Audio Analysis** — Sortformer spustí diarizaci nad celým souborem, aby identifikoval hranice mluvčích. Ukazatel může zůstat blízko 0 %, dokud tato fáze neskončí.
2. **Speech Recognition** — každý segment mluvčího je přepsán. Procento v této fázi plynule roste.

**Režim detekce hlasové aktivity**:

1. **Detecting speech segments** — Silero VAD prohledá soubor a nalezne oblasti s řečí. Tato fáze je rychlá.
2. **Speech Recognition** — každá detekovaná oblast řeči je přepsána.

V obou režimech se živá tabulka segmentů plní průběžně s postupem přepisu.

## Přechod na jinou obrazovku

Kliknutím na `← Back to Home` se vrátíte na domovskou obrazovku bez přerušení úlohy. Úloha pokračuje na pozadí a její stav se aktualizuje v tabulce **Historie přepisů**. Kdykoli znovu klikněte na `Monitor` a vrátíte se do zobrazení Průběh.

---