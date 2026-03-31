---
title: "Monitorovanie úloh"
description: "Ako sledovať priebeh spustenej alebo zaradenej úlohy."
topic_id: operations_monitoring_jobs
---

# Monitorovanie úloh

Zobrazenie **Priebeh** poskytuje živý náhľad na práve prebiehajúcu úlohu prepisu.

## Otvorenie zobrazenia Priebeh

- Po spustení nového prepisu aplikácia automaticky prejde do zobrazenia Priebeh.
- V prípade úlohy, ktorá už beží alebo je zaradená do frontu, ju nájdite v tabuľke **História prepisov** a kliknite na `Monitor` v jej stĺpci **Akcie**.

## Čítanie zobrazenia Priebeh

| Prvok | Popis |
|---|---|
| Ukazovateľ priebehu | Celkové percento dokončenia. Neurčitý (animovaný) počas spúšťania alebo obnovy úlohy. |
| Percentuálny údaj | Číselné percento zobrazené napravo od ukazovateľa. |
| Stavová správa | Aktuálna činnosť — napríklad `Audio Analysis` alebo `Speech Recognition`. Zobrazuje `Waiting in queue…`, ak úloha ešte nezačala. |
| Tabuľka segmentov | Živý záznam prepísaných segmentov so stĺpcami **Hovoriaci**, **Začiatok**, **Koniec** a **Obsah**. Automaticky sa posúva pri príchode nových segmentov. |

## Fázy priebehu

Zobrazené fázy závisejú od **Režimu segmentácie** zvoleného v nastaveniach.

**Režim diarizácie hovorcov** (predvolený):

1. **Audio Analysis** — diarizácia SortFormer prebehne nad celým súborom s cieľom identifikovať hranice hovorcov. Ukazovateľ môže zostať blízko 0 %, kým sa táto fáza nedokončí.
2. **Speech Recognition** — každý segment hovorcu je prepísaný. Percento počas tejto fázy plynule stúpa.

**Režim detekcie hlasovej aktivity**:

1. **Detecting speech segments** — Silero VAD prehľadá súbor a nájde oblasti reči. Táto fáza je rýchla.
2. **Speech Recognition** — každá zistená oblasť reči je prepísaná.

V oboch režimoch sa živá tabuľka segmentov vypĺňa súbežne s prebiehajúcim prepisom.

## Navigácia preč

Kliknite na `← Back to Home` a vráťte sa na domovskú obrazovku bez prerušenia úlohy. Úloha naďalej beží na pozadí a jej stav sa aktualizuje v tabuľke **História prepisov**. Kliknutím na `Monitor` sa kedykoľvek vrátite do zobrazenia Priebeh.

---