---
title: "Pridávanie viacerých zvukových súborov do fronty"
description: "Ako pridať niekoľko zvukových súborov do fronty úloh naraz."
topic_id: operations_bulk_add_jobs
---

# Pridávanie viacerých zvukových súborov do fronty

Použite funkciu **Hromadné pridanie úloh** na zaradenie viacerých zvukových alebo video súborov do fronty na prepis v jednom kroku. Aplikácia ich spracúva jeden po druhom v poradí, v akom boli pridané.

## Predpoklady

- Všetky súbory modelov musia byť stiahnuté. Karta **Stav modelu** musí zobrazovať `All N model file(s) present ✓`. Pozrite si [Sťahovanie modelov](../first_steps/downloading_models.md).

## Ako hromadne pridať úlohy

1. Na domovskej obrazovke kliknite na `Bulk Add Jobs`.
2. Otvorí sa výber súborov. Vyberte jeden alebo viacero zvukových alebo video súborov — podržte `Ctrl` alebo `Shift` na výber viacerých súborov.
3. Kliknite na **Otvoriť**. Každý vybraný súbor sa pridá do tabuľky **História prepisov** ako samostatná úloha.

> **Video súbory s viacerými zvukovými stopami:** Ak video súbor obsahuje viac než jednu zvukovú stopu (napríklad viacero jazykov alebo komentár režiséra), aplikácia automaticky vytvorí jednu úlohu pre každú stopu.

## Názvy úloh

Každá úloha je automaticky pomenovaná podľa názvu jej zvukového súboru. Úlohu môžete kedykoľvek premenovať kliknutím na jej názov v stĺpci **Názov** tabuľky História prepisov, upravením textu a stlačením `Enter` alebo kliknutím mimo poľa.

## Správanie fronty

- Ak momentálne neprebieha žiadna úloha, prvý súbor sa spustí okamžite a zvyšné sa zobrazia ako `queued`.
- Ak už nejaká úloha prebieha, všetky novo pridané súbory sa zobrazia ako `queued` a budú sa spúšťať automaticky v poradí.
- Ak chcete sledovať aktívnu úlohu, kliknite na `Monitor` v jej stĺpci **Akcie**. Pozrite si [Sledovanie úloh](monitoring_jobs.md).
- Ak chcete pozastaviť alebo odstrániť úlohu vo fronte skôr, ako sa spustí, použite tlačidlá `Pause` alebo `Remove` v jej stĺpci **Akcie**. Pozrite si [Pozastavenie, obnovenie alebo odstránenie úloh](pausing_resuming_removing.md).

---