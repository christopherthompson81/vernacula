---
title: "Přidání více zvukových souborů do fronty"
description: "Jak přidat několik zvukových souborů do fronty úloh najednou."
topic_id: operations_bulk_add_jobs
---

# Přidání více zvukových souborů do fronty

Pomocí funkce **Hromadné přidání úloh** můžete zařadit do fronty více zvukových nebo video souborů k přepisu v jednom kroku. Aplikace je zpracovává postupně jeden po druhém v pořadí, v jakém byly přidány.

## Předpoklady

- Všechny soubory modelu musí být staženy. Karta **Stav modelu** musí zobrazovat `All N model file(s) present ✓`. Viz [Stahování modelů](../first_steps/downloading_models.md).

## Jak hromadně přidat úlohy

1. Na domovské obrazovce klikněte na `Bulk Add Jobs`.
2. Otevře se okno pro výběr souborů. Vyberte jeden nebo více zvukových či video souborů — pro výběr více souborů podržte `Ctrl` nebo `Shift`.
3. Klikněte na **Otevřít**. Každý vybraný soubor bude přidán do tabulky **Historie přepisů** jako samostatná úloha.

> **Video soubory s více zvukovými stopami:** Pokud video soubor obsahuje více než jednu zvukovou stopu (například více jazyků nebo komentář režiséra), aplikace automaticky vytvoří jednu úlohu pro každou stopu.

## Názvy úloh

Každá úloha je automaticky pojmenována podle názvu příslušného zvukového souboru. Úlohu lze kdykoli přejmenovat kliknutím na její název ve sloupci **Název** v tabulce Historie přepisů, úpravou textu a stisknutím klávesy `Enter` nebo kliknutím jinam.

## Chování fronty

- Pokud žádná úloha právě neběží, první soubor se spustí okamžitě a zbývající jsou zobrazeny jako `queued`.
- Pokud již nějaká úloha běží, všechny nově přidané soubory jsou zobrazeny jako `queued` a budou spouštěny automaticky v pořadí.
- Chcete-li sledovat aktivní úlohu, klikněte na `Monitor` ve sloupci **Akce** příslušné úlohy. Viz [Sledování úloh](monitoring_jobs.md).
- Chcete-li pozastavit nebo odebrat úlohu ve frontě před jejím spuštěním, použijte tlačítka `Pause` nebo `Remove` ve sloupci **Akce** dané úlohy. Viz [Pozastavení, obnovení nebo odebrání úloh](pausing_resuming_removing.md).

---