---
title: "Pozastavení, obnovení nebo odebrání úloh"
description: "Jak pozastavit spuštěnou úlohu, obnovit zastavenou nebo odstranit úlohu z historie."
topic_id: operations_pausing_resuming_removing
---

# Pozastavení, obnovení nebo odebrání úloh

## Pozastavení úlohy

Spuštěnou nebo zařazenou úlohu můžete pozastavit ze dvou míst:

- **Zobrazení průběhu** — klikněte na `Pause` v pravém dolním rohu při sledování aktivní úlohy.
- **Tabulka historie přepisů** — klikněte na `Pause` ve sloupci **Actions** u libovolného řádku, jehož stav je `running` nebo `queued`.

Po kliknutí na `Pause` se na stavovém řádku zobrazí `Pausing…`, dokud aplikace nedokončí aktuální zpracovávanou jednotku. Stav úlohy se poté v tabulce historie změní na `cancelled`.

> Pozastavení uloží všechny dosud přepsané segmenty. Úlohu můžete kdykoli později obnovit, aniž byste o tuto práci přišli.

## Obnovení úlohy

Postup obnovení pozastavené nebo neúspěšné úlohy:

1. Na domovské obrazovce vyhledejte úlohu v tabulce **Transcription History**. Její stav bude `cancelled` nebo `failed`.
2. Klikněte na `Resume` ve sloupci **Actions**.
3. Aplikace se vrátí do zobrazení **Progress** a pokračuje od místa, kde bylo zpracování přerušeno.

Stavový řádek krátce zobrazí `Resuming…`, zatímco se úloha znovu inicializuje.

## Odebrání úlohy

Postup trvalého odstranění úlohy a jejího přepisu z historie:

1. V tabulce **Transcription History** klikněte na `Remove` ve sloupci **Actions** u úlohy, kterou chcete odstranit.

Úloha je odstraněna ze seznamu a její data jsou vymazána z místní databáze. Tuto akci nelze vrátit zpět. Exportované soubory uložené na disku nejsou ovlivněny.

---