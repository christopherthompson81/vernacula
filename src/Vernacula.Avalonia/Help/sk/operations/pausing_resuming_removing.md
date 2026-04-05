---
title: "Pozastavenie, obnovenie alebo odstránenie úloh"
description: "Ako pozastaviť spustenú úlohu, obnoviť zastavenú alebo odstrániť úlohu z histórie."
topic_id: operations_pausing_resuming_removing
---

# Pozastavenie, obnovenie alebo odstránenie úloh

## Pozastavenie úlohy

Spustenú alebo zaradenú úlohu môžete pozastaviť z dvoch miest:

- **Zobrazenie priebehu** — kliknite na `Pause` v pravom dolnom rohu počas sledovania aktívnej úlohy.
- **Tabuľka histórie prepisov** — kliknite na `Pause` v stĺpci **Actions** v riadku, ktorého stav je `running` alebo `queued`.

Po kliknutí na `Pause` sa na stavovom riadku zobrazí `Pausing…`, kým aplikácia dokončí aktuálnu spracovávanú jednotku. Stav úlohy sa potom v tabuľke histórie zmení na `cancelled`.

> Pozastavením sa uložia všetky doteraz prepísané segmenty. Úlohu môžete neskôr obnoviť bez straty tejto práce.

## Obnovenie úlohy

Postup obnovenia pozastavenej alebo neúspešnej úlohy:

1. Na domovskej obrazovke vyhľadajte úlohu v tabuľke **Transcription History**. Jej stav bude `cancelled` alebo `failed`.
2. Kliknite na `Resume` v stĺpci **Actions**.
3. Aplikácia sa vráti do zobrazenia **Progress** a pokračuje od miesta, kde sa spracovanie zastavilo.

Na stavovom riadku sa krátko zobrazí `Resuming…`, kým sa úloha znovu inicializuje.

## Odstránenie úlohy

Postup trvalého odstránenia úlohy a jej prepisu z histórie:

1. V tabuľke **Transcription History** kliknite na `Remove` v stĺpci **Actions** pri úlohe, ktorú chcete odstrániť.

Úloha sa odstráni zo zoznamu a jej údaje sa vymažú z lokálnej databázy. Túto akciu nie je možné vrátiť späť. Exportované súbory uložené na disku nie sú ovplyvnené.

---