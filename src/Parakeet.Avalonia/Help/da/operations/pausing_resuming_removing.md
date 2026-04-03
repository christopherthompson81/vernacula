---
title: "Pause, genoptag eller fjern job"
description: "Sådan pauser du et kørende job, genoptager et stoppet job eller sletter et job fra historikken."
topic_id: operations_pausing_resuming_removing
---

# Pause, genoptag eller fjern job

## Pause et job

Du kan pause et kørende eller sat-i-kø job fra to steder:

- **Statusvisning** — klik på `Pause` i nederste højre hjørne, mens du følger det aktive job.
- **Transskriptionshistoriktabellen** — klik på `Pause` i kolonnen **Handlinger** på en hvilken som helst række, hvis status er `running` eller `queued`.

Når du klikker på `Pause`, viser statuslinjen `Pausing…`, mens programmet afslutter den aktuelle behandlingsenhed. Jobbets status ændres derefter til `cancelled` i historiktabellen.

> Pausefunktionen gemmer alle segmenter, der er transskriberet indtil videre. Du kan genoptage jobbet senere uden at miste dette arbejde.

## Genoptag et job

Sådan genoptager du et pauseret eller mislykket job:

1. Find jobbet i tabellen **Transskriptionshistorik** på startskærmen. Dets status vil være `cancelled` eller `failed`.
2. Klik på `Resume` i kolonnen **Handlinger**.
3. Programmet vender tilbage til visningen **Status** og fortsætter fra det punkt, hvor behandlingen stoppede.

Statuslinjen viser kortvarigt `Resuming…`, mens jobbet initialiseres igen.

## Fjern et job

Sådan sletter du permanent et job og dets transskription fra historikken:

1. Klik på `Remove` i kolonnen **Handlinger** for det job, du vil slette, i tabellen **Transskriptionshistorik**.

Jobbet fjernes fra listen, og dets data slettes fra den lokale database. Denne handling kan ikke fortrydes. Eksporterede filer, der er gemt på disken, påvirkes ikke.

---