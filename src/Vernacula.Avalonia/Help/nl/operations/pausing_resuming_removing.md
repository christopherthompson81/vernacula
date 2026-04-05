---
title: "Taken pauzeren, hervatten of verwijderen"
description: "Hoe u een actieve taak kunt pauzeren, een gestopte taak kunt hervatten of een taak uit de geschiedenis kunt verwijderen."
topic_id: operations_pausing_resuming_removing
---

# Taken pauzeren, hervatten of verwijderen

## Een taak pauzeren

U kunt een actieve of in de wachtrij staande taak op twee plaatsen pauzeren:

- **Voortgangsweergave** — klik op `Pause` in de rechterbenedenhoek terwijl u de actieve taak bekijkt.
- **Transcriptiegeschiedenistabel** — klik op `Pause` in de kolom **Actions** van een rij waarvan de status `running` of `queued` is.

Nadat u op `Pause` klikt, toont de statusregel `Pausing…` terwijl de toepassing de huidige verwerkingseenheid afrondt. De taakstatus verandert vervolgens in `cancelled` in de geschiedenistabel.

> Door te pauzeren worden alle tot nu toe getranscribeerde segmenten opgeslagen. U kunt de taak later hervatten zonder dat dit werk verloren gaat.

## Een taak hervatten

Een gepauzeerde of mislukte taak hervatten:

1. Zoek op het startscherm de taak op in de tabel **Transcription History**. De status is `cancelled` of `failed`.
2. Klik op `Resume` in de kolom **Actions**.
3. De toepassing keert terug naar de weergave **Progress** en gaat verder vanaf het punt waar de verwerking was gestopt.

De statusregel toont kort `Resuming…` terwijl de taak opnieuw wordt geïnitialiseerd.

## Een taak verwijderen

Een taak en het bijbehorende transcript permanent uit de geschiedenis verwijderen:

1. Klik in de tabel **Transcription History** op `Remove` in de kolom **Actions** van de taak die u wilt verwijderen.

De taak wordt uit de lijst verwijderd en de bijbehorende gegevens worden uit de lokale database gewist. Deze actie kan niet ongedaan worden gemaakt. Geëxporteerde bestanden die op schijf zijn opgeslagen, worden niet beïnvloed.

---