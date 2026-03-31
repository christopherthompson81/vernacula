---
title: "Meerdere audiobestanden in de wachtrij plaatsen"
description: "Hoe u meerdere audiobestanden tegelijk aan de taakrij toevoegt."
topic_id: operations_bulk_add_jobs
---

# Meerdere audiobestanden in de wachtrij plaatsen

Gebruik **Bulk toevoegen** om meerdere audio- of videobestanden in één stap in de wachtrij te plaatsen voor transcriptie. De toepassing verwerkt ze één voor één in de volgorde waarin ze zijn toegevoegd.

## Vereisten

- Alle modelbestanden moeten gedownload zijn. De kaart **Modelstatus** moet `All N model file(s) present ✓` weergeven. Zie [Modellen downloaden](../first_steps/downloading_models.md).

## Meerdere taken tegelijk toevoegen

1. Klik op het startscherm op `Bulk Add Jobs`.
2. Er opent zich een bestandskiezer. Selecteer een of meer audio- of videobestanden — houd `Ctrl` of `Shift` ingedrukt om meerdere bestanden te selecteren.
3. Klik op **Openen**. Elk geselecteerd bestand wordt als een afzonderlijke taak toegevoegd aan de tabel **Transcriptiegeschiedenis**.

> **Videobestanden met meerdere audiostreams:** Als een videobestand meer dan één audiostream bevat (bijvoorbeeld meerdere talen of een regisseurscommentaartrack), maakt de toepassing automatisch één taak per stream aan.

## Taaknamen

Elke taak krijgt automatisch de naam van het bijbehorende audiobestand. U kunt een taak op elk moment hernoemen door op de naam in de kolom **Titel** van de tabel Transcriptiegeschiedenis te klikken, de tekst te bewerken en op `Enter` te drukken of ergens anders te klikken.

## Wachtrijgedrag

- Als er momenteel geen taak actief is, start het eerste bestand onmiddellijk en worden de overige weergegeven als `queued`.
- Als er al een taak actief is, worden alle nieuw toegevoegde bestanden weergegeven als `queued` en starten ze automatisch in volgorde.
- Om de actieve taak te bewaken, klikt u op `Monitor` in de kolom **Acties**. Zie [Taken bewaken](monitoring_jobs.md).
- Om een taak in de wachtrij te pauzeren of te verwijderen voordat deze start, gebruikt u de knoppen `Pause` of `Remove` in de kolom **Acties**. Zie [Taken pauzeren, hervatten of verwijderen](pausing_resuming_removing.md).

---