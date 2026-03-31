---
title: "Nieuwe transcriptieworkflow"
description: "Stapsgewijze handleiding voor het transcriberen van een audiobestand."
topic_id: operations_new_transcription
---

# Nieuwe transcriptieworkflow

Gebruik deze workflow om een enkel audiobestand te transcriberen.

## Vereisten

- Alle modelbestanden moeten zijn gedownload. De kaart **Modelstatus** moet `All N model file(s) present ✓` weergeven. Zie [Modellen downloaden](../first_steps/downloading_models.md).

## Ondersteunde indelingen

### Audio

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Videobestanden worden gedecodeerd via FFmpeg. Als een videobestand **meerdere audiostreams** bevat (bijvoorbeeld meerdere talen of commentaarsporen), wordt er automatisch één transcroptietaak aangemaakt voor elke stream.

## Stappen

### 1. Open het formulier Nieuwe transcriptie

Klik op `New Transcription` op het startscherm, of ga naar `File > New Transcription`.

### 2. Selecteer een mediabestand

Klik op `Browse…` naast het veld **Audio File**. Er wordt een bestandskiezer geopend die gefilterd is op ondersteunde audio- en video-indelingen. Selecteer uw bestand en klik op **Open**. Het bestandspad verschijnt in het veld.

### 3. Geef de taak een naam

Het veld **Job Name** wordt vooraf ingevuld op basis van de bestandsnaam. Bewerk het als u een ander label wilt — deze naam wordt weergegeven in de transcriptiegeschiedenis op het startscherm.

### 4. Start de transcriptie

Klik op `Start Transcription`. De applicatie schakelt over naar de weergave **Progress**.

Klik op `← Back` om terug te gaan zonder te starten.

## Wat er daarna gebeurt

De taak doorloopt twee fasen die worden weergegeven in de voortgangsbalk:

1. **Audio Analysis** — sprekersdiarisatie: bepalen wie er spreekt en wanneer.
2. **Speech Recognition** — spraak segment voor segment omzetten naar tekst.

Getranscribeerde segmenten verschijnen in de live tabel zodra ze worden gegenereerd. Wanneer de verwerking is voltooid, gaat de applicatie automatisch over naar de weergave **Results**.

Als u een taak toevoegt terwijl er al een andere actief is, krijgt de nieuwe taak de status `queued` en wordt deze gestart zodra de huidige taak is voltooid. Zie [Taken bewaken](monitoring_jobs.md).

---