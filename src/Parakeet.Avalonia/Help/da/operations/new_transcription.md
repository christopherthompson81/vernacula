---
title: "Ny transskriptionsarbejdsgang"
description: "Trin-for-trin-guide til transskription af en lydfil."
topic_id: operations_new_transcription
---

# Ny transskriptionsarbejdsgang

Brug denne arbejdsgang til at transskribere en enkelt lydfil.

## Forudsætninger

- Alle modelfiler skal være downloadet. Kortet **Modelstatus** skal vise `All N model file(s) present ✓`. Se [Download af modeller](../first_steps/downloading_models.md).

## Understøttede formater

### Lyd

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Videofiler afkodes via FFmpeg. Hvis en videofil indeholder **flere lydspor** (f.eks. flere sprog eller kommentarspor), oprettes der automatisk ét transskriptionsjob for hvert spor.

## Trin

### 1. Åbn formularen Ny transskription

Klik på `New Transcription` på startskærmen, eller gå til `File > New Transcription`.

### 2. Vælg en mediefil

Klik på `Browse…` ud for feltet **Audio File**. Der åbnes en filvælger, som er filtreret til understøttede lyd- og videoformater. Vælg din fil, og klik på **Open**. Filstien vises i feltet.

### 3. Navngiv jobbet

Feltet **Job Name** udfyldes automatisk med filnavnet. Rediger det, hvis du ønsker en anden betegnelse — dette navn vises i transskriptionshistorikken på startskærmen.

### 4. Start transskription

Klik på `Start Transcription`. Programmet skifter til visningen **Progress**.

Klik på `← Back` for at gå tilbage uden at starte.

## Hvad sker der derefter

Jobbet gennemgår to faser, der vises i statuslinjen:

1. **Audio Analysis** — taleropsporing: identifikation af hvem der taler og hvornår.
2. **Speech Recognition** — omsætning af tale til tekst segment for segment.

Transskriberede segmenter vises i den dynamiske tabel, efterhånden som de produceres. Når behandlingen er fuldført, skifter programmet automatisk til visningen **Results**.

Hvis du tilføjer et job, mens et andet allerede kører, vil det nye job få status `queued` og starte, når det igangværende job er færdigt. Se [Overvågning af job](monitoring_jobs.md).

---