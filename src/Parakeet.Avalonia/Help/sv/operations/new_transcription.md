---
title: "Arbetsflöde för ny transkription"
description: "Steg-för-steg-guide för att transkribera en ljudfil."
topic_id: operations_new_transcription
---

# Arbetsflöde för ny transkription

Använd det här arbetsflödet för att transkribera en enskild ljudfil.

## Förutsättningar

- Alla modellfiler måste vara nedladdade. Kortet **Modellstatus** måste visa `All N model file(s) present ✓`. Se [Ladda ned modeller](../first_steps/downloading_models.md).

## Filformat som stöds

### Ljud

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Videofiler avkodas via FFmpeg. Om en videofil innehåller **flera ljudströmmar** (t.ex. flera språk eller kommentarsspår) skapas automatiskt ett transkriptionsjobb för varje ström.

## Steg

### 1. Öppna formuläret för ny transkription

Klicka på `New Transcription` på startskärmen, eller gå till `File > New Transcription`.

### 2. Välj en mediafil

Klicka på `Browse…` bredvid fältet **Audio File**. En filväljare öppnas filtrerad på ljud- och videoformat som stöds. Välj din fil och klicka på **Open**. Filsökvägen visas i fältet.

### 3. Namnge jobbet

Fältet **Job Name** fylls i automatiskt utifrån filnamnet. Redigera det om du vill använda en annan etikett — det här namnet visas i transkriptionshistoriken på startskärmen.

### 4. Starta transkriptionen

Klicka på `Start Transcription`. Programmet byter till vyn **Progress**.

Klicka på `← Back` om du vill gå tillbaka utan att starta.

## Vad händer härnäst

Jobbet körs igenom två faser som visas i förloppsindikatorn:

1. **Audio Analysis** — talaridentifiering: avgör vem som talar och när.
2. **Speech Recognition** — omvandlar tal till text segment för segment.

Transkriberade segment visas i realtidstabellen allt eftersom de produceras. När bearbetningen är klar övergår programmet automatiskt till vyn **Results**.

Om du lägger till ett jobb medan ett annat redan körs visas statusen `queued` för det nya jobbet, och det startar när det pågående jobbet är klart. Se [Övervaka jobb](monitoring_jobs.md).

---