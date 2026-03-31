---
title: "Pracovní postup nového přepisu"
description: "Podrobný průvodce přepisem zvukového souboru."
topic_id: operations_new_transcription
---

# Pracovní postup nového přepisu

Tento pracovní postup použijte k přepisu jednoho zvukového souboru.

## Předpoklady

- Všechny soubory modelů musí být staženy. Karta **Stav modelu** musí zobrazovat `All N model file(s) present ✓`. Viz [Stahování modelů](../first_steps/downloading_models.md).

## Podporované formáty

### Zvuk

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Videosoubory jsou dekódovány prostřednictvím FFmpeg. Pokud videosoubor obsahuje **více zvukových stop** (např. více jazyků nebo komentářové stopy), je pro každou stopu automaticky vytvořena samostatná úloha přepisu.

## Kroky

### 1. Otevřete formulář nového přepisu

Klikněte na `New Transcription` na domovské obrazovce nebo přejděte na `File > New Transcription`.

### 2. Vyberte mediální soubor

Klikněte na `Browse…` vedle pole **Audio File**. Otevře se výběr souborů filtrovaný na podporované zvukové a video formáty. Vyberte soubor a klikněte na **Open**. Cesta k souboru se zobrazí v poli.

### 3. Pojmenujte úlohu

Pole **Job Name** je předvyplněno názvem souboru. Upravte jej, pokud chcete jiný popisek — tento název se zobrazuje v historii přepisů na domovské obrazovce.

### 4. Spusťte přepis

Klikněte na `Start Transcription`. Aplikace přepne do zobrazení **Progress**.

Chcete-li se vrátit bez spuštění, klikněte na `← Back`.

## Co se stane dále

Úloha probíhá ve dvou fázích zobrazených v průběhové liště:

1. **Audio Analysis** — diarizace mluvčích: identifikace toho, kdo mluví a kdy.
2. **Speech Recognition** — převod řeči na text po jednotlivých segmentech.

Přepsané segmenty se průběžně zobrazují v živé tabulce. Po dokončení zpracování aplikace automaticky přejde do zobrazení **Results**.

Pokud přidáte úlohu ve chvíli, kdy již jiná úloha běží, nová úloha zobrazí stav `queued` a spustí se po dokončení aktuální úlohy. Viz [Sledování úloh](monitoring_jobs.md).

---