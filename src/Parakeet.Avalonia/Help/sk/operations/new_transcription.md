---
title: "Pracovný postup novej transkripcie"
description: "Podrobný návod na transkripciu zvukového súboru."
topic_id: operations_new_transcription
---

# Pracovný postup novej transkripcie

Pomocou tohto pracovného postupu môžete prepísať jeden zvukový súbor.

## Predpoklady

- Všetky súbory modelov musia byť stiahnuté. Karta **Stav modelu** musí zobrazovať `All N model file(s) present ✓`. Pozrite si [Sťahovanie modelov](../first_steps/downloading_models.md).

## Podporované formáty

### Zvuk

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Videosúbory sú dekódované prostredníctvom FFmpeg. Ak videosúbor obsahuje **viacero zvukových stôp** (napr. viacero jazykov alebo komentárové stopy), pre každú stopu sa automaticky vytvorí jedna úloha transkripcie.

## Kroky

### 1. Otvorte formulár novej transkripcie

Kliknite na `New Transcription` na domovskej obrazovke alebo prejdite na `File > New Transcription`.

### 2. Vyberte mediálny súbor

Kliknite na `Browse…` vedľa poľa **Audio File**. Otvorí sa výber súborov filtrovaný na podporované zvukové a video formáty. Vyberte svoj súbor a kliknite na **Open**. Cesta k súboru sa zobrazí v príslušnom poli.

### 3. Pomenujte úlohu

Pole **Job Name** je vopred vyplnené názvom súboru. Upravte ho, ak chcete iný popis — tento názov sa zobrazuje v histórii transkripcií na domovskej obrazovke.

### 4. Spustite transkripciu

Kliknite na `Start Transcription`. Aplikácia sa prepne do zobrazenia **Progress**.

Ak sa chcete vrátiť späť bez spustenia, kliknite na `← Back`.

## Čo sa stane ďalej

Úloha prebieha v dvoch fázach zobrazených na ukazovateli priebehu:

1. **Audio Analysis** — diarizácia rečníkov: identifikácia toho, kto hovorí a kedy.
2. **Speech Recognition** — prevod reči na text po jednotlivých segmentoch.

Prepísané segmenty sa priebežne zobrazujú v živej tabuľke tak, ako sú spracovávané. Po dokončení spracovania sa aplikácia automaticky presunie do zobrazenia **Results**.

Ak pridáte úlohu, kým iná už prebieha, nová úloha bude mať stav `queued` a spustí sa po dokončení aktuálnej úlohy. Pozrite si [Monitorovanie úloh](monitoring_jobs.md).

---