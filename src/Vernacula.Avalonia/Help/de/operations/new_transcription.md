---
title: "Neues Transkriptions-Workflow"
description: "Schritt-für-Schritt-Anleitung zur Transkription einer Audiodatei."
topic_id: operations_new_transcription
---

# Neuer Transkriptions-Workflow

Verwenden Sie diesen Workflow, um eine einzelne Audiodatei zu transkribieren.

## Voraussetzungen

- Alle Modelldateien müssen heruntergeladen sein. Die Karte **Modellstatus** muss `All N model file(s) present ✓` anzeigen. Siehe [Modelle herunterladen](../first_steps/downloading_models.md).

## Unterstützte Formate

### Audio

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Videodateien werden über FFmpeg dekodiert. Enthält eine Videodatei **mehrere Audiostreams** (z. B. mehrere Sprachen oder Kommentarspuren), wird automatisch für jeden Stream ein separater Transkriptionsjob erstellt.

## Schritte

### 1. Das Formular „Neue Transkription" öffnen

Klicken Sie auf der Startseite auf `New Transcription` oder navigieren Sie zu `File > New Transcription`.

### 2. Eine Mediendatei auswählen

Klicken Sie neben dem Feld **Audiodatei** auf `Browse…`. Es öffnet sich ein Dateiauswahldialog, der auf unterstützte Audio- und Videoformate gefiltert ist. Wählen Sie Ihre Datei aus und klicken Sie auf **Open**. Der Dateipfad wird im Feld angezeigt.

### 3. Den Job benennen

Das Feld **Jobname** wird automatisch mit dem Dateinamen vorausgefüllt. Bearbeiten Sie es, wenn Sie eine andere Bezeichnung wünschen – dieser Name erscheint im Transkriptionsverlauf auf der Startseite.

### 4. Transkription starten

Klicken Sie auf `Start Transcription`. Die Anwendung wechselt in die Ansicht **Fortschritt**.

Um ohne Start zurückzugehen, klicken Sie auf `← Back`.

## Was als Nächstes passiert

Der Job durchläuft zwei Phasen, die im Fortschrittsbalken angezeigt werden:

1. **Audioanalyse** — Sprecherdiarisierung: Es wird erkannt, wer wann spricht.
2. **Spracherkennung** — Die Sprache wird Segment für Segment in Text umgewandelt.

Transkribierte Segmente erscheinen in der Live-Tabelle, sobald sie erzeugt werden. Wenn die Verarbeitung abgeschlossen ist, wechselt die Anwendung automatisch in die Ansicht **Ergebnisse**.

Wenn Sie einen Job hinzufügen, während bereits ein anderer läuft, erhält der neue Job den Status `queued` und startet, sobald der aktuelle Job abgeschlossen ist. Siehe [Jobs überwachen](monitoring_jobs.md).

---