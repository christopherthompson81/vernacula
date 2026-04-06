---
title: "Modelle herunterladen"
description: "So laden Sie die KI-Modelldateien herunter, die für die Transkription erforderlich sind."
topic_id: first_steps_downloading_models
---

# Modelle herunterladen

Vernacula-Desktop benötigt KI-Modelldateien, um zu funktionieren. Diese sind nicht im Installationspaket enthalten und müssen vor der ersten Transkription heruntergeladen werden.

## Modellstatus (Startbildschirm)

Eine schmale Statusleiste oben auf dem Startbildschirm zeigt an, ob Ihre Modelle bereit sind. Fehlen Dateien, wird dort auch eine Schaltfläche `Open Settings` angezeigt, über die Sie direkt zur Modellverwaltung gelangen.

| Status | Bedeutung |
|---|---|
| `All N model file(s) present ✓` | Alle erforderlichen Dateien wurden heruntergeladen und sind bereit. |
| `N model file(s) missing: …` | Eine oder mehrere Dateien fehlen; öffnen Sie die Einstellungen, um sie herunterzuladen. |

Sobald die Modelle bereit sind, werden die Schaltflächen `New Transcription` und `Bulk Add Jobs` aktiv.

## So laden Sie Modelle herunter

1. Klicken Sie auf dem Startbildschirm auf `Open Settings` (oder navigieren Sie zu `Settings… > Models`).
2. Klicken Sie im Abschnitt **Models** auf `Download Missing Models`.
3. Ein Fortschrittsbalken und eine Statuszeile werden angezeigt, die die aktuelle Datei, ihre Position in der Warteschlange und die Downloadgröße anzeigen – zum Beispiel: `[1/3] encoder-model.onnx — 42 MB`.
4. Warten Sie, bis der Status `Download complete.` anzeigt.

## Einen Download abbrechen

Um einen laufenden Download zu stoppen, klicken Sie auf `Cancel`. Die Statuszeile zeigt dann `Download cancelled.` an. Teilweise heruntergeladene Dateien bleiben erhalten, sodass der Download beim nächsten Klick auf `Download Missing Models` an der unterbrochenen Stelle fortgesetzt wird.

## Download-Fehler

Schlägt ein Download fehl, zeigt die Statuszeile `Download failed: <reason>` an. Überprüfen Sie Ihre Internetverbindung und klicken Sie erneut auf `Download Missing Models`, um den Vorgang zu wiederholen. Die Anwendung setzt dabei bei der zuletzt erfolgreich abgeschlossenen Datei an.

## Genauigkeit ändern

Welche Modelldateien heruntergeladen werden müssen, hängt von der ausgewählten **Model Precision** ab. Um diese zu ändern, navigieren Sie zu `Settings… > Models > Model Precision`. Wenn Sie die Genauigkeit nach dem Herunterladen wechseln, muss der neue Dateisatz separat heruntergeladen werden. Weitere Informationen finden Sie unter [Modellgewichtgenauigkeit auswählen](model_precision.md).

---