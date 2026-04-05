---
title: "Mehrere Audiodateien in die Warteschlange einreihen"
description: "So fügen Sie mehrere Audiodateien auf einmal zur Auftragswarteschlange hinzu."
topic_id: operations_bulk_add_jobs
---

# Mehrere Audiodateien in die Warteschlange einreihen

Verwenden Sie **Bulk Add Jobs**, um mehrere Audio- oder Videodateien in einem Schritt zur Transkriptionswarteschlange hinzuzufügen. Die Anwendung verarbeitet sie nacheinander in der Reihenfolge, in der sie hinzugefügt wurden.

## Voraussetzungen

- Alle Modelldateien müssen heruntergeladen sein. Die Karte **Model Status** muss `All N model file(s) present ✓` anzeigen. Siehe [Modelle herunterladen](../first_steps/downloading_models.md).

## So fügen Sie mehrere Aufträge auf einmal hinzu

1. Klicken Sie auf dem Startbildschirm auf `Bulk Add Jobs`.
2. Es öffnet sich ein Dateiauswahldialog. Wählen Sie eine oder mehrere Audio- oder Videodateien aus – halten Sie `Ctrl` oder `Shift` gedrückt, um mehrere Dateien auszuwählen.
3. Klicken Sie auf **Open**. Jede ausgewählte Datei wird als separater Auftrag zur Tabelle **Transcription History** hinzugefügt.

> **Videodateien mit mehreren Audiostreams:** Enthält eine Videodatei mehr als einen Audiostream (beispielsweise mehrere Sprachen oder einen Kommentartrack), erstellt die Anwendung automatisch einen Auftrag pro Stream.

## Auftragsnamen

Jeder Auftrag wird automatisch nach dem Namen der zugehörigen Audiodatei benannt. Sie können einen Auftrag jederzeit umbenennen, indem Sie in der Tabelle „Transkriptionsverlauf" auf seinen Namen in der Spalte **Title** klicken, den Text bearbeiten und `Enter` drücken oder an eine andere Stelle klicken.

## Verhalten der Warteschlange

- Wenn gerade kein Auftrag ausgeführt wird, beginnt die erste Datei sofort und die übrigen werden als `queued` angezeigt.
- Wenn bereits ein Auftrag ausgeführt wird, werden alle neu hinzugefügten Dateien als `queued` angezeigt und starten automatisch nacheinander.
- Um den aktiven Auftrag zu überwachen, klicken Sie in seiner Spalte **Actions** auf `Monitor`. Siehe [Aufträge überwachen](monitoring_jobs.md).
- Um einen Auftrag in der Warteschlange anzuhalten oder zu entfernen, bevor er startet, verwenden Sie die Schaltflächen `Pause` oder `Remove` in seiner Spalte **Actions**. Siehe [Aufträge anhalten, fortsetzen oder entfernen](pausing_resuming_removing.md).

---