---
title: "Ergebnisse oder Transkripte exportieren"
description: "So speichern Sie ein Transkript in verschiedenen Formaten als Datei."
topic_id: operations_exporting_results
---

# Ergebnisse oder Transkripte exportieren

Sie können ein abgeschlossenes Transkript in verschiedene Dateiformate exportieren, um es in anderen Anwendungen zu verwenden.

## So exportieren Sie

1. Öffnen Sie einen abgeschlossenen Auftrag. Siehe [Abgeschlossene Aufträge laden](loading_completed_jobs.md).
2. Klicken Sie in der Ansicht **Ergebnisse** auf `Export Transcript`.
3. Der Dialog **Export Transcript** wird geöffnet. Wählen Sie ein Format aus dem Dropdown-Menü **Format** aus.
4. Klicken Sie auf `Save`. Ein Datei-Speichern-Dialog wird geöffnet.
5. Wählen Sie einen Zielordner und einen Dateinamen, und klicken Sie dann auf **Speichern**.

Am unteren Rand des Dialogs wird eine Bestätigungsmeldung mit dem vollständigen Pfad der gespeicherten Datei angezeigt.

## Verfügbare Formate

| Format | Erweiterung | Am besten geeignet für |
|---|---|---|
| Excel | `.xlsx` | Tabellenkalkulationsanalyse mit Spalten für Sprecher, Zeitstempel und Inhalt. |
| CSV | `.csv` | Import in beliebige Tabellenkalkulationen oder Datenwerkzeuge. |
| JSON | `.json` | Programmgesteuerte Verarbeitung. |
| SRT-Untertitel | `.srt` | Laden in Videoeditoren oder Mediaplayer als Untertitel. |
| Markdown | `.md` | Lesbare Nur-Text-Dokumente. |
| Word-Dokument | `.docx` | Weitergabe an Benutzer von Microsoft Word. |
| SQLite-Datenbank | `.db` | Vollständiger Datenbankexport für benutzerdefinierte Abfragen. |

## Sprechernamen in Exporten

Wenn Sie den Sprechern Anzeigenamen zugewiesen haben, werden diese Namen in allen Exportformaten verwendet. Um Namen vor dem Exportieren zu aktualisieren, klicken Sie zuerst auf `Edit Speaker Names`. Siehe [Sprechernamen bearbeiten](editing_speaker_names.md).

---