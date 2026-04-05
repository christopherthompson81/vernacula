---
title: "Sprechernamen bearbeiten"
description: "So ersetzen Sie generische Sprecher-IDs durch echte Namen in einem Transkript."
topic_id: operations_editing_speaker_names
---

# Sprechernamen bearbeiten

Die Transkriptions-Engine beschriftet jeden Sprecher automatisch mit einer generischen ID (zum Beispiel `speaker_0`, `speaker_1`). Sie können diese durch echte Namen ersetzen, die im gesamten Transkript und in allen exportierten Dateien erscheinen.

## So bearbeiten Sie Sprechernamen

1. Öffnen Sie einen abgeschlossenen Auftrag. Siehe [Abgeschlossene Aufträge laden](loading_completed_jobs.md).
2. Klicken Sie in der Ansicht **Ergebnisse** auf `Edit Speaker Names`.
3. Der Dialog **Edit Speaker Names** öffnet sich mit zwei Spalten:
   - **Speaker ID** — die vom Modell zugewiesene ursprüngliche Bezeichnung (schreibgeschützt).
   - **Display Name** — der im Transkript angezeigte Name (bearbeitbar).
4. Klicken Sie auf eine Zelle in der Spalte **Display Name** und geben Sie den Namen des Sprechers ein.
5. Drücken Sie `Tab` oder klicken Sie auf eine andere Zeile, um zum nächsten Sprecher zu wechseln.
6. Klicken Sie auf `Save`, um die Änderungen zu übernehmen, oder auf `Cancel`, um sie zu verwerfen.

## Wo Namen erscheinen

Aktualisierte Anzeigenamen ersetzen die generischen IDs in:

- Der Segmenttabelle in der Ergebnisansicht.
- Allen exportierten Dateien (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Namen erneut bearbeiten

Sie können den Dialog „Sprechernamen bearbeiten" jederzeit erneut öffnen, solange der Auftrag in der Ergebnisansicht geladen ist. Änderungen werden in der lokalen Datenbank gespeichert und bleiben sitzungsübergreifend erhalten.

---