---
title: "Aufträge überwachen"
description: "So verfolgen Sie den Fortschritt eines laufenden oder in der Warteschlange befindlichen Auftrags."
topic_id: operations_monitoring_jobs
---

# Aufträge überwachen

Die Ansicht **Fortschritt** zeigt Ihnen den aktuellen Stand eines laufenden Transkriptionsauftrags in Echtzeit.

## Fortschrittsansicht öffnen

- Wenn Sie eine neue Transkription starten, wechselt die Anwendung automatisch zur Fortschrittsansicht.
- Bei einem bereits laufenden oder in der Warteschlange befindlichen Auftrag suchen Sie ihn in der Tabelle **Transkriptionsverlauf** und klicken Sie in der Spalte **Aktionen** auf `Monitor`.

## Fortschrittsansicht verstehen

| Element | Beschreibung |
|---|---|
| Fortschrittsbalken | Gesamtprozentsatz der Fertigstellung. Unbestimmt (animiert), während der Auftrag gestartet oder fortgesetzt wird. |
| Prozentanzeige | Numerischer Prozentwert, der rechts neben dem Balken angezeigt wird. |
| Statusmeldung | Aktuelle Aktivität – zum Beispiel `Audio Analysis` oder `Speech Recognition`. Zeigt `Waiting in queue…` an, wenn der Auftrag noch nicht begonnen hat. |
| Segmenttabelle | Live-Anzeige der transkribierten Segmente mit den Spalten **Sprecher**, **Start**, **Ende** und **Inhalt**. Scrollt automatisch, wenn neue Segmente eintreffen. |

## Fortschrittsphasen

Die angezeigten Phasen hängen vom in den Einstellungen gewählten **Segmentierungsmodus** ab.

**Modus „Sprecherdiarisierung"** (Standard):

1. **Audio Analysis** — SortFormer-Diarisierung wird über die gesamte Datei ausgeführt, um Sprechergrenzen zu erkennen. Der Balken bleibt möglicherweise bis zum Abschluss dieser Phase nahe 0 %.
2. **Speech Recognition** — jedes Sprechersegment wird transkribiert. Der Prozentsatz steigt in dieser Phase kontinuierlich an.

**Modus „Sprachaktivitätserkennung"**:

1. **Detecting speech segments** — Silero VAD durchsucht die Datei nach Sprachbereichen. Diese Phase verläuft schnell.
2. **Speech Recognition** — jeder erkannte Sprachbereich wird transkribiert.

In beiden Modi füllt sich die Live-Segmenttabelle während der Transkription fortlaufend.

## Ansicht verlassen

Klicken Sie auf `← Back to Home`, um zum Startbildschirm zurückzukehren, ohne den Auftrag zu unterbrechen. Der Auftrag wird im Hintergrund weiter ausgeführt und sein Status wird in der Tabelle **Transkriptionsverlauf** aktualisiert. Klicken Sie jederzeit erneut auf `Monitor`, um zur Fortschrittsansicht zurückzukehren.

---