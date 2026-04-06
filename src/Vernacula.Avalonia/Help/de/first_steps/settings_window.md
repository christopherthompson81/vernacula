---
title: "Einstellungen"
description: "Übersicht aller Optionen im Einstellungsfenster."
topic_id: first_steps_settings_window
---

# Einstellungen

Das Fenster **Einstellungen** bietet Ihnen die Kontrolle über Hardwarekonfiguration, Modellverwaltung, Segmentierungsmodus, Editor-Verhalten, Darstellung und Sprache. Öffnen Sie es über die Menüleiste: `Settings…`.

## Hardware & Leistung

Dieser Abschnitt zeigt den Status Ihrer NVIDIA GPU und des CUDA-Software-Stacks und gibt die dynamische Batch-Obergrenze an, die bei der GPU-Transkription verwendet wird.

| Element | Beschreibung |
|---|---|
| GPU-Name und VRAM | Erkannte NVIDIA GPU und verfügbarer Videospeicher. |
| CUDA Toolkit | Gibt an, ob die CUDA-Laufzeitbibliotheken über `CUDA_PATH` gefunden wurden. |
| cuDNN | Gibt an, ob die cuDNN-Laufzeit-DLLs verfügbar sind. |
| CUDA-Beschleunigung | Gibt an, ob der ONNX Runtime der CUDA-Ausführungsanbieter erfolgreich geladen wurde. |

Klicken Sie auf `Re-check`, um die Hardwareerkennung erneut auszuführen, ohne die Anwendung neu zu starten – nützlich nach der Installation von CUDA oder cuDNN.

Direkte Download-Links für das CUDA Toolkit und cuDNN werden angezeigt, wenn diese Komponenten nicht erkannt werden.

Die Meldung zur **Batch-Obergrenze** gibt an, wie viele Sekunden Audio in jedem GPU-Durchlauf verarbeitet werden. Dieser Wert wird aus dem freien VRAM nach dem Laden der Modelle abgeleitet und passt sich automatisch an.

Eine vollständige Anleitung zur CUDA-Einrichtung finden Sie unter [CUDA und cuDNN installieren](cuda_installation.md).

## Modelle

Dieser Abschnitt verwaltet die KI-Modelldateien, die für die Transkription benötigt werden.

- **Fehlende Modelle herunterladen** — lädt alle Modelldateien herunter, die noch nicht auf dem Datenträger vorhanden sind. Ein Fortschrittsbalken und eine Statuszeile verfolgen jede Datei während des Downloads.
- **Auf Updates prüfen** — prüft, ob neuere Modellgewichte verfügbar sind. Wenn aktualisierte Gewichte erkannt werden, erscheint auf dem Startbildschirm automatisch ein Aktualisierungshinweis.

## Segmentierungsmodus

Steuert, wie das Audio vor der Spracherkennung in Segmente aufgeteilt wird.

| Modus | Beschreibung |
|---|---|
| **Sprecher-Diarisierung** | Verwendet das SortFormer-Modell, um einzelne Sprecher zu identifizieren und jedes Segment zu beschriften. Am besten geeignet für Interviews, Besprechungen und Aufnahmen mit mehreren Sprechern. |
| **Sprachaktivitätserkennung** | Verwendet Silero VAD, um nur Sprachbereiche zu erkennen – ohne Sprecherbezeichnungen. Schneller als die Diarisierung und gut geeignet für Aufnahmen mit einem einzelnen Sprecher. |

## Transkript-Editor

**Standard-Wiedergabemodus** — legt den Wiedergabemodus fest, der beim Öffnen des Transkript-Editors verwendet wird. Sie können diesen auch jederzeit direkt im Editor ändern. Eine Beschreibung der einzelnen Modi finden Sie unter [Transkripte bearbeiten](../operations/editing_transcripts.md).

## Darstellung

Wählen Sie das **Dunkle** oder **Helle** Design. Die Änderung wird sofort übernommen. Weitere Informationen finden Sie unter [Design auswählen](theme.md).

## Sprache

Wählen Sie die Anzeigesprache der Anwendungsoberfläche. Die Änderung wird sofort übernommen. Weitere Informationen finden Sie unter [Sprache auswählen](language.md).

---