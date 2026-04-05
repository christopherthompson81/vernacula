---
title: "Transkripte bearbeiten"
description: "So können Sie transkribierte Segmente im Transkripteditor überprüfen, korrigieren und bestätigen."
topic_id: operations_editing_transcripts
---

# Transkripte bearbeiten

Der **Transkripteditor** ermöglicht es Ihnen, ASR-Ausgaben zu überprüfen, Text zu korrigieren, Sprecher direkt umzubenennen, Segmentzeiten anzupassen und Segmente als überprüft zu markieren – alles während Sie die Originalaufnahme anhören.

## Editor öffnen

1. Laden Sie einen abgeschlossenen Auftrag (siehe [Abgeschlossene Aufträge laden](loading_completed_jobs.md)).
2. Klicken Sie in der Ansicht **Ergebnisse** auf `Edit Transcript`.

Der Editor öffnet sich als separates Fenster und kann neben der Hauptanwendung geöffnet bleiben.

## Layout

Jedes Segment wird als Karte mit zwei nebeneinander angeordneten Bereichen dargestellt:

- **Linker Bereich** – die ursprüngliche ASR-Ausgabe mit wortgenauer Konfidenzfärbung. Wörter, bei denen das Modell weniger sicher war, erscheinen in Rot; Wörter mit hoher Konfidenz werden in der normalen Textfarbe angezeigt.
- **Rechter Bereich** – ein bearbeitbares Textfeld. Nehmen Sie hier Korrekturen vor; die Unterschiede zum Original werden beim Tippen hervorgehoben.

Die Sprecherbezeichnung und der Zeitbereich erscheinen oberhalb jeder Karte. Klicken Sie auf eine Karte, um sie zu fokussieren und die zugehörigen Aktionssymbole anzuzeigen. Fahren Sie mit der Maus über ein Symbol, um einen Tooltip mit einer Beschreibung der Funktion zu sehen.

## Symbolübersicht

### Wiedergabeleiste

| Symbol | Aktion |
|--------|--------|
| ▶ | Wiedergabe |
| ⏸ | Pause |
| ⏮ | Zum vorherigen Segment springen |
| ⏭ | Zum nächsten Segment springen |

### Aktionen für Segmentkarten

| Symbol | Aktion |
|--------|--------|
| <mdl2 ch="E77B"/> | Segment einem anderen Sprecher zuweisen |
| <mdl2 ch="E916"/> | Start- und Endzeit des Segments anpassen |
| <mdl2 ch="EA39"/> | Segment unterdrücken oder Unterdrückung aufheben |
| <mdl2 ch="E72B"/> | Mit dem vorherigen Segment zusammenführen |
| <mdl2 ch="E72A"/> | Mit dem nächsten Segment zusammenführen |
| <mdl2 ch="E8C6"/> | Segment teilen |
| <mdl2 ch="E72C"/> | ASR für dieses Segment wiederholen |

## Audiowiedergabe

Eine Wiedergabeleiste befindet sich am oberen Rand des Editorfensters:

| Steuerelement | Aktion |
|---------------|--------|
| Symbol „Wiedergabe/Pause" | Wiedergabe starten oder anhalten |
| Suchleiste | Ziehen, um zu einer beliebigen Position in der Aufnahme zu springen |
| Geschwindigkeitsregler | Wiedergabegeschwindigkeit anpassen (0,5× – 2×) |
| Symbole „Zurück/Weiter" | Zum vorherigen oder nächsten Segment springen |
| Dropdown „Wiedergabemodus" | Einen von drei Wiedergabemodi auswählen (siehe unten) |
| Lautstärkeregler | Wiedergabelautstärke anpassen |

Während der Wiedergabe wird das aktuell gesprochene Wort im linken Bereich hervorgehoben. Bei einer Pause nach einem Sprung in der Aufnahme aktualisiert sich die Hervorhebung auf das Wort an der neuen Position.

### Wiedergabemodi

| Modus | Verhalten |
|-------|-----------|
| `Single` | Das aktuelle Segment einmal abspielen, dann stoppen. |
| `Auto-advance` | Das aktuelle Segment abspielen; am Ende wird es als überprüft markiert und zum nächsten gewechselt. |
| `Continuous` | Alle Segmente nacheinander abspielen, ohne eines als überprüft zu markieren. |

Wählen Sie den aktiven Modus aus dem Dropdown in der Wiedergabeleiste.

## Ein Segment bearbeiten

1. Klicken Sie auf eine Karte, um sie zu fokussieren.
2. Bearbeiten Sie den Text im rechten Bereich. Änderungen werden automatisch gespeichert, wenn Sie den Fokus auf eine andere Karte verschieben.

## Einen Sprecher umbenennen

Klicken Sie auf die Sprecherbezeichnung in der fokussierten Karte und geben Sie einen neuen Namen ein. Drücken Sie `Enter` oder klicken Sie auf eine andere Stelle, um zu speichern. Der neue Name wird nur auf diese Karte angewendet. Um einen Sprecher global umzubenennen, verwenden Sie [Sprechernamen bearbeiten](editing_speaker_names.md) in der Ergebnisansicht.

## Ein Segment als überprüft markieren

Klicken Sie auf das Kontrollkästchen `Verified` einer fokussierten Karte, um sie als überprüft zu markieren. Der Überprüfungsstatus wird in der Datenbank gespeichert und beim nächsten Öffnen im Editor angezeigt.

## Ein Segment unterdrücken

Klicken Sie auf `Suppress` einer fokussierten Karte, um das Segment aus Exporten auszublenden (nützlich für Geräusche, Musik oder andere Nicht-Sprach-Abschnitte). Klicken Sie auf `Unsuppress`, um es wiederherzustellen.

## Segmentzeiten anpassen

Klicken Sie auf `Adjust Times` einer fokussierten Karte, um den Dialog zur Zeitanpassung zu öffnen. Verwenden Sie das Scrollrad über dem Feld **Start** oder **End**, um den Wert in 0,1-Sekunden-Schritten zu ändern, oder geben Sie einen Wert direkt ein. Klicken Sie auf `Save`, um die Änderungen zu übernehmen.

## Segmente zusammenführen

- Klicken Sie auf `⟵ Merge`, um das fokussierte Segment mit dem unmittelbar vorherigen Segment zusammenzuführen.
- Klicken Sie auf `Merge ⟶`, um das fokussierte Segment mit dem unmittelbar folgenden Segment zusammenzuführen.

Der kombinierte Text und der Zeitbereich beider Karten werden vereint. Dies ist nützlich, wenn eine einzelne gesprochene Äußerung auf zwei Segmente aufgeteilt wurde.

## Ein Segment teilen

Klicken Sie auf `Split…` einer fokussierten Karte, um den Teilen-Dialog zu öffnen. Legen Sie die Teilungsposition im Text fest und bestätigen Sie. Es werden zwei neue Segmente erstellt, die den ursprünglichen Zeitbereich abdecken. Dies ist nützlich, wenn zwei unterschiedliche Äußerungen in einem Segment zusammengefasst wurden.

## ASR wiederholen

Klicken Sie auf `Redo ASR` einer fokussierten Karte, um die Spracherkennung für den Audioabschnitt dieses Segments erneut auszuführen. Das Modell verarbeitet nur den Audioausschnitt des betreffenden Segments und erstellt eine neue, einheitliche Transkription.

Verwenden Sie diese Funktion, wenn:

- Ein Segment durch eine Zusammenführung entstanden ist und nicht geteilt werden kann (zusammengeführte Segmente umfassen mehrere ASR-Quellen; „Redo ASR" fasst sie zu einer einzigen zusammen, woraufhin `Split…` verfügbar wird).
- Die ursprüngliche Transkription fehlerhaft ist und Sie ohne manuelles Bearbeiten eine saubere zweite Auswertung wünschen.

**Hinweis:** Bereits im rechten Bereich eingegebener Text wird verworfen und durch die neue ASR-Ausgabe ersetzt. Der Vorgang erfordert, dass die Audiodatei geladen ist; die Schaltfläche ist deaktiviert, wenn keine Audiodatei verfügbar ist.