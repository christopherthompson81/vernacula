---
title: "Aufträge anhalten, fortsetzen oder entfernen"
description: "So halten Sie einen laufenden Auftrag an, setzen einen gestoppten fort oder löschen einen Auftrag aus dem Verlauf."
topic_id: operations_pausing_resuming_removing
---

# Aufträge anhalten, fortsetzen oder entfernen

## Einen Auftrag anhalten

Sie können einen laufenden oder in der Warteschlange befindlichen Auftrag an zwei Stellen anhalten:

- **Fortschrittsansicht** — klicken Sie auf `Pause` in der unteren rechten Ecke, während Sie den aktiven Auftrag beobachten.
- **Tabelle „Transkriptionsverlauf"** — klicken Sie auf `Pause` in der Spalte **Aktionen** einer beliebigen Zeile, deren Status `running` oder `queued` ist.

Nachdem Sie auf `Pause` geklickt haben, zeigt die Statuszeile `Pausing…` an, während die Anwendung die aktuelle Verarbeitungseinheit abschließt. Der Auftragsstatus ändert sich anschließend in `cancelled` in der Verlaufstabelle.

> Beim Anhalten werden alle bisher transkribierten Segmente gespeichert. Sie können den Auftrag später fortsetzen, ohne diese Arbeit zu verlieren.

## Einen Auftrag fortsetzen

So setzen Sie einen angehaltenen oder fehlgeschlagenen Auftrag fort:

1. Suchen Sie auf dem Startbildschirm den Auftrag in der Tabelle **Transkriptionsverlauf**. Sein Status lautet `cancelled` oder `failed`.
2. Klicken Sie auf `Resume` in der Spalte **Aktionen**.
3. Die Anwendung kehrt zur Ansicht **Fortschritt** zurück und setzt die Verarbeitung an der Stelle fort, an der sie unterbrochen wurde.

Die Statuszeile zeigt kurz `Resuming…` an, während der Auftrag neu initialisiert wird.

## Einen Auftrag entfernen

So löschen Sie einen Auftrag und sein Transkript dauerhaft aus dem Verlauf:

1. Klicken Sie in der Tabelle **Transkriptionsverlauf** auf `Remove` in der Spalte **Aktionen** des Auftrags, den Sie löschen möchten.

Der Auftrag wird aus der Liste entfernt und seine Daten werden aus der lokalen Datenbank gelöscht. Diese Aktion kann nicht rückgängig gemacht werden. Exportierte Dateien, die auf dem Datenträger gespeichert wurden, sind davon nicht betroffen.

---