---
title: "Auswahl der Modellgewichtspräzision"
description: "Wie Sie zwischen INT8 und FP32 Modellpräzision wählen und welche Kompromisse damit verbunden sind."
topic_id: first_steps_model_precision
---

# Auswahl der Modellgewichtspräzision

Die Modellpräzision steuert das numerische Format, das für die KI-Modellgewichte verwendet wird. Sie beeinflusst die Download-Größe, den Speicherbedarf und die Genauigkeit.

## Präzisionsoptionen

### INT8 (kleinerer Download)

- Kleinere Modelldateien — schnellerer Download und geringerer Speicherplatzbedarf.
- Etwas geringere Genauigkeit bei manchen Audiodaten.
- Empfohlen, wenn Sie über begrenzten Speicherplatz oder eine langsamere Internetverbindung verfügen.

### FP32 (genauer)

- Größere Modelldateien.
- Höhere Genauigkeit, insbesondere bei schwierigen Audiodaten mit Akzenten oder Hintergrundgeräuschen.
- Empfohlen, wenn Genauigkeit Vorrang hat und ausreichend Speicherplatz vorhanden ist.
- Erforderlich für die CUDA GPU-Beschleunigung — der GPU-Pfad verwendet immer FP32, unabhängig von dieser Einstellung.

## Präzision ändern

Öffnen Sie `Settings…` in der Menüleiste, wechseln Sie dann zum Abschnitt **Models** und wählen Sie entweder `INT8 (smaller download)` oder `FP32 (more accurate)`.

## Nach dem Ändern der Präzision

Eine Änderung der Präzision erfordert einen anderen Satz von Modelldateien. Falls die Modelldateien für die neue Präzision noch nicht heruntergeladen wurden, klicken Sie in den Einstellungen auf `Download Missing Models`. Bereits heruntergeladene Dateien für die andere Präzision verbleiben auf dem Datenträger und müssen nicht erneut heruntergeladen werden, wenn Sie zurückwechseln.

---