---
title: "CUDA und cuDNN für GPU-Beschleunigung installieren"
description: "So richten Sie NVIDIA CUDA und cuDNN ein, damit Vernacula-Desktop Ihre GPU nutzen kann."
topic_id: first_steps_cuda_installation
---

# CUDA und cuDNN für GPU-Beschleunigung installieren

Vernacula-Desktop kann eine NVIDIA GPU verwenden, um die Transkription erheblich zu beschleunigen. Die GPU-Beschleunigung erfordert, dass das NVIDIA CUDA Toolkit und die cuDNN-Laufzeitbibliotheken auf Ihrem System installiert sind.

## Voraussetzungen

- Eine NVIDIA GPU mit CUDA-Unterstützung (empfohlen wird eine GeForce GTX 10-Serie oder neuer).
- Windows 10 oder 11 (64-Bit).
- Die Modelldateien müssen bereits heruntergeladen sein. Siehe [Modelle herunterladen](downloading_models.md).

## Installationsschritte

### 1. Das CUDA Toolkit installieren

Laden Sie das CUDA Toolkit-Installationsprogramm von der NVIDIA-Entwicklerwebsite herunter und führen Sie es aus. Akzeptieren Sie während der Installation die Standardpfade. Das Installationsprogramm setzt die Umgebungsvariable `CUDA_PATH` automatisch — Vernacula-Desktop verwendet diese Variable, um die CUDA-Bibliotheken zu finden.

### 2. cuDNN installieren

Laden Sie das cuDNN-ZIP-Archiv für Ihre installierte CUDA-Version von der NVIDIA-Entwicklerwebsite herunter. Entpacken Sie das Archiv und kopieren Sie den Inhalt der Ordner `bin`, `include` und `lib` in die entsprechenden Ordner innerhalb Ihres CUDA Toolkit-Installationsverzeichnisses (der durch `CUDA_PATH` angezeigte Pfad).

Alternativ können Sie cuDNN mithilfe des NVIDIA cuDNN-Installationsprogramms installieren, sofern eines für Ihre CUDA-Version verfügbar ist.

### 3. Die Anwendung neu starten

Schließen Sie Vernacula-Desktop nach der Installation und öffnen Sie es erneut. Die Anwendung prüft beim Start, ob CUDA vorhanden ist.

## GPU-Status in den Einstellungen

Öffnen Sie `Settings…` in der Menüleiste und sehen Sie sich den Abschnitt **Hardware & Performance** an. Jede Komponente zeigt ein Häkchen (✓), sobald sie erkannt wurde:

| Element | Bedeutung |
|---|---|
| GPU-Name und VRAM | Ihre NVIDIA GPU wurde gefunden |
| CUDA Toolkit ✓ | CUDA-Bibliotheken wurden über `CUDA_PATH` gefunden |
| cuDNN ✓ | cuDNN-Laufzeit-DLLs wurden gefunden |
| CUDA Acceleration ✓ | ONNX Runtime hat den CUDA-Ausführungsanbieter geladen |

Fehlt ein Eintrag nach der Installation, klicken Sie auf `Re-check`, um die Hardwareerkennung erneut auszuführen, ohne die Anwendung neu zu starten.

Das Einstellungsfenster enthält außerdem direkte Download-Links für das CUDA Toolkit und cuDNN, sofern diese noch nicht installiert sind.

### Fehlerbehebung

Wenn `CUDA Acceleration` kein Häkchen anzeigt, überprüfen Sie Folgendes:

- Die Umgebungsvariable `CUDA_PATH` ist gesetzt (prüfen Sie `System > Erweiterte Systemeinstellungen > Umgebungsvariablen`).
- Die cuDNN-DLLs befinden sich in einem Verzeichnis, das in Ihrem System-`PATH` eingetragen ist, oder im CUDA-Ordner `bin`.
- Ihr GPU-Treiber ist aktuell.

### Stapelverarbeitung

Wenn CUDA aktiv ist, zeigt der Abschnitt **Hardware & Performance** außerdem die aktuelle dynamische Stapelgrenze an — die maximale Anzahl an Audiosekunden, die in einem GPU-Durchlauf verarbeitet werden. Dieser Wert wird anhand des freien VRAMs nach dem Laden der Modelle berechnet und passt sich automatisch an, wenn sich Ihr verfügbarer Arbeitsspeicher ändert.

## Betrieb ohne GPU

Wenn CUDA nicht verfügbar ist, wechselt Vernacula-Desktop automatisch zur CPU-Verarbeitung. Die Transkription funktioniert weiterhin, ist jedoch langsamer — insbesondere bei langen Audiodateien.

---