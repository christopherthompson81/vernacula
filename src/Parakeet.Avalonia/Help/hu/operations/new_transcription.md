---
title: "Új átírási munkafolyamat"
description: "Lépésről lépésre útmutató egy hangfájl átírásához."
topic_id: operations_new_transcription
---

# Új átírási munkafolyamat

Ezzel a munkafolyamattal egyetlen hangfájlt írhat át.

## Előfeltételek

- Minden modellfájlt le kell tölteni. A **Modell állapota** kártyán az `All N model file(s) present ✓` üzenetnek kell megjelennie. Lásd: [Modellek letöltése](../first_steps/downloading_models.md).

## Támogatott formátumok

### Hang

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Videó

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

A videófájlok dekódolása FFmpeg segítségével történik. Ha egy videófájl **több hangsávot** tartalmaz (pl. több nyelv vagy kommentársáv), a rendszer automatikusan külön átírási feladatot hoz létre minden egyes hangsávhoz.

## Lépések

### 1. Nyissa meg az Új átírás űrlapot

Kattintson a `New Transcription` gombra a kezdőképernyőn, vagy lépjen a `File > New Transcription` menüpontra.

### 2. Válasszon médiafájlt

Kattintson a `Browse…` gombra a **Hangfájl** mező mellett. Megnyílik egy fájlválasztó párbeszédablak, amely a támogatott hang- és videóformátumokra van szűrve. Válassza ki a fájlt, majd kattintson a **Megnyitás** gombra. A fájl elérési útvonala megjelenik a mezőben.

### 3. Nevezze el a feladatot

A **Feladat neve** mező automatikusan kitöltődik a fájlnévből. Szerkessze meg, ha más nevet szeretne megadni — ez a név jelenik meg az átírási előzményekben a kezdőképernyőn.

### 4. Indítsa el az átírást

Kattintson a `Start Transcription` gombra. Az alkalmazás átvált a **Folyamat** nézetbe.

Ha visszalép az indítás nélkül, kattintson a `← Back` gombra.

## Mi történik ezután

A feladat két fázison megy keresztül, amelyek az állapotjelző sávon is láthatók:

1. **Hanganalízis** — hangszóró-szétválasztás (diarizáció): annak azonosítása, ki beszél és mikor.
2. **Beszédfelismerés** — a beszéd szöveggé alakítása szegmensről szegmensre.

Az átírt szegmensek az élő táblázatban jelennek meg, ahogy elkészülnek. A feldolgozás befejezésekor az alkalmazás automatikusan átlép az **Eredmények** nézetbe.

Ha egy másik feladat futása közben ad hozzá új feladatot, az új feladat `queued` állapotban jelenik meg, és akkor indul el, amikor az aktuális feladat befejeződik. Lásd: [Feladatok figyelése](monitoring_jobs.md).

---