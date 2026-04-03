---
title: "Feladatok figyelése"
description: "Hogyan követhető egy futó vagy várakozó feladat előrehaladása."
topic_id: operations_monitoring_jobs
---

# Feladatok figyelése

A **Folyamat** nézet élő betekintést nyújt egy futó átírási feladat állapotába.

## A Folyamat nézet megnyitása

- Új átírás indításakor az alkalmazás automatikusan a Folyamat nézetre vált.
- Már futó vagy várakozó feladat esetén keresse meg azt az **Átírási előzmények** táblában, majd kattintson a **Műveletek** oszlopban található `Monitor` gombra.

## A Folyamat nézet értelmezése

| Elem | Leírás |
|---|---|
| Folyamatjelző sáv | A teljes befejezettség százalékos aránya. Határozatlan (animált) állapotban jelenik meg, amíg a feladat elindul vagy folytatódik. |
| Százalékos felirat | A sáv jobb oldalán megjelenő numerikus százalékos érték. |
| Állapotüzenet | Az aktuális tevékenység — például `Audio Analysis` vagy `Speech Recognition`. `Waiting in queue…` üzenetet jelenít meg, ha a feladat még nem indult el. |
| Szegmenstábla | Az átírt szegmensek élő nézete a **Beszélő**, **Kezdet**, **Vég** és **Tartalom** oszlopokkal. Automatikusan görget, ahogy új szegmensek érkeznek. |

## A folyamat fázisai

A megjelenített fázisok a Beállításokban kiválasztott **Szegmentálási módtól** függnek.

**Beszélő-diarizáció mód** (alapértelmezett):

1. **Audio Analysis** — a SortFormer diarizáció a teljes fájlon végigfut a beszélők határainak azonosítása érdekében. A sáv közel maradhat a 0%-hoz, amíg ez a fázis be nem fejeződik.
2. **Speech Recognition** — minden egyes beszélői szegmens átírásra kerül. A százalékos érték egyenletesen emelkedik ebben a fázisban.

**Hangaktivitás-érzékelési mód**:

1. **Detecting speech segments** — a Silero VAD végigvizsgálja a fájlt a beszédet tartalmazó szakaszok megtalálásához. Ez a fázis gyors.
2. **Speech Recognition** — minden egyes érzékelt beszédszakasz átírásra kerül.

Mindkét módban az élő szegmenstábla fokozatosan töltődik fel az átírás előrehaladásával.

## Navigálás el a nézetből

Kattintson a `← Back to Home` gombra a kezdőképernyőre való visszatéréshez a feladat megszakítása nélkül. A feladat továbbra is fut a háttérben, és állapota frissül az **Átírási előzmények** táblában. Bármikor kattintson ismét a `Monitor` gombra a Folyamat nézethez való visszatéréshez.

---