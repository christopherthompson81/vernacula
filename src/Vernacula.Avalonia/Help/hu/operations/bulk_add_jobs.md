---
title: "Több hangfájl sorba állítása"
description: "Hogyan adhatunk több hangfájlt egyszerre a feladatsorhoz."
topic_id: operations_bulk_add_jobs
---

# Több hangfájl sorba állítása

A **Tömeges feladatfelvétel** funkcióval egyszerre több hang- vagy videófájlt is beküldhet átírásra. Az alkalmazás a hozzáadás sorrendjében, egyenként dolgozza fel őket.

## Előfeltételek

- Az összes modellfájlnak le kell töltve lennie. A **Modell állapota** kártyán a következőnek kell megjelennie: `All N model file(s) present ✓`. Lásd: [Modellek letöltése](../first_steps/downloading_models.md).

## A tömeges feladatfelvétel menete

1. A főképernyőn kattintson a `Bulk Add Jobs` gombra.
2. Megnyílik egy fájlböngésző. Válasszon ki egy vagy több hang- vagy videófájlt — több fájl kijelöléséhez tartsa lenyomva a `Ctrl` vagy a `Shift` billentyűt.
3. Kattintson az **Megnyitás** gombra. Minden kijelölt fájl külön feladatként jelenik meg az **Átírási előzmények** táblázatban.

> **Több hangsávot tartalmazó videófájlok:** Ha egy videófájl egynél több hangsávot tartalmaz (például több nyelvet vagy rendezői kommentárt), az alkalmazás automatikusan minden hangsávhoz külön feladatot hoz létre.

## Feladatnevek

Minden feladat neve automatikusan a hozzá tartozó hangfájl nevéből származik. A feladatot bármikor átnevezheti: kattintson a nevére az Átírási előzmények táblázat **Cím** oszlopában, szerkessze a szöveget, majd nyomja meg az `Enter` billentyűt, vagy kattintson máshova.

## A sor működése

- Ha éppen nem fut feladat, az első fájl feldolgozása azonnal elindul, a többi `queued` állapotban jelenik meg.
- Ha már fut egy feladat, az összes újonnan hozzáadott fájl `queued` állapotban jelenik meg, és automatikusan, sorban egymás után indul el.
- Az aktív feladat nyomon követéséhez kattintson a `Monitor` gombra az adott feladat **Műveletek** oszlopában. Lásd: [Feladatok figyelése](monitoring_jobs.md).
- Ha egy sorban álló feladatot szüneteltetni vagy eltávolítani szeretne még az indulása előtt, használja a `Pause` vagy a `Remove` gombot az adott feladat **Műveletek** oszlopában. Lásd: [Feladatok szüneteltetése, folytatása vagy eltávolítása](pausing_resuming_removing.md).

---