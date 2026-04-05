---
title: "Feladatok szüneteltetése, folytatása vagy eltávolítása"
description: "Hogyan lehet szüneteltetni egy futó feladatot, folytatni egy leállított feladatot, vagy törölni egy feladatot az előzményekből."
topic_id: operations_pausing_resuming_removing
---

# Feladatok szüneteltetése, folytatása vagy eltávolítása

## Feladat szüneteltetése

Egy futó vagy várakozó feladatot két helyről lehet szüneteltetni:

- **Folyamatjelző nézet** — kattintson a `Pause` gombra a jobb alsó sarokban, miközben az aktív feladatot figyeli.
- **Átiratok előzményei táblázat** — kattintson a `Pause` gombra bármelyik sor **Actions** oszlopában, amelynek állapota `running` vagy `queued`.

A `Pause` gombra kattintás után az állapotsor `Pausing…` feliratot jelenít meg, amíg az alkalmazás befejezi az aktuális feldolgozási egységet. Ezután a feladat állapota `cancelled` értékre változik az előzmények táblázatában.

> A szüneteltetés menti az eddig elkészített összes szegmenst. A feladatot később folytathatja anélkül, hogy elveszítené az elvégzett munkát.

## Feladat folytatása

Egy szüneteltetett vagy sikertelen feladat folytatásához:

1. A kezdőképernyőn keresse meg a feladatot az **Átiratok előzményei** táblázatban. Állapota `cancelled` vagy `failed` lesz.
2. Kattintson a `Resume` gombra az **Actions** oszlopban.
3. Az alkalmazás visszatér a **Progress** nézetbe, és onnan folytatja a feldolgozást, ahol abbahagyta.

Az állapotsor röviden `Resuming…` feliratot jelenít meg, amíg a feladat újrainicializálódik.

## Feladat eltávolítása

Egy feladat és az átiratának végleges törléséhez az előzményekből:

1. Az **Átiratok előzményei** táblázatban kattintson a `Remove` gombra a törölni kívánt feladat **Actions** oszlopában.

A feladat eltűnik a listából, és adatai törlődnek a helyi adatbázisból. Ez a művelet nem vonható vissza. A lemezre mentett exportált fájlokat ez nem érinti.

---