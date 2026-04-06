---
title: "Modellek letöltése"
description: "Az átíráshoz szükséges AI-modellfájlok letöltésének módja."
topic_id: first_steps_downloading_models
---

# Modellek letöltése

A Vernacula-Desktop működéséhez AI-modellfájlokra van szükség. Ezek nem részei az alkalmazáscsomagnak, ezért az első átírás előtt le kell tölteni őket.

## Modell állapota (kezdőképernyő)

A kezdőképernyő tetején egy vékony állapotsor jelzi, hogy a modellek készen állnak-e. Ha fájlok hiányoznak, megjelenik egy `Open Settings` gomb is, amely közvetlenül a modellkezelési felületre vezet.

| Állapot | Jelentés |
|---|---|
| `All N model file(s) present ✓` | Az összes szükséges fájl le van töltve és készen áll. |
| `N model file(s) missing: …` | Egy vagy több fájl hiányzik; a letöltéshez nyissa meg a Beállításokat. |

Ha a modellek készen állnak, a `New Transcription` és a `Bulk Add Jobs` gombok aktívvá válnak.

## A modellek letöltése

1. A kezdőképernyőn kattintson az `Open Settings` gombra (vagy lépjen a `Settings… > Models` menüpontra).
2. A **Models** szakaszban kattintson a `Download Missing Models` gombra.
3. Megjelenik egy folyamatjelző sáv és egy állapotsor, amely mutatja az aktuális fájlt, annak helyzetét a sorban, valamint a letöltési méretet – például: `[1/3] encoder-model.onnx — 42 MB`.
4. Várja meg, amíg az állapot a következőre vált: `Download complete.`

## Letöltés megszakítása

A folyamatban lévő letöltés leállításához kattintson a `Cancel` gombra. Az állapotsor a `Download cancelled.` üzenetet jeleníti meg. A részlegesen letöltött fájlok megmaradnak, így a következő alkalommal, amikor a `Download Missing Models` gombra kattint, a letöltés onnan folytatódik, ahol abbahagyta.

## Letöltési hibák

Ha egy letöltés meghiúsul, az állapotsor a `Download failed: <reason>` üzenetet jeleníti meg. Ellenőrizze az internetkapcsolatát, majd kattintson ismét a `Download Missing Models` gombra az újrapróbálkozáshoz. Az alkalmazás az utoljára sikeresen befejezett fájltól folytatja a letöltést.
