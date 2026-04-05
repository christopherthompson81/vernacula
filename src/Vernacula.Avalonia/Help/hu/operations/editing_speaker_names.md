---
title: "Előadók neveinek szerkesztése"
description: "Hogyan lehet az általános előadóazonosítókat valódi nevekre cserélni az átiratban."
topic_id: operations_editing_speaker_names
---

# Előadók neveinek szerkesztése

Az átírási motor automatikusan általános azonosítóval látja el az egyes előadókat (például `speaker_0`, `speaker_1`). Ezeket valódi nevekre cserélheti, amelyek az átirat egészében és az exportált fájlokban is megjelennek.

## Az előadók nevének szerkesztése

1. Nyisson meg egy befejezett feladatot. Lásd: [Befejezett feladatok betöltése](loading_completed_jobs.md).
2. Az **Eredmények** nézetben kattintson az `Edit Speaker Names` gombra.
3. Megnyílik az **Edit Speaker Names** párbeszédablak, amelyben két oszlop látható:
   - **Speaker ID** — a modell által hozzárendelt eredeti azonosító (csak olvasható).
   - **Display Name** — az átiratban megjelenített név (szerkeszthető).
4. Kattintson egy cellára a **Display Name** oszlopban, és írja be az előadó nevét.
5. Nyomja meg a `Tab` billentyűt, vagy kattintson egy másik sorra a következő előadóra lépéshez.
6. Kattintson a `Save` gombra a módosítások alkalmazásához, vagy a `Cancel` gombra az elvetéshez.

## Hol jelennek meg a nevek

A frissített megjelenítési nevek az általános azonosítók helyére kerülnek a következő helyeken:

- Az Eredmények nézet szegmenstáblájában.
- Az összes exportált fájlban (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Nevek ismételt szerkesztése

Az Edit Speaker Names párbeszédablakot bármikor újra megnyithatja, amíg a feladat betöltve van az Eredmények nézetben. A módosítások a helyi adatbázisba kerülnek mentésre, és munkamenetek között is megmaradnak.

---