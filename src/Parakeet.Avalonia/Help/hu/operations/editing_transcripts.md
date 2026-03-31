---
title: "Átiratok szerkesztése"
description: "Az átiratok szerkesztőjében hogyan lehet áttekinteni, javítani és ellenőrzöttnek jelölni az átírt szegmenseket."
topic_id: operations_editing_transcripts
---

# Átiratok szerkesztése

Az **Átiratszerkesztő** lehetővé teszi az ASR kimenet áttekintését, a szöveg javítását, a hangszórók nevének helyi átírását, a szegmensek időzítésének módosítását, és a szegmensek ellenőrzöttnek jelölését — mindezt az eredeti hang meghallgatása közben.

## A szerkesztő megnyitása

1. Töltsön be egy befejezett feladatot (lásd: [Befejezett feladatok betöltése](loading_completed_jobs.md)).
2. Az **Eredmények** nézetben kattintson az `Edit Transcript` gombra.

A szerkesztő külön ablakban nyílik meg, és a fő alkalmazás mellett nyitva maradhat.

## Elrendezés

Minden szegmens egy kártyaként jelenik meg, két egymás melletti panellel:

- **Bal panel** — az eredeti ASR kimenet szavanként megjelenített megbízhatósági színezéssel. Azok a szavak, amelyekben a modell kevésbé volt biztos, pirossal jelennek meg; a nagy megbízhatóságú szavak a normál szövegszínnel láthatók.
- **Jobb panel** — szerkeszthető szövegmező. Itt végezze el a javításokat; az eredeti szövegtől való eltérés gépelés közben ki van emelve.

A hangszóró neve és az időtartomány minden kártya felett megjelenik. Kattintson egy kártyára a fókuszáláshoz és a műveleti ikonok megjelenítéséhez. Vigye az egeret bármely ikon fölé, hogy megjelenjen a funkcióját leíró eszköztipp.

## Ikonok magyarázata

### Lejátszási sáv

| Ikon | Művelet |
|------|--------|
| ▶ | Lejátszás |
| ⏸ | Szünet |
| ⏮ | Ugrás az előző szegmensre |
| ⏭ | Ugrás a következő szegmensre |

### Szegmenskártya-műveletek

| Ikon | Művelet |
|------|--------|
| <mdl2 ch="E77B"/> | Szegmens átrendelése másik hangszóróhoz |
| <mdl2 ch="E916"/> | Szegmens kezdési és befejezési idejének módosítása |
| <mdl2 ch="EA39"/> | Szegmens elnyomása vagy visszaállítása |
| <mdl2 ch="E72B"/> | Összevonás az előző szegmenssel |
| <mdl2 ch="E72A"/> | Összevonás a következő szegmenssel |
| <mdl2 ch="E8C6"/> | Szegmens felosztása |
| <mdl2 ch="E72C"/> | ASR újrafuttatása ezen a szegmensen |

## Hanglejátszás

Egy lejátszási sáv húzódik végig a szerkesztőablak tetején:

| Vezérlő | Művelet |
|---------|--------|
| Lejátszás / Szünet ikon | Lejátszás indítása vagy szüneteltetése |
| Keresősáv | Húzza a kívánt pozícióra a hangban |
| Sebességcsúszka | Lejátszási sebesség beállítása (0,5× – 2×) |
| Előző / Következő ikonok | Ugrás az előző vagy a következő szegmensre |
| Lejátszási mód legördülő | Három lejátszási mód egyikének kiválasztása (lásd alább) |
| Hangerőcsúszka | Lejátszási hangerő beállítása |

Lejátszás közben az éppen elhangzó szó ki van emelve a bal panelen. Ha keresés után szüneteltetjük, a kiemelés a keresési pozícióhoz tartozó szóra frissül.

### Lejátszási módok

| Mód | Viselkedés |
|------|-----------|
| `Single` | Az aktuális szegmens lejátszása egyszer, majd megállás. |
| `Auto-advance` | Az aktuális szegmens lejátszása; befejezésekor ellenőrzöttnek jelöli, és a következőre lép. |
| `Continuous` | Az összes szegmens egymás utáni lejátszása anélkül, hogy bármelyiket ellenőrzöttként jelölné. |

Az aktív módot a lejátszási sáv legördülő menüjéből lehet kiválasztani.

## Szegmens szerkesztése

1. Kattintson egy kártyára a fókuszáláshoz.
2. Szerkessze a szöveget a jobb panelen. A módosítások automatikusan mentődnek, amikor a fókuszt egy másik kártyára viszi.

## Hangszóró átnevezése

Kattintson a hangszóró nevére a fókuszált kártyán belül, és írjon be egy új nevet. Nyomja meg az `Enter` billentyűt, vagy kattintson el a mentéshez. Az új név csak arra a kártyára vonatkozik; egy hangszóró globális átnevezéséhez használja a [Hangszórónevek szerkesztése](editing_speaker_names.md) funkciót az Eredmények nézetből.

## Szegmens ellenőrzöttnek jelölése

Kattintson a `Verified` jelölőnégyzetre a fókuszált kártyán, hogy áttekintettként jelölje meg. Az ellenőrzött állapot az adatbázisba mentődik, és a jövőbeli betöltések során a szerkesztőben is látható lesz.

## Szegmens elnyomása

Kattintson a `Suppress` gombra egy fókuszált kártyán, hogy elrejtse a szegmenst az exportálásból (hasznos zajok, zene vagy egyéb nem beszédet tartalmazó szakaszok esetén). Kattintson az `Unsuppress` gombra a visszaállításhoz.

## Szegmensidők módosítása

Kattintson az `Adjust Times` gombra egy fókuszált kártyán az időmódosítás párbeszédpanel megnyitásához. Görgesse az egérkereket a **Start** vagy az **End** mező felett az érték 0,1 másodperces léptékű finomhangolásához, vagy írja be az értéket közvetlenül. Kattintson a `Save` gombra az alkalmazáshoz.

## Szegmensek összevonása

- Kattintson a `⟵ Merge` gombra, hogy az aktuálisan fókuszált szegmenst az előtte lévővel vonja össze.
- Kattintson a `Merge ⟶` gombra, hogy az aktuálisan fókuszált szegmenst az utána következővel vonja össze.

A két kártya szövege és időtartománya összekapcsolódik. Ez akkor hasznos, ha egyetlen elhangzott megnyilatkozást két szegmensre osztott fel a rendszer.

## Szegmens felosztása

Kattintson a `Split…` gombra egy fókuszált kártyán a felosztás párbeszédpanel megnyitásához. Helyezze a felosztási pontot a szövegen belülre, majd erősítse meg. Két új szegmens jön létre, amelyek az eredeti időtartományt fedik le. Ez akkor hasznos, ha két különálló megnyilatkozást egy szegmensbe vontak össze.

## ASR újrafuttatása

Kattintson a `Redo ASR` gombra egy fókuszált kártyán, hogy újrafuttassa a beszédfelismerést az adott szegmens hanganyagán. A modell csak az adott szegmens hangszeletét dolgozza fel, és egy új, egyforrású átírást készít.

Használja ezt, ha:

- Egy szegmens összevonásból keletkezett, és nem lehet felosztani (az összevont szegmensek több ASR forrást ölelnek fel; az ASR újrafuttatása ezeket eggyé vonja össze, ezt követően a `Split…` funkció elérhetővé válik).
- Az eredeti átírás gyenge minőségű, és manuális szerkesztés helyett egy tiszta második feldolgozást szeretne.

**Megjegyzés:** A jobb panelbe már begépelt szöveg elvész, és az új ASR kimenet váltja fel. A művelethez szükséges, hogy a hangfájl be legyen töltve; ha a hang nem elérhető, a gomb le van tiltva.