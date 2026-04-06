---
title: "Beállítások"
description: "Az összes lehetőség áttekintése a Beállítások ablakban."
topic_id: first_steps_settings_window
---

# Beállítások

A **Beállítások** ablakban vezérelheted a hardverkonfigurációt, a modellkezelést, a szegmentálási módot, a szerkesztő viselkedését, a megjelenést és a nyelvet. A menüsorból nyitható meg: `Settings…`.

## Hardver és teljesítmény

Ez a szakasz megjeleníti az NVIDIA GPU és a CUDA szoftververem állapotát, valamint a GPU-s átiratozás során használt dinamikus kötegméret-korlátot.

| Elem | Leírás |
|---|---|
| GPU neve és VRAM | Az érzékelt NVIDIA GPU és az elérhető videomemória. |
| CUDA Toolkit | Megtalálhatók-e a CUDA futásidejű könyvtárak a `CUDA_PATH` útvonalon. |
| cuDNN | Elérhetők-e a cuDNN futásidejű DLL-fájlok. |
| CUDA gyorsítás | Az ONNX Runtime sikeresen betöltötte-e a CUDA végrehajtási szolgáltatót. |

Kattints a `Re-check` gombra a hardverészlelés újrafuttatásához az alkalmazás újraindítása nélkül — hasznos a CUDA vagy cuDNN telepítése után.

Ha ezek az összetevők nem észlelhetők, a CUDA Toolkit és a cuDNN közvetlen letöltési hivatkozásai is megjelennek.

A **kötegméret-korlát** üzenete arról tájékoztat, hogy hány másodpercnyi hanganyagot dolgoz fel egyszerre a GPU. Ez az érték a modellek betöltése után szabad VRAM alapján kerül meghatározásra, és automatikusan módosul.

A teljes CUDA-telepítési útmutatóért lásd: [A CUDA és cuDNN telepítése](cuda_installation.md).

## Modellek

Ez a szakasz az átiratozáshoz szükséges AI-modellfájlokat kezeli.

- **Hiányzó modellek letöltése** — letölti azokat a modellfájlokat, amelyek még nem találhatók meg a lemezen. Egy folyamatjelző sáv és egy állapotsor követi az egyes fájlok letöltését.
- **Frissítések keresése** — ellenőrzi, hogy elérhetők-e újabb modellsúlyok. Ha frissített súlyok észlelhetők, a kezdőképernyőn automatikusan megjelenik egy frissítési értesítés is.

## Szegmentálási mód

Meghatározza, hogyan osztja fel az alkalmazás a hanganyagot szegmensekre a beszédfelismerés előtt.

| Mód | Leírás |
|---|---|
| **Hangszóró-azonosítás** | A SortFormer modellt használja az egyes hangszórók azonosítására és az egyes szegmensek felcímkézésére. Interjúkhoz, megbeszélésekhez és több hangszórót tartalmazó felvételekhez ajánlott. |
| **Hangaktivitás-észlelés** | A Silero VAD segítségével csak a beszéd régióit érzékeli — hangszóró-azonosítás nélkül. Gyorsabb a hangszóró-azonosításnál, és egyhangszórós hanganyaghoz jól alkalmazható. |

## Átirat-szerkesztő

**Alapértelmezett lejátszási mód** — beállítja azt a lejátszási módot, amelyet az átirat-szerkesztő megnyitásakor használ a program. A szerkesztőben bármikor közvetlenül is módosítható. Az egyes módok leírásáért lásd: [Átiratok szerkesztése](../operations/editing_transcripts.md).

## Megjelenés

Válassz **Sötét** vagy **Világos** témát. A változtatás azonnal érvénybe lép. Lásd: [Téma kiválasztása](theme.md).

## Nyelv

Válaszd ki az alkalmazás felületének megjelenítési nyelvét. A változtatás azonnal érvénybe lép. Lásd: [Nyelv kiválasztása](language.md).

---