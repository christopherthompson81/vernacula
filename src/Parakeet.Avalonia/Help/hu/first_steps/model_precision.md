---
title: "A modell súlyainak pontosságának kiválasztása"
description: "Hogyan válasszunk az INT8 és az FP32 modellpontosság között, és mik az egyes lehetőségek előnyei és hátrányai."
topic_id: first_steps_model_precision
---

# A modell súlyainak pontosságának kiválasztása

A modell pontossága azt a számformátumot szabályozza, amelyet a mesterséges intelligencia modell súlyai használnak. Ez befolyásolja a letöltési méretet, a memóriahasználatot és a pontosságot.

## Pontossági lehetőségek

### INT8 (kisebb letöltés)

- Kisebb modellfájlok — gyorsabban letölthető, és kevesebb lemezterületet igényel.
- Néhány hanganyag esetén kissé alacsonyabb pontosság.
- Ajánlott, ha korlátozott lemezterülettel vagy lassabb internetkapcsolattal rendelkezik.

### FP32 (pontosabb)

- Nagyobb modellfájlok.
- Magasabb pontosság, különösen nehéz hanganyagoknál, ahol akcentus vagy háttérzaj van.
- Ajánlott, ha a pontosság az elsődleges szempont, és elegendő lemezterület áll rendelkezésre.
- Szükséges a CUDA GPU gyorsításhoz — a GPU-s útvonal mindig FP32-t használ, függetlenül ettől a beállítástól.

## A pontosság módosítása

Nyissa meg a `Settings…` menüpontot a menüsorból, majd lépjen a **Models** szakaszra, és válassza ki az `INT8 (smaller download)` vagy az `FP32 (more accurate)` lehetőséget.

## A pontosság módosítása után

A pontosság megváltoztatásához eltérő modellfájlokra van szükség. Ha az új pontossághoz tartozó modellek még nem töltődtek le, kattintson a `Download Missing Models` gombra a Beállításokban. A másik pontossághoz korábban letöltött fájlok megmaradnak a lemezen, és nem kell újra letölteni azokat, ha visszavált.

---