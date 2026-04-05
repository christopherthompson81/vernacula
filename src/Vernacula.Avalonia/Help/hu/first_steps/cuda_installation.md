---
title: "A CUDA és cuDNN telepítése GPU-gyorsításhoz"
description: "Útmutató az NVIDIA CUDA és cuDNN beállításához, hogy a Parakeet Transcription használhassa a GPU-t."
topic_id: first_steps_cuda_installation
---

# A CUDA és cuDNN telepítése GPU-gyorsításhoz

A Parakeet Transcription képes NVIDIA GPU-t használni az átírás jelentős felgyorsításához. A GPU-gyorsításhoz az NVIDIA CUDA Toolkit és a cuDNN futtatókörnyezeti könyvtárak telepítése szükséges a rendszeren.

## Követelmények

- CUDA-t támogató NVIDIA GPU (GeForce GTX 10-es sorozat vagy újabb ajánlott).
- Windows 10 vagy 11 (64 bites).
- A modellfájloknak már letöltöttnek kell lenniük. Lásd: [Modellek letöltése](downloading_models.md).

## Telepítési lépések

### 1. A CUDA Toolkit telepítése

Töltse le és futtassa a CUDA Toolkit telepítőjét az NVIDIA fejlesztői webhelyéről. A telepítés során fogadja el az alapértelmezett útvonalakat. A telepítő automatikusan beállítja a `CUDA_PATH` környezeti változót — a Parakeet ezt a változót használja a CUDA könyvtárak megkereséséhez.

### 2. A cuDNN telepítése

Töltse le a telepített CUDA-verziójához tartozó cuDNN ZIP-archívumot az NVIDIA fejlesztői webhelyéről. Csomagolja ki az archívumot, majd másolja a `bin`, `include` és `lib` mappák tartalmát a CUDA Toolkit telepítési könyvtárán belüli megfelelő mappákba (az útvonalat a `CUDA_PATH` jelzi).

Alternatív megoldásként telepítse a cuDNN-t az NVIDIA cuDNN telepítőjével, ha az elérhető az Ön CUDA-verziójához.

### 3. Az alkalmazás újraindítása

A telepítés után zárja be, majd nyissa meg újra a Parakeet Transcription alkalmazást. Az alkalmazás indításkor ellenőrzi a CUDA jelenlétét.

## GPU-állapot a beállításokban

Nyissa meg a `Settings…` menüpontot a menüsávból, és tekintse meg a **Hardware & Performance** szakaszt. Minden összetevőnél pipa (✓) jelenik meg, ha észlelésre került:

| Elem | Mit jelent |
|---|---|
| GPU neve és VRAM | Az NVIDIA GPU megtalálható |
| CUDA Toolkit ✓ | A CUDA könyvtárak megtalálhatók a `CUDA_PATH` alapján |
| cuDNN ✓ | A cuDNN futtatókörnyezeti DLL-ek megtalálhatók |
| CUDA Acceleration ✓ | Az ONNX Runtime betöltötte a CUDA végrehajtási szolgáltatót |

Ha valamely elem hiányzik a telepítés után, kattintson a `Re-check` gombra a hardverészlelés újrafuttatásához az alkalmazás újraindítása nélkül.

A Beállítások ablak közvetlen letöltési hivatkozásokat is tartalmaz a CUDA Toolkithez és a cuDNN-hez, ha azok még nincsenek telepítve.

### Hibaelhárítás

Ha a `CUDA Acceleration` nem mutat pipát, ellenőrizze a következőket:

- A `CUDA_PATH` környezeti változó be van-e állítva (ellenőrizze a `System > Advanced system settings > Environment Variables` menüpontban).
- A cuDNN DLL-ek egy, a rendszer `PATH`-jában szereplő könyvtárban vagy a CUDA `bin` mappájában találhatók-e.
- A GPU-illesztőprogram naprakész-e.

### Kötegméret

Ha a CUDA aktív, a **Hardware & Performance** szakasz az aktuális dinamikus kötegplafonértéket is megjeleníti — azt a maximális másodpercnyi hanganyagot, amelyet egy GPU-futtatás során dolgoz fel. Ezt az értéket a modellek betöltése után rendelkezésre álló szabad VRAM alapján számítja ki a rendszer, és automatikusan igazodik, ha a rendelkezésre álló memória megváltozik.

## Futtatás GPU nélkül

Ha a CUDA nem érhető el, a Parakeet automatikusan visszavált CPU-alapú feldolgozásra. Az átírás ekkor is működik, de lassabb lesz, különösen hosszú hangfájlok esetén.

---