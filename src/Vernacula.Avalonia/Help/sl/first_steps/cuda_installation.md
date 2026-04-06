---
title: "Namestitev CUDA in cuDNN za pospeševanje z GPU"
description: "Kako nastaviti NVIDIA CUDA in cuDNN, da lahko Vernacula-Desktop uporablja vaš GPU."
topic_id: first_steps_cuda_installation
---

# Namestitev CUDA in cuDNN za pospeševanje z GPU

Vernacula-Desktop lahko za občutno pospeševanje prepisovanja uporabi grafično kartico NVIDIA GPU. Pospeševanje z GPU zahteva, da sta na vašem sistemu nameščena NVIDIA CUDA Toolkit in knjižnice cuDNN.

## Zahteve

- Grafična kartica NVIDIA GPU z podporo za CUDA (priporočena je serija GeForce GTX 10 ali novejša).
- Windows 10 ali 11 (64-bitni).
- Datoteke modelov morajo biti že prenesene. Glejte [Prenos modelov](downloading_models.md).

## Koraki namestitve

### 1. Namestite CUDA Toolkit

Prenesite in zaženite namestitveni program CUDA Toolkit s spletnega mesta za razvijalce NVIDIA. Med namestitvijo sprejmite privzete poti. Namestitveni program samodejno nastavi okoljsko spremenljivko `CUDA_PATH` — Vernacula-Desktop jo uporablja za iskanje knjižnic CUDA.

### 2. Namestite cuDNN

S spletnega mesta za razvijalce NVIDIA prenesite arhiv ZIP za cuDNN, ki ustreza nameščeni različici CUDA. Razpakirajte arhiv in vsebino njegovih map `bin`, `include` in `lib` kopirajte v ustrezne mape v namestitvenem imeniku CUDA Toolkit (pot, ki jo prikazuje `CUDA_PATH`).

Druga možnost je, da cuDNN namestite z namestitvenim programom NVIDIA cuDNN, če je na voljo za vašo različico CUDA.

### 3. Znova zaženite aplikacijo

Po namestitvi zaprite in znova odprite Vernacula-Desktop. Aplikacija ob zagonu preveri prisotnost CUDA.

## Stanje GPU v nastavitvah

V menijski vrstici odprite `Settings…` in poiščite razdelek **Hardware & Performance**. Vsaka komponenta ob zaznavi prikaže kljukico (✓):

| Element | Pomen |
|---|---|
| Ime GPU in VRAM | Najdena je bila vaša grafična kartica NVIDIA GPU |
| CUDA Toolkit ✓ | Knjižnice CUDA so bile najdene prek `CUDA_PATH` |
| cuDNN ✓ | Najdene so bile knjižnice DLL za cuDNN |
| CUDA Acceleration ✓ | ONNX Runtime je naložil ponudnika izvajanja CUDA |

Če kateri element po namestitvi manjka, kliknite `Re-check`, da znova zaženete zaznavanje strojne opreme brez ponovnega zagona aplikacije.

Okno z nastavitvami prav tako ponuja neposredne povezave za prenos CUDA Toolkit in cuDNN, če še nista nameščena.

### Odpravljanje težav

Če `CUDA Acceleration` ne prikazuje kljukice, preverite:

- Ali je nastavljena okoljska spremenljivka `CUDA_PATH` (preverite `System > Advanced system settings > Environment Variables`).
- Ali se knjižnice DLL za cuDNN nahajajo v imeniku, ki je naveden v sistemski spremenljivki `PATH`, ali v mapi `bin` znotraj imenika CUDA.
- Ali so gonilniki vaše grafične kartice posodobljeni.

### Velikost serije

Ko je CUDA aktiven, razdelek **Hardware & Performance** prikaže tudi trenutno dinamično zgornjo mejo serije — največje število sekund zvoka, ki se obdela v enem zagonu GPU. Ta vrednost se izračuna na podlagi prostega pomnilnika VRAM po nalaganju modelov in se samodejno prilagodi, če se razpoložljivi pomnilnik spremeni.

## Delovanje brez GPU

Če CUDA ni na voljo, Vernacula-Desktop samodejno preklopi na obdelavo s procesorjem (CPU). Prepisovanje še vedno deluje, vendar bo počasnejše, zlasti pri dolgih zvočnih datotekah.

---