---
title: "Nastavitve"
description: "Pregled vseh možnosti v oknu Nastavitve."
topic_id: first_steps_settings_window
---

# Nastavitve

Okno **Nastavitve** vam omogoča nadzor nad konfiguracijo strojne opreme, upravljanjem modelov, načinom segmentacije, obnašanjem urejevalnika, videzom in jezikom. Odprite ga iz menijske vrstice: `Settings…`.

## Strojna oprema in zmogljivost

Ta razdelek prikazuje stanje vaše NVIDIA GPU in programskega sklada CUDA ter poroča o dinamični meji paketov, ki se uporablja med prepisovanjem z GPU.

| Element | Opis |
|---|---|
| Ime GPU in VRAM | Zaznana NVIDIA GPU in razpoložljivi video pomnilnik. |
| CUDA Toolkit | Ali so bile knjižnice CUDA zaznane prek `CUDA_PATH`. |
| cuDNN | Ali so na voljo knjižnice DLL za cuDNN. |
| Pospeševanje CUDA | Ali je ONNX Runtime uspešno naložil ponudnika izvajanja CUDA. |

Kliknite `Re-check`, da znova zaženete zaznavanje strojne opreme brez ponovnega zagona aplikacije — koristno po namestitvi CUDA ali cuDNN.

Ko katera od teh komponent ni zaznana, so prikazane neposredne povezave za prenos CUDA Toolkit in cuDNN.

Sporočilo o **meji paketov** poroča, koliko sekund zvoka se obdela v vsakem zagonu GPU. Ta vrednost se izračuna na podlagi prostega VRAM po nalaganju modelov in se samodejno prilagaja.

Za celotna navodila za namestitev CUDA glejte [Namestitev CUDA in cuDNN](cuda_installation.md).

## Modeli

Ta razdelek upravlja datoteke modelov UI, potrebne za prepisovanje.

- **Prenos manjkajočih modelov** — prenese vse datoteke modelov, ki še niso prisotne na disku. Vrstica napredka in vrstica stanja sledita vsaki datoteki med prenosom.
- **Preverjanje posodobitev** — preveri, ali so na voljo novejše uteži modela. Ko so zaznane posodobljene uteži, se na domačem zaslonu samodejno prikaže tudi pasica za posodobitev.

## Način segmentacije

Nadzoruje, kako se zvok razdeli na segmente pred prepoznavanjem govora.

| Način | Opis |
|---|---|
| **Diarizacija govorca** | Uporablja model SortFormer za prepoznavanje posameznih govorcev in označi vsak segment. Najprimernejše za intervjuje, sestanke in posnetke z več govorci. |
| **Zaznavanje glasovne aktivnosti** | Uporablja Silero VAD za zaznavanje govornih območij brez oznak govorcev. Hitrejše od diarizacije in primerno za zvok z enim govorcem. |

## Urejevalnik prepisov

**Privzeti način predvajanja** — nastavi način predvajanja, ki se uporabi ob odpiranju urejevalnika prepisov. Kadar koli ga lahko spremenite tudi neposredno v urejevalniku. Opis posameznih načinov najdete v razdelku [Urejanje prepisov](../operations/editing_transcripts.md).

## Videz

Izberite temo **Temna** ali **Svetla**. Sprememba se uveljavi takoj. Glejte [Izbira teme](theme.md).

## Jezik

Izberite jezik prikaza za vmesnik aplikacije. Sprememba se uveljavi takoj. Glejte [Izbira jezika](language.md).

---