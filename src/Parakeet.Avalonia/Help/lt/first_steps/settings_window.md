---
title: "Nustatymai"
description: "Visų nustatymų lango parinkčių apžvalga."
topic_id: first_steps_settings_window
---

# Nustatymai

**Nustatymų** langas suteikia galimybę valdyti aparatinės įrangos konfigūraciją, modelių tvarkymą, segmentavimo režimą, redaktoriaus elgseną, išvaizdą ir kalbą. Atidarykite jį iš meniu juostos: `Settings…`.

## Aparatinė įranga ir našumas

Šiame skyriuje rodoma NVIDIA GPU ir CUDA programinės įrangos rietuvės būsena bei pranešamas dinaminis paketų limitas, naudojamas GPU transkribavimo metu.

| Elementas | Aprašymas |
|---|---|
| GPU pavadinimas ir VRAM | Aptiktas NVIDIA GPU ir turima vaizdo atmintis. |
| CUDA Toolkit | Ar CUDA vykdymo laiko bibliotekos rastos per `CUDA_PATH`. |
| cuDNN | Ar cuDNN vykdymo laiko DLL failai yra prieinami. |
| CUDA spartinimas | Ar ONNX Runtime sėkmingai įkėlė CUDA vykdymo teikėją. |

Spustelėkite `Re-check`, kad iš naujo patikrintumėte aparatinę įrangą nepaleidę programos iš naujo — tai naudinga po CUDA arba cuDNN įdiegimo.

Tiesioginės atsisiuntimo nuorodos į CUDA Toolkit ir cuDNN rodomos, kai šie komponentai neaptinkami.

**Paketų limito** pranešimas nurodo, kiek sekundžių garso apdorojama per vieną GPU vykdymo ciklą. Ši reikšmė apskaičiuojama pagal laisvą VRAM po modelių įkėlimo ir koreguojama automatiškai.

Išsamias CUDA diegimo instrukcijas rasite skyriuje [CUDA ir cuDNN diegimas](cuda_installation.md).

## Modeliai

Šiame skyriuje tvarkomos transkribavimui reikalingos DI modelių bylos.

- **Modelio tikslumas** — pasirinkite `INT8 (smaller download)` arba `FP32 (more accurate)`. Žr. [Modelio svorių tikslumo pasirinkimas](model_precision.md).
- **Atsisiųsti trūkstamus modelius** — atsisiunčia modelių bylas, kurių dar nėra diske. Eigos juosta ir būsenos eilutė rodo kiekvienos bylos atsisiuntimo eigą.
- **Tikrinti naujinimus** — tikrina, ar yra naujesnių modelių svorių. Naujinimo pranešimas taip pat automatiškai rodomas pradžios ekrane, kai aptinkami atnaujinti svoriai.

## Segmentavimo režimas

Valdo tai, kaip garsas padalijamas į segmentus prieš kalbos atpažinimą.

| Režimas | Aprašymas |
|---|---|
| **Kalbėtojų atskyrimas** | Naudoja SortFormer modelį atskirų kalbėtojų identifikavimui ir kiekvieno segmento žymėjimui. Geriausiai tinka pokalbiams, susitikimams ir kelių kalbėtojų įrašams. |
| **Balso aktyvumo aptikimas** | Naudoja Silero VAD kalbos sritims aptikti — be kalbėtojų žymių. Greitesnis nei kalbėtojų atskyrimas ir puikiai tinka vieno kalbėtojo garsui. |

## Transkriptų redaktorius

**Numatytasis atkūrimo režimas** — nustato atkūrimo režimą, naudojamą atidarant transkriptų redaktorių. Jį taip pat galite keisti tiesiogiai redaktoriuje bet kuriuo metu. Kiekvieno režimo aprašymą rasite skyriuje [Transkriptų redagavimas](../operations/editing_transcripts.md).

## Išvaizda

Pasirinkite **tamsią** arba **šviesią** temą. Pakeitimas pritaikomas nedelsiant. Žr. [Temos pasirinkimas](theme.md).

## Kalba

Pasirinkite programos sąsajos rodymo kalbą. Pakeitimas pritaikomas nedelsiant. Žr. [Kalbos pasirinkimas](language.md).

---