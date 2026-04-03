---
title: "CUDA ir cuDNN įdiegimas GPU spartinimui"
description: "Kaip sukonfigūruoti NVIDIA CUDA ir cuDNN, kad Parakeet Transcription galėtų naudoti jūsų GPU."
topic_id: first_steps_cuda_installation
---

# CUDA ir cuDNN įdiegimas GPU spartinimui

Parakeet Transcription gali naudoti NVIDIA GPU, kad žymiai pagreitintų transkripciją. GPU spartinimui reikalinga, kad jūsų sistemoje būtų įdiegti NVIDIA CUDA Toolkit ir cuDNN vykdymo bibliotekos.

## Reikalavimai

- NVIDIA GPU, palaikantis CUDA (rekomenduojama GeForce GTX 10 serija arba naujesnė).
- Windows 10 arba 11 (64 bitų).
- Modelių failai turi būti jau atsisiųsti. Žr. [Modelių atsisiuntimas](downloading_models.md).

## Diegimo veiksmai

### 1. Įdiekite CUDA Toolkit

Atsisiųskite ir paleiskite CUDA Toolkit diegimo programą iš NVIDIA kūrėjų svetainės. Diegimo metu palikite numatytuosius kelius. Diegimo programa automatiškai nustato aplinkos kintamąjį `CUDA_PATH` — Parakeet naudoja šį kintamąjį CUDA bibliotekoms rasti.

### 2. Įdiekite cuDNN

Atsisiųskite cuDNN ZIP archyvą, skirtą jūsų įdiegtai CUDA versijai, iš NVIDIA kūrėjų svetainės. Išskleiskite archyvą ir nukopijuokite jo aplankų `bin`, `include` ir `lib` turinį į atitinkamus aplankus CUDA Toolkit diegimo kataloge (kelyje, kurį nurodo `CUDA_PATH`).

Arba įdiekite cuDNN naudodami NVIDIA cuDNN diegimo programą, jei ji prieinama jūsų CUDA versijai.

### 3. Paleiskite programą iš naujo

Po įdiegimo uždarykite ir vėl atidarykite Parakeet Transcription. Programa tikrina CUDA prieinamumą paleidimo metu.

## GPU būsena nustatymuose

Atidarykite `Settings…` meniu juostoje ir pažvelkite į skyrių **Hardware & Performance**. Kiekvienas komponentas rodo varnelę (✓), kai aptinkamas:

| Elementas | Ką tai reiškia |
|---|---|
| GPU pavadinimas ir VRAM | Jūsų NVIDIA GPU rastas |
| CUDA Toolkit ✓ | CUDA bibliotekos rastos per `CUDA_PATH` |
| cuDNN ✓ | cuDNN vykdymo DLL failai rasti |
| CUDA Acceleration ✓ | ONNX Runtime įkėlė CUDA vykdymo teikėją |

Jei po įdiegimo kuris nors elementas nerodomas, spustelėkite `Re-check`, kad iš naujo paleistumėte aparatinės įrangos aptikimą nepaleidžiant programos iš naujo.

Nustatymų lange taip pat pateiktos tiesioginės CUDA Toolkit ir cuDNN atsisiuntimo nuorodos, jei jos dar neįdiegtos.

### Trikčių šalinimas

Jei `CUDA Acceleration` nerodo varnelės, patikrinkite, ar:

- Aplinkos kintamasis `CUDA_PATH` yra nustatytas (patikrinkite `System > Advanced system settings > Environment Variables`).
- cuDNN DLL failai yra kataloge, esančiame sistemos `PATH` kintamajame, arba CUDA aplanke `bin`.
- Jūsų GPU tvarkyklė yra atnaujinta.

### Paketų dydis

Kai CUDA veikia, skyriuje **Hardware & Performance** taip pat rodoma dabartinė dinamiška paketo riba — maksimalus garso sekundžių skaičius, apdorojamas per vieną GPU vykdymą. Ši reikšmė apskaičiuojama pagal laisvą VRAM po modelių įkėlimo ir automatiškai koreguojasi, jei keičiasi turima atmintis.

## Naudojimas be GPU

Jei CUDA nėra prieinama, Parakeet automatiškai pereina prie apdorojimo naudojant CPU. Transkripcija vis tiek veikia, tačiau bus lėtesnė, ypač ilgiems garso failams.

---