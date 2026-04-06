---
title: "Modelių atsisiuntimas"
description: "Kaip atsisiųsti AI modelių failus, reikalingus transkripcijai."
topic_id: first_steps_downloading_models
---

# Modelių atsisiuntimas

Norint paleisti „Vernacula-Desktop", reikalingi AI modelių failai. Jie nėra įtraukti į programos diegimo paketą ir turi būti atsisiųsti prieš pirmąją transkripciją.

## Modelių būsena (pagrindinis ekranas)

Viršutinėje pagrindinio ekrano dalyje rodoma siaura būsenos juosta, kurioje nurodoma, ar jūsų modeliai yra paruošti. Kai trūksta failų, taip pat rodomas mygtukas `Open Settings`, kuris nukreipia tiesiai į modelių valdymą.

| Būsena | Reikšmė |
|---|---|
| `All N model file(s) present ✓` | Visi reikalingi failai yra atsisiųsti ir paruošti. |
| `N model file(s) missing: …` | Vieno ar kelių failų trūksta; atidarykite nustatymus, kad atsisiųstumėte. |

Kai modeliai paruošti, mygtukai `New Transcription` ir `Bulk Add Jobs` tampa aktyvūs.

## Kaip atsisiųsti modelius

1. Pagrindiniame ekrane spustelėkite `Open Settings` (arba eikite į `Settings… > Models`).
2. Skyriuje **Models** spustelėkite `Download Missing Models`.
3. Pasirodo eigos juosta ir būsenos eilutė, kurioje rodomas dabartinis failas, jo vieta eilėje ir atsisiuntimo dydis — pavyzdžiui: `[1/3] encoder-model.onnx — 42 MB`.
4. Palaukite, kol būsena pasikeis į `Download complete.`

## Atsisiuntimo atšaukimas

Norėdami sustabdyti vykdomą atsisiuntimą, spustelėkite `Cancel`. Būsenos eilutėje bus rodoma `Download cancelled.` Iš dalies atsisiųsti failai išsaugomi, todėl kitą kartą spustelėjus `Download Missing Models` atsisiuntimas tęsiamas nuo tos vietos, kurioje buvo sustabdytas.

## Atsisiuntimo klaidos

Jei atsisiuntimas nepavyksta, būsenos eilutėje rodoma `Download failed: <reason>`. Patikrinkite interneto ryšį ir dar kartą spustelėkite `Download Missing Models`, kad bandytumėte iš naujo. Programa tęsia atsisiuntimą nuo paskutinio sėkmingai užbaigto failo.
