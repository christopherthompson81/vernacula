---
title: "Mudelite allalaadimine"
description: "Kuidas alla laadida transkriptsiooniks vajalikud AI-mudelite failid."
topic_id: first_steps_downloading_models
---

# Mudelite allalaadimine

Parakeet Transcription vajab töötamiseks AI-mudelite faile. Need ei ole rakendusega kaasas ja tuleb enne esimest transkriptsiooni alla laadida.

## Mudeli olek (avakuva)

Avakuva ülaosas olev õhuke olekuriba näitab, kas teie mudelid on valmis. Kui failid puuduvad, kuvatakse ka nupp `Open Settings`, mis viib teid otse mudelite haldamise vaatesse.

| Olek | Tähendus |
|---|---|
| `All N model file(s) present ✓` | Kõik vajalikud failid on alla laaditud ja kasutamiseks valmis. |
| `N model file(s) missing: …` | Üks või mitu faili puudub; avage allalaadimiseks Seaded. |

Kui mudelid on valmis, muutuvad nupud `New Transcription` ja `Bulk Add Jobs` aktiivseks.

## Kuidas mudeleid alla laadida

1. Klõpsake avakuval `Open Settings` (või minge `Settings… > Models`).
2. Jaotises **Models** klõpsake `Download Missing Models`.
3. Ilmub edenemisnäidik ja olekurida, mis näitavad praegust faili, selle järjekorranumbrit ja allalaadimise suurust — näiteks: `[1/3] encoder-model.onnx — 42 MB`.
4. Oodake, kuni olek muutub `Download complete.`

## Allalaadimise tühistamine

Allalaadimise peatamiseks klõpsake `Cancel`. Olekureal kuvatakse `Download cancelled.` Osaliselt alla laaditud failid säilitatakse, nii et järgmisel korral, kui klõpsate `Download Missing Models`, jätkub allalaadimine sealt, kus see pooleli jäi.

## Allalaadimise vead

Kui allalaadimine ebaõnnestub, kuvatakse olekureal `Download failed: <reason>`. Kontrollige oma internetiühendust ja klõpsake uuesti `Download Missing Models`, et proovida uuesti. Rakendus jätkab viimasest edukalt lõpetatud failist.

## Täpsuse muutmine

Alla laaditavad mudelifailid sõltuvad valitud **Model Precision** seadistusest. Selle muutmiseks minge `Settings… > Models > Model Precision`. Kui vahetate täpsust pärast allalaadimist, tuleb uus failikomplekt eraldi alla laadida. Vaadake [Mudeli kaalu täpsuse valimine](model_precision.md).

---