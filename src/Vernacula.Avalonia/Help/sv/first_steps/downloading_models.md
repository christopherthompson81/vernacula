---
title: "Ladda ned modeller"
description: "Hur du laddar ned de AI-modellfiler som krävs för transkribering."
topic_id: first_steps_downloading_models
---

# Ladda ned modeller

Vernacula-Desktop kräver AI-modellfiler för att fungera. Dessa medföljer inte programmet och måste laddas ned innan din första transkribering.

## Modellstatus (startskärmen)

En smal statusrad längst upp på startskärmen visar om dina modeller är redo. Om filer saknas visas även en `Open Settings`-knapp som tar dig direkt till modellhanteringen.

| Status | Betydelse |
|---|---|
| `All N model file(s) present ✓` | Alla nödvändiga filer är nedladdade och redo att användas. |
| `N model file(s) missing: …` | En eller flera filer saknas; öppna Inställningar för att ladda ned. |

När modellerna är redo aktiveras knapparna `New Transcription` och `Bulk Add Jobs`.

## Så här laddar du ned modeller

1. På startskärmen klickar du på `Open Settings` (eller går till `Settings… > Models`).
2. I avsnittet **Models** klickar du på `Download Missing Models`.
3. En förloppsindikator och en statusrad visas med information om aktuell fil, dess position i kön och filstorleken — till exempel: `[1/3] encoder-model.onnx — 42 MB`.
4. Vänta tills statusen visar `Download complete.`

## Avbryta en nedladdning

Om du vill stoppa en pågående nedladdning klickar du på `Cancel`. Statusraden visar då `Download cancelled.` Delvis nedladdade filer bevaras så att nedladdningen återupptas från samma ställe nästa gång du klickar på `Download Missing Models`.

## Nedladdningsfel

Om en nedladdning misslyckas visas `Download failed: <reason>` i statusraden. Kontrollera din internetanslutning och klicka på `Download Missing Models` igen för att försöka på nytt. Programmet återupptar nedladdningen från den senast slutförda filen.

## Ändra precision

Vilka modellfiler som behöver laddas ned beror på vald **Model Precision**. Om du vill ändra den går du till `Settings… > Models > Model Precision`. Om du byter precision efter att ha laddat ned filer måste den nya uppsättningen filer laddas ned separat. Se [Välja precision för modellvikter](model_precision.md).

---