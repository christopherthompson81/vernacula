---
title: "Välja precision för modellvikter"
description: "Hur du väljer mellan INT8 och FP32 modellprecision och vilka avvägningar som gäller."
topic_id: first_steps_model_precision
---

# Välja precision för modellvikter

Modellprecision styr det numeriska format som används av AI-modellens vikter. Det påverkar nedladdningsstorlek, minnesanvändning och noggrannhet.

## Precisionsalternativ

### INT8 (mindre nedladdning)

- Mindre modellfiler — snabbare att ladda ned och kräver mindre diskutrymme.
- Något lägre noggrannhet för vissa typer av ljud.
- Rekommenderas om du har begränsat diskutrymme eller en långsammare internetanslutning.

### FP32 (mer noggrann)

- Större modellfiler.
- Högre noggrannhet, särskilt för svårt ljud med accenter eller bakgrundsljud.
- Rekommenderas när noggrannhet är prioritet och du har tillräckligt med diskutrymme.
- Krävs för CUDA GPU-acceleration — GPU-sökvägen använder alltid FP32 oavsett den här inställningen.

## Så här ändrar du precision

Öppna `Settings…` från menyraden, gå sedan till avsnittet **Models** och välj antingen `INT8 (smaller download)` eller `FP32 (more accurate)`.

## Efter att du ändrat precision

En ändrad precision kräver en annan uppsättning modellfiler. Om modellfilerna för den nya precisionen inte har laddats ned ännu klickar du på `Download Missing Models` i inställningarna. Tidigare nedladdade filer för den andra precisionen behålls på disken och behöver inte laddas ned på nytt om du byter tillbaka.

---