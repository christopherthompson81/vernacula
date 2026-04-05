---
title: "Modelgewichtprecisie kiezen"
description: "Hoe u kiest tussen INT8- en FP32-modelprecisie en wat de afwegingen zijn."
topic_id: first_steps_model_precision
---

# Modelgewichtprecisie kiezen

Modelprecisie bepaalt het numerieke formaat dat wordt gebruikt door de AI-modelgewichten. Het is van invloed op de downloadgrootte, het geheugengebruik en de nauwkeurigheid.

## Precisie-opties

### INT8 (kleinere download)

- Kleinere modelbestanden — sneller te downloaden en minder schijfruimte vereist.
- Iets lagere nauwkeurigheid bij bepaald audiomateriaal.
- Aanbevolen als u weinig schijfruimte heeft of een tragere internetverbinding gebruikt.

### FP32 (nauwkeuriger)

- Grotere modelbestanden.
- Hogere nauwkeurigheid, vooral bij moeilijk audiomateriaal met accenten of achtergrondgeluid.
- Aanbevolen wanneer nauwkeurigheid prioriteit heeft en u voldoende schijfruimte beschikbaar heeft.
- Vereist voor CUDA GPU-versnelling — het GPU-pad gebruikt altijd FP32, ongeacht deze instelling.

## Precisie wijzigen

Open `Settings…` via de menubalk, ga vervolgens naar het gedeelte **Models** en selecteer `INT8 (smaller download)` of `FP32 (more accurate)`.

## Na het wijzigen van de precisie

Het wijzigen van de precisie vereist een andere set modelbestanden. Als de modelbestanden voor de nieuwe precisie nog niet zijn gedownload, klikt u op `Download Missing Models` in Instellingen. Eerder gedownloade bestanden voor de andere precisie blijven op schijf bewaard en hoeven niet opnieuw te worden gedownload als u terugschakelt.

---