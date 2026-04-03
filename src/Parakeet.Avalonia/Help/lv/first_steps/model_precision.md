---
title: "Modeļa svaru precizitātes izvēle"
description: "Kā izvēlēties starp INT8 un FP32 modeļa precizitāti un kādi ir kompromisi."
topic_id: first_steps_model_precision
---

# Modeļa svaru precizitātes izvēle

Modeļa precizitāte nosaka skaitlisko formātu, ko izmanto AI modeļa svari. Tā ietekmē lejupielādes izmēru, atmiņas patēriņu un precizitāti.

## Precizitātes opcijas

### INT8 (mazāka lejupielāde)

- Mazāki modeļa faili — ātrāka lejupielāde un mazāks nepieciešamais diska vietas apjoms.
- Nedaudz zemāka precizitāte ar dažiem audio ierakstiem.
- Ieteicams, ja jums ir ierobežota diska vieta vai lēnāks interneta savienojums.

### FP32 (lielāka precizitāte)

- Lielāki modeļa faili.
- Augstāka precizitāte, īpaši ar sarežģītiem audio ierakstiem, kuros ir akcents vai fona troksnis.
- Ieteicams, ja precizitāte ir prioritāte un jums ir pietiekami daudz diska vietas.
- Nepieciešams CUDA GPU paātrinājumam — GPU ceļš vienmēr izmanto FP32 neatkarīgi no šī iestatījuma.

## Kā mainīt precizitāti

Atveriet `Settings…` izvēlņu joslā, pēc tam dodieties uz sadaļu **Models** un atlasiet `INT8 (smaller download)` vai `FP32 (more accurate)`.

## Pēc precizitātes maiņas

Precizitātes maiņai nepieciešams cits modeļa failu komplekts. Ja jaunās precizitātes modeļi vēl nav lejupielādēti, noklikšķiniet uz `Download Missing Models` sadaļā Settings. Iepriekš lejupielādētie otras precizitātes faili tiek saglabāti diskā un nav atkārtoti jālejupielādē, ja pārslēdzaties atpakaļ.

---