---
title: "Modelio svorių tikslumo pasirinkimas"
description: "Kaip pasirinkti tarp INT8 ir FP32 modelio tikslumo ir kokie yra kompromisai."
topic_id: first_steps_model_precision
---

# Modelio svorių tikslumo pasirinkimas

Modelio tikslumas nustato skaitinį formatą, naudojamą dirbtinio intelekto modelio svoriams. Jis turi įtakos atsisiuntimo dydžiui, atminties naudojimui ir tikslumui.

## Tikslumo parinktys

### INT8 (mažesnis atsisiuntimas)

- Mažesni modelio failai — greičiau atsisiunčiama ir reikia mažiau vietos diske.
- Šiek tiek mažesnis tikslumas su kai kuriuo garso įrašu.
- Rekomenduojama, jei turite ribotą vietos diske arba lėtesnį interneto ryšį.

### FP32 (tikslesnis)

- Didesni modelio failai.
- Didesnis tikslumas, ypač su sudėtingu garso įrašu, kuriame yra akcentų arba fono triukšmo.
- Rekomenduojama, kai tikslumas yra prioritetas ir turite pakankamai vietos diske.
- Būtina CUDA GPU spartinimui — GPU kelias visada naudoja FP32, neatsižvelgiant į šį nustatymą.

## Kaip pakeisti tikslumą

Atidarykite `Settings…` meniu juostoje, tada eikite į skyrių **Models** ir pasirinkite `INT8 (smaller download)` arba `FP32 (more accurate)`.

## Po tikslumo pakeitimo

Pakeitus tikslumą reikalingas kitas modelio failų rinkinys. Jei naujo tikslumo modeliai dar nebuvo atsisiųsti, spustelėkite `Download Missing Models` nustatymuose. Anksčiau atsisiųsti kito tikslumo failai išsaugomi diske ir nereikia jų atsisiųsti iš naujo, jei grįšite prie ankstesnio nustatymo.

---