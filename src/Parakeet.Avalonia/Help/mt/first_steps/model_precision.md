---
title: "Għażla tal-Preċiżjoni tal-Piżijiet tal-Mudell"
description: "Kif tagħżel bejn il-preċiżjoni tal-mudell INT8 u FP32 u x'inhuma l-kompromessi."
topic_id: first_steps_model_precision
---

# Għażla tal-Preċiżjoni tal-Piżijiet tal-Mudell

Il-preċiżjoni tal-mudell tikkontrolla l-format numeriku użat mill-piżijiet tal-mudell tal-AI. Taffettwa d-daqs tat-tniżżil, l-użu tal-memorja, u l-preċiżjoni.

## Għażliet ta' Preċiżjoni

### INT8 (tniżżil iżgħar)

- Fajls tal-mudell iżgħar — itwal biex jitniżżlu u jeħtieġu inqas spazju fuq id-diska.
- Preċiżjoni ftit inqas għolja fuq xi awdjo.
- Rakkomandata jekk għandek spazju limitat fuq id-diska jew konnessjoni tal-internet aktar bil-mod.

### FP32 (aktar preċiż)

- Fajls tal-mudell akbar.
- Preċiżjoni ogħla, speċjalment fuq awdjo diffiċli b'aċċenti jew ħoss tal-isfond.
- Rakkomandata meta l-preċiżjoni hija l-prijorità u għandek biżżejjed spazju fuq id-diska.
- Meħtieġa għall-aċċelerazzjoni GPU CUDA — il-mogħdija GPU dejjem tuża FP32 irrispettivament minn din l-issettjar.

## Kif Tibdel il-Preċiżjoni

Iftaħ `Settings…` mill-bar tal-menu, imbagħad mur fit-taqsima **Models** u agħżel jew `INT8 (smaller download)` jew `FP32 (more accurate)`.

## Wara li Tibdel il-Preċiżjoni

Il-bidla fil-preċiżjoni teħtieġ sett differenti ta' fajls tal-mudell. Jekk il-mudelli tal-preċiżjoni l-ġdida għadhom ma tniżżlux, ikklikkja `Download Missing Models` fis-Settings. Il-fajls li tniżżlu qabel għall-preċiżjoni l-oħra jinżammu fuq id-diska u ma jkollhomx bżonn jitniżżlu mill-ġdid jekk terġa' lura.

---