---
title: "Mudeli kaalude täpsuse valimine"
description: "Kuidas valida INT8 ja FP32 mudeli täpsuse vahel ning millised on nende kompromissid."
topic_id: first_steps_model_precision
---

# Mudeli kaalude täpsuse valimine

Mudeli täpsus määrab, millist arvuformaati kasutatakse tehisintellekti mudeli kaaludes. See mõjutab allalaadimise suurust, mälukasutust ja täpsust.

## Täpsuse valikud

### INT8 (väiksem allalaadimine)

- Väiksemad mudelivormid — kiirem allalaadida ja nõuab vähem kettaruumi.
- Veidi madalam täpsus mõne helisalvestise puhul.
- Soovitatav, kui kettaruum on piiratud või internetiühendus aeglasem.

### FP32 (täpsem)

- Suuremad mudelivormid.
- Kõrgem täpsus, eriti raske heli puhul, kus esineb aktsente või taustamüra.
- Soovitatav, kui täpsus on esmatähtis ja kettaruumi on piisavalt.
- Nõutav CUDA GPU kiirenduse kasutamiseks — GPU tee kasutab alati FP32-t, olenemata sellest sättest.

## Täpsuse muutmine

Ava menüüribalt `Settings…`, seejärel mine jaotisse **Models** ja vali `INT8 (smaller download)` või `FP32 (more accurate)`.

## Pärast täpsuse muutmist

Täpsuse muutmine nõuab teistsugust mudelivormide komplekti. Kui uue täpsuse mudelid pole veel alla laaditud, klõpsa sätetes nupul `Download Missing Models`. Teise täpsuse jaoks varem alla laaditud failid jäävad kettale alles ja neid ei pea uuesti alla laadima, kui soovid tagasi lülituda.

---