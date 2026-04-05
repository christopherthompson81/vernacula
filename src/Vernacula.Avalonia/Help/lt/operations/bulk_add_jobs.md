---
title: "Kelių garso failų įtraukimas į eilę"
description: "Kaip vienu žingsniu įtraukti kelis garso failus į užduočių eilę."
topic_id: operations_bulk_add_jobs
---

# Kelių garso failų įtraukimas į eilę

Naudokite **Bulk Add Jobs**, kad vienu veiksmu į eilę įtrauktumėte kelis garso arba vaizdo failus transkripcijai. Programa juos apdoroja po vieną ta tvarka, kuria jie buvo pridėti.

## Būtinosios sąlygos

- Visi modelių failai turi būti atsisiųsti. **Model Status** kortelėje turi būti rodoma `All N model file(s) present ✓`. Žr. [Modelių atsisiuntimas](../first_steps/downloading_models.md).

## Kaip naudoti massinį užduočių pridėjimą

1. Pagrindiniame ekrane spustelėkite `Bulk Add Jobs`.
2. Atsidaro failų parinkiklis. Pasirinkite vieną ar kelis garso arba vaizdo failus — laikykite `Ctrl` arba `Shift`, kad pasirinktumėte kelis failus.
3. Spustelėkite **Open**. Kiekvienas pasirinktas failas įtraukiamas į **Transcription History** lentelę kaip atskira užduotis.

> **Vaizdo failai su keliais garso srautais:** jei vaizdo failas turi daugiau nei vieną garso srautą (pavyzdžiui, kelias kalbas arba režisieriaus komentarų takelį), programa automatiškai sukuria po vieną užduotį kiekvienam srautui.

## Užduočių pavadinimai

Kiekvienos užduoties pavadinimas suteikiamas automatiškai pagal garso failo pavadinimą. Užduotį galite pervardyti bet kuriuo metu — spustelėkite jos pavadinimą **Title** stulpelyje, esančiame Transcription History lentelėje, redaguokite tekstą ir paspauskite `Enter` arba spustelėkite kitur.

## Eilės veikimas

- Jei šiuo metu nevykdoma jokia užduotis, pirmasis failas pradedamas apdoroti iš karto, o likę rodomi kaip `queued`.
- Jei užduotis jau vykdoma, visi naujai pridėti failai rodomi kaip `queued` ir bus pradėti apdoroti automatiškai paeiliui.
- Norėdami stebėti aktyvią užduotį, jos **Actions** stulpelyje spustelėkite `Monitor`. Žr. [Užduočių stebėjimas](monitoring_jobs.md).
- Norėdami pristabdyti arba pašalinti užduotį iš eilės prieš jai pradedant vykti, naudokite jos **Actions** stulpelyje esančius mygtukus `Pause` arba `Remove`. Žr. [Užduočių pristabdymas, tęsimas ar šalinimas](pausing_resuming_removing.md).

---