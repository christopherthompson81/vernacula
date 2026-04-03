---
title: "Užduočių pristabdymas, tęsimas arba šalinimas"
description: "Kaip pristabdyti vykdomą užduotį, tęsti sustabdytą arba ištrinti užduotį iš istorijos."
topic_id: operations_pausing_resuming_removing
---

# Užduočių pristabdymas, tęsimas arba šalinimas

## Užduoties pristabdymas

Vykdomą arba eilėje esančią užduotį galite pristabdyti iš dviejų vietų:

- **Eigos rodinys** — spustelėkite `Pause` apatiniame dešiniajame kampe, stebėdami aktyvią užduotį.
- **Transkripcijų istorijos lentelė** — spustelėkite `Pause` **Veiksmai** stulpelyje eilutėje, kurios būsena yra `running` arba `queued`.

Spustelėjus `Pause`, būsenos eilutėje rodoma `Pausing…`, kol programa užbaigia apdoroti dabartinį vienetą. Tada užduoties būsena istorijos lentelėje pasikeičia į `cancelled`.

> Pristabdžius užduotį, visi iki tol transkribuoti segmentai išsaugomi. Vėliau galite tęsti užduotį neprarasdami atlikto darbo.

## Užduoties tęsimas

Norėdami tęsti pristabdytą arba nepavykusią užduotį:

1. Pagrindiniame ekrane susiraskite užduotį **Transkripcijų istorijos** lentelėje. Jos būsena bus `cancelled` arba `failed`.
2. Spustelėkite `Resume` **Veiksmai** stulpelyje.
3. Programa grįžta į **Eigos** rodinį ir tęsia apdorojimą nuo tos vietos, kur buvo sustota.

Būsenos eilutėje trumpam rodoma `Resuming…`, kol užduotis inicijuojama iš naujo.

## Užduoties šalinimas

Norėdami visam laikui ištrinti užduotį ir jos transkripciją iš istorijos:

1. **Transkripcijų istorijos** lentelėje spustelėkite `Remove` **Veiksmai** stulpelyje prie užduoties, kurią norite ištrinti.

Užduotis pašalinama iš sąrašo, o jos duomenys ištrinami iš vietinės duomenų bazės. Šio veiksmo atšaukti neįmanoma. Į diską išsaugoti eksportuoti failai lieka nepaveikti.

---