---
title: "Užduočių stebėjimas"
description: "Kaip stebėti vykdomos arba eilėje laukiančios užduoties eigą."
topic_id: operations_monitoring_jobs
---

# Užduočių stebėjimas

**Eigos** rodinys suteikia galimybę stebėti vykdomą transkripcijos užduotį realiuoju laiku.

## Eigos rodinio atidarymas

- Kai pradedate naują transkripciją, programa automatiškai pereina į Eigos rodinį.
- Jei užduotis jau vykdoma arba laukia eilėje, suraskite ją **Transkripcijos istorijos** lentelėje ir spustelėkite `Monitor` jos **Veiksmų** skiltyje.

## Eigos rodinio skaitymas

| Elementas | Aprašymas |
|---|---|
| Eigos juosta | Bendras užbaigtumo procentas. Neapibrėžta (animuota), kol užduotis paleidžiama arba atnaujinama. |
| Procento žyma | Skaitmeninis procentas, rodomas dešinėje juostos pusėje. |
| Būsenos pranešimas | Dabartinė veikla — pavyzdžiui, `Audio Analysis` arba `Speech Recognition`. Rodo `Waiting in queue…`, jei užduotis dar nepradėta. |
| Segmentų lentelė | Transkribuotų segmentų srautas realiuoju laiku su **Kalbėtojo**, **Pradžios**, **Pabaigos** ir **Turinio** stulpeliais. Automatiškai slenkama, kai atsiranda nauji segmentai. |

## Eigos fazės

Rodomos fazės priklauso nuo **Segmentavimo režimo**, pasirinkto nustatymuose.

**Kalbėtojų diarizacijos režimas** (numatytasis):

1. **Audio Analysis** — SortFormer diarizacija apdoroja visą failą ir nustato kalbėtojų ribas. Juosta gali išlikti ties 0 %, kol ši fazė bus baigta.
2. **Speech Recognition** — kiekvienas kalbėtojo segmentas yra transkribuojamas. Šios fazės metu procentas nuosekliai didėja.

**Balso aktyvumo aptikimo režimas**:

1. **Detecting speech segments** — Silero VAD nuskaito failą ir suranda kalbos sritis. Ši fazė yra greita.
2. **Speech Recognition** — kiekviena aptikta kalbos sritis yra transkribuojama.

Abiejuose režimuose realiojo laiko segmentų lentelė pildoma transkripcijos metu.

## Naršymas kitur

Spustelėkite `← Back to Home`, kad grįžtumėte į pagrindinį ekraną nepertraukiant užduoties. Užduotis toliau vykdoma fone, o jos būsena atnaujinama **Transkripcijos istorijos** lentelėje. Bet kuriuo metu vėl spustelėkite `Monitor`, kad grįžtumėte į Eigos rodinį.

---