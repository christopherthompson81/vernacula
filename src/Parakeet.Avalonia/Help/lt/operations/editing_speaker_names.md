---
title: "Kalbėtojų vardų redagavimas"
description: "Kaip pakeisti bendrinius kalbėtojų ID tikrais vardais transkripcijoje."
topic_id: operations_editing_speaker_names
---

# Kalbėtojų vardų redagavimas

Transkripcijos variklis automatiškai suteikia kiekvienam kalbėtojui bendrinio ID etiketę (pavyzdžiui, `speaker_0`, `speaker_1`). Šiuos ID galite pakeisti tikrais vardais, kurie bus rodomi visoje transkripcijoje ir visuose eksportuotuose failuose.

## Kaip redaguoti kalbėtojų vardus

1. Atidarykite užbaigtą užduotį. Žr. [Užbaigtų užduočių įkėlimas](loading_completed_jobs.md).
2. **Rezultatų** rodinyje spustelėkite `Edit Speaker Names`.
3. Atidaromas dialogo langas **Edit Speaker Names** su dviem stulpeliais:
   - **Speaker ID** — originalus modelio priskirtas žymuo (tik skaitomas).
   - **Display Name** — transkripcijoje rodomas vardas (redaguojamas).
4. Spustelėkite langelį stulpelyje **Display Name** ir įveskite kalbėtojo vardą.
5. Paspauskite `Tab` arba spustelėkite kitą eilutę, kad pereitumėte prie kito kalbėtojo.
6. Spustelėkite `Save`, kad pritaikytumėte pakeitimus, arba `Cancel`, kad juos atmestumėte.

## Kur rodomi vardai

Atnaujinti rodomieji vardai pakeičia bendrinius ID šiose vietose:

- Rezultatų rodinio segmentų lentelėje.
- Visuose eksportuotuose failuose (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Pakartotinis vardų redagavimas

Dialogo langą „Edit Speaker Names" galite atidaryti bet kuriuo metu, kol užduotis yra įkelta į Rezultatų rodinį. Pakeitimai išsaugomi vietinėje duomenų bazėje ir išlieka per visas sesijas.

---