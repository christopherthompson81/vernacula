---
title: "Overvågning af job"
description: "Sådan følger du et kørende eller kø-sat jobs fremdrift."
topic_id: operations_monitoring_jobs
---

# Overvågning af job

Visningen **Fremdrift** giver dig et live-overblik over et kørende transskriptionsjob.

## Åbning af fremdriftsvisningen

- Når du starter en ny transskription, skifter programmet automatisk til fremdriftsvisningen.
- For et job, der allerede kører eller venter i kø, finder du det i tabellen **Transskriptionshistorik** og klikker på `Monitor` i kolonnen **Handlinger**.

## Aflæsning af fremdriftsvisningen

| Element | Beskrivelse |
|---|---|
| Fremdriftslinje | Samlet fuldførelsesprocent. Ubestemt (animeret), mens jobbet starter eller genoptages. |
| Procentlabel | Numerisk procent vist til højre for linjen. |
| Statusbesked | Aktuel aktivitet — for eksempel `Audio Analysis` eller `Speech Recognition`. Viser `Waiting in queue…`, hvis jobbet endnu ikke er startet. |
| Segmenttabel | Live-feed af transskriberede segmenter med kolonnerne **Taler**, **Start**, **Slut** og **Indhold**. Ruller automatisk, efterhånden som nye segmenter ankommer. |

## Fremdriftsfaser

De viste faser afhænger af den **segmenteringstilstand**, der er valgt i Indstillinger.

**Tilstanden Talerdiarisering** (standard):

1. **Audio Analysis** — SortFormer-diarisering køres over hele filen for at identificere talergrænser. Linjen kan blive tæt på 0%, indtil denne fase er afsluttet.
2. **Speech Recognition** — hvert talerssegment transskriberes. Procenttallet stiger jævnt i denne fase.

**Tilstanden Stemmeaktivitetsdetektion**:

1. **Detecting speech segments** — Silero VAD scanner filen for at finde taleregioner. Denne fase er hurtig.
2. **Speech Recognition** — hver registreret taleregion transskriberes.

I begge tilstande udfyldes live-segmenttabellen, efterhånden som transskriptionen skrider frem.

## Navigation væk fra visningen

Klik på `← Back to Home` for at vende tilbage til startskærmen uden at afbryde jobbet. Jobbet fortsætter med at køre i baggrunden, og dets status opdateres i tabellen **Transskriptionshistorik**. Klik på `Monitor` igen når som helst for at vende tilbage til fremdriftsvisningen.

---