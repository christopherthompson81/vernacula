---
title: "Praćenje zadataka"
description: "Kako pratiti napredak pokrenutog ili zadatka u redu čekanja."
topic_id: operations_monitoring_jobs
---

# Praćenje zadataka

Prikaz **Napredak** pruža vam prikaz pokrenutog zadatka transkripcije uživo.

## Otvaranje prikaza Napredak

- Kada pokrenete novu transkripciju, aplikacija se automatski prebacuje na prikaz Napredak.
- Za zadatak koji je već pokrenut ili u redu čekanja, pronađite ga u tablici **Povijest transkripcija** i kliknite `Monitor` u njegovom stupcu **Akcije**.

## Čitanje prikaza Napredak

| Element | Opis |
|---|---|
| Traka napretka | Ukupni postotak dovršenosti. Neodređena (animirana) dok se zadatak pokreće ili nastavlja. |
| Oznaka postotka | Numerički postotak prikazan desno od trake. |
| Poruka statusa | Trenutna aktivnost — na primjer `Audio Analysis` ili `Speech Recognition`. Prikazuje `Waiting in queue…` ako zadatak još nije pokrenut. |
| Tablica segmenata | Prikaz transkribirana segmenata uživo sa stupcima **Govornik**, **Početak**, **Kraj** i **Sadržaj**. Automatski se pomiče kako novi segmenti pristižu. |

## Faze napretka

Prikazane faze ovise o **načinu segmentacije** odabranom u Postavkama.

**Način diarizacije govornika** (zadano):

1. **Audio Analysis** — SortFormer diarizacija se izvodi nad cijelom datotekom radi prepoznavanja granica između govornika. Traka može ostati blizu 0% dok ova faza ne završi.
2. **Speech Recognition** — svaki segment govornika se transkribira. Postotak stabilno raste tijekom ove faze.

**Način otkrivanja glasovne aktivnosti**:

1. **Detecting speech segments** — Silero VAD skenira datoteku kako bi pronašao dijelove s govorom. Ova faza je brza.
2. **Speech Recognition** — svaki otkriveni govorni dio se transkribira.

U oba načina tablica segmenata uživo se popunjava kako transkripcija napreduje.

## Napuštanje prikaza

Kliknite `← Back to Home` za povratak na početni zaslon bez prekidanja zadatka. Zadatak nastavlja s izvođenjem u pozadini i njegov se status ažurira u tablici **Povijest transkripcija**. Kliknite `Monitor` u bilo kojem trenutku za povratak na prikaz Napredak.

---