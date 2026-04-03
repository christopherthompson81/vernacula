---
title: "Učitavanje dovršenih zadataka"
description: "Kako otvoriti rezultate prethodno dovršenog transkripcijskog zadatka."
topic_id: operations_loading_completed_jobs
---

# Učitavanje dovršenih zadataka

Svi dovršeni transkripcijski zadaci spremaju se u lokalnu bazu podataka i ostaju dostupni u tablici **Povijest transkripcija** na početnom zaslonu.

## Kako učitati dovršeni zadatak

1. Na početnom zaslonu pronađite zadatak u tablici **Povijest transkripcija**. Dovršeni zadaci prikazuju oznaku statusa `complete`.
2. Kliknite `Load` u stupcu **Akcije** željenog zadatka.
3. Aplikacija se prebacuje na prikaz **Rezultati**, koji prikazuje sve transkribirane segmente za taj zadatak.

## Prikaz rezultata

Prikaz rezultata prikazuje:

- Naziv audio datoteke kao naslov stranice.
- Podnaslov s brojem segmenata (na primjer, `42 segment(s)`).
- Tablicu segmenata s stupcima **Speaker**, **Start**, **End** i **Content**.

Iz prikaza rezultata možete:

- [Uređivati transkripciju](editing_transcripts.md) — pregledati i ispraviti tekst, prilagoditi vremenski raspored, spajati ili dijeliti segmente te provjeravati segmente uz slušanje zvuka.
- [Uređivati imena govornika](editing_speaker_names.md) — zamijeniti generičke oznake poput `speaker_0` stvarnim imenima.
- [Izvesti transkripciju](exporting_results.md) — spremiti transkripciju u obliku Excel, CSV, JSON, SRT, Markdown, Word ili SQLite.

Za povratak na popis povijesti kliknite `← Back to History`.

---