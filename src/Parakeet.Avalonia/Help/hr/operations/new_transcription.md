---
title: "Tijek rada nove transkripcije"
description: "Vodič korak po korak za transkripciju audio datoteke."
topic_id: operations_new_transcription
---

# Tijek rada nove transkripcije

Koristite ovaj tijek rada za transkripciju jedne audio datoteke.

## Preduvjeti

- Sve datoteke modela moraju biti preuzete. Kartica **Status modela** mora prikazivati `All N model file(s) present ✓`. Pogledajte [Preuzimanje modela](../first_steps/downloading_models.md).

## Podržani formati

### Audio

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Video datoteke se dekodiraju putem FFmpeg. Ako video datoteka sadrži **više audio tokova** (npr. više jezika ili komentatorsih zapisa), automatski se stvara jedan zadatak transkripcije za svaki tok.

## Koraci

### 1. Otvorite obrazac za novu transkripciju

Kliknite `New Transcription` na početnom zaslonu ili idite na `File > New Transcription`.

### 2. Odaberite medijsku datoteku

Kliknite `Browse…` pokraj polja **Audio File**. Otvara se birač datoteka filtriran na podržane audio i video formate. Odaberite svoju datoteku i kliknite **Open**. Put do datoteke prikazuje se u polju.

### 3. Imenujte zadatak

Polje **Job Name** unaprijed je ispunjeno nazivom datoteke. Uredite ga ako želite drugačiju oznaku — ovaj naziv pojavljuje se u povijesti transkripcija na početnom zaslonu.

### 4. Pokrenite transkripciju

Kliknite `Start Transcription`. Aplikacija prelazi na prikaz **Progress**.

Za povratak bez pokretanja kliknite `← Back`.

## Što se događa sljedeće

Zadatak prolazi kroz dvije faze prikazane na traci napretka:

1. **Analiza zvuka** — dijariazacija govornika: identificiranje tko govori i kada.
2. **Prepoznavanje govora** — pretvaranje govora u tekst segment po segment.

Transkribirani segmenti pojavljuju se u tablici uživo kako se proizvode. Kada obrada završi, aplikacija automatski prelazi na prikaz **Results**.

Ako dodate zadatak dok je drugi već pokrenut, novi zadatak prikazivat će status `queued` i pokrenuti se kada trenutni zadatak završi. Pogledajte [Nadzor zadataka](monitoring_jobs.md).

---