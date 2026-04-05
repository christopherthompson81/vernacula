---
title: "Potek dela za novo transkripcijo"
description: "Postopni vodič za transkripcijo zvočne datoteke."
topic_id: operations_new_transcription
---

# Potek dela za novo transkripcijo

Ta potek dela uporabite za transkripcijo posamezne zvočne datoteke.

## Predpogoji

- Vse datoteke modelov morajo biti prenesene. Kartica **Stanje modela** mora prikazovati `All N model file(s) present ✓`. Glejte [Prenos modelov](../first_steps/downloading_models.md).

## Podprte oblike

### Zvok

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Video datoteke se dekodirajo prek FFmpeg. Če video datoteka vsebuje **več zvočnih tokov** (npr. več jezikov ali komentarske sledi), se za vsak tok samodejno ustvari ena transkripcijska naloga.

## Koraki

### 1. Odprite obrazec za novo transkripcijo

Kliknite `New Transcription` na začetnem zaslonu ali pojdite na `File > New Transcription`.

### 2. Izberite medijsko datoteko

Kliknite `Browse…` poleg polja **Zvočna datoteka**. Odpre se izbirnik datotek, filtriran na podprte zvočne in video oblike. Izberite svojo datoteko in kliknite **Odpri**. Pot do datoteke se prikaže v polju.

### 3. Poimenujte nalogo

Polje **Ime naloge** je vnaprej izpolnjeno z imenom datoteke. Uredite ga, če želite drugačno oznako — to ime se prikaže v Zgodovini transkripcij na začetnem zaslonu.

### 4. Zaženite transkripcijo

Kliknite `Start Transcription`. Aplikacija preklopi na pogled **Napredek**.

Če se želite vrniti brez zagona, kliknite `← Back`.

## Kaj se zgodi nato

Naloga se izvede v dveh fazah, prikazanih v vrstici napredka:

1. **Analiza zvoka** — diarizacija govornikov: prepoznavanje, kdo govori in kdaj.
2. **Prepoznavanje govora** — pretvorba govora v besedilo, segment za segmentom.

Prepisani segmenti se sproti prikazujejo v živi tabeli, ko so ustvarjeni. Ko je obdelava končana, se aplikacija samodejno premakne na pogled **Rezultati**.

Če dodate nalogo, medtem ko že teče druga, bo nova naloga prikazovala stanje `queued` in se bo zagnala, ko se trenutna naloga konča. Glejte [Nadzor nalog](monitoring_jobs.md).

---