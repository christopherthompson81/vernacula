---
title: "Uuden litteroinnin työnkulku"
description: "Vaiheittainen opas äänitiedoston litterointiin."
topic_id: operations_new_transcription
---

# Uuden litteroinnin työnkulku

Käytä tätä työnkulkua yksittäisen äänitiedoston litterointiin.

## Edellytykset

- Kaikki mallitiedostot on oltava ladattuna. **Mallin tila** -kortin on näytettävä `All N model file(s) present ✓`. Katso [Mallien lataaminen](../first_steps/downloading_models.md).

## Tuetut tiedostomuodot

### Ääni

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Videotiedostot puretaan FFmpeg-ohjelmiston avulla. Jos videotiedosto sisältää **useita ääniraitoja** (esim. useita kieliä tai kommentaariraitoja), jokaiselle raidalle luodaan automaattisesti oma litterointityö.

## Vaiheet

### 1. Avaa Uusi litterointi -lomake

Napsauta `New Transcription` aloitusnäytöllä tai siirry kohtaan `File > New Transcription`.

### 2. Valitse mediatiedosto

Napsauta `Browse…` **Äänitiedosto**-kentän vieressä. Tiedostonvalitsin avautuu suodatettuna tuetuille ääni- ja videotiedostomuodoille. Valitse tiedosto ja napsauta **Avaa**. Tiedostopolku näkyy kentässä.

### 3. Nimeä työ

**Työn nimi** -kenttä täytetään automaattisesti tiedoston nimen perusteella. Muokkaa sitä, jos haluat käyttää eri nimeä — tämä nimi näkyy aloitusnäytön litterointihistoriassa.

### 4. Käynnistä litterointi

Napsauta `Start Transcription`. Sovellus siirtyy **Edistyminen**-näkymään.

Palataksesi takaisin käynnistämättä litterointia napsauta `← Back`.

## Mitä tapahtuu seuraavaksi

Työ käy läpi kaksi vaihetta, jotka näkyvät edistymispalkissa:

1. **Äänianalyysi** — puhujien diarisaatio: tunnistetaan, kuka puhuu ja milloin.
2. **Puheentunnistus** — puheen muuntaminen tekstiksi segmentti kerrallaan.

Litteroidut segmentit ilmestyvät reaaliaikaiseen taulukkoon sitä mukaa kuin ne valmistuvat. Kun käsittely on valmis, sovellus siirtyy automaattisesti **Tulokset**-näkymään.

Jos lisäät työn, kun toinen on jo käynnissä, uusi työ näyttää tilana `queued` ja käynnistyy, kun nykyinen työ valmistuu. Katso [Töiden seuraaminen](monitoring_jobs.md).

---