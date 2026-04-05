---
title: "Uue transkriptsiooni töövoog"
description: "Samm-sammuline juhend helifaili transkribeerimiseks."
topic_id: operations_new_transcription
---

# Uue transkriptsiooni töövoog

Kasutage seda töövooga ühe helifaili transkribeerimiseks.

## Eeltingimused

- Kõik mudelifailid peavad olema alla laaditud. **Mudeli olek** kaardil peab olema kuvatud `All N model file(s) present ✓`. Vaadake [Mudelite allalaadimine](../first_steps/downloading_models.md).

## Toetatud vormingud

### Heli

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Videofailid dekodeeritakse FFmpeg abil. Kui videofail sisaldab **mitu helivoogu** (nt mitu keelt või kommentaariradasid), luuakse iga voo jaoks automaatselt eraldi transkriptsiooni töö.

## Sammud

### 1. Avage uue transkriptsiooni vorm

Klõpsake avakuval nuppu `New Transcription` või valige `File > New Transcription`.

### 2. Valige meediumifail

Klõpsake välja **Audio File** kõrval nuppu `Browse…`. Avaneb failivalija, mis on filtreeritud toetatud heli- ja videovormingute järgi. Valige soovitud fail ja klõpsake **Open**. Faili tee kuvatakse väljal.

### 3. Andke tööle nimi

**Job Name** väli täidetakse automaatselt failinime põhjal. Muutke seda, kui soovite teistsugust silti — see nimi kuvatakse avakuva transkriptsioonide ajaloos.

### 4. Alustage transkribeerimist

Klõpsake `Start Transcription`. Rakendus lülitub **Progress** vaatesse.

Ilma alustamata tagasi minekuks klõpsake `← Back`.

## Mis juhtub edasi

Töö läbib edenemisribal kaks faasi:

1. **Audio Analysis** — kõneleja tuvastamine: määratakse, kes räägib ja millal.
2. **Speech Recognition** — kõne teisendamine tekstiks segment-segmendi haaval.

Transkribeeritud segmendid ilmuvad töötlemise ajal reaalajas tabelisse. Kui töötlemine on lõpule jõudnud, liigub rakendus automaatselt **Results** vaatesse.

Kui lisate töö ajal, mil teine töö on juba käimas, kuvatakse uue töö olekuks `queued` ning see käivitub pärast praeguse töö lõpetamist. Vaadake [Tööde jälgimine](monitoring_jobs.md).

---