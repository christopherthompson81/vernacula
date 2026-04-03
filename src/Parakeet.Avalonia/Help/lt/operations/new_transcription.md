---
title: "Naujos Transkribavimo Darbo Eiga"
description: "Žingsnis po žingsnio vadovas garso failo transkribavimui."
topic_id: operations_new_transcription
---

# Naujos Transkribavimo Darbo Eiga

Naudokite šią darbo eigą vienam garso failui transkribuoti.

## Išankstinės Sąlygos

- Visi modelio failai turi būti atsisiųsti. **Modelio Būsenos** kortelėje turi būti rodoma `All N model file(s) present ✓`. Žr. [Modelių Atsisiuntimas](../first_steps/downloading_models.md).

## Palaikomi Formatai

### Garso

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Vaizdo

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Vaizdo failai dekoduo­jami per FFmpeg. Jei vaizdo faile yra **keli garso srautai** (pvz., kelios kalbos arba komentarų takelius), kiekvienam srautui automatiškai sukuriama atskira transkribavimo užduotis.

## Veiksmai

### 1. Atidarykite naują transkribavimo formą

Pagrindiniame ekrane spustelėkite `New Transcription` arba eikite į `File > New Transcription`.

### 2. Pasirinkite medijos failą

Spustelėkite `Browse…` šalia lauko **Audio File**. Atsidaro failų parinkiklis, filtruotas pagal palaikomus garso ir vaizdo formatus. Pasirinkite failą ir spustelėkite **Open**. Failo kelias atsiranda lauke.

### 3. Pavadinkite užduotį

Laukas **Job Name** yra iš anksto užpildytas failo pavadinimu. Pakeiskite jį, jei norite kito pavadinimo — šis pavadinimas rodomas pagrindinio ekrano transkribavimo istorijoje.

### 4. Pradėkite transkribavimą

Spustelėkite `Start Transcription`. Programa persijungia į **Pažangos** rodinį.

Norėdami grįžti nepradėję, spustelėkite `← Back`.

## Kas Vyksta Toliau

Užduotis vykdoma dviem fazėmis, rodomomis eigos juostoje:

1. **Garso Analizė** — kalbėtojų diarizacija: nustatoma, kas kalba ir kada.
2. **Kalbos Atpažinimas** — kalbos konvertavimas į tekstą segmentas po segmento.

Transkribuoti segmentai atsiranda tiesioginėje lentelėje, kai tik jie sukuriami. Apdorojimui pasibaigus, programa automatiškai pereina į **Rezultatų** rodinį.

Jei pridedате užduotį, kol kita jau vykdoma, nauja užduotis rodys būseną `queued` ir prasidės, kai baigsis dabartinė užduotis. Žr. [Užduočių Stebėjimas](monitoring_jobs.md).

---