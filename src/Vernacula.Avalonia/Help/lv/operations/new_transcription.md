---
title: "Jaunas Transkribācijas Darbplūsma"
description: "Soli pa solim ceļvedis audio faila transkribēšanai."
topic_id: operations_new_transcription
---

# Jaunas Transkribācijas Darbplūsma

Izmantojiet šo darbplūsmu, lai transkribētu vienu audio failu.

## Priekšnosacījumi

- Visiem modeļu failiem jābūt lejupielādētiem. **Modeļa statusa** kartē jāparādās `All N model file(s) present ✓`. Skatiet [Modeļu lejupielāde](../first_steps/downloading_models.md).

## Atbalstītie Formāti

### Audio

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Video faili tiek dekodēti ar FFmpeg palīdzību. Ja video fails satur **vairākus audio straumjus** (piemēram, vairākas valodas vai komentāru skaņu celiņus), katrai straumei automātiski tiek izveidots atsevišķs transkribācijas uzdevums.

## Soļi

### 1. Atveriet jaunu transkribācijas veidlapu

Noklikšķiniet uz `New Transcription` sākuma ekrānā vai dodieties uz `File > New Transcription`.

### 2. Izvēlieties multivides failu

Noklikšķiniet uz `Browse…` blakus laukam **Audio File**. Tiek atvērts failu atlasītājs, kas filtrēts pēc atbalstītajiem audio un video formātiem. Atlasiet failu un noklikšķiniet uz **Open**. Faila ceļš parādās laukā.

### 3. Nosauciet uzdevumu

Lauks **Job Name** tiek automātiski aizpildīts ar faila nosaukumu. Rediģējiet to, ja vēlaties citu apzīmējumu — šis nosaukums tiks parādīts Transkribācijas vēsturē sākuma ekrānā.

### 4. Sāciet transkribāciju

Noklikšķiniet uz `Start Transcription`. Lietojumprogramma pārslēdzas uz skatu **Progress**.

Lai atgrieztos, neuzsākot transkribāciju, noklikšķiniet uz `← Back`.

## Kas Notiek Tālāk

Uzdevums tiek veikts divos posmos, kas parādīti progresa joslā:

1. **Audio analīze** — runātāju diarizācija: tiek noteikts, kurš runā un kad.
2. **Runas atpazīšana** — runas pārvēršana tekstā segments pa segmentam.

Transkribētie segmenti parādās tiešraides tabulā, tiklīdz tie tiek sagatavoti. Kad apstrāde ir pabeigta, lietojumprogramma automātiski pāriet uz skatu **Results**.

Ja pievienojat uzdevumu, kamēr vēl darbojas cits, jaunajam uzdevumam tiks piešķirts statuss `queued` un tas sāksies, tiklīdz pašreizējais uzdevums pabeigs darbu. Skatiet [Uzdevumu uzraudzība](monitoring_jobs.md).

---