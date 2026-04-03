---
title: "Fluss tax-Xogħol għal Traskrizzjoni Ġdida"
description: "Gwida pass pass biex tittraskrixxi fajl awdjo."
topic_id: operations_new_transcription
---

# Fluss tax-Xogħol għal Traskrizzjoni Ġdida

Uża dan il-fluss tax-xogħol biex tittraskrixxi fajl awdjo wieħed.

## Prerekwiżiti

- Il-fajls kollha tal-mudell iridu jkunu mniżżla. Il-kard **Model Status** trid turi `All N model file(s) present ✓`. Ara [Niżżel il-Mudelli](../first_steps/downloading_models.md).

## Formati Supportati

### Awdjo

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Vidjo

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Il-fajls tal-vidjo jiġu dekodjati permezz ta' FFmpeg. Jekk fajl tal-vidjo fih **streams awdjo multipli** (pereżempju lingwi differenti jew tracks ta' kummenti), jinħoloq awtomatikament xogħol ta' traskrizzjoni wieħed għal kull stream.

## Passi

### 1. Iftaħ il-formola għal Traskrizzjoni Ġdida

Ikklikkja `New Transcription` fuq l-iskrin ewlieni, jew mur `File > New Transcription`.

### 2. Agħżel fajl tal-midja

Ikklikkja `Browse…` ħdejn il-qasam **Audio File**. Jinfetaħ għażlier ta' fajls iffiltrjat għall-formati awdjo u vidjo supportati. Agħżel il-fajl tiegħek u kklikkja **Open**. Il-mogħdija tal-fajl tidher fil-qasam.

### 3. Semmi x-xogħol

Il-qasam **Job Name** jiġi mimli minn qabel bl-isem tal-fajl. Editjah jekk trid tikkeja differenti — dan l-isem jidher fl-Istorja tat-Traskrizzjoni fuq l-iskrin ewlieni.

### 4. Ibda t-traskrizzjoni

Ikklikkja `Start Transcription`. L-applikazzjoni tbiddel għall-veduta **Progress**.

Biex tmur lura mingħajr ma tibda, ikklikkja `← Back`.

## X'Jiġri Wara

Ix-xogħol jgħaddi minn żewġ fażijiet li jinsabu fil-barra tal-progress:

1. **Analiżi Awdjo** — diarization tal-ispeaker: li tidentifika min qed jitkellem u meta.
2. **Rikonoxximent tal-Kliem** — tibdil tal-kliem f'test segment wara segment.

Is-segmenti traskritti jidhru fit-tabella diretta hekk kif jiġu prodotti. Meta l-ipproċessar ikun lest, l-applikazzjoni tiċċaqlaq awtomatikament għall-veduta **Results**.

Jekk iżżid xogħol filwaqt li ieħor diġà qed jiddekorri, ix-xogħol il-ġdid juri l-istatus `queued` u jibda meta x-xogħol attwali jispiċċa. Ara [Monitoraġġ tax-Xogħlijiet](monitoring_jobs.md).

---