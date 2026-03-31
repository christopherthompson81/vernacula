---
title: "Il-Monitoraġġ tal-Impjiegi"
description: "Kif tara l-progress ta' impjieg li qiegħed jaħdem jew jistenna."
topic_id: operations_monitoring_jobs
---

# Il-Monitoraġġ tal-Impjiegi

Il-veduta **Progress** tagħtik veduta diretta ta' impjieg ta' traskrozzjoni li qiegħed jaħdem.

## Kif Tiftaħ il-Veduta tal-Progress

- Meta tibda traskrozzjoni ġdida, l-applikazzjoni tmur awtomatikament għall-veduta tal-Progress.
- Għal impjieg li diġà qiegħed jaħdem jew jistenna, sibtu fit-tabella **Transcription History** u kklikkja `Monitor` fil-kolonna **Actions** tiegħu.

## Kif Taqra l-Veduta tal-Progress

| Element | Deskrizzjoni |
|---|---|
| Progress bar | Il-perċentwali globali tal-ilħiq. Tkun indeterminata (animata) waqt li l-impjieg qiegħed jibda jew jerġa' jibda. |
| Tikketta tal-perċentwali | Il-perċentwali numerika murija fuq il-lemin tal-bar. |
| Messaġġ tal-istat | L-attività attwali — pereżempju `Audio Analysis` jew `Speech Recognition`. Turi `Waiting in queue…` jekk l-impjieg għadu ma bdax. |
| Tabella tas-segmenti | Għajn diretta ta' segmenti traskritti bil-kolonni **Speaker**, **Start**, **End**, u **Content**. Tiskrollja awtomatikament hekk kif jaslu segmenti ġodda. |

## Il-Fażijiet tal-Progress

Il-fażijiet murija jiddependu fuq il-**Segmentation Mode** magħżula fis-Settings.

**Modalità Speaker Diarization** (default):

1. **Audio Analysis** — id-diarizzazzjoni Sortformer taħdem fuq il-fajl kollu biex tidentifika l-konfini bejn il-kelliema. Il-bar tista' tibqa' qrib 0% sakemm din il-fażi titlesta.
2. **Speech Recognition** — kull segment ta' kelliem jiġi traskritt. Il-perċentwali tikber b'mod kostanti matul din il-fażi.

**Modalità Voice Activity Detection**:

1. **Detecting speech segments** — Silero VAD jagħmel skaner tal-fajl biex isib ir-reġjuni tal-kelma. Din il-fażi hija rapida.
2. **Speech Recognition** — kull reġjun tal-kelma identifikat jiġi traskritt.

Fiż-żewġ modalitajiet, it-tabella diretta tas-segmenti timtela hekk kif tipproċedi t-traskrozzjoni.

## Kif Titlaq mill-Veduta

Kklikkja `← Back to Home` biex tirritorna għall-iskrin ewlieni mingħajr ma tinterrompi l-impjieg. L-impjieg ikompli jaħdem fil-background u l-istat tiegħu jiġi aġġornat fit-tabella **Transcription History**. Kklikkja `Monitor` mill-ġdid fi kwalunkwe ħin biex tirritorna għall-veduta tal-Progress.

---