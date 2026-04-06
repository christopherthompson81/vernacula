---
title: "Settings"
description: "Ħarsa ġenerali lejn l-għażliet kollha fit-tieqa Settings."
topic_id: first_steps_settings_window
---

# Settings

It-tieqa **Settings** tagħtik kontroll fuq il-konfigurazzjoni tal-hardware, il-ġestjoni tal-mudelli, il-modalità ta' segmentazzjoni, l-imġieba tal-editur, id-dehra, u l-lingwa. Iftaħha mill-menu bar: `Settings…`.

## Hardware u Prestazzjoni

Din it-taqsima turi l-istatus tal-GPU NVIDIA u s-software stack CUDA, u tirrapporta l-limitu massimu tal-batch użat waqt it-traskrizzjoni bil-GPU.

| Element | Deskrizzjoni |
|---|---|
| Isem tal-GPU u VRAM | Il-GPU NVIDIA detettata u l-memorja tal-vidjo disponibbli. |
| CUDA Toolkit | Jekk il-libreriji runtime tal-CUDA nstabu permezz ta' `CUDA_PATH`. |
| cuDNN | Jekk id-DLLs runtime tal-cuDNN huma disponibbli. |
| CUDA Acceleration | Jekk l-ONNX Runtime iċċarġa l-fornitur ta' eżekuzzjoni CUDA b'suċċess. |

Ikklikkja `Re-check` biex terġa' tħaddem id-detezzjoni tal-hardware mingħajr ma terġa' tiftaħ l-applikazzjoni — utli wara l-installazzjoni ta' CUDA jew cuDNN.

Links ta' tniżżil dirett għal CUDA Toolkit u cuDNN jintwerew meta dawk il-komponenti ma jkunux detettati.

Il-messaġġ tal-**batch ceiling** jirrapporta kemm sekondi ta' awdjo jiġu pproċessati f'kull ħidma tal-GPU. Din il-valur tiġi derivata mill-VRAM ħielsa wara li jitgħabbew il-mudelli u taġġusta ruħha awtomatikament.

Għal istruzzjonijiet sħaħ dwar is-setup ta' CUDA, ara [L-Installazzjoni ta' CUDA u cuDNN](cuda_installation.md).

## Mudelli

Din it-taqsima timmaniġġja l-fajls tal-mudell AI meħtieġa għat-traskrizzjoni.

- **Download Missing Models** — itniżżel kull fajl tal-mudell li mhuwiex preżenti fuq id-diska. Bar tal-progress u linja ta' status isegwu kull fajl waqt it-tniżżil.
- **Check for Updates** — jiċċekkja jekk hemmx piżijiet tal-mudell aktar ġodda disponibbli. Banner ta' aġġornament jidher ukoll fuq l-iskrin ewlieni awtomatikament meta jiġu detettati piżijiet aġġornati.

## Modalità ta' Segmentazzjoni

Tikkontrolla kif l-awdjo jiġi maqsum f'segmenti qabel ir-rikonoxximent tal-kelma.

| Modalità | Deskrizzjoni |
|---|---|
| **Speaker Diarization** | Juża l-mudell SortFormer biex jidentifika kelliema individwali u jittikketta kull segement. L-aħjar għal intervisti, laqgħat, u reġistrazzjonijiet b'ħafna kelliema. |
| **Voice Activity Detection** | Juża Silero VAD biex jiddetetta reġjuni tal-kelma biss — mingħajr tikketti ta' kelliema. Aktar mgħaġġel mid-diarization u addattat sewwa għall-awdjo ta' kelliem wieħed. |

## Editur tat-Traskrizzjoni

**Default Playback Mode** — jistabbilixxi l-modalità ta' riproduzzjoni użata meta tiftaħ l-editur tat-traskrizzjoni. Tista' tbiddilha direttament fl-editur ukoll fi kwalunkwe ħin. Ara [L-Editjar tat-Traskrizzjonijiet](../operations/editing_transcripts.md) għal deskrizzjoni ta' kull modalità.

## Dehra

Agħżel it-tema **Skura** jew **Ċara**. Il-bidla tapplika immedjatament. Ara [L-Għażla ta' Tema](theme.md).

## Lingwa

Agħżel il-lingwa tal-wiri għall-interface tal-applikazzjoni. Il-bidla tapplika immedjatament. Ara [L-Għażla ta' Lingwa](language.md).

---