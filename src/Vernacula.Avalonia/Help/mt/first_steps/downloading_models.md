---
title: "Qed Tniżżel il-Mudelli"
description: "Kif tniżżel il-fajls tal-mudell tal-AI meħtieġa għat-traskrizzjoni."
topic_id: first_steps_downloading_models
---

# Qed Tniżżel il-Mudelli

Vernacula-Desktop jeħtieġ fajls tal-mudell tal-AI biex jaħdem. Dawn ma jinkludux mal-applikazzjoni u jridu jitniżżlu qabel l-ewwel traskrizzjoni tiegħek.

## Status tal-Mudell (Skrin tad-Dar)

Linja ta' status żgħira fil-vrieħ tal-iskrin tad-dar turi jekk il-mudelli tiegħek humiex lesti. Meta jkun hemm fajls nieqsa, turi wkoll buttuna `Open Settings` li tieħdok direttament għall-ġestjoni tal-mudelli.

| Status | Tifsira |
|---|---|
| `All N model file(s) present ✓` | Il-fajls kollha meħtieġa huma mniżżlin u lesti. |
| `N model file(s) missing: …` | Fajl wieħed jew aktar mhumiex preżenti; iftaħ is-Settings biex tniżżilhom. |

Meta l-mudelli jkunu lesti, il-buttuni `New Transcription` u `Bulk Add Jobs` isiru attivi.

## Kif Tniżżel il-Mudelli

1. Fuq l-iskrin tad-dar, ikklikkja `Open Settings` (jew mur `Settings… > Models`).
2. Fit-taqsima **Models**, ikklikkja `Download Missing Models`.
3. Jidher progress bar u linja ta' status li juru l-fajl attwali, il-pożizzjoni tiegħu fil-kju, u d-daqs tat-tniżżil — pereżempju: `[1/3] encoder-model.onnx — 42 MB`.
4. Stenna sakemm l-istatus jaqra `Download complete.`

## Ikkanċellar Tniżżil

Biex tieqaf tniżżil li jkun għaddej, ikklikkja `Cancel`. Il-linja ta' status se turi `Download cancelled.` Il-fajls imniżżlin parzjalment jinżammu, b'hekk it-tniżżil jibda minn fejn waqaf id-darba li jmiss li tikklikkja `Download Missing Models`.

## Żbalji fit-Tniżżil

Jekk tniżżil jiffranka, il-linja ta' status turi `Download failed: <reason>`. Iċċekkja l-konnessjoni tal-internet tiegħek u ikklikkja `Download Missing Models` mill-ġdid biex terġa' tipprova. L-applikazzjoni tibda mill-ġdid mill-aħħar fajl li ntemm b'suċċess.

## Bidel il-Preċiżjoni

Il-fajls tal-mudell li jeħtieġu jitniżżlu jiddependu fuq il-**Model Precision** magħżula. Biex tibdilha, mur `Settings… > Models > Model Precision`. Jekk tbiddel il-preċiżjoni wara t-tniżżil, is-sett ġdid ta' fajls irid jitniżżel separatament. Ara [Tagħżel il-Preċiżjoni tal-Piż tal-Mudell](model_precision.md).

---