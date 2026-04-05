---
title: "Żieda ta' Fajls Awdjo Multipli fil-Kju"
description: "Kif iżżid diversi fajls awdjo fil-kju tax-xogħlijiet f'daqqa waħda."
topic_id: operations_bulk_add_jobs
---

# Żieda ta' Fajls Awdjo Multipli fil-Kju

Uża **Bulk Add Jobs** biex tpoġġi fil-kju fajls awdjo jew vidjo multipli għat-traskrizzjoni f'pass wieħed. L-applikazzjoni tipproċessahom wieħed wieħed fl-ordni li fih inżdiedu.

## Prerekwiżiti

- Il-fajls kollha tal-mudell iridu jkunu mtella'. Il-kard **Model Status** trid turi `All N model file(s) present ✓`. Ara [Tniżżil tal-Mudelli](../first_steps/downloading_models.md).

## Kif Iżżid Xogħlijiet bl-Ingrossa

1. Fuq l-iskrin ewlieni, ikklikkja `Bulk Add Jobs`.
2. Jinfetaħ selezzjonatur ta' fajls. Agħżel fajl wieħed jew aktar ta' awdjo jew vidjo — żomm `Ctrl` jew `Shift` biex tagħżel fajls multipli.
3. Ikklikkja **Open**. Kull fajl magħżul jiżdied fit-tabella **Transcription History** bħala xogħol separat.

> **Fajls vidjo b'flussi awdjo multipli:** Jekk fajl vidjo fih aktar minn fluss awdjo wieħed (per eżempju, lingwi multipli jew track ta' kummentarju tad-direttur), l-applikazzjoni toħloq xogħol wieħed għal kull fluss awtomatikament.

## Ismijiet tax-Xogħlijiet

Kull xogħol jissemma awtomatikament mill-isem tal-fajl awdjo tiegħu. Tista' tagħti isem ġdid lil xogħol fi kwalunkwe ħin billi tikklikkja fuq ismu fil-kolonna **Title** tat-tabella Transcription History, teditja t-test, u tagħfas `Enter` jew tikklikkja 'l barra.

## Imġiba tal-Kju

- Jekk l-ebda xogħol ma jkun qiegħed jidħol bħalissa, l-ewwel fajl jibda immedjatament u l-bqija jintwera bħala `queued`.
- Jekk xogħol ikun diġà qed jaħdem, il-fajls kollha miżjuda ġodda jintwera bħala `queued` u jibdew awtomatikament fi sfera.
- Biex tissorvelja x-xogħol attiv, ikklikkja `Monitor` fil-kolonna **Actions** tiegħu. Ara [Monitoraġġ tax-Xogħlijiet](monitoring_jobs.md).
- Biex twaqqaf jew tneħħi xogħol fil-kju qabel ma jibda, uża l-buttuni `Pause` jew `Remove` fil-kolonna **Actions** tiegħu. Ara [Waqfien, Tkomplija, jew Tneħħija ta' Xogħlijiet](pausing_resuming_removing.md).

---