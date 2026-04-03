---
title: "Waqfien, Tkomplija, jew Tneħħija ta' Xogħlijiet"
description: "Kif twaqqaf xogħol li qed jopera, tibda mill-ġdid wieħed li twaqqaf, jew tħassar xogħol mill-istorja."
topic_id: operations_pausing_resuming_removing
---

# Waqfien, Tkomplija, jew Tneħħija ta' Xogħlijiet

## Waqfien ta' Xogħol

Tista' twaqqaf xogħol li qed jopera jew li jinsab fil-kju minn żewġ postijiet:

- **Veduta tal-Progress** — ikklikkja `Pause` fil-kantuniera t'isfel fuq il-lemin waqt li tkun qed issegwi x-xogħol attiv.
- **Tabella tal-Istorja tat-Traskrizzjoni** — ikklikkja `Pause` fil-kolonna **Actions** ta' kwalunkwe ringiela li l-istat tagħha huwa `running` jew `queued`.

Wara li tikklikkja `Pause`, il-linja tal-istat turi `Pausing…` waqt li l-applikazzjoni tispiċċa l-unità ta' pproċessar kurrenti. L-istat tax-xogħol imbagħad jinbidel għal `cancelled` fit-tabella tal-istorja.

> Il-waqfien jiffranka s-segmenti kollha traskrivuti sa dak il-ħin. Tista' tkompli x-xogħol aktar tard mingħajr ma titlef dak ix-xogħol.

## Tkomplija ta' Xogħol

Biex tkompli xogħol li twaqqaf jew li falla:

1. Fuq l-iskrin ewlieni, sib ix-xogħol fit-tabella **Transcription History**. L-istat tiegħu jkun `cancelled` jew `failed`.
2. Ikklikkja `Resume` fil-kolonna **Actions**.
3. L-applikazzjoni terġa' tmur għall-veduta tal-**Progress** u tkompli minn fejn waqaf l-ipproċessar.

Il-linja tal-istat turi `Resuming…` għal ftit waqt li x-xogħol jerġa' jibda.

## Tneħħija ta' Xogħol

Biex tħassar xogħol u t-traskrizzjoni tiegħu mill-istorja b'mod permanenti:

1. Fit-tabella **Transcription History**, ikklikkja `Remove` fil-kolonna **Actions** tax-xogħol li trid tħassar.

Ix-xogħol jitneħħa mil-lista u d-dejta tiegħu titħassar mid-database lokali. Din l-azzjoni ma tistax tiġi annullata. Il-fajls esportati salvati fuq id-disk ma jiġux affettwati.

---