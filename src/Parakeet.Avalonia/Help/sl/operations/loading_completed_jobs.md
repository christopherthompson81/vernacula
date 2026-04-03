---
title: "Nalaganje dokončanih opravil"
description: "Kako odpreti rezultate predhodno dokončanega prepisovanja."
topic_id: operations_loading_completed_jobs
---

# Nalaganje dokončanih opravil

Vsa dokončana opravila prepisovanja so shranjena v lokalno zbirko podatkov in ostanejo dostopna v tabeli **Zgodovina prepisovanja** na domačem zaslonu.

## Kako naložiti dokončano opravilo

1. Na domačem zaslonu poiščite opravilo v tabeli **Zgodovina prepisovanja**. Dokončana opravila imajo označbo stanja `complete`.
2. Kliknite `Load` v stolpcu **Actions** za to opravilo.
3. Aplikacija se preklopi na pogled **Results**, kjer so prikazani vsi prepisani segmenti tega opravila.

## Pogled Results

Pogled Results prikazuje:

- Ime zvočne datoteke kot naslov strani.
- Podnaslov s številom segmentov (na primer `42 segment(s)`).
- Tabelo segmentov s stolpci **Speaker**, **Start**, **End** in **Content**.

V pogledu Results lahko:

- [Urejate prepis](editing_transcripts.md) — pregledate in popravite besedilo, prilagodite časovne oznake, združite ali razdelite segmente ter preverite segmente med poslušanjem zvoka.
- [Urejate imena govorcev](editing_speaker_names.md) — zamenjate splošne identifikatorje, kot je `speaker_0`, z dejanskimi imeni.
- [Izvozite prepis](exporting_results.md) — shranite prepis v obliki Excel, CSV, JSON, SRT, Markdown, Word ali SQLite.

Če se želite vrniti na seznam zgodovine, kliknite `← Back to History`.

---