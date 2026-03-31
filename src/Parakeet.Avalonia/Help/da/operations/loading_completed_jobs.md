---
title: "Indlæsning af fuldførte job"
description: "Sådan åbner du resultaterne af en tidligere fuldført transskription."
topic_id: operations_loading_completed_jobs
---

# Indlæsning af fuldførte job

Alle fuldførte transskriptionsjob gemmes i den lokale database og er fortsat tilgængelige i tabellen **Transskriptionshistorik** på startskærmen.

## Sådan indlæser du et fuldført job

1. Find jobbet i tabellen **Transskriptionshistorik** på startskærmen. Fuldførte job vises med statusmærket `complete`.
2. Klik på `Load` i jobbets kolonne **Actions**.
3. Applikationen skifter til visningen **Results**, som viser alle transskriberede segmenter for det pågældende job.

## Visningen Results

Visningen Results viser:

- Lydfiltnavnet som sideoverskrift.
- En undertekst med antal segmenter (f.eks. `42 segment(s)`).
- En tabel over segmenter med kolonnerne **Speaker**, **Start**, **End** og **Content**.

Fra visningen Results kan du:

- [Redigere transskriptionen](editing_transcripts.md) — gennemgå og ret tekst, juster tidspunkter, flet eller opdel segmenter, og bekræft segmenter mens du lytter til lyden.
- [Redigere talerenavn](editing_speaker_names.md) — erstat generiske id'er som `speaker_0` med rigtige navne.
- [Eksportere transskriptionen](exporting_results.md) — gem transskriptionen til Excel, CSV, JSON, SRT, Markdown, Word eller SQLite.

Klik på `← Back to History` for at vende tilbage til historiklisten.

---