---
title: "Läsa in slutförda jobb"
description: "Hur du öppnar resultaten från en tidigare slutförd transkription."
topic_id: operations_loading_completed_jobs
---

# Läsa in slutförda jobb

Alla slutförda transkriptionsjobb sparas i den lokala databasen och är tillgängliga i tabellen **Transkriptionshistorik** på startskärmen.

## Hur du läser in ett slutfört jobb

1. På startskärmen letar du upp jobbet i tabellen **Transkriptionshistorik**. Slutförda jobb visas med statusmärket `complete`.
2. Klicka på `Load` i jobbets kolumn **Actions**.
3. Programmet växlar till vyn **Results**, där alla transkriberade segment för det jobbet visas.

## Vyn Results

Vyn Results visar:

- Ljudfilens namn som sidrubrik.
- En underrubrik med antalet segment (till exempel `42 segment(s)`).
- En tabell med segment som innehåller kolumnerna **Speaker**, **Start**, **End** och **Content**.

Från vyn Results kan du:

- [Redigera transkriptionen](editing_transcripts.md) — granska och korrigera text, justera tidpunkter, slå ihop eller dela segment samt verifiera segment medan du lyssnar på ljudet.
- [Redigera talarnamn](editing_speaker_names.md) — ersätt generiska ID:n som `speaker_0` med riktiga namn.
- [Exportera transkriptionen](exporting_results.md) — spara transkriptionen till Excel, CSV, JSON, SRT, Markdown, Word eller SQLite.

Klicka på `← Back to History` för att återgå till historiklistan.

---