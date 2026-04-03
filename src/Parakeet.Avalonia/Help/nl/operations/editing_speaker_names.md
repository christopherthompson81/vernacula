---
title: "Sprekernamen bewerken"
description: "Hoe u generieke sprekers-ID's vervangt door echte namen in een transcript."
topic_id: operations_editing_speaker_names
---

# Sprekernamen bewerken

De transcriptiemotor labelt elke spreker automatisch met een generieke ID (bijvoorbeeld `speaker_0`, `speaker_1`). U kunt deze vervangen door echte namen die in het gehele transcript en in eventuele geëxporteerde bestanden worden weergegeven.

## Sprekernamen bewerken

1. Open een voltooide taak. Zie [Voltooide taken laden](loading_completed_jobs.md).
2. Klik in de weergave **Resultaten** op `Edit Speaker Names`.
3. Het dialoogvenster **Edit Speaker Names** wordt geopend met twee kolommen:
   - **Speaker ID** — het oorspronkelijke label dat door het model is toegewezen (alleen-lezen).
   - **Display Name** — de naam die in het transcript wordt weergegeven (bewerkbaar).
4. Klik op een cel in de kolom **Display Name** en typ de naam van de spreker.
5. Druk op `Tab` of klik op een andere rij om naar de volgende spreker te gaan.
6. Klik op `Save` om de wijzigingen toe te passen, of op `Cancel` om ze te verwerpen.

## Waar namen verschijnen

Bijgewerkte weergavenamen vervangen de generieke ID's in:

- De segmententabel in de weergave Resultaten.
- Alle geëxporteerde bestanden (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Namen opnieuw bewerken

U kunt het dialoogvenster Sprekernamen bewerken op elk gewenst moment opnieuw openen zolang de taak is geladen in de weergave Resultaten. Wijzigingen worden opgeslagen in de lokale database en blijven behouden tussen sessies.

---