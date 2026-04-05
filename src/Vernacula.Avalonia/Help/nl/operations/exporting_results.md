---
title: "Resultaten of transcripten exporteren"
description: "Hoe u een transcript naar een bestand in verschillende indelingen kunt opslaan."
topic_id: operations_exporting_results
---

# Resultaten of transcripten exporteren

U kunt een voltooid transcript exporteren naar verschillende bestandsindelingen voor gebruik in andere toepassingen.

## Exporteren

1. Open een voltooid taak. Zie [Voltooide taken laden](loading_completed_jobs.md).
2. Klik in de weergave **Resultaten** op `Export Transcript`.
3. Het dialoogvenster **Export Transcript** wordt geopend. Kies een indeling in de vervolgkeuzelijst **Format**.
4. Klik op `Save`. Er wordt een dialoogvenster voor opslaan geopend.
5. Kies een doelmap en bestandsnaam en klik vervolgens op **Opslaan**.

Onderaan het dialoogvenster verschijnt een bevestigingsbericht met het volledige pad van het opgeslagen bestand.

## Beschikbare indelingen

| Indeling | Extensie | Geschikt voor |
|---|---|---|
| Excel | `.xlsx` | Spreadsheetanalyse met kolommen voor spreker, tijdstempels en inhoud. |
| CSV | `.csv` | Importeren in elk spreadsheet- of gegevensprogramma. |
| JSON | `.json` | Programmatische verwerking. |
| SRT-ondertitels | `.srt` | Laden in video-editors of mediaspelers als ondertitels. |
| Markdown | `.md` | Leesbare tekstdocumenten zonder opmaak. |
| Word-document | `.docx` | Delen met gebruikers van Microsoft Word. |
| SQLite-database | `.db` | Volledige database-export voor aangepaste query's. |

## Sprekernamen in exports

Als u weergavenamen aan sprekers hebt toegewezen, worden deze namen gebruikt in alle exportindelingen. Als u namen wilt bijwerken vóór het exporteren, klikt u eerst op `Edit Speaker Names`. Zie [Sprekernamen bewerken](editing_speaker_names.md).

---