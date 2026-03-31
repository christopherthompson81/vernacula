---
title: "Exportera resultat eller transkript"
description: "Hur du sparar ett transkript till en fil i olika format."
topic_id: operations_exporting_results
---

# Exportera resultat eller transkript

Du kan exportera ett färdigt transkript till flera filformat för användning i andra program.

## Så här exporterar du

1. Öppna ett avslutat jobb. Se [Läsa in avslutade jobb](loading_completed_jobs.md).
2. I vyn **Resultat**, klicka på `Export Transcript`.
3. Dialogrutan **Export Transcript** öppnas. Välj ett format i listrutan **Format**.
4. Klicka på `Save`. En dialogruta för att spara filen öppnas.
5. Välj en målmapp och ett filnamn och klicka sedan på **Save**.

Ett bekräftelsemeddelande visas längst ned i dialogrutan med den fullständiga sökvägen till den sparade filen.

## Tillgängliga format

| Format | Ändelse | Passar bäst för |
|---|---|---|
| Excel | `.xlsx` | Kalkylbladsanalys med kolumner för talare, tidsstämplar och innehåll. |
| CSV | `.csv` | Import till valfritt kalkylblad eller dataverktyg. |
| JSON | `.json` | Programmatisk bearbetning. |
| SRT Subtitles | `.srt` | Inläsning i videoredigerare eller mediaspelare som undertexter. |
| Markdown | `.md` | Lättlästa dokument i klartext. |
| Word Document | `.docx` | Delning med användare av Microsoft Word. |
| SQLite Database | `.db` | Fullständig databasexport för anpassade frågor. |

## Talarnamn i exporter

Om du har tilldelat visningsnamn till talare används dessa namn i alla exportformat. Om du vill uppdatera namn innan du exporterar klickar du först på `Edit Speaker Names`. Se [Redigera talarnamn](editing_speaker_names.md).

---