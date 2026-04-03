---
title: "Eksport af resultater eller transskriptioner"
description: "Sådan gemmer du en transskription til en fil i forskellige formater."
topic_id: operations_exporting_results
---

# Eksport af resultater eller transskriptioner

Du kan eksportere en færdig transskription til flere filformater til brug i andre programmer.

## Sådan eksporterer du

1. Åbn et fuldført job. Se [Indlæsning af fuldførte job](loading_completed_jobs.md).
2. I visningen **Resultater** skal du klikke på `Export Transcript`.
3. Dialogboksen **Export Transcript** åbnes. Vælg et format i rullemenuen **Format**.
4. Klik på `Save`. En dialogboks til at gemme filer åbnes.
5. Vælg en destinationsmappe og et filnavn, og klik derefter på **Save**.

En bekræftelsesmeddelelse vises nederst i dialogboksen med den fulde sti til den gemte fil.

## Tilgængelige formater

| Format | Filendelse | Bedst til |
|---|---|---|
| Excel | `.xlsx` | Regnearksanalyse med kolonner til taler, tidsstempler og indhold. |
| CSV | `.csv` | Import i et hvilket som helst regneark eller dataværktøj. |
| JSON | `.json` | Programmatisk behandling. |
| SRT-undertekster | `.srt` | Indlæsning i videoredigeringsprogrammer eller medieafspillere som undertekster. |
| Markdown | `.md` | Læsbare dokumenter i almindelig tekst. |
| Word-dokument | `.docx` | Deling med brugere af Microsoft Word. |
| SQLite-database | `.db` | Fuld databaseeksport til brugerdefinerede forespørgsler. |

## Telernavne i eksporter

Hvis du har tildelt visningsnavne til talere, bruges disse navne i alle eksportformater. Hvis du vil opdatere navne før eksport, skal du først klikke på `Edit Speaker Names`. Se [Redigering af telernavne](editing_speaker_names.md).

---