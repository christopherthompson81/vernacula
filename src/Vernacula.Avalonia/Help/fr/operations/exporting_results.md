---
title: "Exporter les résultats ou les transcriptions"
description: "Comment enregistrer une transcription dans un fichier en différents formats."
topic_id: operations_exporting_results
---

# Exporter les résultats ou les transcriptions

Vous pouvez exporter une transcription terminée vers plusieurs formats de fichier pour l'utiliser dans d'autres applications.

## Comment exporter

1. Ouvrez un travail terminé. Voir [Charger des travaux terminés](loading_completed_jobs.md).
2. Dans la vue **Résultats**, cliquez sur `Export Transcript`.
3. La boîte de dialogue **Export Transcript** s'ouvre. Choisissez un format dans le menu déroulant **Format**.
4. Cliquez sur `Save`. Une boîte de dialogue d'enregistrement s'ouvre.
5. Choisissez un dossier de destination et un nom de fichier, puis cliquez sur **Save**.

Un message de confirmation s'affiche en bas de la boîte de dialogue, indiquant le chemin complet du fichier enregistré.

## Formats disponibles

| Format | Extension | Idéal pour |
|---|---|---|
| Excel | `.xlsx` | Analyse dans un tableur avec des colonnes pour le locuteur, les horodatages et le contenu. |
| CSV | `.csv` | Importation dans n'importe quel tableur ou outil de données. |
| JSON | `.json` | Traitement programmatique. |
| Sous-titres SRT | `.srt` | Chargement dans des éditeurs vidéo ou des lecteurs multimédias en tant que sous-titres. |
| Markdown | `.md` | Documents en texte brut lisibles. |
| Document Word | `.docx` | Partage avec des utilisateurs de Microsoft Word. |
| Base de données SQLite | `.db` | Export complet de la base de données pour des requêtes personnalisées. |

## Noms des locuteurs dans les exports

Si vous avez attribué des noms d'affichage aux locuteurs, ces noms sont utilisés dans tous les formats d'export. Pour mettre à jour les noms avant d'exporter, cliquez d'abord sur `Edit Speaker Names`. Voir [Modifier les noms des locuteurs](editing_speaker_names.md).

---