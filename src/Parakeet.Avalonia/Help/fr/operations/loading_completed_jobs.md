---
title: "Chargement des tâches terminées"
description: "Comment ouvrir les résultats d'une transcription précédemment terminée."
topic_id: operations_loading_completed_jobs
---

# Chargement des tâches terminées

Toutes les tâches de transcription terminées sont enregistrées dans la base de données locale et restent accessibles dans le tableau **Historique des transcriptions** sur l'écran d'accueil.

## Comment charger une tâche terminée

1. Sur l'écran d'accueil, localisez la tâche dans le tableau **Historique des transcriptions**. Les tâches terminées affichent un badge de statut `complete`.
2. Cliquez sur `Load` dans la colonne **Actions** de la tâche.
3. L'application bascule vers la vue **Résultats**, affichant tous les segments transcrits pour cette tâche.

## Vue Résultats

La vue Résultats affiche :

- Le nom du fichier audio comme titre de la page.
- Un sous-titre indiquant le nombre de segments (par exemple, `42 segment(s)`).
- Un tableau de segments avec les colonnes **Speaker**, **Start**, **End** et **Content**.

Depuis la vue Résultats, vous pouvez :

- [Modifier la transcription](editing_transcripts.md) — vérifier et corriger le texte, ajuster le minutage, fusionner ou diviser des segments, et valider les segments tout en écoutant l'audio.
- [Modifier les noms des intervenants](editing_speaker_names.md) — remplacer les identifiants génériques tels que `speaker_0` par de véritables noms.
- [Exporter la transcription](exporting_results.md) — enregistrer la transcription au format Excel, CSV, JSON, SRT, Markdown, Word ou SQLite.

Pour revenir à la liste de l'historique, cliquez sur `← Back to History`.

---