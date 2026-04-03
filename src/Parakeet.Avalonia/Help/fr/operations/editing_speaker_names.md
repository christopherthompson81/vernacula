---
title: "Modification des noms des intervenants"
description: "Comment remplacer les identifiants génériques des intervenants par de vrais noms dans une transcription."
topic_id: operations_editing_speaker_names
---

# Modification des noms des intervenants

Le moteur de transcription attribue automatiquement à chaque intervenant un identifiant générique (par exemple, `speaker_0`, `speaker_1`). Vous pouvez remplacer ces identifiants par de vrais noms, qui apparaîtront dans l'ensemble de la transcription ainsi que dans tous les fichiers exportés.

## Comment modifier les noms des intervenants

1. Ouvrez un travail terminé. Voir [Chargement des travaux terminés](loading_completed_jobs.md).
2. Dans la vue **Résultats**, cliquez sur `Edit Speaker Names`.
3. La boîte de dialogue **Edit Speaker Names** s'ouvre avec deux colonnes :
   - **Speaker ID** — l'étiquette d'origine attribuée par le modèle (en lecture seule).
   - **Display Name** — le nom affiché dans la transcription (modifiable).
4. Cliquez sur une cellule dans la colonne **Display Name** et saisissez le nom de l'intervenant.
5. Appuyez sur `Tab` ou cliquez sur une autre ligne pour passer à l'intervenant suivant.
6. Cliquez sur `Save` pour appliquer les modifications, ou sur `Cancel` pour les annuler.

## Où les noms apparaissent

Les noms d'affichage mis à jour remplacent les identifiants génériques dans :

- Le tableau des segments dans la vue Résultats.
- Tous les fichiers exportés (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Modifier les noms à nouveau

Vous pouvez rouvrir la boîte de dialogue Edit Speaker Names à tout moment lorsque le travail est chargé dans la vue Résultats. Les modifications sont enregistrées dans la base de données locale et persistent d'une session à l'autre.

---