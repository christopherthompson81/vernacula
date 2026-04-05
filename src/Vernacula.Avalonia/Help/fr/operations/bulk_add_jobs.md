---
title: "Mise en file d'attente de plusieurs fichiers audio"
description: "Comment ajouter plusieurs fichiers audio à la file d'attente en une seule fois."
topic_id: operations_bulk_add_jobs
---

# Mise en file d'attente de plusieurs fichiers audio

Utilisez **Bulk Add Jobs** pour ajouter plusieurs fichiers audio ou vidéo à la file d'attente de transcription en une seule étape. L'application les traite un par un dans l'ordre où ils ont été ajoutés.

## Prérequis

- Tous les fichiers de modèle doivent être téléchargés. La carte **Model Status** doit afficher `All N model file(s) present ✓`. Consultez [Téléchargement des modèles](../first_steps/downloading_models.md).

## Comment ajouter plusieurs tâches en une fois

1. Sur l'écran d'accueil, cliquez sur `Bulk Add Jobs`.
2. Un sélecteur de fichiers s'ouvre. Sélectionnez un ou plusieurs fichiers audio ou vidéo — maintenez `Ctrl` ou `Shift` pour en sélectionner plusieurs.
3. Cliquez sur **Open**. Chaque fichier sélectionné est ajouté au tableau **Transcription History** en tant que tâche distincte.

> **Fichiers vidéo comportant plusieurs pistes audio :** Si un fichier vidéo contient plus d'une piste audio (par exemple, plusieurs langues ou une piste de commentaire du réalisateur), l'application crée automatiquement une tâche par piste.

## Noms des tâches

Chaque tâche est nommée automatiquement d'après le nom de son fichier audio. Vous pouvez renommer une tâche à tout moment en cliquant sur son nom dans la colonne **Title** du tableau Transcription History, en modifiant le texte, puis en appuyant sur `Enter` ou en cliquant ailleurs.

## Comportement de la file d'attente

- Si aucune tâche n'est en cours d'exécution, le premier fichier démarre immédiatement et les suivants sont affichés comme `queued`.
- Si une tâche est déjà en cours d'exécution, tous les fichiers nouvellement ajoutés sont affichés comme `queued` et démarreront automatiquement les uns après les autres.
- Pour surveiller la tâche active, cliquez sur `Monitor` dans sa colonne **Actions**. Consultez [Surveillance des tâches](monitoring_jobs.md).
- Pour mettre en pause ou supprimer une tâche en attente avant qu'elle ne démarre, utilisez les boutons `Pause` ou `Remove` dans sa colonne **Actions**. Consultez [Mettre en pause, reprendre ou supprimer des tâches](pausing_resuming_removing.md).

---