---
title: "Nouveau flux de travail de transcription"
description: "Guide étape par étape pour transcrire un fichier audio."
topic_id: operations_new_transcription
---

# Nouveau flux de travail de transcription

Utilisez ce flux de travail pour transcrire un seul fichier audio.

## Prérequis

- Tous les fichiers de modèles doivent être téléchargés. La carte **État des modèles** doit afficher `All N model file(s) present ✓`. Voir [Téléchargement des modèles](../first_steps/downloading_models.md).

## Formats pris en charge

### Audio

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Vidéo

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Les fichiers vidéo sont décodés via FFmpeg. Si un fichier vidéo contient **plusieurs flux audio** (par exemple, plusieurs langues ou pistes de commentaires), une tâche de transcription est créée automatiquement pour chaque flux.

## Étapes

### 1. Ouvrir le formulaire Nouvelle transcription

Cliquez sur `New Transcription` dans l'écran d'accueil, ou accédez à `File > New Transcription`.

### 2. Sélectionner un fichier multimédia

Cliquez sur `Browse…` à côté du champ **Audio File**. Un sélecteur de fichiers s'ouvre, filtré sur les formats audio et vidéo pris en charge. Sélectionnez votre fichier et cliquez sur **Open**. Le chemin d'accès au fichier apparaît dans le champ.

### 3. Nommer la tâche

Le champ **Job Name** est prérempli à partir du nom du fichier. Modifiez-le si vous souhaitez un libellé différent — ce nom apparaît dans l'historique des transcriptions sur l'écran d'accueil.

### 4. Lancer la transcription

Cliquez sur `Start Transcription`. L'application bascule vers la vue **Progress**.

Pour revenir en arrière sans lancer la transcription, cliquez sur `← Back`.

## Ce qui se passe ensuite

La tâche s'exécute en deux phases affichées dans la barre de progression :

1. **Audio Analysis** — diarisation des locuteurs : identification de qui parle et à quel moment.
2. **Speech Recognition** — conversion de la parole en texte, segment par segment.

Les segments transcrits apparaissent dans le tableau en direct au fur et à mesure de leur production. Une fois le traitement terminé, l'application passe automatiquement à la vue **Results**.

Si vous ajoutez une tâche alors qu'une autre est déjà en cours d'exécution, la nouvelle tâche affichera le statut `queued` et démarrera lorsque la tâche en cours sera terminée. Voir [Surveillance des tâches](monitoring_jobs.md).

---