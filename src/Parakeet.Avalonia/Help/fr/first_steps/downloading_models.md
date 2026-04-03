---
title: "Téléchargement des modèles"
description: "Comment télécharger les fichiers de modèles IA requis pour la transcription."
topic_id: first_steps_downloading_models
---

# Téléchargement des modèles

Parakeet Transcription nécessite des fichiers de modèles IA pour fonctionner. Ceux-ci ne sont pas inclus dans l'application et doivent être téléchargés avant votre première transcription.

## État des modèles (écran d'accueil)

Une fine ligne d'état en haut de l'écran d'accueil indique si vos modèles sont prêts. Lorsque des fichiers sont manquants, elle affiche également un bouton `Open Settings` qui vous amène directement à la gestion des modèles.

| État | Signification |
|---|---|
| `All N model file(s) present ✓` | Tous les fichiers requis sont téléchargés et prêts. |
| `N model file(s) missing: …` | Un ou plusieurs fichiers sont absents ; ouvrez les Paramètres pour les télécharger. |

Lorsque les modèles sont prêts, les boutons `New Transcription` et `Bulk Add Jobs` deviennent actifs.

## Comment télécharger les modèles

1. Sur l'écran d'accueil, cliquez sur `Open Settings` (ou accédez à `Settings… > Models`).
2. Dans la section **Models**, cliquez sur `Download Missing Models`.
3. Une barre de progression et une ligne d'état apparaissent, indiquant le fichier en cours, sa position dans la file d'attente et la taille du téléchargement — par exemple : `[1/3] encoder-model.onnx — 42 MB`.
4. Attendez que l'état affiche `Download complete.`

## Annuler un téléchargement

Pour interrompre un téléchargement en cours, cliquez sur `Cancel`. La ligne d'état affichera `Download cancelled.` Les fichiers partiellement téléchargés sont conservés, de sorte que le téléchargement reprend là où il s'était arrêté la prochaine fois que vous cliquez sur `Download Missing Models`.

## Erreurs de téléchargement

Si un téléchargement échoue, la ligne d'état affiche `Download failed: <reason>`. Vérifiez votre connexion Internet et cliquez à nouveau sur `Download Missing Models` pour réessayer. L'application reprend à partir du dernier fichier téléchargé avec succès.

## Modifier la précision

Les fichiers de modèles à télécharger dépendent de la **Model Precision** sélectionnée. Pour la modifier, accédez à `Settings… > Models > Model Precision`. Si vous changez la précision après avoir téléchargé les fichiers, le nouvel ensemble de fichiers doit être téléchargé séparément. Consultez [Choisir la précision des poids du modèle](model_precision.md).

---