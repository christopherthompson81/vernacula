---
title: "Paramètres"
description: "Vue d'ensemble de toutes les options de la fenêtre Paramètres."
topic_id: first_steps_settings_window
---

# Paramètres

La fenêtre **Paramètres** vous permet de contrôler la configuration matérielle, la gestion des modèles, le mode de segmentation, le comportement de l'éditeur, l'apparence et la langue. Ouvrez-la depuis la barre de menu : `Settings…`.

## Matériel et performances

Cette section affiche l'état de votre GPU NVIDIA et de la pile logicielle CUDA, et indique le plafond de lots dynamique utilisé lors de la transcription sur GPU.

| Élément | Description |
|---|---|
| Nom du GPU et VRAM | GPU NVIDIA détecté et mémoire vidéo disponible. |
| CUDA Toolkit | Indique si les bibliothèques d'exécution CUDA ont été trouvées via `CUDA_PATH`. |
| cuDNN | Indique si les DLL d'exécution cuDNN sont disponibles. |
| Accélération CUDA | Indique si ONNX Runtime a chargé avec succès le fournisseur d'exécution CUDA. |

Cliquez sur `Re-check` pour relancer la détection matérielle sans redémarrer l'application — utile après l'installation de CUDA ou cuDNN.

Des liens de téléchargement directs pour le CUDA Toolkit et cuDNN s'affichent lorsque ces composants ne sont pas détectés.

Le message du **plafond de lots** indique combien de secondes d'audio sont traitées lors de chaque exécution sur GPU. Cette valeur est calculée à partir de la VRAM libre après le chargement des modèles et s'ajuste automatiquement.

Pour les instructions complètes de configuration CUDA, consultez [Installer CUDA et cuDNN](cuda_installation.md).

## Modèles

Cette section permet de gérer les fichiers de modèles d'IA nécessaires à la transcription.

- **Télécharger les modèles manquants** — télécharge les fichiers de modèles non encore présents sur le disque. Une barre de progression et une ligne d'état suivent chaque fichier au cours du téléchargement.
- **Vérifier les mises à jour** — vérifie si des poids de modèles plus récents sont disponibles. Une bannière de mise à jour s'affiche également automatiquement sur l'écran d'accueil lorsque des poids mis à jour sont détectés.

## Mode de segmentation

Contrôle la manière dont l'audio est divisé en segments avant la reconnaissance vocale.

| Mode | Description |
|---|---|
| **Diarisation des locuteurs** | Utilise le modèle SortFormer pour identifier les locuteurs individuels et étiqueter chaque segment. Idéal pour les entretiens, les réunions et les enregistrements avec plusieurs locuteurs. |
| **Détection d'activité vocale** | Utilise Silero VAD pour détecter uniquement les zones de parole — sans étiquettes de locuteur. Plus rapide que la diarisation et bien adapté aux enregistrements à locuteur unique. |

## Éditeur de transcription

**Mode de lecture par défaut** — définit le mode de lecture utilisé à l'ouverture de l'éditeur de transcription. Vous pouvez également le modifier directement dans l'éditeur à tout moment. Voir [Modifier les transcriptions](../operations/editing_transcripts.md) pour une description de chaque mode.

## Apparence

Sélectionnez le thème **Sombre** ou **Clair**. La modification s'applique immédiatement. Voir [Choisir un thème](theme.md).

## Langue

Sélectionnez la langue d'affichage de l'interface de l'application. La modification s'applique immédiatement. Voir [Choisir une langue](language.md).

---