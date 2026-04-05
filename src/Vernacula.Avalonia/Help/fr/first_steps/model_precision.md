---
title: "Choisir la précision des poids du modèle"
description: "Comment choisir entre la précision INT8 et FP32 et quels sont les compromis."
topic_id: first_steps_model_precision
---

# Choisir la précision des poids du modèle

La précision du modèle contrôle le format numérique utilisé par les poids du modèle d'IA. Elle influe sur la taille du téléchargement, l'utilisation de la mémoire et la précision des résultats.

## Options de précision

### INT8 (téléchargement plus léger)

- Fichiers de modèle plus petits — téléchargement plus rapide et espace disque réduit.
- Précision légèrement inférieure sur certains fichiers audio.
- Recommandé si vous disposez d'un espace disque limité ou d'une connexion Internet lente.

### FP32 (plus précis)

- Fichiers de modèle plus volumineux.
- Précision plus élevée, notamment sur les fichiers audio difficiles comportant des accents ou des bruits de fond.
- Recommandé lorsque la précision est prioritaire et que vous disposez d'un espace disque suffisant.
- Requis pour l'accélération GPU CUDA — le chemin GPU utilise toujours FP32, quel que soit ce paramètre.

## Modifier la précision

Ouvrez `Settings…` depuis la barre de menus, puis accédez à la section **Models** et sélectionnez `INT8 (smaller download)` ou `FP32 (more accurate)`.

## Après avoir modifié la précision

Changer de précision nécessite un ensemble différent de fichiers de modèle. Si les modèles correspondant à la nouvelle précision n'ont pas encore été téléchargés, cliquez sur `Download Missing Models` dans les paramètres. Les fichiers précédemment téléchargés pour l'autre précision sont conservés sur le disque et n'ont pas besoin d'être re-téléchargés si vous revenez à l'ancienne précision.

---