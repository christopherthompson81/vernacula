---
title: "Installation de CUDA et cuDNN pour l'accélération GPU"
description: "Comment configurer NVIDIA CUDA et cuDNN pour que Parakeet Transcription puisse utiliser votre GPU."
topic_id: first_steps_cuda_installation
---

# Installation de CUDA et cuDNN pour l'accélération GPU

Parakeet Transcription peut utiliser un GPU NVIDIA pour accélérer considérablement la transcription. L'accélération GPU nécessite que le kit d'outils NVIDIA CUDA et les bibliothèques d'exécution cuDNN soient installés sur votre système.

## Configuration requise

- Un GPU NVIDIA compatible CUDA (une GeForce GTX série 10 ou ultérieure est recommandée).
- Windows 10 ou 11 (64 bits).
- Les fichiers de modèles doivent déjà être téléchargés. Voir [Téléchargement des modèles](downloading_models.md).

## Étapes d'installation

### 1. Installer le kit d'outils CUDA

Téléchargez et exécutez le programme d'installation du kit d'outils CUDA depuis le site des développeurs NVIDIA. Pendant l'installation, acceptez les chemins par défaut. Le programme d'installation définit automatiquement la variable d'environnement `CUDA_PATH` — Parakeet utilise cette variable pour localiser les bibliothèques CUDA.

### 2. Installer cuDNN

Téléchargez l'archive ZIP cuDNN correspondant à votre version de CUDA installée depuis le site des développeurs NVIDIA. Extrayez l'archive et copiez le contenu de ses dossiers `bin`, `include` et `lib` dans les dossiers correspondants du répertoire d'installation du kit d'outils CUDA (le chemin indiqué par `CUDA_PATH`).

Vous pouvez également installer cuDNN à l'aide du programme d'installation NVIDIA cuDNN, s'il est disponible pour votre version de CUDA.

### 3. Redémarrer l'application

Fermez et rouvrez Parakeet Transcription après l'installation. L'application vérifie la présence de CUDA au démarrage.

## État du GPU dans les paramètres

Ouvrez `Settings…` depuis la barre de menus et consultez la section **Hardware & Performance**. Chaque composant affiche une coche (✓) lorsqu'il est détecté :

| Élément | Signification |
|---|---|
| Nom du GPU et VRAM | Votre GPU NVIDIA a été trouvé |
| CUDA Toolkit ✓ | Bibliothèques CUDA localisées via `CUDA_PATH` |
| cuDNN ✓ | DLL d'exécution cuDNN trouvées |
| CUDA Acceleration ✓ | ONNX Runtime a chargé le fournisseur d'exécution CUDA |

Si un élément est manquant après l'installation, cliquez sur `Re-check` pour relancer la détection du matériel sans redémarrer l'application.

La fenêtre Paramètres fournit également des liens de téléchargement directs pour le kit d'outils CUDA et cuDNN s'ils ne sont pas encore installés.

### Résolution des problèmes

Si `CUDA Acceleration` n'affiche pas de coche, vérifiez que :

- La variable d'environnement `CUDA_PATH` est bien définie (vérifiez dans `System > Advanced system settings > Environment Variables`).
- Les DLL cuDNN se trouvent dans un répertoire inclus dans le `PATH` système ou à l'intérieur du dossier `bin` de CUDA.
- Le pilote de votre GPU est à jour.

### Dimensionnement des lots

Lorsque CUDA est actif, la section **Hardware & Performance** affiche également le plafond dynamique de lots actuel — la durée maximale en secondes d'audio traitée en une seule exécution GPU. Cette valeur est calculée à partir de la VRAM disponible après le chargement des modèles et s'ajuste automatiquement si votre mémoire disponible change.

## Fonctionnement sans GPU

Si CUDA n'est pas disponible, Parakeet bascule automatiquement vers le traitement par CPU. La transcription fonctionne toujours, mais sera plus lente, en particulier pour les fichiers audio de longue durée.

---