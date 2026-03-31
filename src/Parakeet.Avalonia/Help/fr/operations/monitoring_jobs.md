---
title: "Surveillance des tâches"
description: "Comment suivre la progression d'une tâche en cours d'exécution ou en file d'attente."
topic_id: operations_monitoring_jobs
---

# Surveillance des tâches

La vue **Progression** vous offre un suivi en temps réel d'une tâche de transcription en cours.

## Ouvrir la vue Progression

- Lorsque vous démarrez une nouvelle transcription, l'application accède automatiquement à la vue Progression.
- Pour une tâche déjà en cours d'exécution ou en file d'attente, recherchez-la dans le tableau **Historique des transcriptions** et cliquez sur `Monitor` dans sa colonne **Actions**.

## Lire la vue Progression

| Élément | Description |
|---|---|
| Barre de progression | Pourcentage d'avancement global. Indéterminée (animée) pendant le démarrage ou la reprise de la tâche. |
| Étiquette de pourcentage | Pourcentage numérique affiché à droite de la barre. |
| Message d'état | Activité en cours — par exemple `Audio Analysis` ou `Speech Recognition`. Affiche `Waiting in queue…` si la tâche n'a pas encore démarré. |
| Tableau des segments | Flux en direct des segments transcrits avec les colonnes **Intervenant**, **Début**, **Fin** et **Contenu**. Défile automatiquement à mesure que de nouveaux segments arrivent. |

## Phases de progression

Les phases affichées dépendent du **Mode de segmentation** sélectionné dans les paramètres.

**Mode Diarisation des intervenants** (par défaut) :

1. **Audio Analysis** — la diarisation SortFormer s'exécute sur l'intégralité du fichier pour identifier les limites entre intervenants. La barre peut rester proche de 0 % jusqu'à la fin de cette phase.
2. **Speech Recognition** — chaque segment par intervenant est transcrit. Le pourcentage augmente régulièrement durant cette phase.

**Mode Détection d'activité vocale** :

1. **Detecting speech segments** — Silero VAD analyse le fichier pour repérer les zones de parole. Cette phase est rapide.
2. **Speech Recognition** — chaque zone de parole détectée est transcrite.

Dans les deux modes, le tableau de segments en direct se remplit au fur et à mesure de la transcription.

## Quitter la vue

Cliquez sur `← Back to Home` pour revenir à l'écran d'accueil sans interrompre la tâche. La tâche continue de s'exécuter en arrière-plan et son état est mis à jour dans le tableau **Historique des transcriptions**. Cliquez à nouveau sur `Monitor` à tout moment pour revenir à la vue Progression.

---