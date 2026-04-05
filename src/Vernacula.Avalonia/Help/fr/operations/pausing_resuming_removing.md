---
title: "Suspendre, reprendre ou supprimer des tâches"
description: "Comment suspendre une tâche en cours, reprendre une tâche arrêtée ou supprimer une tâche de l'historique."
topic_id: operations_pausing_resuming_removing
---

# Suspendre, reprendre ou supprimer des tâches

## Suspendre une tâche

Vous pouvez suspendre une tâche en cours d'exécution ou en file d'attente depuis deux endroits :

- **Vue de progression** — cliquez sur `Pause` dans le coin inférieur droit pendant que vous suivez la tâche active.
- **Tableau de l'historique de transcription** — cliquez sur `Pause` dans la colonne **Actions** de n'importe quelle ligne dont le statut est `running` ou `queued`.

Après avoir cliqué sur `Pause`, la ligne de statut affiche `Pausing…` pendant que l'application termine l'unité de traitement en cours. Le statut de la tâche passe ensuite à `cancelled` dans le tableau de l'historique.

> La suspension sauvegarde tous les segments transcrits jusqu'à présent. Vous pouvez reprendre la tâche ultérieurement sans perdre ce travail.

## Reprendre une tâche

Pour reprendre une tâche suspendue ou échouée :

1. Sur l'écran d'accueil, localisez la tâche dans le tableau **Transcription History**. Son statut sera `cancelled` ou `failed`.
2. Cliquez sur `Resume` dans la colonne **Actions**.
3. L'application revient à la vue **Progress** et reprend là où le traitement s'était arrêté.

La ligne de statut affiche brièvement `Resuming…` pendant la réinitialisation de la tâche.

## Supprimer une tâche

Pour supprimer définitivement une tâche et sa transcription de l'historique :

1. Dans le tableau **Transcription History**, cliquez sur `Remove` dans la colonne **Actions** de la tâche que vous souhaitez supprimer.

La tâche est retirée de la liste et ses données sont effacées de la base de données locale. Cette action est irréversible. Les fichiers exportés enregistrés sur le disque ne sont pas affectés.

---