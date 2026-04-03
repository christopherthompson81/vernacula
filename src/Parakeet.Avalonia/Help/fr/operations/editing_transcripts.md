---
title: "Modification des transcriptions"
description: "Comment examiner, corriger et vérifier les segments transcrits dans l'éditeur de transcription."
topic_id: operations_editing_transcripts
---

# Modification des transcriptions

L'**éditeur de transcription** vous permet d'examiner la sortie ASR, de corriger le texte, de renommer les intervenants directement, d'ajuster le minutage des segments et de marquer des segments comme vérifiés — tout en écoutant l'audio original.

## Ouverture de l'éditeur

1. Chargez une tâche terminée (voir [Chargement des tâches terminées](loading_completed_jobs.md)).
2. Dans la vue **Résultats**, cliquez sur `Edit Transcript`.

L'éditeur s'ouvre dans une fenêtre séparée et peut rester ouvert en parallèle de l'application principale.

## Disposition

Chaque segment est affiché sous forme de carte avec deux panneaux côte à côte :

- **Panneau gauche** — la sortie ASR d'origine avec une coloration de confiance mot par mot. Les mots pour lesquels le modèle était moins certain apparaissent en rouge ; les mots à haute confiance s'affichent dans la couleur de texte normale.
- **Panneau droit** — une zone de texte modifiable. Apportez vos corrections ici ; les différences par rapport à l'original sont mises en évidence au fur et à mesure de la saisie.

L'étiquette de l'intervenant et la plage horaire apparaissent au-dessus de chaque carte. Cliquez sur une carte pour la sélectionner et afficher ses icônes d'action. Survolez n'importe quelle icône pour afficher une info-bulle décrivant sa fonction.

## Légende des icônes

### Barre de lecture

| Icône | Action |
|-------|--------|
| ▶ | Lecture |
| ⏸ | Pause |
| ⏮ | Aller au segment précédent |
| ⏭ | Aller au segment suivant |

### Actions sur la carte de segment

| Icône | Action |
|-------|--------|
| <mdl2 ch="E77B"/> | Réaffecter le segment à un autre intervenant |
| <mdl2 ch="E916"/> | Ajuster les temps de début et de fin du segment |
| <mdl2 ch="EA39"/> | Supprimer ou restaurer le segment |
| <mdl2 ch="E72B"/> | Fusionner avec le segment précédent |
| <mdl2 ch="E72A"/> | Fusionner avec le segment suivant |
| <mdl2 ch="E8C6"/> | Diviser le segment |
| <mdl2 ch="E72C"/> | Relancer l'ASR sur ce segment |

## Lecture audio

Une barre de lecture s'étend en haut de la fenêtre de l'éditeur :

| Contrôle | Action |
|----------|--------|
| Icône Lecture / Pause | Démarrer ou mettre en pause la lecture |
| Barre de défilement | Faire glisser pour accéder à n'importe quelle position dans l'audio |
| Curseur de vitesse | Ajuster la vitesse de lecture (0,5× – 2×) |
| Icônes Préc. / Suiv. | Aller au segment précédent ou suivant |
| Menu déroulant du mode de lecture | Sélectionner l'un des trois modes de lecture (voir ci-dessous) |
| Curseur de volume | Ajuster le volume de lecture |

Pendant la lecture, le mot en cours de prononciation est mis en évidence dans le panneau gauche. Lorsque la lecture est en pause après un déplacement, la mise en évidence se met à jour pour indiquer le mot à la position de déplacement.

### Modes de lecture

| Mode | Comportement |
|------|-------------|
| `Single` | Lit le segment actuel une fois, puis s'arrête. |
| `Auto-advance` | Lit le segment actuel ; lorsqu'il se termine, le marque comme vérifié et passe au suivant. |
| `Continuous` | Lit tous les segments en séquence sans en marquer aucun comme vérifié. |

Sélectionnez le mode actif dans le menu déroulant de la barre de lecture.

## Modification d'un segment

1. Cliquez sur une carte pour la sélectionner.
2. Modifiez le texte dans le panneau droit. Les modifications sont enregistrées automatiquement lorsque vous déplacez le focus vers une autre carte.

## Renommer un intervenant

Cliquez sur l'étiquette de l'intervenant dans la carte sélectionnée et saisissez un nouveau nom. Appuyez sur `Enter` ou cliquez ailleurs pour enregistrer. Le nouveau nom est appliqué uniquement à cette carte ; pour renommer un intervenant de manière globale, utilisez [Modifier les noms des intervenants](editing_speaker_names.md) depuis la vue Résultats.

## Vérification d'un segment

Cochez la case `Verified` sur une carte sélectionnée pour la marquer comme révisée. L'état de vérification est enregistré dans la base de données et est visible dans l'éditeur lors des chargements ultérieurs.

## Suppression d'un segment

Cliquez sur `Suppress` sur une carte sélectionnée pour masquer le segment lors des exportations (utile pour les bruits, la musique ou d'autres sections sans parole). Cliquez sur `Unsuppress` pour le restaurer.

## Ajustement des temps de segment

Cliquez sur `Adjust Times` sur une carte sélectionnée pour ouvrir la boîte de dialogue d'ajustement des temps. Utilisez la molette de la souris sur le champ **Start** ou **End** pour modifier la valeur par incréments de 0,1 seconde, ou saisissez une valeur directement. Cliquez sur `Save` pour appliquer.

## Fusion de segments

- Cliquez sur `⟵ Merge` pour fusionner le segment sélectionné avec le segment qui le précède immédiatement.
- Cliquez sur `Merge ⟶` pour fusionner le segment sélectionné avec le segment qui le suit immédiatement.

Le texte combiné et la plage horaire des deux cartes sont réunis. Cette fonction est utile lorsqu'une seule prise de parole a été répartie sur deux segments.

## Division d'un segment

Cliquez sur `Split…` sur une carte sélectionnée pour ouvrir la boîte de dialogue de division. Positionnez le point de division dans le texte et confirmez. Deux nouveaux segments sont créés couvrant la plage horaire d'origine. Cette fonction est utile lorsque deux prises de parole distinctes ont été regroupées dans un seul segment.

## Relancer l'ASR

Cliquez sur `Redo ASR` sur une carte sélectionnée pour relancer la reconnaissance vocale sur l'audio de ce segment. Le modèle traite uniquement la tranche audio de ce segment et produit une nouvelle transcription à source unique.

Utilisez cette option dans les cas suivants :

- Un segment provient d'une fusion et ne peut pas être divisé (les segments fusionnés couvrent plusieurs sources ASR ; Relancer l'ASR les regroupe en une seule, après quoi `Split…` devient disponible).
- La transcription d'origine est de mauvaise qualité et vous souhaitez effectuer une nouvelle passe propre sans modification manuelle.

**Remarque :** Tout texte déjà saisi dans le panneau droit est supprimé et remplacé par la nouvelle sortie ASR. L'opération nécessite que le fichier audio soit chargé ; le bouton est désactivé si l'audio n'est pas disponible.