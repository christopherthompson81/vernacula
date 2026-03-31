---
title: "Runātāju vārdu rediģēšana"
description: "Kā aizstāt vispārīgos runātāju ID ar īstiem vārdiem transkriptā."
topic_id: operations_editing_speaker_names
---

# Runātāju vārdu rediģēšana

Transkripcijas dzinējs katram runātājam automātiski piešķir vispārīgu ID (piemēram, `speaker_0`, `speaker_1`). Šos ID var aizstāt ar īstiem vārdiem, kas parādīsies visā transkriptā un visos eksportētajos failos.

## Kā rediģēt runātāju vārdus

1. Atveriet pabeigtu uzdevumu. Skatiet [Pabeigto uzdevumu ielāde](loading_completed_jobs.md).
2. Skatījumā **Results** noklikšķiniet uz `Edit Speaker Names`.
3. Tiek atvērts dialoglodziņš **Edit Speaker Names** ar divām kolonnām:
   - **Speaker ID** — modelim piešķirtā sākotnējā etiķete (tikai lasīšanai).
   - **Display Name** — transkriptā rādāmais vārds (rediģējams).
4. Noklikšķiniet uz šūnas kolonnā **Display Name** un ierakstiet runātāja vārdu.
5. Nospiediet `Tab` vai noklikšķiniet uz citas rindas, lai pārietu uz nākamo runātāju.
6. Noklikšķiniet uz `Save`, lai lietotu izmaiņas, vai uz `Cancel`, lai tās atceltu.

## Kur vārdi parādās

Atjauninātie attēlojamie vārdi aizstāj vispārīgos ID šādās vietās:

- Segmentu tabulā skatījumā Results.
- Visos eksportētajos failos (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Vārdu atkārtota rediģēšana

Dialoglodziņu Edit Speaker Names var atkārtoti atvērt jebkurā laikā, kamēr uzdevums ir ielādēts skatījumā Results. Izmaiņas tiek saglabātas lokālajā datubāzē un saglabājas starp sesijām.

---