---
title: "Uzdevumu apturēšana, atsākšana vai noņemšana"
description: "Kā apturēt aktīvu uzdevumu, atsākt apstādinātu vai dzēst uzdevumu no vēstures."
topic_id: operations_pausing_resuming_removing
---

# Uzdevumu apturēšana, atsākšana vai noņemšana

## Uzdevuma apturēšana

Darbojošos vai rindā esošu uzdevumu var apturēt no divām vietām:

- **Progresa skats** — noklikšķiniet `Pause` apakšējā labajā stūrī, kamēr vērojat aktīvo uzdevumu.
- **Transkripcijas vēstures tabula** — noklikšķiniet `Pause` kolonnas **Darbības** rindā, kuras statuss ir `running` vai `queued`.

Pēc noklikšķināšanas uz `Pause` statusa rindā rādās `Pausing…`, kamēr lietojumprogramma pabeidz pašreizējo apstrādes vienību. Pēc tam uzdevuma statuss vēstures tabulā mainās uz `cancelled`.

> Apturēšana saglabā visus līdz šim transkribētos segmentus. Vēlāk uzdevumu var atsākt, nezaudējot paveikto darbu.

## Uzdevuma atsākšana

Lai atsāktu apturētu vai neizdevušos uzdevumu:

1. Sākuma ekrānā atrodiet uzdevumu **Transkripcijas vēstures** tabulā. Tā statuss būs `cancelled` vai `failed`.
2. Noklikšķiniet `Resume` kolonnā **Darbības**.
3. Lietojumprogramma atgriežas **Progresa** skatā un turpina no vietas, kur apstrāde tika pārtraukta.

Statusa rindā uz brīdi rādās `Resuming…`, kamēr uzdevums tiek atkārtoti inicializēts.

## Uzdevuma noņemšana

Lai neatgriezeniski dzēstu uzdevumu un tā transkriptu no vēstures:

1. **Transkripcijas vēstures** tabulā noklikšķiniet `Remove` kolonnā **Darbības** pie uzdevuma, kuru vēlaties dzēst.

Uzdevums tiek noņemts no saraksta un tā dati tiek izdzēsti no lokālās datubāzes. Šo darbību nevar atsaukt. Eksportētie faili, kas saglabāti diskā, netiek ietekmēti.

---