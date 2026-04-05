---
title: "Darbu uzraudzība"
description: "Kā sekot līdzi darbojošā vai rindā esošā darba norisei."
topic_id: operations_monitoring_jobs
---

# Darbu uzraudzība

Skats **Progrese** sniedz reāllaika pārskatu par pašlaik izpildāmo transkribēšanas darbu.

## Progrese skata atvēršana

- Uzsākot jaunu transkribēšanu, lietojumprogramma automātiski pārslēdzas uz skatu Progrese.
- Ja darbs jau tiek izpildīts vai atrodas rindā, atrodiet to tabulā **Transkribēšanas vēsture** un kolonnā **Darbības** noklikšķiniet uz `Monitor`.

## Progrese skata lasīšana

| Elements | Apraksts |
|---|---|
| Progresa josla | Kopējais izpildes procentuālais rādītājs. Nenoteikta (animēta), kamēr darbs tiek startēts vai atsākts. |
| Procentuālā atzīme | Skaitliskais procentuālais rādītājs, kas redzams pa labi no joslas. |
| Statusa ziņojums | Pašreizējā darbība — piemēram, `Audio Analysis` vai `Speech Recognition`. Rāda `Waiting in queue…`, ja darbs vēl nav sācies. |
| Segmentu tabula | Reāllaika atpazīto segmentu plūsma ar kolonnām **Speaker**, **Start**, **End** un **Content**. Ritina automātiski, pievienojoties jauniem segmentiem. |

## Progresa posmi

Rādāmie posmi ir atkarīgi no iestatījumos izvēlētā **segmentēšanas režīma**.

**Runātāju diarizācijas režīms** (noklusējuma):

1. **Audio Analysis** — SortFormer diarizācija apstrādā visu failu, lai noteiktu runātāju robežas. Josla var palikt tuvu 0%, līdz šis posms tiek pabeigts.
2. **Speech Recognition** — katrs runātāja segments tiek transkribēts. Procentuālais rādītājs šajā posmā pakāpeniski pieaug.

**Balss aktivitātes noteikšanas režīms**:

1. **Detecting speech segments** — Silero VAD skenē failu, lai atrastu runas apgabalus. Šis posms ir ātrs.
2. **Speech Recognition** — katrs atklātais runas apgabals tiek transkribēts.

Abos režīmos reāllaika segmentu tabula tiek aizpildīta, transkribēšanai turpinoties.

## Pāreja uz citiem skatiem

Noklikšķiniet uz `← Back to Home`, lai atgrieztos sākuma ekrānā, nepārtraucot darbu. Darbs turpinās fonā, un tā statuss tiek atjaunināts tabulā **Transkribēšanas vēsture**. Jebkurā laikā noklikšķiniet uz `Monitor` vēlreiz, lai atgrieztos skatā Progrese.

---