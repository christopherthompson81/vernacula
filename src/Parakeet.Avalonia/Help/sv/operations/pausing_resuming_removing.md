---
title: "Pausa, återuppta eller ta bort jobb"
description: "Hur du pausar ett pågående jobb, återupptar ett stoppat jobb eller tar bort ett jobb från historiken."
topic_id: operations_pausing_resuming_removing
---

# Pausa, återuppta eller ta bort jobb

## Pausa ett jobb

Du kan pausa ett pågående eller köat jobb från två platser:

- **Förloppsvyn** — klicka på `Pause` i det nedre högra hörnet medan du ser på det aktiva jobbet.
- **Transkriptionshistoriktabellen** — klicka på `Pause` i kolumnen **Actions** på valfri rad vars status är `running` eller `queued`.

När du har klickat på `Pause` visar statusraden `Pausing…` medan programmet slutför den aktuella bearbetningsenheten. Jobbstatusen ändras sedan till `cancelled` i historiktabellen.

> Pausning sparar alla segment som har transkriberats hittills. Du kan återuppta jobbet senare utan att förlora det arbetet.

## Återuppta ett jobb

Så här återupptar du ett pausat eller misslyckat jobb:

1. På startskärmen letar du upp jobbet i tabellen **Transcription History**. Dess status är `cancelled` eller `failed`.
2. Klicka på `Resume` i kolumnen **Actions**.
3. Programmet återgår till vyn **Progress** och fortsätter från där bearbetningen avbröts.

Statusraden visar kort `Resuming…` medan jobbet initieras på nytt.

## Ta bort ett jobb

Så här tar du bort ett jobb och dess transkript permanent från historiken:

1. I tabellen **Transcription History** klickar du på `Remove` i kolumnen **Actions** för det jobb du vill ta bort.

Jobbet tas bort från listan och dess data raderas från den lokala databasen. Den här åtgärden kan inte ångras. Exporterade filer som har sparats på disk påverkas inte.

---