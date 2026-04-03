---
title: "Tilføjelse af flere lydfiler på én gang"
description: "Sådan tilføjer du flere lydfiler til jobkøen på én gang."
topic_id: operations_bulk_add_jobs
---

# Tilføjelse af flere lydfiler på én gang

Brug **Tilføj flere job** til at sætte flere lyd- eller videofiler i kø til transskription i ét trin. Programmet behandler dem én ad gangen i den rækkefølge, de blev tilføjet.

## Forudsætninger

- Alle modelfiler skal være downloadet. Kortet **Modelstatus** skal vise `All N model file(s) present ✓`. Se [Downloading af modeller](../first_steps/downloading_models.md).

## Sådan tilføjer du flere job på én gang

1. På startskærmen klikker du på `Bulk Add Jobs`.
2. En filvælger åbnes. Vælg en eller flere lyd- eller videofiler — hold `Ctrl` eller `Shift` nede for at vælge flere filer.
3. Klik på **Åbn**. Hver valgt fil tilføjes til tabellen **Transskriptionshistorik** som et separat job.

> **Videofiler med flere lydspor:** Hvis en videofil indeholder mere end ét lydspor (f.eks. flere sprog eller en instruktørkommentar), opretter programmet automatisk ét job pr. lydspor.

## Jobnavne

Hvert job navngives automatisk ud fra dets lydfils navn. Du kan omdøbe et job når som helst ved at klikke på dets navn i kolonnen **Titel** i tabellen Transskriptionshistorik, redigere teksten og trykke på `Enter` eller klikke et andet sted.

## Køadfærd

- Hvis der ikke kører et job i øjeblikket, starter den første fil med det samme, og resten vises som `queued`.
- Hvis der allerede kører et job, vises alle nyligt tilføjede filer som `queued` og starter automatisk i rækkefølge.
- Hvis du vil overvåge det aktive job, skal du klikke på `Monitor` i kolonnen **Handlinger** for det pågældende job. Se [Overvågning af job](monitoring_jobs.md).
- Hvis du vil sætte et job i kø på pause eller fjerne det, inden det starter, skal du bruge knapperne `Pause` eller `Remove` i kolonnen **Handlinger**. Se [Pause, genoptagelse eller fjernelse af job](pausing_resuming_removing.md).

---