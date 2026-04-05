---
title: "Övervaka jobb"
description: "Hur du följer förloppet för ett pågående eller köat jobb."
topic_id: operations_monitoring_jobs
---

# Övervaka jobb

Vyn **Förlopp** ger dig en realtidsvy över ett pågående transkribbingsjobb.

## Öppna förloppsvyn

- När du startar en ny transkribering går programmet automatiskt till förloppsvyn.
- För ett jobb som redan körs eller är köat, hitta det i tabellen **Transkriberingshistorik** och klicka på `Monitor` i kolumnen **Åtgärder**.

## Läsa förloppsvyn

| Element | Beskrivning |
|---|---|
| Förloppsindikator | Övergripande färdigställandegrad i procent. Obestämd (animerad) medan jobbet startar eller återupptas. |
| Procentetikett | Numerisk procentsats som visas till höger om indikatorn. |
| Statusmeddelande | Aktuell aktivitet — till exempel `Audio Analysis` eller `Speech Recognition`. Visar `Waiting in queue…` om jobbet inte har startat ännu. |
| Segmenttabell | Realtidsflöde av transkriberade segment med kolumnerna **Talare**, **Start**, **Slut** och **Innehåll**. Rullar automatiskt när nya segment anländer. |

## Förloppsfaser

Vilka faser som visas beror på vilket **segmenteringsläge** som valts i inställningarna.

**Läget Talaridentifiering** (standard):

1. **Audio Analysis** — SortFormer-diarisering körs över hela filen för att identifiera talargränser. Indikatorn kan stanna nära 0 % tills den här fasen är klar.
2. **Speech Recognition** — varje talarsegment transkriberas. Procentsatsen stiger stadigt under den här fasen.

**Läget Röstsaktivitetsdetektering**:

1. **Detecting speech segments** — Silero VAD genomsöker filen för att hitta talavsnitt. Den här fasen är snabb.
2. **Speech Recognition** — varje identifierat talavsnitt transkriberas.

I båda lägena fylls segmenttabellen i realtid i takt med att transkriberingen fortskrider.

## Navigera bort

Klicka på `← Back to Home` för att återgå till startskärmen utan att avbryta jobbet. Jobbet fortsätter att köras i bakgrunden och dess status uppdateras i tabellen **Transkriberingshistorik**. Klicka på `Monitor` igen när som helst för att återvända till förloppsvyn.

---