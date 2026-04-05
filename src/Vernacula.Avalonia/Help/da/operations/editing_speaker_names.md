---
title: "Redigering af taler-navne"
description: "Sådan erstatter du generiske taler-id'er med rigtige navne i en transskription."
topic_id: operations_editing_speaker_names
---

# Redigering af taler-navne

Transskriptionsmotoren tildeler automatisk hver taler et generisk id (f.eks. `speaker_0`, `speaker_1`). Du kan erstatte disse med rigtige navne, som vil fremgå i hele transskriptionen og i alle eksporterede filer.

## Sådan redigerer du taler-navne

1. Åbn et fuldført job. Se [Indlæsning af fuldførte job](loading_completed_jobs.md).
2. Klik på `Edit Speaker Names` i **Resultater**-visningen.
3. Dialogboksen **Edit Speaker Names** åbnes med to kolonner:
   - **Speaker ID** — den oprindelige betegnelse tildelt af modellen (skrivebeskyttet).
   - **Display Name** — det navn, der vises i transskriptionen (redigerbart).
4. Klik på en celle i kolonnen **Display Name**, og skriv taler-navnet.
5. Tryk på `Tab`, eller klik på en anden række for at gå videre til næste taler.
6. Klik på `Save` for at anvende ændringerne, eller `Cancel` for at forkaste dem.

## Hvor navnene vises

Opdaterede visningsnavne erstatter de generiske id'er i:

- Segmenttabellen i Resultater-visningen.
- Alle eksporterede filer (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Redigering af navne igen

Du kan til enhver tid genåbne dialogboksen Edit Speaker Names, mens jobbet er indlæst i Resultater-visningen. Ændringer gemmes i den lokale database og bevares på tværs af sessioner.

---