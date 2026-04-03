---
title: "Redigera talarnamn"
description: "Hur du ersätter generiska talar-ID:n med riktiga namn i ett transkript."
topic_id: operations_editing_speaker_names
---

# Redigera talarnamn

Transkriptionsmotorn märker automatiskt varje talare med ett generiskt ID (till exempel `speaker_0`, `speaker_1`). Du kan ersätta dessa med riktiga namn som visas genomgående i transkriptet och i alla exporterade filer.

## Så här redigerar du talarnamn

1. Öppna ett avslutat jobb. Se [Läsa in avslutade jobb](loading_completed_jobs.md).
2. I vyn **Resultat** klickar du på `Edit Speaker Names`.
3. Dialogrutan **Edit Speaker Names** öppnas med två kolumner:
   - **Speaker ID** — den ursprungliga etikett som modellen tilldelat (skrivskyddad).
   - **Display Name** — det namn som visas i transkriptet (redigerbart).
4. Klicka på en cell i kolumnen **Display Name** och skriv talarens namn.
5. Tryck på `Tab` eller klicka på en annan rad för att gå vidare till nästa talare.
6. Klicka på `Save` för att tillämpa ändringarna, eller `Cancel` för att ångra dem.

## Var namnen visas

Uppdaterade visningsnamn ersätter de generiska ID:na i:

- Segmenttabellen i vyn Resultat.
- Alla exporterade filer (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Redigera namn igen

Du kan när som helst öppna dialogrutan Edit Speaker Names på nytt medan jobbet är inläst i vyn Resultat. Ändringar sparas i den lokala databasen och bevaras mellan sessioner.

---