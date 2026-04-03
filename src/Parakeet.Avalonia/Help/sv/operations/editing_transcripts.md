---
title: "Redigera transkript"
description: "Hur du granskar, korrigerar och verifierar transkriberade segment i transkriptredigeraren."
topic_id: operations_editing_transcripts
---

# Redigera transkript

**Transkriptredigeraren** låter dig granska ASR-utdata, korrigera text, byta namn på talare direkt i redigeraren, justera segmenttider och markera segment som verifierade — allt medan du lyssnar på det ursprungliga ljudet.

## Öppna redigeraren

1. Läs in ett slutfört jobb (se [Läsa in slutförda jobb](loading_completed_jobs.md)).
2. I vyn **Resultat**, klicka på `Edit Transcript`.

Redigeraren öppnas i ett separat fönster och kan vara öppet parallellt med huvudprogrammet.

## Layout

Varje segment visas som ett kort med två paneler sida vid sida:

- **Vänster panel** — det ursprungliga ASR-utdatat med färgkodning per ord baserat på konfidens. Ord som modellen var mindre säker på visas i rött; ord med hög konfidens visas i den normala textfärgen.
- **Höger panel** — en redigerbar textruta. Gör korrigeringar här; skillnaden mot originalet markeras allteftersom du skriver.

Talarens etikett och tidsintervall visas ovanför varje kort. Klicka på ett kort för att fokusera det och visa dess åtgärdsikonerna. Håll muspekaren över en ikon för att se ett verktygstips som beskriver dess funktion.

## Ikonförklaring

### Uppspelningsfält

| Ikon | Åtgärd |
|------|--------|
| ▶ | Spela upp |
| ⏸ | Pausa |
| ⏮ | Hoppa till föregående segment |
| ⏭ | Hoppa till nästa segment |

### Åtgärder för segmentkort

| Ikon | Åtgärd |
|------|--------|
| <mdl2 ch="E77B"/> | Tilldela om segmentet till en annan talare |
| <mdl2 ch="E916"/> | Justera segmentets start- och sluttider |
| <mdl2 ch="EA39"/> | Dölj eller visa segmentet |
| <mdl2 ch="E72B"/> | Slå ihop med föregående segment |
| <mdl2 ch="E72A"/> | Slå ihop med nästa segment |
| <mdl2 ch="E8C6"/> | Dela upp segmentet |
| <mdl2 ch="E72C"/> | Kör om ASR på detta segment |

## Ljuduppspelning

Ett uppspelningsfält löper längs toppen av redigerarfönstret:

| Kontroll | Åtgärd |
|---------|--------|
| Ikonen Spela upp / Pausa | Starta eller pausa uppspelningen |
| Sökfält | Dra för att hoppa till valfri position i ljudet |
| Hastighetsskjutreglage | Justera uppspelningshastigheten (0,5× – 2×) |
| Ikonerna Föregående / Nästa | Hoppa till föregående eller nästa segment |
| Listruta för uppspelningsläge | Välj ett av tre uppspelningslägen (se nedan) |
| Volymskjutreglage | Justera uppspelningsvolymen |

Under uppspelning markeras det ord som just talas i den vänstra panelen. När uppspelningen pausas efter en sökning uppdateras markeringen till det ord som finns vid sökpositionen.

### Uppspelningslägen

| Läge | Beteende |
|------|-----------|
| `Single` | Spela upp det aktuella segmentet en gång och stoppa sedan. |
| `Auto-advance` | Spela upp det aktuella segmentet; när det är klart markeras det som verifierat och nästa segment visas. |
| `Continuous` | Spela upp alla segment i följd utan att markera något som verifierat. |

Välj aktivt läge från listrutan i uppspelningsfältet.

## Redigera ett segment

1. Klicka på ett kort för att fokusera det.
2. Redigera texten i den högra panelen. Ändringar sparas automatiskt när du flyttar fokus till ett annat kort.

## Byta namn på en talare

Klicka på talaretiketten i det fokuserade kortet och skriv ett nytt namn. Tryck på `Enter` eller klicka någon annanstans för att spara. Det nya namnet tillämpas bara på det kortet. Om du vill byta namn på en talare globalt, använd [Redigera talarnamn](editing_speaker_names.md) från vyn Resultat.

## Verifiera ett segment

Klicka på kryssrutan `Verified` på ett fokuserat kort för att markera det som granskat. Verifieringsstatus sparas i databasen och visas i redigeraren vid framtida inläsningar.

## Dölja ett segment

Klicka på `Suppress` på ett fokuserat kort för att dölja segmentet från exporter (användbart för brus, musik eller andra avsnitt utan tal). Klicka på `Unsuppress` för att återställa det.

## Justera segmenttider

Klicka på `Adjust Times` på ett fokuserat kort för att öppna dialogen för tidsjustering. Använd scrollhjulet över fältet **Start** eller **End** för att ändra värdet i steg om 0,1 sekund, eller skriv ett värde direkt. Klicka på `Save` för att tillämpa.

## Slå ihop segment

- Klicka på `⟵ Merge` för att slå ihop det fokuserade segmentet med det segment som kommer omedelbart före det.
- Klicka på `Merge ⟶` för att slå ihop det fokuserade segmentet med det segment som kommer omedelbart efter det.

Den kombinerade texten och tidsintervallet från båda korten fogas samman. Detta är användbart när ett enda talat yttrande delats upp i två segment.

## Dela upp ett segment

Klicka på `Split…` på ett fokuserat kort för att öppna dialogen för uppdelning. Placera delningspunkten i texten och bekräfta. Två nya segment skapas som täcker det ursprungliga tidsintervallet. Detta är användbart när två separata yttranden slagits samman till ett segment.

## Kör om ASR

Klicka på `Redo ASR` på ett fokuserat kort för att köra taligenkänning igen på det segmentets ljud. Modellen bearbetar bara det ljud som hör till det segmentet och producerar en ny, enhetlig transkription.

Använd detta när:

- Ett segment härstammar från en sammanslagning och inte kan delas upp (sammanslagna segment sträcker sig över flera ASR-källor; Redo ASR slår ihop dem till ett, varefter `Split…` blir tillgänglig).
- Den ursprungliga transkriptionen är bristfällig och du vill ha en ny, ren genomgång utan att redigera manuellt.

**Obs!** All text du redan har skrivit i den högra panelen tas bort och ersätts med det nya ASR-utdatat. Åtgärden kräver att ljudfilen är inläst; knappen är inaktiverad om inget ljud är tillgängligt.