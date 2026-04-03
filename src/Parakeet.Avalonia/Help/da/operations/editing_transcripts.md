---
title: "Redigering af transskriptioner"
description: "Sådan gennemgår, retter og verificerer du transskriberede segmenter i transskriptionseditoren."
topic_id: operations_editing_transcripts
---

# Redigering af transskriptioner

**Transskriptionseditoren** giver dig mulighed for at gennemgå ASR-output, rette tekst, omdøbe talere direkte i editoren, justere segmenttiming og markere segmenter som verificerede — alt imens du lytter til den originale lyd.

## Åbning af editoren

1. Indlæs et afsluttet job (se [Indlæsning af afsluttede job](loading_completed_jobs.md)).
2. Klik på `Edit Transcript` i visningen **Results**.

Editoren åbnes som et separat vindue og kan forblive åben ved siden af hovedprogrammet.

## Layout

Hvert segment vises som et kort med to paneler side om side:

- **Venstre panel** — det originale ASR-output med farvekodning af tillid pr. ord. Ord, som modellen var mindre sikker på, vises med rødt; ord med høj tillid vises i den normale tekstfarve.
- **Højre panel** — et redigerbart tekstfelt. Foretag rettelser her; forskellen i forhold til originalen fremhæves, mens du skriver.

Talerlabel og tidsinterval vises over hvert kort. Klik på et kort for at fokusere det og få vist dets handlingsikoner. Hold musen over et ikon for at se et værktøjstip, der beskriver dets funktion.

## Ikonforklaring

### Afspilningslinje

| Ikon | Handling |
|------|----------|
| ▶ | Afspil |
| ⏸ | Pause |
| ⏮ | Gå til forrige segment |
| ⏭ | Gå til næste segment |

### Handlinger på segmentkort

| Ikon | Handling |
|------|----------|
| <mdl2 ch="E77B"/> | Tildel segmentet til en anden taler |
| <mdl2 ch="E916"/> | Juster segmentets start- og sluttidspunkt |
| <mdl2 ch="EA39"/> | Undertryk eller fjern undertrykkelse af segmentet |
| <mdl2 ch="E72B"/> | Flet med det forrige segment |
| <mdl2 ch="E72A"/> | Flet med det næste segment |
| <mdl2 ch="E8C6"/> | Opdel segmentet |
| <mdl2 ch="E72C"/> | Kør ASR igen på dette segment |

## Lydafspilning

En afspilningslinje løber hen over toppen af editorvinduet:

| Kontrol | Handling |
|---------|----------|
| Afspil/pause-ikon | Start eller sæt afspilning på pause |
| Søgelinje | Træk for at hoppe til en vilkårlig position i lyden |
| Hastighedsskyder | Juster afspilningshastighed (0,5× – 2×) |
| Forrige/næste-ikoner | Gå til det forrige eller næste segment |
| Rullemenu for afspilningstilstand | Vælg en af tre afspilningstilstande (se nedenfor) |
| Lydstyrkesskyder | Juster afspilningslydstyrken |

Under afspilning fremhæves det ord, der aktuelt udtales, i venstre panel. Når afspilningen er sat på pause efter en søgning, opdateres fremhævningen til det ord, der befinder sig ved søgeposition.

### Afspilningstilstande

| Tilstand | Adfærd |
|----------|--------|
| `Single` | Afspil det aktuelle segment én gang, og stop derefter. |
| `Auto-advance` | Afspil det aktuelle segment; når det slutter, markeres det som verificeret, og der springes til det næste. |
| `Continuous` | Afspil alle segmenter i rækkefølge uden at markere nogen som verificerede. |

Vælg den aktive tilstand i rullemenuen i afspilningslinjen.

## Redigering af et segment

1. Klik på et kort for at fokusere det.
2. Rediger teksten i højre panel. Ændringer gemmes automatisk, når du flytter fokus til et andet kort.

## Omdøbning af en taler

Klik på talerlabelen inde i det fokuserede kort, og skriv et nyt navn. Tryk på `Enter`, eller klik et andet sted for at gemme. Det nye navn anvendes kun på dette kort; for at omdøbe en taler globalt skal du bruge [Rediger talernavne](editing_speaker_names.md) fra visningen Results.

## Verificering af et segment

Klik på afkrydsningsfeltet `Verified` på et fokuseret kort for at markere det som gennemgået. Verificeringsstatus gemmes i databasen og er synlig i editoren ved fremtidige indlæsninger.

## Undertrykkelse af et segment

Klik på `Suppress` på et fokuseret kort for at skjule segmentet fra eksporter (nyttigt ved støj, musik eller andre ikke-tale-afsnit). Klik på `Unsuppress` for at gendanne det.

## Justering af segmenttidspunkter

Klik på `Adjust Times` på et fokuseret kort for at åbne dialogen til tidsjustering. Brug musehjulet over feltet **Start** eller **End** for at ændre værdien i trin på 0,1 sekund, eller skriv en værdi direkte. Klik på `Save` for at anvende.

## Fletning af segmenter

- Klik på `⟵ Merge` for at flette det fokuserede segment med segmentet umiddelbart før det.
- Klik på `Merge ⟶` for at flette det fokuserede segment med segmentet umiddelbart efter det.

Den kombinerede tekst og det kombinerede tidsinterval fra begge kort sammenføjes. Dette er nyttigt, når en enkelt talt ytring blev opdelt på tværs af to segmenter.

## Opdeling af et segment

Klik på `Split…` på et fokuseret kort for at åbne opdelingsdialogen. Placér opdelingspunktet i teksten, og bekræft. Der oprettes to nye segmenter, der dækker det originale tidsinterval. Dette er nyttigt, når to separate ytringer blev flettet sammen til ét segment.

## Kør ASR igen

Klik på `Redo ASR` på et fokuseret kort for at køre talegenkendelse igen på dette segments lyd. Modellen behandler kun det lydudsnit, der tilhører segmentet, og producerer en ny transskription fra én enkelt kilde.

Brug dette når:

- Et segment stammer fra en fletning og ikke kan opdeles (flettede segmenter dækker flere ASR-kilder; Redo ASR sammensmelter dem til én, hvorefter `Split…` bliver tilgængelig).
- Den originale transskription er dårlig, og du ønsker et nyt gennemløb uden at redigere manuelt.

**Bemærk:** Al tekst, du allerede har skrevet i højre panel, kasseres og erstattes med det nye ASR-output. Handlingen kræver, at lydfilen er indlæst; knappen er deaktiveret, hvis lyden ikke er tilgængelig.