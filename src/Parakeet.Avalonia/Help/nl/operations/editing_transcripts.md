---
title: "Transcripties bewerken"
description: "Hoe u getranscribeerde segmenten in de transcriptieeditor kunt bekijken, corrigeren en verifiëren."
topic_id: operations_editing_transcripts
---

# Transcripties bewerken

Met de **Transcriptieeditor** kunt u ASR-uitvoer bekijken, tekst corrigeren, sprekers inline hernoemen, segmenttiming aanpassen en segmenten als geverifieerd markeren — alles terwijl u naar de originele audio luistert.

## De editor openen

1. Laad een voltooid project (zie [Voltooide projecten laden](loading_completed_jobs.md)).
2. Klik in de weergave **Resultaten** op `Edit Transcript`.

De editor wordt geopend als een apart venster en kan naast de hoofdapplicatie open blijven.

## Indeling

Elk segment wordt weergegeven als een kaart met twee panelen naast elkaar:

- **Linker paneel** — de originele ASR-uitvoer met een betrouwbaarheidskleur per woord. Woorden waarover het model minder zeker was, worden rood weergegeven; woorden met een hoge betrouwbaarheid verschijnen in de normale tekstkleur.
- **Rechter paneel** — een bewerkbaar tekstvak. Breng hier correcties aan; de verschillen ten opzichte van het origineel worden gemarkeerd terwijl u typt.

Het sprekerlabel en het tijdsbereik worden boven elke kaart weergegeven. Klik op een kaart om deze te focussen en de actiepictogrammen zichtbaar te maken. Beweeg de muis over een pictogram om een tooltip te zien die de functie ervan beschrijft.

## Pictogramlegenda

### Afspeelbalk

| Pictogram | Actie |
|-----------|-------|
| ▶ | Afspelen |
| ⏸ | Pauzeren |
| ⏮ | Naar vorig segment springen |
| ⏭ | Naar volgend segment springen |

### Acties op segmentkaarten

| Pictogram | Actie |
|-----------|-------|
| <mdl2 ch="E77B"/> | Segment aan een andere spreker toewijzen |
| <mdl2 ch="E916"/> | Begin- en eindtijd van het segment aanpassen |
| <mdl2 ch="EA39"/> | Segment onderdrukken of onderdrukking opheffen |
| <mdl2 ch="E72B"/> | Samenvoegen met het vorige segment |
| <mdl2 ch="E72A"/> | Samenvoegen met het volgende segment |
| <mdl2 ch="E8C6"/> | Segment splitsen |
| <mdl2 ch="E72C"/> | ASR opnieuw uitvoeren op dit segment |

## Audio afspelen

Een afspeelbalk loopt langs de bovenkant van het editorvenster:

| Besturing | Actie |
|-----------|-------|
| Afspelen/pauzeren-pictogram | Afspelen starten of pauzeren |
| Zoekbalk | Slepen om naar een willekeurige positie in de audio te springen |
| Snelheidsschuifregelaar | Afspeelsnelheid aanpassen (0,5× – 2×) |
| Pictogrammen voor vorig/volgend | Naar het vorige of volgende segment springen |
| Vervolgkeuzelijst voor afspeelmodus | Een van de drie afspeelmodi selecteren (zie hieronder) |
| Volumeschuifregelaar | Afspeelvolume aanpassen |

Tijdens het afspelen wordt het woord dat momenteel wordt uitgesproken gemarkeerd in het linker paneel. Wanneer het afspelen gepauzeerd is na het zoeken, wordt de markering bijgewerkt naar het woord op de zoekpositie.

### Afspeelmodi

| Modus | Gedrag |
|-------|--------|
| `Single` | Het huidige segment eenmalig afspelen en vervolgens stoppen. |
| `Auto-advance` | Het huidige segment afspelen; wanneer het eindigt, wordt het als geverifieerd gemarkeerd en wordt het volgende segment geselecteerd. |
| `Continuous` | Alle segmenten achtereenvolgens afspelen zonder ze als geverifieerd te markeren. |

Selecteer de actieve modus via de vervolgkeuzelijst in de afspeelbalk.

## Een segment bewerken

1. Klik op een kaart om deze te focussen.
2. Bewerk de tekst in het rechter paneel. Wijzigingen worden automatisch opgeslagen wanneer u de focus naar een andere kaart verplaatst.

## Een spreker hernoemen

Klik op het sprekerlabel in de gefocuste kaart en typ een nieuwe naam. Druk op `Enter` of klik ergens anders om op te slaan. De nieuwe naam wordt alleen op die kaart toegepast; om een spreker globaal te hernoemen, gebruikt u [Sprekernamen bewerken](editing_speaker_names.md) vanuit de weergave Resultaten.

## Een segment verifiëren

Klik op het selectievakje `Verified` op een gefocuste kaart om het als gecontroleerd te markeren. De verificatiestatus wordt opgeslagen in de database en is bij toekomstige laadacties zichtbaar in de editor.

## Een segment onderdrukken

Klik op `Suppress` op een gefocuste kaart om het segment te verbergen uit exports (handig voor ruis, muziek of andere niet-spraakgedeelten). Klik op `Unsuppress` om het te herstellen.

## Segmenttijden aanpassen

Klik op `Adjust Times` op een gefocuste kaart om het dialoogvenster voor tijdsaanpassing te openen. Gebruik het scrollwiel boven het veld **Start** of **End** om de waarde in stappen van 0,1 seconde aan te passen, of typ een waarde direct in. Klik op `Save` om toe te passen.

## Segmenten samenvoegen

- Klik op `⟵ Merge` om het gefocuste segment samen te voegen met het segment dat er direct aan voorafgaat.
- Klik op `Merge ⟶` om het gefocuste segment samen te voegen met het segment dat er direct op volgt.

De gecombineerde tekst en het tijdsbereik van beide kaarten worden samengevoegd. Dit is handig wanneer één gesproken uiting over twee segmenten was verdeeld.

## Een segment splitsen

Klik op `Split…` op een gefocuste kaart om het splitsingsdialoogvenster te openen. Plaats het splitspunt in de tekst en bevestig. Er worden twee nieuwe segmenten aangemaakt die het originele tijdsbereik beslaan. Dit is handig wanneer twee afzonderlijke uitingen in één segment waren samengevoegd.

## ASR opnieuw uitvoeren

Klik op `Redo ASR` op een gefocuste kaart om spraakherkenning opnieuw uit te voeren op de audio van dat segment. Het model verwerkt alleen het audiofragment van dat segment en produceert een nieuwe transcriptie vanuit één bron.

Gebruik dit wanneer:

- Een segment afkomstig is van een samenvoeging en niet kan worden gesplitst (samengevoegde segmenten omvatten meerdere ASR-bronnen; Redo ASR voegt ze samen tot één, waarna `Split…` beschikbaar wordt).
- De originele transcriptie slecht is en u een schone tweede doorgang wilt zonder handmatig te bewerken.

**Opmerking:** Tekst die u al in het rechter paneel heeft getypt, wordt verwijderd en vervangen door de nieuwe ASR-uitvoer. Voor de bewerking moet het audiobestand geladen zijn; de knop is uitgeschakeld als audio niet beschikbaar is.