---
title: "Úprava prepisov"
description: "Ako kontrolovať, opravovať a overovať prepísané segmenty v editore prepisov."
topic_id: operations_editing_transcripts
---

# Úprava prepisov

**Editor prepisov** umožňuje kontrolovať výstup ASR, opravovať text, premenovávať rečníkov priamo v editore, upravovať časovanie segmentov a označovať segmenty ako overené — to všetko počas prehrávania pôvodného zvuku.

## Otvorenie editora

1. Načítajte dokončenú úlohu (pozri [Načítanie dokončených úloh](loading_completed_jobs.md)).
2. V zobrazení **Výsledky** kliknite na `Edit Transcript`.

Editor sa otvorí ako samostatné okno a môže zostať otvorené vedľa hlavnej aplikácie.

## Rozloženie

Každý segment je zobrazený ako karta s dvoma panelmi umiestnenými vedľa seba:

- **Ľavý panel** — pôvodný výstup ASR s farebným zvýraznením spoľahlivosti jednotlivých slov. Slová, v ktorých si model nebol istý, sú zobrazené červenou farbou; slová s vysokou spoľahlivosťou sú zobrazené v bežnej farbe textu.
- **Pravý panel** — upraviteľné textové pole. Opravy vykonávajte tu; rozdiel oproti originálu sa počas písania farebne zvýrazní.

Nad každou kartou sa zobrazuje označenie rečníka a časový rozsah. Kliknutím na kartu ju aktivujete a zobrazia sa ikony akcií. Podržaním kurzora nad ľubovoľnou ikonou zobrazíte popis jej funkcie.

## Prehľad ikon

### Panel prehrávania

| Ikona | Akcia |
|-------|-------|
| ▶ | Prehrať |
| ⏸ | Pozastaviť |
| ⏮ | Preskočiť na predchádzajúci segment |
| ⏭ | Preskočiť na nasledujúci segment |

### Akcie karty segmentu

| Ikona | Akcia |
|-------|-------|
| <mdl2 ch="E77B"/> | Priradiť segment inému rečníkovi |
| <mdl2 ch="E916"/> | Upraviť čas začiatku a konca segmentu |
| <mdl2 ch="EA39"/> | Potlačiť alebo obnoviť segment |
| <mdl2 ch="E72B"/> | Zlúčiť s predchádzajúcim segmentom |
| <mdl2 ch="E72A"/> | Zlúčiť s nasledujúcim segmentom |
| <mdl2 ch="E8C6"/> | Rozdeliť segment |
| <mdl2 ch="E72C"/> | Znovu spustiť ASR na tomto segmente |

## Prehrávanie zvuku

Panel prehrávania sa nachádza pozdĺž hornej časti okna editora:

| Ovládací prvok | Akcia |
|----------------|-------|
| Ikona Prehrať / Pozastaviť | Spustiť alebo pozastaviť prehrávanie |
| Posuvník polohy | Potiahnutím preskočíte na ľubovoľné miesto v zvuku |
| Posuvník rýchlosti | Upraviť rýchlosť prehrávania (0,5× – 2×) |
| Ikony Predch. / Ďalší | Preskočiť na predchádzajúci alebo nasledujúci segment |
| Rozbaľovací zoznam režimu prehrávania | Vybrať jeden z troch režimov prehrávania (pozri nižšie) |
| Posuvník hlasitosti | Upraviť hlasitosť prehrávania |

Počas prehrávania je aktuálne vyslovené slovo zvýraznené v ľavom paneli. Po pozastavení po posunutí polohy sa zvýraznenie aktualizuje na slovo zodpovedajúce novej pozícii.

### Režimy prehrávania

| Režim | Správanie |
|-------|-----------|
| `Single` | Prehrá aktuálny segment raz a zastaví sa. |
| `Auto-advance` | Prehrá aktuálny segment; po jeho skončení ho označí ako overený a prejde na nasledujúci. |
| `Continuous` | Prehráva všetky segmenty za sebou bez toho, aby ktorýkoľvek z nich označil ako overený. |

Aktívny režim vyberte z rozbaľovacieho zoznamu v paneli prehrávania.

## Úprava segmentu

1. Kliknutím na kartu ju aktivujete.
2. Upravte text v pravom paneli. Zmeny sa ukladajú automaticky, keď presuniete fokus na inú kartu.

## Premenovanie rečníka

Kliknite na označenie rečníka v aktívnej karte a zadajte nové meno. Stlačte `Enter` alebo kliknite inam, čím uložíte zmenu. Nové meno sa použije iba pre túto kartu; ak chcete rečníka premenovať globálne, použite funkciu [Upraviť mená rečníkov](editing_speaker_names.md) v zobrazení Výsledky.

## Overenie segmentu

Kliknutím na začiarkavacie políčko `Verified` na aktívnej karte ju označíte ako skontrolovanú. Stav overenia sa uloží do databázy a bude viditeľný v editore pri budúcich načítaniach.

## Potlačenie segmentu

Kliknutím na `Suppress` na aktívnej karte skryjete segment z exportov (užitočné pri šume, hudbe alebo iných častiach bez reči). Kliknutím na `Unsuppress` segment obnovíte.

## Úprava časov segmentu

Kliknutím na `Adjust Times` na aktívnej karte otvoríte dialóg úpravy časov. Rolovacím kolieskom nad poľom **Start** alebo **End** upravujte hodnotu po krokoch 0,1 sekundy, alebo zadajte hodnotu priamo. Kliknutím na `Save` zmeny uložíte.

## Zlúčenie segmentov

- Kliknutím na `⟵ Merge` zlúčite aktívny segment s bezprostredne predchádzajúcim segmentom.
- Kliknutím na `Merge ⟶` zlúčite aktívny segment s bezprostredne nasledujúcim segmentom.

Text a časový rozsah oboch kariet sa spoja. Je to užitočné vtedy, keď bola jedna hovorená výpoveď rozdelená na dva segmenty.

## Rozdelenie segmentu

Kliknutím na `Split…` na aktívnej karte otvoríte dialóg rozdelenia. Umiestnite bod rozdelenia v texte a potvrďte. Vytvoria sa dva nové segmenty pokrývajúce pôvodný časový rozsah. Je to užitočné vtedy, keď boli dve odlišné výpovede zlúčené do jedného segmentu.

## Opakované spustenie ASR

Kliknutím na `Redo ASR` na aktívnej karte znovu spustíte rozpoznávanie reči pre zvuk daného segmentu. Model spracuje iba zvukovú časť tohto segmentu a vytvorí nový prepis z jediného zdroja.

Túto funkciu použite v týchto prípadoch:

- Segment vznikol zlúčením a nedá sa rozdeliť (zlúčené segmenty zahŕňajú viacero zdrojov ASR; funkcia Redo ASR ich skonsoliduje do jedného, po čom bude dostupná možnosť `Split…`).
- Pôvodný prepis je nekvalitný a chcete čistý druhý pokus bez manuálnych úprav.

**Poznámka:** Akýkoľvek text, ktorý ste už zadali v pravom paneli, bude zmazaný a nahradený novým výstupom ASR. Operácia vyžaduje načítaný zvukový súbor; tlačidlo je deaktivované, ak zvuk nie je dostupný.