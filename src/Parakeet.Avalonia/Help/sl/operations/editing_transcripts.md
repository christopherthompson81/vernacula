---
title: "Urejanje prepisov"
description: "Kako pregledati, popraviti in preveriti prepisane segmente v urejevalniku prepisov."
topic_id: operations_editing_transcripts
---

# Urejanje prepisov

**Urejevalnik prepisov** vam omogoča pregled izpisa ASR, popravljanje besedila, preimenovanje govorcev neposredno v urejevalniku, prilagajanje časovnih oznak segmentov in označevanje segmentov kot preverjenih — vse to med poslušanjem izvirnega zvoka.

## Odpiranje urejevalnika

1. Naložite dokončano opravilo (glejte [Nalaganje dokončanih opravil](loading_completed_jobs.md)).
2. V pogledu **Rezultati** kliknite `Edit Transcript`.

Urejevalnik se odpre v ločenem oknu in lahko ostane odprt vzporedno z glavno aplikacijo.

## Postavitev

Vsak segment je prikazan kot kartica z dvema vzporednima podoknom:

- **Levo podokno** — izvirni izpis ASR z barvnim označevanjem zaupanja za posamezne besede. Besede, glede katerih model ni bil dovolj prepričan, so prikazane z rdečo barvo; besede z visoko stopnjo zaupanja so prikazane v navadni barvi besedila.
- **Desno podokno** — polje za urejanje besedila. Popravke vnašajte tukaj; razlike glede na izvirnik so med tipkanjem sproti označene.

Oznaka govorca in časovni razpon sta prikazana nad vsako kartico. Kliknite kartico, da jo izberete in prikažete njene ikone dejanj. Premaknite kazalec miške nad katero koli ikono, da se prikaže namig z opisom njene funkcije.

## Legenda ikon

### Vrstica za predvajanje

| Ikona | Dejanje |
|-------|---------|
| ▶ | Predvajaj |
| ⏸ | Premor |
| ⏮ | Skok na prejšnji segment |
| ⏭ | Skok na naslednji segment |

### Dejanja na kartici segmenta

| Ikona | Dejanje |
|-------|---------|
| <mdl2 ch="E77B"/> | Dodelitev segmenta drugemu govorcu |
| <mdl2 ch="E916"/> | Prilagoditev začetnega in končnega časa segmenta |
| <mdl2 ch="EA39"/> | Skrivanje ali razkrivanje segmenta |
| <mdl2 ch="E72B"/> | Združitev s prejšnjim segmentom |
| <mdl2 ch="E72A"/> | Združitev z naslednjim segmentom |
| <mdl2 ch="E8C6"/> | Razdelitev segmenta |
| <mdl2 ch="E72C"/> | Ponoven ASR za ta segment |

## Predvajanje zvoka

Vrstica za predvajanje se razteza vzdolž vrha okna urejevalnika:

| Kontrolnik | Dejanje |
|------------|---------|
| Ikona Predvajaj / Premor | Zažene ali zaustavi predvajanje |
| Drsnik za pomikanje | Povlecite za skok na poljubno mesto v zvočnem posnetku |
| Drsnik za hitrost | Prilagoditev hitrosti predvajanja (0,5× – 2×) |
| Ikoni Prejšnji / Naslednji | Skok na prejšnji ali naslednji segment |
| Spustni meni načina predvajanja | Izberite enega od treh načinov predvajanja (glejte spodaj) |
| Drsnik za glasnost | Prilagoditev glasnosti predvajanja |

Med predvajanjem je beseda, ki je trenutno izgovorjena, označena v levem podoknu. Ko je predvajanje zaustavljeno po premiku drsnika, se označitev posodobi na besedo na položaju drsnika.

### Načini predvajanja

| Način | Obnašanje |
|-------|-----------|
| `Single` | Predvaja trenutni segment enkrat, nato se ustavi. |
| `Auto-advance` | Predvaja trenutni segment; ko se konča, ga označi kot preverjenega in nadaljuje z naslednjim. |
| `Continuous` | Predvaja vse segmente zaporedoma, ne da bi katerega koli označilo kot preverjenega. |

Aktivni način izberite v spustnem meniju v vrstici za predvajanje.

## Urejanje segmenta

1. Kliknite kartico, da jo izberete.
2. Uredite besedilo v desnem podoknu. Spremembe se samodejno shranijo, ko premaknete fokus na drugo kartico.

## Preimenovanje govorca

Kliknite oznako govorca na izbrani kartici in vpišite novo ime. Pritisnite `Enter` ali kliknite drugam, da shranite. Novo ime se uporabi samo za to kartico; za globalno preimenovanje govorca uporabite [Urejanje imen govorcev](editing_speaker_names.md) v pogledu Rezultati.

## Preverjanje segmenta

Kliknite potrditveno polje `Verified` na izbrani kartici, da jo označite kot pregledano. Stanje preverjanja se shrani v podatkovno zbirko in je ob prihodnjih nalanjih vidno v urejevalniku.

## Skrivanje segmenta

Kliknite `Suppress` na izbrani kartici, da skrijete segment iz izvozov (koristno za šum, glasbo ali druge negovorne odseke). Kliknite `Unsuppress`, da ga obnovite.

## Prilagajanje časov segmenta

Kliknite `Adjust Times` na izbrani kartici, da odprete pogovorno okno za prilagajanje časa. Z vrtenjem kolesca miške nad poljem **Start** ali **End** prilagajajte vrednost v korakih po 0,1 sekunde ali vrednost vnesite neposredno. Kliknite `Save`, da uveljavite spremembe.

## Združevanje segmentov

- Kliknite `⟵ Merge`, da združite izbrani segment s segmentom neposredno pred njim.
- Kliknite `Merge ⟶`, da združite izbrani segment s segmentom neposredno za njim.

Besedilo in časovni razpon obeh kartic se združita. To je koristno, kadar je bila ena izgovorjena izjava razdeljena na dva segmenta.

## Razdelitev segmenta

Kliknite `Split…` na izbrani kartici, da odprete pogovorno okno za razdelitev. Določite točko razdelitve v besedilu in potrdite. Ustvarita se dva nova segmenta, ki pokrivata izvirni časovni razpon. To je koristno, kadar sta bili dve ločeni izjavi združeni v en segment.

## Ponoven ASR

Kliknite `Redo ASR` na izbrani kartici, da znova zaženete prepoznavanje govora na zvoku tega segmenta. Model obdela samo zvočni odsek tega segmenta in ustvari svež prepis iz enega vira.

Uporabite to, kadar:

- Je segment nastal z združitvijo in ga ni mogoče razdeliti (združeni segmenti zajemajo več virov ASR; Redo ASR jih strne v enega, nakar postane na voljo možnost `Split…`).
- Je izvirni prepis slabe kakovosti in želite nov čist prehod brez ročnega urejanja.

**Opomba:** Vse besedilo, ki ste ga že vnesli v desno podokno, se zavrže in nadomesti z novim izpisom ASR. Operacija zahteva naloženo zvočno datoteko; gumb je onemogočen, če zvok ni na voljo.