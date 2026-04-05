---
title: "Uređivanje transkripata"
description: "Kako pregledavati, ispravljati i verificirati transkribirane segmente u uređivaču transkripata."
topic_id: operations_editing_transcripts
---

# Uređivanje transkripata

**Uređivač transkripata** omogućuje vam pregled ASR izlaza, ispravljanje teksta, preimenovanje govornika izravno u sučelju, podešavanje vremenskog raspona segmenata i označavanje segmenata kao verificiranih — sve to uz istodobno slušanje originalnog zvuka.

## Otvaranje uređivača

1. Učitajte dovršeni posao (pogledajte [Učitavanje dovršenih poslova](loading_completed_jobs.md)).
2. U prikazu **Rezultati** kliknite `Edit Transcript`.

Uređivač se otvara kao zaseban prozor i može ostati otvoren uz glavni prozor aplikacije.

## Izgled sučelja

Svaki segment prikazan je kao kartica s dvije ploče postavljene jedna uz drugu:

- **Lijeva ploča** — originalni ASR izlaz s bojanjem pouzdanosti po riječima. Riječi koje je model prepoznao s manjom sigurnošću prikazuju se crvenom bojom, a visoko pouzdane riječi prikazuju se u normalnoj boji teksta.
- **Desna ploča** — polje za uređivanje teksta. Ispravke unosite ovdje; razlike u odnosu na original ističu se dok tipkate.

Oznaka govornika i vremenski raspon prikazani su iznad svake kartice. Kliknite karticu kako biste je fokusirali i prikazali njezine ikone radnji. Zadržite pokazivač iznad bilo koje ikone kako biste vidjeli opis njezine funkcije.

## Legenda ikona

### Traka za reprodukciju

| Ikona | Radnja |
|-------|--------|
| ▶ | Reprodukcija |
| ⏸ | Pauza |
| ⏮ | Skok na prethodni segment |
| ⏭ | Skok na sljedeći segment |

### Radnje na karticama segmenata

| Ikona | Radnja |
|-------|--------|
| <mdl2 ch="E77B"/> | Dodjela segmenta drugom govorniku |
| <mdl2 ch="E916"/> | Podešavanje vremena početka i kraja segmenta |
| <mdl2 ch="EA39"/> | Skrivanje ili otkrivanje segmenta |
| <mdl2 ch="E72B"/> | Spajanje s prethodnim segmentom |
| <mdl2 ch="E72A"/> | Spajanje sa sljedećim segmentom |
| <mdl2 ch="E8C6"/> | Dijeljenje segmenta |
| <mdl2 ch="E72C"/> | Ponovni ASR za ovaj segment |

## Reprodukcija zvuka

Traka za reprodukciju proteže se duž vrha prozora uređivača:

| Kontrola | Radnja |
|----------|--------|
| Ikona Reprodukcija / Pauza | Pokretanje ili zaustavljanje reprodukcije |
| Traka za premotavanje | Povucite kako biste prešli na bilo koji položaj u zvučnom zapisu |
| Klizač brzine | Podešavanje brzine reprodukcije (0,5× – 2×) |
| Ikone Prethodni / Sljedeći | Skok na prethodni ili sljedeći segment |
| Padajući izbornik načina reprodukcije | Odabir jednog od tri načina reprodukcije (pogledajte dolje) |
| Klizač glasnoće | Podešavanje glasnoće reprodukcije |

Tijekom reprodukcije, riječ koja se trenutno izgovara istaknuta je u lijevoj ploči. Kada je reprodukcija pauzirana nakon premotavanja, isticanje se ažurira na riječ na položaju premotavanja.

### Načini reprodukcije

| Način | Ponašanje |
|-------|-----------|
| `Single` | Reproducira trenutni segment jednom, zatim se zaustavlja. |
| `Auto-advance` | Reproducira trenutni segment; kada završi, označava ga kao verificiran i prelazi na sljedeći. |
| `Continuous` | Reproducira sve segmente redom bez označavanja ijednog kao verificiranog. |

Odaberite aktivni način iz padajućeg izbornika na traci za reprodukciju.

## Uređivanje segmenta

1. Kliknite karticu kako biste je fokusirali.
2. Uredite tekst u desnoj ploči. Promjene se automatski spremaju kada premjestite fokus na drugu karticu.

## Preimenovanje govornika

Kliknite oznaku govornika unutar fokusirane kartice i upišite novo ime. Pritisnite `Enter` ili kliknite drugdje kako biste spremili. Novo ime primjenjuje se samo na tu karticu; za globalno preimenovanje govornika koristite [Uredi nazive govornika](editing_speaker_names.md) iz prikaza Rezultati.

## Verificiranje segmenta

Kliknite potvrdni okvir `Verified` na fokusiranoj kartici kako biste je označili kao pregledanu. Status verificiranosti sprema se u bazu podataka i vidljiv je u uređivaču pri budućim učitavanjima.

## Skrivanje segmenta

Kliknite `Suppress` na fokusiranoj kartici kako biste sakrili segment iz izvoza (korisno za šum, glazbu ili druge dijelove bez govora). Kliknite `Unsuppress` kako biste ga vratili.

## Podešavanje vremena segmenta

Kliknite `Adjust Times` na fokusiranoj kartici kako biste otvorili dijaloški okvir za podešavanje vremena. Koristite kotačić miša iznad polja **Start** ili **End** za pomicanje vrijednosti u koracima od 0,1 sekunde, ili izravno upišite vrijednost. Kliknite `Save` za primjenu.

## Spajanje segmenata

- Kliknite `⟵ Merge` kako biste spojili fokusirani segment s prethodnim segmentom.
- Kliknite `Merge ⟶` kako biste spojili fokusirani segment sa sljedećim segmentom.

Kombiniraju se tekst i vremenski raspon obje kartice. Ovo je korisno kada je jedan izgovoreni iskaz bio podijeljen na dva segmenta.

## Dijeljenje segmenta

Kliknite `Split…` na fokusiranoj kartici kako biste otvorili dijaloški okvir za dijeljenje. Postavite točku podjele unutar teksta i potvrdite. Stvaraju se dva nova segmenta koji pokrivaju originalni vremenski raspon. Ovo je korisno kada su dva zasebna iskaza spojena u jedan segment.

## Ponovni ASR

Kliknite `Redo ASR` na fokusiranoj kartici kako biste ponovo pokrenuli prepoznavanje govora za zvuk tog segmenta. Model obrađuje samo zvučni isječak za taj segment i izrađuje novu transkripciju iz jednog izvora.

Koristite ovo kada:

- Segment nastao spajanjem nije moguće podijeliti (spojeni segmenti obuhvaćaju više ASR izvora; Redo ASR ih sažima u jedan, nakon čega `Split…` postaje dostupan).
- Originalna transkripcija je loše kvalitete i želite čisti drugi prolaz bez ručnog uređivanja.

**Napomena:** Sav tekst koji ste već unijeli u desnoj ploči bit će odbačen i zamijenjen novim ASR izlazom. Operacija zahtijeva da zvučna datoteka bude učitana; gumb je onemogućen ako zvuk nije dostupan.