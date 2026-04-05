---
title: "Transkriptioiden muokkaaminen"
description: "Ohjeet transkriptoitujen segmenttien tarkistamiseen, korjaamiseen ja vahvistamiseen transkriptioeditorissa."
topic_id: operations_editing_transcripts
---

# Transkriptioiden muokkaaminen

**Transkriptioeditorin** avulla voit tarkistaa ASR-tulostetta, korjata tekstiä, nimetä puhujia uudelleen suoraan editorissa, säätää segmenttien ajoitusta ja merkitä segmentit vahvistetuiksi — kaiken tämän samalla kun kuuntelet alkuperäistä ääntä.

## Editorin avaaminen

1. Lataa valmis työ (katso [Valmiiden töiden lataaminen](loading_completed_jobs.md)).
2. Napsauta **Tulokset**-näkymässä `Edit Transcript`.

Editori avautuu erillisenä ikkunana ja voi pysyä auki rinnakkain pääsovelluksen kanssa.

## Asettelu

Jokainen segmentti näytetään korttina, jossa on kaksi vierekkäistä paneelia:

- **Vasen paneeli** — alkuperäinen ASR-tuloste, jossa yksittäisten sanojen luottamustaso on väritetty. Sanat, joista malli oli epävarmempi, näkyvät punaisina; korkean luottamustason sanat näkyvät normaalilla tekstivärillä.
- **Oikea paneeli** — muokattava tekstikenttä. Tee korjaukset tässä; ero alkuperäiseen korostetaan kirjoittaessasi.

Puhujan tunniste ja aikaväli näkyvät jokaisen kortin yläpuolella. Napsauta korttia kohdistamalla se ja paljastamalla sen toimintokuvakkeet. Vie hiiri minkä tahansa kuvakkeen päälle nähdäksesi sen toimintoa kuvaavan työkaluvihjeen.

## Kuvakeopas

### Toistopalkin kuvakkeet

| Kuvake | Toiminto |
|--------|----------|
| ▶ | Toista |
| ⏸ | Keskeytä |
| ⏮ | Siirry edelliseen segmenttiin |
| ⏭ | Siirry seuraavaan segmenttiin |

### Segmenttikortin toiminnot

| Kuvake | Toiminto |
|--------|----------|
| <mdl2 ch="E77B"/> | Määritä segmentti eri puhujalle |
| <mdl2 ch="E916"/> | Säädä segmentin alku- ja loppuaikoja |
| <mdl2 ch="EA39"/> | Piilota segmentti viennistä tai palauta se |
| <mdl2 ch="E72B"/> | Yhdistä edelliseen segmenttiin |
| <mdl2 ch="E72A"/> | Yhdistä seuraavaan segmenttiin |
| <mdl2 ch="E8C6"/> | Jaa segmentti |
| <mdl2 ch="E72C"/> | Suorita ASR uudelleen tälle segmentille |

## Äänen toistaminen

Toistopalki kulkee editori-ikkunan yläosassa:

| Ohjain | Toiminto |
|--------|----------|
| Toista/Keskeytä-kuvake | Aloita tai keskeytä toisto |
| Hakupalkki | Vedä siirtyäksesi mihin tahansa kohtaan äänessä |
| Nopeudenliukusäädin | Säädä toistonopeutta (0,5× – 2×) |
| Edellinen/Seuraava-kuvakkeet | Siirry edelliseen tai seuraavaan segmenttiin |
| Toistomodin alasvetovalikko | Valitse yksi kolmesta toistomodista (katso alla) |
| Äänenvoimakkuuden liukusäädin | Säädä toistoäänenvoimakkuutta |

Toiston aikana parhaillaan puhuttu sana korostetaan vasemmassa paneelissa. Kun toisto on keskeytetty haun jälkeen, korostus siirtyy haun kohtaan vastaavaan sanaan.

### Toistomodit

| Modi | Toiminta |
|------|----------|
| `Single` | Toistaa nykyisen segmentin kerran, sitten pysähtyy. |
| `Auto-advance` | Toistaa nykyisen segmentin; kun se päättyy, merkitsee sen vahvistetuksi ja siirtyy seuraavaan. |
| `Continuous` | Toistaa kaikki segmentit peräkkäin merkitsemättä yhtään vahvistetuksi. |

Valitse aktiivinen modi toistopalkin alasvetovalikosta.

## Segmentin muokkaaminen

1. Napsauta korttia kohdistamalla se.
2. Muokkaa tekstiä oikeassa paneelissa. Muutokset tallennetaan automaattisesti, kun siirrät kohdistuksen toiseen korttiin.

## Puhujan nimen muuttaminen

Napsauta puhujan tunnistetta kohdistetun kortin sisällä ja kirjoita uusi nimi. Tallenna painamalla `Enter` tai napsauttamalla muualle. Uusi nimi otetaan käyttöön vain kyseisessä kortissa. Jos haluat nimetä puhujan uudelleen kaikkialla, käytä [Puhujien nimien muokkaaminen](editing_speaker_names.md) -toimintoa Tulokset-näkymässä.

## Segmentin vahvistaminen

Napsauta kohdistetun kortin `Verified`-valintaruutua merkitäksesi sen tarkistetuksi. Vahvistettu tila tallennetaan tietokantaan ja näkyy editorissa tulevissa latauksissa.

## Segmentin piilottaminen viennistä

Napsauta kohdistetun kortin `Suppress`-painiketta piilottaaksesi segmentin vienneistä (hyödyllinen melulle, musiikille tai muille ei-puheosioille). Palauta segmentti napsauttamalla `Unsuppress`.

## Segmentin aikojen säätäminen

Napsauta kohdistetun kortin `Adjust Times` -painiketta avataksesi aikojen säätövalintaikkunan. Käytä vierityspyörää **Alku**- tai **Loppu**-kentän päällä siirtääksesi arvoa 0,1 sekunnin askelin tai kirjoita arvo suoraan. Ota muutokset käyttöön napsauttamalla `Save`.

## Segmenttien yhdistäminen

- Napsauta `⟵ Merge` yhdistääksesi kohdistetun segmentin välittömästi edeltävään segmenttiin.
- Napsauta `Merge ⟶` yhdistääksesi kohdistetun segmentin välittömästi seuraavaan segmenttiin.

Molempien korttien teksti ja aikaväli yhdistetään. Tämä on hyödyllistä, kun yksi puhuttu lausuma on jaettu kahteen segmenttiin.

## Segmentin jakaminen

Napsauta kohdistetun kortin `Split…`-painiketta avataksesi jakovalintaikkunan. Aseta jakopiste tekstin sisälle ja vahvista. Kaksi uutta segmenttiä luodaan kattamaan alkuperäinen aikaväli. Tämä on hyödyllistä, kun kaksi erillistä lausumaa on yhdistetty yhteen segmenttiin.

## ASR:n suorittaminen uudelleen

Napsauta kohdistetun kortin `Redo ASR` -painiketta suorittaaksesi puheentunnistuksen uudelleen kyseisen segmentin äänelle. Malli käsittelee vain kyseisen segmentin äänipätkän ja tuottaa uuden, yksittäislähteisen transkription.

Käytä tätä kun:

- Segmentti on peräisin yhdistämisestä eikä sitä voi jakaa (yhdistetyt segmentit kattavat useita ASR-lähteitä; Redo ASR kokoaa ne yhteen, minkä jälkeen `Split…` tulee käytettäväksi).
- Alkuperäinen transkriptio on heikkolaatuinen ja haluat puhtaan toisen kierroksen ilman manuaalista muokkausta.

**Huomio:** Kaikki oikeaan paneeliin jo kirjoittamasi teksti poistetaan ja korvataan uudella ASR-tulosteella. Toiminto edellyttää, että äänitiedosto on ladattu; painike on poistettu käytöstä, jos ääni ei ole saatavilla.