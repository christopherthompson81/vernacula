---
title: "Töiden keskeyttäminen, jatkaminen tai poistaminen"
description: "Kuinka keskeyttää käynnissä oleva työ, jatkaa pysäytettyä työtä tai poistaa työ historiatiedoista."
topic_id: operations_pausing_resuming_removing
---

# Töiden keskeyttäminen, jatkaminen tai poistaminen

## Työn keskeyttäminen

Voit keskeyttää käynnissä olevan tai jonossa olevan työn kahdesta paikasta:

- **Edistymänäkymä** — napsauta `Pause` oikeassa alakulmassa, kun seuraat aktiivista työtä.
- **Transkriptiohistoria-taulukko** — napsauta `Pause` sen rivin **Actions**-sarakkeessa, jonka tila on `running` tai `queued`.

Kun napsautat `Pause`, tilarivi näyttää tekstin `Pausing…`, kun sovellus viimeistelee nykyisen käsittely-yksikön. Tämän jälkeen työn tilaksi muuttuu `cancelled` historiataulukossa.

> Keskeyttäminen tallentaa kaikki tähän mennessä litteroidut segmentit. Voit jatkaa työtä myöhemmin menettämättä tehtyä työtä.

## Työn jatkaminen

Keskeytetyn tai epäonnistuneen työn jatkaminen:

1. Etsi työ aloitusnäytön **Transcription History** -taulukosta. Sen tilana on `cancelled` tai `failed`.
2. Napsauta **Actions**-sarakkeen `Resume`-painiketta.
3. Sovellus palaa **Progress**-näkymään ja jatkaa siitä kohdasta, johon käsittely pysähtyi.

Tilarivi näyttää hetken tekstin `Resuming…`, kun työ alustaa itsensä uudelleen.

## Työn poistaminen

Työn ja sen litteraatin pysyvä poistaminen historiatiedoista:

1. Napsauta **Transcription History** -taulukon **Actions**-sarakkeessa poistettavan työn kohdalla olevaa `Remove`-painiketta.

Työ poistetaan luettelosta ja sen tiedot poistetaan paikallisesta tietokannasta. Tätä toimintoa ei voi kumota. Levylle tallennetut viedyt tiedostot eivät ole vaikutuksen alaisia.

---