---
title: "Useiden äänitiedostojen lisääminen jonoon"
description: "Kuinka lisätä useita äänitiedostoja työjonoon kerralla."
topic_id: operations_bulk_add_jobs
---

# Useiden äänitiedostojen lisääminen jonoon

Käytä **Bulk Add Jobs** -toimintoa lisätäksesi useita ääni- tai videotiedostoja transkriptiota varten jonoon yhdellä kertaa. Sovellus käsittelee ne yksi kerrallaan siinä järjestyksessä, jossa ne lisättiin.

## Edellytykset

- Kaikki mallitiedostot on ladattava. **Model Status** -kortin on näytettävä `All N model file(s) present ✓`. Katso [Mallien lataaminen](../first_steps/downloading_models.md).

## Töiden lisääminen joukkona

1. Napsauta aloitusnäytöllä `Bulk Add Jobs`.
2. Tiedostonvalitsin avautuu. Valitse yksi tai useampi ääni- tai videotiedosto — pidä `Ctrl` tai `Shift` painettuna valitaksesi useita tiedostoja.
3. Napsauta **Open**. Jokainen valittu tiedosto lisätään **Transcription History** -taulukkoon erillisenä työnä.

> **Videotiedostot, joissa on useita äänivirtoja:** Jos videotiedosto sisältää useamman kuin yhden äänivirtauksen (esimerkiksi useita kieliä tai ohjaajan kommentaariraita), sovellus luo automaattisesti yhden työn kutakin äänivirtaa kohden.

## Töiden nimet

Kullekin työlle annetaan automaattisesti nimi sen äänitiedoston nimen perusteella. Voit nimetä työn uudelleen milloin tahansa napsauttamalla sen nimeä Transcription History -taulukon **Title**-sarakkeessa, muokkaamalla tekstiä ja painamalla `Enter` tai napsauttamalla muualle.

## Jonon toiminta

- Jos yhtään työtä ei ole käynnissä, ensimmäinen tiedosto käynnistyy välittömästi ja loput näytetään tilassa `queued`.
- Jos jokin työ on jo käynnissä, kaikki juuri lisätyt tiedostot näytetään tilassa `queued` ja ne käynnistyvät automaattisesti peräkkäin.
- Voit seurata aktiivista työtä napsauttamalla `Monitor` sen **Actions**-sarakkeessa. Katso [Töiden seuraaminen](monitoring_jobs.md).
- Voit keskeyttää tai poistaa jonossa olevan työn ennen sen käynnistymistä käyttämällä `Pause`- tai `Remove`-painikkeita sen **Actions**-sarakkeessa. Katso [Töiden keskeyttäminen, jatkaminen tai poistaminen](pausing_resuming_removing.md).

---