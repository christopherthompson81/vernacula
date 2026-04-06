---
title: "Mallien lataaminen"
description: "Ohjeet transkription edellyttämien tekoälymallitiedostojen lataamiseen."
topic_id: first_steps_downloading_models
---

# Mallien lataaminen

Vernacula-Desktop tarvitsee toimiakseen tekoälymallitiedostoja. Nämä eivät sisälly sovellukseen, vaan ne on ladattava ennen ensimmäistä transkriptiota.

## Mallien tila (aloitusnäyttö)

Aloitusnäytön yläreunassa oleva kapea tilarivi näyttää, ovatko mallisi valmiina. Jos tiedostoja puuttuu, tilarivissä näkyy myös `Open Settings` -painike, joka vie sinut suoraan mallien hallintaan.

| Tila | Merkitys |
|---|---|
| `All N model file(s) present ✓` | Kaikki tarvittavat tiedostot on ladattu ja ne ovat valmiina. |
| `N model file(s) missing: …` | Yksi tai useampi tiedosto puuttuu; avaa Asetukset ladataksesi ne. |

Kun mallit ovat valmiina, `New Transcription`- ja `Bulk Add Jobs` -painikkeet aktivoituvat.

## Mallien lataaminen

1. Napsauta aloitusnäytöllä `Open Settings` (tai siirry kohtaan `Settings… > Models`).
2. Napsauta **Models**-osiossa `Download Missing Models`.
3. Näkyviin tulee edistymispalkki ja tilarivi, jotka näyttävät nykyisen tiedoston, sen paikan jonossa sekä latauksen koon — esimerkiksi: `[1/3] encoder-model.onnx — 42 MB`.
4. Odota, kunnes tilariville ilmestyy teksti `Download complete.`

## Latauksen peruuttaminen

Voit keskeyttää käynnissä olevan latauksen napsauttamalla `Cancel`. Tilariviin ilmestyy teksti `Download cancelled.` Osittain ladatut tiedostot säilyvät, joten lataus jatkuu siitä, mihin se jäi, kun seuraavan kerran napsautat `Download Missing Models`.

## Latausvirheet

Jos lataus epäonnistuu, tilarivissä näkyy `Download failed: <reason>`. Tarkista internetyhteytesi ja napsauta `Download Missing Models` uudelleen yrittääksesi uudelleen. Sovellus jatkaa viimeksi onnistuneesti valmistuneen tiedoston jälkeen.

## Tarkkuuden muuttaminen

Ladattavat mallitiedostot riippuvat valitusta **Model Precision** -asetuksesta. Voit muuttaa sitä menemällä kohtaan `Settings… > Models > Model Precision`. Jos vaihdat tarkkuutta lataamisen jälkeen, uusi tiedostosarja on ladattava erikseen. Katso [Mallipainon tarkkuuden valitseminen](model_precision.md).

---