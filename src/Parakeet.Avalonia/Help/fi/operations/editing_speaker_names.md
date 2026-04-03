---
title: "Puhujan nimien muokkaaminen"
description: "Kuinka korvata yleiset puhujatunnisteet oikeilla nimillä transkriptiossa."
topic_id: operations_editing_speaker_names
---

# Puhujan nimien muokkaaminen

Transkriptiokone merkitsee jokaisen puhujan automaattisesti yleisellä tunnisteella (esimerkiksi `speaker_0`, `speaker_1`). Voit korvata nämä oikeilla nimillä, jotka näkyvät koko transkriptiossa ja kaikissa viedyissä tiedostoissa.

## Puhujan nimien muokkaaminen

1. Avaa valmis työ. Katso [Valmiiden töiden lataaminen](loading_completed_jobs.md).
2. Napsauta **Tulokset**-näkymässä `Edit Speaker Names`.
3. **Edit Speaker Names** -valintaikkuna avautuu kahdella sarakkeella:
   - **Speaker ID** — mallin määrittämä alkuperäinen tunniste (vain luku).
   - **Display Name** — transkriptiossa näytettävä nimi (muokattavissa).
4. Napsauta solua **Display Name** -sarakkeessa ja kirjoita puhujan nimi.
5. Siirry seuraavaan puhujaan painamalla `Tab` tai napsauttamalla toista riviä.
6. Tallenna muutokset napsauttamalla `Save` tai hylkää ne napsauttamalla `Cancel`.

## Missä nimet näkyvät

Päivitetyt näyttönimet korvaavat yleiset tunnisteet seuraavissa paikoissa:

- Tulokset-näkymän segmenttitaulukossa.
- Kaikissa viedyissä tiedostoissa (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Nimien muokkaaminen uudelleen

Voit avata Edit Speaker Names -valintaikkunan uudelleen milloin tahansa, kun työ on ladattuna Tulokset-näkymässä. Muutokset tallennetaan paikalliseen tietokantaan ja säilyvät istuntojen välillä.

---