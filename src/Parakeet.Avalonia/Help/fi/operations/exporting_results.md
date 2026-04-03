---
title: "Tulosten tai litteraattien vienti"
description: "Ohje litteraatin tallentamiseen tiedostoon eri muodoissa."
topic_id: operations_exporting_results
---

# Tulosten tai litteraattien vienti

Voit viedä valmiin litteraatin useisiin tiedostomuotoihin käytettäväksi muissa sovelluksissa.

## Vieminen

1. Avaa valmis työ. Katso [Valmiiden töiden lataaminen](loading_completed_jobs.md).
2. Napsauta **Tulokset**-näkymässä `Export Transcript`.
3. **Export Transcript** -valintaikkuna avautuu. Valitse muoto **Format**-pudotusvalikosta.
4. Napsauta `Save`. Tiedoston tallennusvalintaikkuna avautuu.
5. Valitse kohdekansio ja tiedostonimi, ja napsauta sitten **Save**.

Valintaikkunan alaosaan ilmestyy vahvistusviesti, jossa näkyy tallennetun tiedoston koko polku.

## Käytettävissä olevat muodot

| Muoto | Tunniste | Sopii parhaiten |
|---|---|---|
| Excel | `.xlsx` | Taulukkolaskenta-analyysi puhujan, aikaleimoja ja sisältöä varten tarkoitetuilla sarakkeilla. |
| CSV | `.csv` | Tuonti mihin tahansa taulukkolaskentaohjelmaan tai datatiedostoon. |
| JSON | `.json` | Ohjelmallinen käsittely. |
| SRT Subtitles | `.srt` | Lataaminen videoeditoreihin tai mediasoittimiin tekstityksenä. |
| Markdown | `.md` | Luettavat pelkkää tekstiä sisältävät asiakirjat. |
| Word Document | `.docx` | Jakaminen Microsoft Wordin käyttäjien kanssa. |
| SQLite Database | `.db` | Täydellinen tietokantavienti mukautettuja kyselyjä varten. |

## Puhujien nimet vienneissä

Jos olet määrittänyt puhujille näyttönimet, niitä käytetään kaikissa vientimuodoissa. Jos haluat päivittää nimet ennen vientiä, napsauta ensin `Edit Speaker Names`. Katso [Puhujien nimien muokkaaminen](editing_speaker_names.md).

---