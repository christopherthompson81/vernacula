---
title: "Asetukset"
description: "Yleiskatsaus kaikista Asetukset-ikkunan valinnoista."
topic_id: first_steps_settings_window
---

# Asetukset

**Asetukset**-ikkuna antaa sinulle hallinnan laitteistokokoonpanon, mallien hallinnan, segmentointitilan, editorin toiminnan, ulkoasun ja kielen suhteen. Avaa se valikkoriviltä: `Settings…`.

## Laitteisto ja suorituskyky

Tässä osiossa näkyy NVIDIA GPU:n ja CUDA-ohjelmistopinon tila sekä GPU-litteroinnin aikana käytettävä dynaaminen eräkoko.

| Kohde | Kuvaus |
|---|---|
| GPU:n nimi ja VRAM | Tunnistettu NVIDIA GPU ja käytettävissä oleva näyttömuisti. |
| CUDA Toolkit | Löytyivätkö CUDA-suorituskirjastot `CUDA_PATH`-muuttujan kautta. |
| cuDNN | Ovatko cuDNN-suorituskirjastot (DLL) saatavilla. |
| CUDA-kiihdytys | Lataisiko ONNX Runtime CUDA-suorituksen tarjoajan onnistuneesti. |

Napsauta `Re-check` suorittaaksesi laitteiston tunnistuksen uudelleen käynnistämättä sovellusta — hyödyllistä CUDA:n tai cuDNN:n asentamisen jälkeen.

Suorat latauslinkit CUDA Toolkitille ja cuDNN:lle näytetään, kun kyseisiä komponentteja ei havaita.

**Eräkoko**-viesti kertoo, kuinka monta sekuntia ääntä käsitellään kussakin GPU-ajossa. Tämä arvo johdetaan mallien lataamisen jälkeen vapaana olevasta VRAM-muistista ja mukautuu automaattisesti.

Täydelliset CUDA-asennusohjeet löydät sivulta [CUDA:n ja cuDNN:n asentaminen](cuda_installation.md).

## Mallit

Tässä osiossa hallitaan litterointiin tarvittavia tekoälymallin tiedostoja.

- **Mallin tarkkuus** — valitse `INT8 (smaller download)` tai `FP32 (more accurate)`. Katso [Mallin painotarkkuuden valitseminen](model_precision.md).
- **Lataa puuttuvat mallit** — lataa mallitiedostot, joita ei vielä ole levyllä. Edistymispalkki ja tilarivi seuraavat jokaisen tiedoston latautumista.
- **Tarkista päivitykset** — tarkistaa, onko uudempia mallipainoja saatavilla. Päivitysilmoitus näkyy myös automaattisesti aloitusnäytöllä, kun uusia painoja havaitaan.

## Segmentointitila

Määrittää, miten ääni jaetaan segmentteihin ennen puheentunnistusta.

| Tila | Kuvaus |
|---|---|
| **Puhujien diarisaatio** | Käyttää SortFormer-mallia tunnistaakseen yksittäiset puhujat ja merkitäkseen jokaisen segmentin. Parhaiten sopii haastatteluihin, kokouksiin ja useamman puhujan nauhoituksiin. |
| **Puheaktiivisuuden tunnistus** | Käyttää Silero VAD:ia havaitakseen puheenkohdat — ei puhujamerkintöjä. Nopeampi kuin diarisaatio ja sopii hyvin yhden puhujan ääneen. |

## Litterointieditori

**Oletustoistotila** — asettaa toistotilan, jota käytetään, kun avaat litterointieditorin. Voit myös vaihtaa sitä suoraan editorissa milloin tahansa. Katso [Litterointien muokkaaminen](../operations/editing_transcripts.md) kunkin tilan kuvauksesta.

## Ulkoasu

Valitse **Tumma** tai **Vaalea** teema. Muutos astuu voimaan välittömästi. Katso [Teeman valitseminen](theme.md).

## Kieli

Valitse sovelluksen käyttöliittymän näyttökieli. Muutos astuu voimaan välittömästi. Katso [Kielen valitseminen](language.md).

---