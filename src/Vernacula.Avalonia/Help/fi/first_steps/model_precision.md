---
title: "Mallin painotarkkuuden valinta"
description: "Kuinka valita INT8- ja FP32-mallitarkkuuden välillä ja mitkä ovat niiden väliset kompromissit."
topic_id: first_steps_model_precision
---

# Mallin painotarkkuuden valinta

Mallitarkkuus määrittää tekoälymallin painojen käyttämän numeerisen esitysmuodon. Se vaikuttaa latauksen kokoon, muistinkäyttöön ja tarkkuuteen.

## Tarkkuusvaihtoehdot

### INT8 (pienempi lataus)

- Pienemmät mallitiedostot — nopeampi ladata ja vaatii vähemmän levytilaa.
- Hieman heikompi tarkkuus joillakin äänillä.
- Suositellaan, jos levytila on rajallinen tai internet-yhteys on hidas.

### FP32 (tarkempi)

- Suuremmat mallitiedostot.
- Parempi tarkkuus, erityisesti vaikeassa äänessä, kuten aksenttien tai taustamelun kanssa.
- Suositellaan, kun tarkkuus on etusijalla ja levytilaa on riittävästi.
- Vaaditaan CUDA GPU -kiihdytykseen — GPU-polku käyttää aina FP32-tarkkuutta tästä asetuksesta riippumatta.

## Tarkkuuden vaihtaminen

Avaa `Settings…` valikkopalkista, siirry sitten **Models**-osioon ja valitse joko `INT8 (smaller download)` tai `FP32 (more accurate)`.

## Tarkkuuden vaihtamisen jälkeen

Tarkkuuden vaihtaminen edellyttää erilaista mallitiedostojoukkoa. Jos uuden tarkkuuden mallit eivät ole vielä ladattuna, napsauta `Download Missing Models` Asetuksissa. Aiemmin ladatut tiedostot toiselle tarkkuudelle säilytetään levyllä, eikä niitä tarvitse ladata uudelleen, jos vaihdat takaisin.

---