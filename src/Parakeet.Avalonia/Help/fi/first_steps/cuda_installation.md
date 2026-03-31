---
title: "CUDA:n ja cuDNN:n asentaminen GPU-kiihdytystä varten"
description: "Kuinka määrittää NVIDIA CUDA ja cuDNN, jotta Parakeet Transcription voi käyttää GPU:ta."
topic_id: first_steps_cuda_installation
---

# CUDA:n ja cuDNN:n asentaminen GPU-kiihdytystä varten

Parakeet Transcription voi käyttää NVIDIA GPU:ta transkription merkittävään nopeuttamiseen. GPU-kiihdytys edellyttää, että NVIDIA CUDA Toolkit ja cuDNN-ajonaikaiset kirjastot on asennettu järjestelmääsi.

## Vaatimukset

- NVIDIA GPU, joka tukee CUDA:ta (GeForce GTX 10 -sarja tai uudempi suositellaan).
- Windows 10 tai 11 (64-bittinen).
- Mallitiedostojen on oltava jo ladattu. Katso [Mallien lataaminen](downloading_models.md).

## Asennusvaiheet

### 1. Asenna CUDA Toolkit

Lataa ja suorita CUDA Toolkit -asennusohjelma NVIDIA:n kehittäjäsivustolta. Hyväksy asennuksen aikana oletuspolut. Asennusohjelma asettaa `CUDA_PATH`-ympäristömuuttujan automaattisesti — Parakeet käyttää tätä muuttujaa CUDA-kirjastojen paikantamiseen.

### 2. Asenna cuDNN

Lataa NVIDIA:n kehittäjäsivustolta asennettuun CUDA-versiooisi sopiva cuDNN ZIP -arkisto. Pura arkisto ja kopioi sen `bin`-, `include`- ja `lib`-kansioiden sisältö CUDA Toolkit -asennushakemistosi vastaaviin kansioihin (polku näkyy `CUDA_PATH`-muuttujassa).

Vaihtoehtoisesti voit asentaa cuDNN:n NVIDIA:n cuDNN-asennusohjelmalla, jos se on saatavilla käyttämällesi CUDA-versiolle.

### 3. Käynnistä sovellus uudelleen

Sulje ja avaa Parakeet Transcription uudelleen asennuksen jälkeen. Sovellus tarkistaa CUDA:n olemassaolon käynnistyksen yhteydessä.

## GPU-tila asetuksissa

Avaa `Settings…` valikkoriviltä ja tarkastele **Hardware & Performance** -osiota. Jokaisen komponentin kohdalla näkyy valintamerkki (✓), kun se on havaittu:

| Kohde | Mitä se tarkoittaa |
|---|---|
| GPU:n nimi ja VRAM | NVIDIA GPU löydettiin |
| CUDA Toolkit ✓ | CUDA-kirjastot paikannettu `CUDA_PATH`-muuttujan avulla |
| cuDNN ✓ | cuDNN-ajonaikaiset DLL-tiedostot löydetty |
| CUDA Acceleration ✓ | ONNX Runtime latasi CUDA-suorituspalveluntarjoajan |

Jos jokin kohde puuttuu asennuksen jälkeen, napsauta `Re-check` suorittaaksesi laitteistontunnistuksen uudelleen käynnistämättä sovellusta.

Asetusikkuna tarjoaa myös suorat latauslinkit CUDA Toolkitille ja cuDNN:lle, jos niitä ei ole vielä asennettu.

### Vianmääritys

Jos `CUDA Acceleration` ei näytä valintamerkkiä, varmista seuraavat asiat:

- `CUDA_PATH`-ympäristömuuttuja on asetettu (tarkista `System > Advanced system settings > Environment Variables`).
- cuDNN DLL-tiedostot sijaitsevat hakemistossa, joka on järjestelmäsi `PATH`-muuttujassa, tai CUDA:n `bin`-kansiossa.
- GPU-ajurisi on ajan tasalla.

### Eräkoko

Kun CUDA on aktiivinen, **Hardware & Performance** -osiossa näkyy myös nykyinen dynaaminen eräkaton arvo — enimmäismäärä sekunteja, jotka käsitellään yhdessä GPU-ajossa. Tämä arvo lasketaan vapaasta VRAM-muistista mallien lataamisen jälkeen ja mukautuu automaattisesti, jos käytettävissä oleva muisti muuttuu.

## Käyttö ilman GPU:ta

Jos CUDA ei ole käytettävissä, Parakeet siirtyy automaattisesti käyttämään suoritinta. Transkriptio toimii edelleen, mutta on hitaampaa — erityisesti pitkien äänitiedostojen kohdalla.

---