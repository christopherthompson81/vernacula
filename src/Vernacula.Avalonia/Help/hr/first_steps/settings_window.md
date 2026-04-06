---
title: "Postavke"
description: "Pregled svih opcija u prozoru Postavke."
topic_id: first_steps_settings_window
---

# Postavke

Prozor **Postavke** omogućuje vam kontrolu nad konfiguracijom hardvera, upravljanjem modelima, načinom segmentacije, ponašanjem uređivača, izgledom i jezikom. Otvorite ga putem trake izbornika: `Settings…`.

## Hardver i performanse

Ovaj odjeljak prikazuje status vašeg NVIDIA GPU-a i programskog skupa CUDA, te izvještava o dinamičnom gornjem ograničenju serije koja se koristi tijekom GPU transkripcije.

| Stavka | Opis |
|---|---|
| Naziv GPU-a i VRAM | Otkriveni NVIDIA GPU i dostupna video memorija. |
| CUDA Toolkit | Jesu li CUDA biblioteke izvođenja pronađene putem `CUDA_PATH`. |
| cuDNN | Jesu li cuDNN DLL datoteke izvođenja dostupne. |
| CUDA ubrzanje | Je li ONNX Runtime uspješno učitao CUDA izvršnog davatelja. |

Kliknite `Re-check` za ponovnu provjeru hardvera bez ponovnog pokretanja aplikacije — korisno nakon instalacije CUDA-e ili cuDNN-a.

Izravne veze za preuzimanje CUDA Toolkita i cuDNN-a prikazuju se kada ti komponenti nisu otkriveni.

Poruka o **gornjem ograničenju serije** izvještava koliko sekundi zvuka se obrađuje u svakom GPU pokretanju. Ova vrijednost izvedena je iz slobodnog VRAM-a nakon učitavanja modela i automatski se prilagođava.

Za potpune upute za postavljanje CUDA-e, pogledajte [Instalacija CUDA-e i cuDNN-a](cuda_installation.md).

## Modeli

Ovaj odjeljak upravlja datotekama AI modela potrebnim za transkripciju.

- **Preuzimanje nedostajućih modela** — preuzima sve datoteke modela koje još nisu prisutne na disku. Traka napretka i statusna linija prate svaku datoteku tijekom preuzimanja.
- **Provjera ažuriranja** — provjerava jesu li dostupne novije težine modela. Natpis o ažuriranju također se automatski pojavljuje na početnom zaslonu kada se otkriju ažurirane težine.

## Način segmentacije

Kontrolira kako se zvuk dijeli na segmente prije prepoznavanja govora.

| Način | Opis |
|---|---|
| **Dijarizacija govornika** | Koristi model SortFormer za identifikaciju pojedinih govornika i označavanje svakog segmenta. Najprikladnije za intervjue, sastanke i snimke s više govornika. |
| **Detekcija glasovne aktivnosti** | Koristi Silero VAD za otkrivanje samo govornih područja — bez oznaka govornika. Brže od dijarizacije i pogodno za zvuk s jednim govornikom. |

## Uređivač transkripta

**Zadani način reprodukcije** — postavlja način reprodukcije koji se koristi kada otvorite uređivač transkripta. Možete ga promijeniti i izravno u uređivaču u bilo kojem trenutku. Pogledajte [Uređivanje transkripata](../operations/editing_transcripts.md) za opis svakog načina.

## Izgled

Odaberite **tamnu** ili **svijetlu** temu. Promjena se primjenjuje odmah. Pogledajte [Odabir teme](theme.md).

## Jezik

Odaberite jezik prikaza za sučelje aplikacije. Promjena se primjenjuje odmah. Pogledajte [Odabir jezika](language.md).

---