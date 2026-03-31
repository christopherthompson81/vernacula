---
title: "Töiden seuranta"
description: "Käynnissä olevan tai jonossa odottavan työn edistymisen seuraaminen."
topic_id: operations_monitoring_jobs
---

# Töiden seuranta

**Edistyminen**-näkymä näyttää käynnissä olevan litterointityön tilan reaaliajassa.

## Edistyminen-näkymän avaaminen

- Kun aloitat uuden litteroinnin, sovellus siirtyy automaattisesti Edistyminen-näkymään.
- Jos työ on jo käynnissä tai jonossa, etsi se **Litterointihistoria**-taulukosta ja napsauta `Monitor` sen **Toiminnot**-sarakkeessa.

## Edistyminen-näkymän lukeminen

| Elementti | Kuvaus |
|---|---|
| Edistymispalkki | Kokonaisvalmiusprosentti. Animoitu (epämääräinen) työn käynnistyessä tai jatkuessa. |
| Prosenttimerkintä | Numeerinen prosentti näytetään palkin oikealla puolella. |
| Tilaviesti | Nykyinen toiminto — esimerkiksi `Audio Analysis` tai `Speech Recognition`. Näyttää `Waiting in queue…`, jos työtä ei ole vielä aloitettu. |
| Segmenttitaulukko | Reaaliaikainen syöte litteroiduista segmenteistä, jossa on **Puhuja**-, **Alku**-, **Loppu**- ja **Sisältö**-sarakkeet. Vierittää automaattisesti uusien segmenttien saapuessa. |

## Edistymisen vaiheet

Näytettävät vaiheet riippuvat Asetuksissa valitusta **Segmentointitilasta**.

**Puhujaerottelu-tila** (oletus):

1. **Audio Analysis** — SortFormer-puhujaerottelu käy koko tiedoston läpi tunnistaakseen puhujien rajat. Palkki saattaa pysyä lähellä 0 %:a, kunnes tämä vaihe on valmis.
2. **Speech Recognition** — jokainen puhujasegmentti litteroidaan. Prosentti nousee tasaisesti tässä vaiheessa.

**Puheaktiivisuuden tunnistus -tila**:

1. **Detecting speech segments** — Silero VAD skannaa tiedoston löytääkseen puhetta sisältävät kohdat. Tämä vaihe on nopea.
2. **Speech Recognition** — jokainen tunnistettu puhealue litteroidaan.

Molemmissa tiloissa reaaliaikainen segmenttitaulukko täyttyy litteroinnin edetessä.

## Näkymästä poistuminen

Napsauta `← Back to Home` palataksesi aloitusnäyttöön keskeyttämättä työtä. Työ jatkuu taustalla ja sen tila päivittyy **Litterointihistoria**-taulukossa. Napsauta `Monitor` milloin tahansa palataksesi Edistyminen-näkymään.

---