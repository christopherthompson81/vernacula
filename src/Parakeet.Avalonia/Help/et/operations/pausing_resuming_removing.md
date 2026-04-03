---
title: "Tööde peatamine, jätkamine või eemaldamine"
description: "Kuidas peatada töötav töö, jätkata peatatud tööd või kustutada töö ajaloost."
topic_id: operations_pausing_resuming_removing
---

# Tööde peatamine, jätkamine või eemaldamine

## Töö peatamine

Töötavat või järjekorras olevat tööd saab peatada kahest kohast:

- **Edenemisvaade** — klõpsake `Pause` paremas alanurgas, kui jälgite aktiivset tööd.
- **Transkriptsiooniloo tabel** — klõpsake `Pause` **Toimingud** veerus mis tahes real, mille olek on `running` või `queued`.

Pärast `Pause` klõpsamist näitab olekurida `Pausing…`, kuni rakendus lõpetab praeguse töötlusühiku. Seejärel muutub töö olek ajalootabelis `cancelled`.

> Peatamine salvestab kõik seni transkribeeritud segmendid. Tööd saab hiljem jätkata ilma tehtud tööd kaotamata.

## Töö jätkamine

Peatatud või ebaõnnestunud töö jätkamiseks tehke järgmist.

1. Leidke avalehel **Transkriptsiooniloo** tabelist soovitud töö. Selle olek on `cancelled` või `failed`.
2. Klõpsake **Toimingud** veerus `Resume`.
3. Rakendus naaseb **Edenemise** vaatesse ja jätkab sealt, kus töötlemine katkes.

Olekurida näitab töö uuesti käivitumise ajal lühidalt `Resuming…`.

## Töö eemaldamine

Töö ja selle transkriptsiooni jäädavalt ajaloost kustutamiseks tehke järgmist.

1. Klõpsake **Transkriptsiooniloo** tabelis kustutatava töö **Toimingud** veerus `Remove`.

Töö eemaldatakse loendist ja selle andmed kustutatakse kohalikust andmebaasist. Seda toimingut ei saa tagasi võtta. Kettale salvestatud eksporditud faile see ei mõjuta.

---