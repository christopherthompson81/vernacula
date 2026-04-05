---
title: "Dodajanje več zvočnih datotek v vrsto"
description: "Kako dodati več zvočnih datotek v vrsto opravil naenkrat."
topic_id: operations_bulk_add_jobs
---

# Dodajanje več zvočnih datotek v vrsto

Uporabite **Skupinsko dodajanje opravil**, da v enem koraku dodate več zvočnih ali video datotek v vrsto za prepisovanje. Aplikacija jih obdeluje eno za drugo v vrstnem redu, v katerem so bile dodane.

## Predpogoji

- Vse datoteke modela morajo biti prenesene. Kartica **Stanje modela** mora prikazovati `All N model file(s) present ✓`. Glejte [Prenos modelov](../first_steps/downloading_models.md).

## Kako skupinsko dodati opravila

1. Na začetnem zaslonu kliknite `Bulk Add Jobs`.
2. Odpre se izbirnik datotek. Izberite eno ali več zvočnih ali video datotek — držite `Ctrl` ali `Shift`, da izberete več datotek hkrati.
3. Kliknite **Open**. Vsaka izbrana datoteka se doda v tabelo **Zgodovina prepisovanja** kot ločeno opravilo.

> **Video datoteke z več zvočnimi tokovi:** Če video datoteka vsebuje več kot en zvočni tok (na primer več jezikov ali komentar režiserja), aplikacija samodejno ustvari eno opravilo za vsak tok.

## Imena opravil

Vsako opravilo je samodejno poimenovano po imenu zvočne datoteke. Opravilo lahko kadar koli preimenujete, tako da kliknete njegovo ime v stolpcu **Title** tabele Zgodovina prepisovanja, uredite besedilo in pritisnete `Enter` ali kliknete drugje.

## Obnašanje vrste

- Če trenutno ni nobeno opravilo v teku, se prva datoteka začne takoj, preostale pa so prikazane kot `queued`.
- Če opravilo že poteka, so vse na novo dodane datoteke prikazane kot `queued` in se bodo samodejno začele ena za drugo.
- Za nadzor aktivnega opravila kliknite `Monitor` v njegovem stolpcu **Actions**. Glejte [Nadzor opravil](monitoring_jobs.md).
- Če želite zaustaviti ali odstraniti opravilo v vrsti, preden se začne, uporabite gumba `Pause` ali `Remove` v njegovem stolpcu **Actions**. Glejte [Zaustavitev, nadaljevanje ali odstranitev opravil](pausing_resuming_removing.md).

---