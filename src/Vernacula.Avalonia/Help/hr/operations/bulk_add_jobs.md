---
title: "Dodavanje više audio datoteka u red"
description: "Kako dodati nekoliko audio datoteka u red zadataka odjednom."
topic_id: operations_bulk_add_jobs
---

# Dodavanje više audio datoteka u red

Koristite **Skupno dodavanje zadataka** kako biste u jednom koraku stavili više audio ili video datoteka u red za transkripciju. Aplikacija ih obrađuje jednu po jednu, redoslijedom kojim su dodane.

## Preduvjeti

- Sve datoteke modela moraju biti preuzete. Kartica **Status modela** mora prikazivati `All N model file(s) present ✓`. Pogledajte [Preuzimanje modela](../first_steps/downloading_models.md).

## Kako skupno dodati zadatke

1. Na početnom zaslonu kliknite `Bulk Add Jobs`.
2. Otvori se odabir datoteka. Odaberite jednu ili više audio ili video datoteka — držite `Ctrl` ili `Shift` za odabir više datoteka.
3. Kliknite **Open**. Svaka odabrana datoteka dodaje se u tablicu **Transcription History** kao zasebni zadatak.

> **Video datoteke s više audio tokova:** Ako video datoteka sadrži više od jednog audio toka (na primjer, više jezika ili komentar redatelja), aplikacija automatski stvara jedan zadatak po toku.

## Nazivi zadataka

Svaki zadatak automatski dobiva naziv prema nazivu odgovarajuće audio datoteke. Zadatak možete preimenovati u bilo kojem trenutku klikom na njegov naziv u stupcu **Title** tablice Transcription History, uređivanjem teksta te pritiskom na `Enter` ili klikom izvan polja.

## Ponašanje reda

- Ako trenutno nije pokrenut nijedan zadatak, prva datoteka počinje se obrađivati odmah, a ostale su prikazane kao `queued`.
- Ako je zadatak već pokrenut, sve novo dodane datoteke prikazuju se kao `queued` i automatski će se pokretati jedna za drugom.
- Za praćenje aktivnog zadatka kliknite `Monitor` u stupcu **Actions**. Pogledajte [Praćenje zadataka](monitoring_jobs.md).
- Za pauziranje ili uklanjanje zadatka iz reda prije pokretanja koristite gumbe `Pause` ili `Remove` u stupcu **Actions**. Pogledajte [Pauziranje, nastavljanje ili uklanjanje zadataka](pausing_resuming_removing.md).

---