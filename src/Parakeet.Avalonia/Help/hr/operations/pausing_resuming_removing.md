---
title: "Pauziranje, nastavljanje ili uklanjanje zadataka"
description: "Kako pauzirati zadatak koji je u tijeku, nastaviti zaustavljeni ili izbrisati zadatak iz povijesti."
topic_id: operations_pausing_resuming_removing
---

# Pauziranje, nastavljanje ili uklanjanje zadataka

## Pauziranje zadatka

Zadatak koji je pokrenut ili čeka u redu možete pauzirati na dva načina:

- **Prikaz napretka** — kliknite `Pause` u donjem desnom kutu dok pratite aktivni zadatak.
- **Tablica povijesti transkripcija** — kliknite `Pause` u stupcu **Actions** u retku čiji je status `running` ili `queued`.

Nakon što kliknete `Pause`, statusna traka prikazuje `Pausing…` dok aplikacija dovršava trenutnu jedinicu obrade. Status zadatka zatim se mijenja u `cancelled` u tablici povijesti.

> Pauziranjem se spremaju svi segmenti transkribirani do tog trenutka. Zadatak možete nastaviti kasnije bez gubitka toga rada.

## Nastavljanje zadatka

Za nastavak pauziranog ili neuspjelog zadatka:

1. Na početnom zaslonu pronađite zadatak u tablici **Transcription History**. Njegov status bit će `cancelled` ili `failed`.
2. Kliknite `Resume` u stupcu **Actions**.
3. Aplikacija se vraća na prikaz **Progress** i nastavlja od mjesta gdje je obrada stala.

Statusna traka nakratko prikazuje `Resuming…` dok se zadatak ponovno inicijalizira.

## Uklanjanje zadatka

Za trajno brisanje zadatka i njegove transkripcije iz povijesti:

1. U tablici **Transcription History** kliknite `Remove` u stupcu **Actions** u retku zadatka koji želite izbrisati.

Zadatak se uklanja s popisa i njegovi podaci brišu se iz lokalne baze podataka. Ova se radnja ne može poništiti. Izvezene datoteke spremljene na disk nisu zahvaćene.

---