---
title: "Začasna ustavitev, nadaljevanje ali odstranitev poslov"
description: "Kako začasno ustaviti delujoči posel, nadaljevati ustavljenega ali izbrisati posel iz zgodovine."
topic_id: operations_pausing_resuming_removing
---

# Začasna ustavitev, nadaljevanje ali odstranitev poslov

## Začasna ustavitev posla

Delujoči ali čakajoči posel lahko začasno ustavite na dveh mestih:

- **Pogled napredka** — kliknite `Pause` v spodnjem desnem kotu med spremljanjem aktivnega posla.
- **Tabela zgodovine prepisov** — kliknite `Pause` v stolpcu **Actions** v vrstici, katere status je `running` ali `queued`.

Ko kliknete `Pause`, vrstica stanja prikaže `Pausing…`, medtem ko aplikacija dokonča trenutno enoto obdelave. Status posla se nato v tabeli zgodovine spremeni v `cancelled`.

> Začasna ustavitev shrani vse do sedaj prepisane segmente. Posel lahko nadaljujete pozneje, ne da bi izgubili opravljeno delo.

## Nadaljevanje posla

Če želite nadaljevati začasno ustavljen ali neuspešen posel:

1. Na začetnem zaslonu poiščite posel v tabeli **Transcription History**. Njegov status bo `cancelled` ali `failed`.
2. Kliknite `Resume` v stolpcu **Actions**.
3. Aplikacija se vrne na pogled **Progress** in nadaljuje od mesta, kjer se je obdelava ustavila.

Vrstica stanja za kratek čas prikaže `Resuming…`, medtem ko se posel znova inicializira.

## Odstranitev posla

Če želite trajno izbrisati posel in njegov prepis iz zgodovine:

1. V tabeli **Transcription History** kliknite `Remove` v stolpcu **Actions** pri poslu, ki ga želite izbrisati.

Posel je odstranjen s seznama in njegovi podatki so izbrisani iz lokalne podatkovne baze. Tega dejanja ni mogoče razveljaviti. Izvožene datoteke, shranjene na disku, niso prizadete.

---