---
title: "Spremljanje opravil"
description: "Kako spremljati napredek tekočega ali čakajočega opravila."
topic_id: operations_monitoring_jobs
---

# Spremljanje opravil

Pogled **Napredek** prikazuje stanje tekočega opravila transkripcije v živo.

## Odpiranje pogleda Napredek

- Ko zaženete novo transkripcijo, vas aplikacija samodejno preusmeri v pogled Napredek.
- Za opravilo, ki že teče ali čaka v vrsti, ga poiščite v tabeli **Zgodovina transkripcij** in kliknite `Monitor` v stolpcu **Dejanja**.

## Branje pogleda Napredek

| Element | Opis |
|---|---|
| Vrstica napredka | Skupni odstotek dokončanja. Nedoločena (animirana) med zagonom ali nadaljevanjem opravila. |
| Oznaka odstotka | Numerični odstotek, prikazan desno od vrstice. |
| Sporočilo o stanju | Trenutna dejavnost — na primer `Audio Analysis` ali `Speech Recognition`. Prikazuje `Waiting in queue…`, če opravilo še ni začeto. |
| Tabela segmentov | Prenos prepisanih segmentov v živo s stolpci **Govornik**, **Začetek**, **Konec** in **Vsebina**. Samodejno se pomika, ko prihajajo novi segmenti. |

## Faze napredka

Prikazane faze so odvisne od **načina segmentacije**, izbranega v nastavitvah.

**Način govornikove diarizacije** (privzeto):

1. **Analiza zvoka** — diarizacija Sortformer se izvaja čez celotno datoteko, da prepozna meje med govorniki. Vrstica lahko ostane blizu 0 %, dokler ta faza ni zaključena.
2. **Prepoznavanje govora** — vsak segment govornika je prepisan. Odstotek enakomerno narašča med to fazo.

**Način zaznavanja glasovne dejavnosti**:

1. **Zaznavanje govornih segmentov** — Silero VAD preišče datoteko in poišče območja z govorom. Ta faza je hitra.
2. **Prepoznavanje govora** — vsako zaznano govorno območje je prepisano.

V obeh načinih se tabela segmentov v živo polni, ko poteka transkripcija.

## Navigacija stran

Kliknite `← Back to Home`, da se vrnete na domači zaslon, ne da bi prekinili opravilo. Opravilo se nadaljuje v ozadju in njegovo stanje se posodablja v tabeli **Zgodovina transkripcij**. Kadarkoli znova kliknite `Monitor`, da se vrnete v pogled Napredek.

---