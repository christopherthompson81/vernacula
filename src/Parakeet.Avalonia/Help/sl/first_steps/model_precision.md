---
title: "Izbira natančnosti uteži modela"
description: "Kako izbrati med natančnostjo modela INT8 in FP32 ter kakšne so prednosti in slabosti posamezne možnosti."
topic_id: first_steps_model_precision
---

# Izbira natančnosti uteži modela

Natančnost modela določa številski format, ki ga AI model uporablja za svoje uteži. Vpliva na velikost prenosa, porabo pomnilnika in točnost prepoznavanja.

## Možnosti natančnosti

### INT8 (manjši prenos)

- Manjše datoteke modela — hitrejši prenos in manjša poraba prostora na disku.
- Nekoliko nižja točnost pri nekaterih zvočnih posnetkih.
- Priporočeno, če imate omejen prostor na disku ali počasnejšo internetno povezavo.

### FP32 (večja točnost)

- Večje datoteke modela.
- Višja točnost, zlasti pri zahtevnih zvočnih posnetkih z naglasom ali hrupom v ozadju.
- Priporočeno, kadar je točnost najpomembnejša in imate dovolj prostora na disku.
- Obvezno za pospeševanje z GPU CUDA — pot prek GPU vedno uporablja FP32 ne glede na to nastavitev.

## Kako spremeniti natančnost

Odprite `Settings…` v menijski vrstici, nato pojdite v razdelek **Models** in izberite `INT8 (smaller download)` ali `FP32 (more accurate)`.

## Po spremembi natančnosti

Sprememba natančnosti zahteva drugačen nabor datotek modela. Če datoteke za novo natančnost še niso bile prenesene, kliknite `Download Missing Models` v nastavitvah. Predhodno prenesene datoteke za drugo natančnost ostanejo na disku in jih pri morebitni vrnitvi ni treba znova prenesti.

---