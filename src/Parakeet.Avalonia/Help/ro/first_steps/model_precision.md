---
title: "Alegerea Preciziei Ponderilor Modelului"
description: "Cum să alegi între precizia INT8 și FP32 a modelului și care sunt compromisurile."
topic_id: first_steps_model_precision
---

# Alegerea Preciziei Ponderilor Modelului

Precizia modelului controlează formatul numeric utilizat de ponderile modelului AI. Aceasta influențează dimensiunea descărcării, utilizarea memoriei și acuratețea.

## Opțiuni de Precizie

### INT8 (descărcare mai mică)

- Fișiere de model mai mici — descărcare mai rapidă și spațiu pe disc mai redus necesar.
- Acuratețe ușor mai scăzută pentru anumite materiale audio.
- Recomandat dacă ai spațiu limitat pe disc sau o conexiune la internet mai lentă.

### FP32 (mai precis)

- Fișiere de model mai mari.
- Acuratețe mai ridicată, în special pentru audio dificil cu accente sau zgomot de fundal.
- Recomandat când acuratețea este prioritară și ai suficient spațiu pe disc.
- Necesar pentru accelerarea GPU CUDA — calea GPU utilizează întotdeauna FP32, indiferent de această setare.

## Cum să Schimbi Precizia

Deschide `Settings…` din bara de meniu, apoi accesează secțiunea **Models** și selectează fie `INT8 (smaller download)`, fie `FP32 (more accurate)`.

## După Schimbarea Preciziei

Schimbarea preciziei necesită un set diferit de fișiere de model. Dacă fișierele pentru noua precizie nu au fost încă descărcate, apasă `Download Missing Models` în Setări. Fișierele descărcate anterior pentru cealaltă precizie sunt păstrate pe disc și nu trebuie descărcate din nou dacă revii la aceasta.

---