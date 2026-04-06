---
title: "Instalacija CUDA i cuDNN za GPU ubrzanje"
description: "Kako postaviti NVIDIA CUDA i cuDNN kako bi Vernacula-Desktop mogao koristiti vaš GPU."
topic_id: first_steps_cuda_installation
---

# Instalacija CUDA i cuDNN za GPU ubrzanje

Vernacula-Desktop može koristiti NVIDIA GPU za značajno ubrzanje transkripcije. GPU ubrzanje zahtijeva instalaciju NVIDIA CUDA Toolkita i cuDNN biblioteka za izvođenje na vašem sustavu.

## Preduvjeti

- NVIDIA GPU koji podržava CUDA (preporučuje se GeForce GTX serija 10 ili novija).
- Windows 10 ili 11 (64-bitni).
- Datoteke modela moraju već biti preuzete. Pogledajte [Preuzimanje modela](downloading_models.md).

## Koraci instalacije

### 1. Instalirajte CUDA Toolkit

Preuzmite i pokrenite instalacijski program CUDA Toolkita s NVIDIA-ine stranice za razvojne programere. Tijekom instalacije prihvatite zadane putanje. Instalacijski program automatski postavlja varijablu okruženja `CUDA_PATH` — Vernacula-Desktop koristi tu varijablu za pronalazak CUDA biblioteka.

### 2. Instalirajte cuDNN

Preuzmite cuDNN ZIP arhivu za vašu instaliranu verziju CUDA-e s NVIDIA-ine stranice za razvojne programere. Raspakirajte arhivu i kopirajte sadržaj mapa `bin`, `include` i `lib` u odgovarajuće mape unutar direktorija instalacije CUDA Toolkita (putanja prikazana varijablom `CUDA_PATH`).

Alternativno, instalirajte cuDNN koristeći NVIDIA-in cuDNN instalacijski program ako je dostupan za vašu verziju CUDA-e.

### 3. Ponovno pokrenite aplikaciju

Zatvorite i ponovo otvorite Vernacula-Desktop nakon instalacije. Aplikacija provjerava dostupnost CUDA-e pri pokretanju.

## Status GPU-a u postavkama

Otvorite `Settings…` iz trake izbornika i pogledajte odjeljak **Hardware & Performance**. Svaka komponenta prikazuje kvačicu (✓) kada je otkrivena:

| Stavka | Značenje |
|---|---|
| Naziv GPU-a i VRAM | Pronađen je vaš NVIDIA GPU |
| CUDA Toolkit ✓ | CUDA biblioteke pronađene putem `CUDA_PATH` |
| cuDNN ✓ | Pronađeni cuDNN DLL-ovi za izvođenje |
| CUDA Acceleration ✓ | ONNX Runtime učitao CUDA davatelja izvođenja |

Ako neka stavka nedostaje nakon instalacije, kliknite `Re-check` kako biste ponovo pokrenuli otkrivanje hardvera bez ponovnog pokretanja aplikacije.

Prozor postavki također pruža izravne poveznice za preuzimanje CUDA Toolkita i cuDNN-a ako još nisu instalirani.

### Rješavanje problema

Ako `CUDA Acceleration` ne prikazuje kvačicu, provjerite sljedeće:

- Je li varijabla okruženja `CUDA_PATH` postavljena (provjerite `System > Advanced system settings > Environment Variables`).
- Nalaze li se cuDNN DLL-ovi u direktoriju na sistemskom `PATH` ili unutar CUDA mape `bin`.
- Je li upravljački program vašeg GPU-a ažuran.

### Veličina grupe

Kada je CUDA aktivan, odjeljak **Hardware & Performance** također prikazuje trenutni dinamički strop grupe — maksimalni broj sekundi zvuka koji se obrađuje u jednom GPU izvođenju. Ta vrijednost izračunava se na temelju slobodnog VRAM-a nakon učitavanja modela i automatski se prilagođava ako se promijeni raspoloživa memorija.

## Rad bez GPU-a

Ako CUDA nije dostupan, Vernacula-Desktop automatski prelazi na CPU obradu. Transkripcija i dalje funkcionira, ali će biti sporija, posebno za dulje zvučne datoteke.

---