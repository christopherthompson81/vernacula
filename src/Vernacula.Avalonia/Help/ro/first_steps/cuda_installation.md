---
title: "Instalarea CUDA și cuDNN pentru accelerare GPU"
description: "Cum să configurați NVIDIA CUDA și cuDNN astfel încât Vernacula-Desktop să poată utiliza placa dvs. grafică."
topic_id: first_steps_cuda_installation
---

# Instalarea CUDA și cuDNN pentru accelerare GPU

Vernacula-Desktop poate utiliza un GPU NVIDIA pentru a accelera semnificativ transcrierea. Accelerarea GPU necesită instalarea pe sistemul dvs. a NVIDIA CUDA Toolkit și a bibliotecilor de execuție cuDNN.

## Cerințe

- Un GPU NVIDIA care acceptă CUDA (se recomandă GeForce GTX seria 10 sau mai recent).
- Windows 10 sau 11 (64 de biți).
- Fișierele de model trebuie să fie deja descărcate. Consultați [Descărcarea modelelor](downloading_models.md).

## Pași de instalare

### 1. Instalați CUDA Toolkit

Descărcați și rulați programul de instalare CUDA Toolkit de pe site-ul pentru dezvoltatori NVIDIA. În timpul instalării, acceptați căile implicite. Programul de instalare setează automat variabila de mediu `CUDA_PATH` — Vernacula-Desktop utilizează această variabilă pentru a localiza bibliotecile CUDA.

### 2. Instalați cuDNN

Descărcați arhiva ZIP cuDNN pentru versiunea CUDA instalată de pe site-ul pentru dezvoltatori NVIDIA. Extrageți arhiva și copiați conținutul folderelor `bin`, `include` și `lib` în folderele corespunzătoare din directorul de instalare al CUDA Toolkit (calea indicată de `CUDA_PATH`).

Alternativ, instalați cuDNN folosind programul de instalare NVIDIA cuDNN, dacă acesta este disponibil pentru versiunea dvs. de CUDA.

### 3. Reporniți aplicația

Închideți și redeschideți Vernacula-Desktop după instalare. Aplicația verifică prezența CUDA la pornire.

## Starea GPU în Setări

Deschideți `Settings…` din bara de meniu și consultați secțiunea **Hardware & Performance**. Fiecare componentă afișează o bifă (✓) atunci când este detectată:

| Element | Semnificație |
|---|---|
| Numele GPU și VRAM | GPU-ul dvs. NVIDIA a fost găsit |
| CUDA Toolkit ✓ | Bibliotecile CUDA au fost localizate prin `CUDA_PATH` |
| cuDNN ✓ | DLL-urile de execuție cuDNN au fost găsite |
| CUDA Acceleration ✓ | ONNX Runtime a încărcat furnizorul de execuție CUDA |

Dacă vreun element lipsește după instalare, faceți clic pe `Re-check` pentru a relua detectarea hardware fără a reporni aplicația.

Fereastra Setări oferă, de asemenea, linkuri directe de descărcare pentru CUDA Toolkit și cuDNN, dacă acestea nu sunt încă instalate.

### Depanare

Dacă `CUDA Acceleration` nu afișează o bifă, verificați că:

- Variabila de mediu `CUDA_PATH` este setată (verificați `System > Advanced system settings > Environment Variables`).
- DLL-urile cuDNN se află într-un director inclus în `PATH`-ul sistemului sau în folderul `bin` al CUDA.
- Driverul GPU-ului dvs. este actualizat.

### Dimensionarea loturilor

Când CUDA este activ, secțiunea **Hardware & Performance** afișează și plafonul dinamic curent al loturilor — numărul maxim de secunde de audio procesate într-o singură rulare pe GPU. Această valoare este calculată pe baza VRAM-ului disponibil după încărcarea modelelor și se ajustează automat dacă memoria disponibilă se modifică.

## Rulare fără GPU

Dacă CUDA nu este disponibil, Vernacula-Desktop comută automat la procesarea pe CPU. Transcrierea funcționează în continuare, dar va fi mai lentă, în special pentru fișierele audio lungi.

---