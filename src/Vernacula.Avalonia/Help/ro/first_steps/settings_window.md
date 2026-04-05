---
title: "Setări"
description: "Prezentare generală a tuturor opțiunilor din fereastra Setări."
topic_id: first_steps_settings_window
---

# Setări

Fereastra **Setări** vă oferă control asupra configurației hardware, gestionării modelelor, modului de segmentare, comportamentului editorului, aspectului vizual și limbii. Deschideți-o din bara de meniu: `Settings…`.

## Hardware și Performanță

Această secțiune afișează starea GPU-ului NVIDIA și a stivei software CUDA, precum și pragul dinamic al lotului utilizat în timpul transcrierii pe GPU.

| Element | Descriere |
|---|---|
| Numele GPU-ului și VRAM | GPU-ul NVIDIA detectat și memoria video disponibilă. |
| CUDA Toolkit | Dacă bibliotecile de rulare CUDA au fost găsite prin `CUDA_PATH`. |
| cuDNN | Dacă fișierele DLL de rulare cuDNN sunt disponibile. |
| Accelerare CUDA | Dacă ONNX Runtime a încărcat cu succes furnizorul de execuție CUDA. |

Faceți clic pe `Re-check` pentru a reporni detectarea hardware fără a reporni aplicația — util după instalarea CUDA sau cuDNN.

Linkuri de descărcare directă pentru CUDA Toolkit și cuDNN sunt afișate atunci când aceste componente nu sunt detectate.

Mesajul privind **pragul lotului** raportează câte secunde de audio sunt procesate în fiecare rulare pe GPU. Această valoare este derivată din VRAM-ul liber după încărcarea modelelor și se ajustează automat.

Pentru instrucțiuni complete de configurare CUDA, consultați [Instalarea CUDA și cuDNN](cuda_installation.md).

## Modele

Această secțiune gestionează fișierele modelelor AI necesare pentru transcriere.

- **Precizia modelului** — alegeți `INT8 (smaller download)` sau `FP32 (more accurate)`. Consultați [Alegerea preciziei greutăților modelului](model_precision.md).
- **Descărcarea modelelor lipsă** — descarcă orice fișiere de model care nu se află încă pe disc. O bară de progres și o linie de stare urmăresc fiecare fișier pe măsură ce se descarcă.
- **Verificarea actualizărilor** — verifică dacă sunt disponibile greutăți de model mai noi. Un banner de actualizare apare, de asemenea, automat pe ecranul principal când sunt detectate greutăți actualizate.

## Modul de Segmentare

Controlează modul în care audioul este împărțit în segmente înainte de recunoașterea vorbirii.

| Mod | Descriere |
|---|---|
| **Diarizare după vorbitor** | Utilizează modelul SortFormer pentru a identifica vorbitorii individuali și a eticheta fiecare segment. Cel mai potrivit pentru interviuri, întâlniri și înregistrări cu mai mulți vorbitori. |
| **Detectarea activității vocale** | Utilizează Silero VAD pentru a detecta doar regiunile de vorbire — fără etichete de vorbitor. Mai rapidă decât diarizarea și bine adaptată pentru audioul cu un singur vorbitor. |

## Editorul de transcriere

**Modul de redare implicit** — setează modul de redare utilizat când deschideți editorul de transcriere. Îl puteți modifica și direct în editor în orice moment. Consultați [Editarea transcrierilor](../operations/editing_transcripts.md) pentru o descriere a fiecărui mod.

## Aspect vizual

Selectați tema **Întunecat** sau **Deschis**. Modificarea se aplică imediat. Consultați [Alegerea unei teme](theme.md).

## Limbă

Selectați limba de afișare pentru interfața aplicației. Modificarea se aplică imediat. Consultați [Alegerea unei limbi](language.md).

---