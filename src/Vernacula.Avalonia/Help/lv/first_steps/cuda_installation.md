---
title: "CUDA un cuDNN instalēšana GPU paātrināšanai"
description: "Kā iestatīt NVIDIA CUDA un cuDNN, lai Vernacula-Desktop varētu izmantot jūsu GPU."
topic_id: first_steps_cuda_installation
---

# CUDA un cuDNN instalēšana GPU paātrināšanai

Vernacula-Desktop var izmantot NVIDIA GPU, lai ievērojami paātrinātu transkripciju. GPU paātrināšanai nepieciešams, lai jūsu sistēmā būtu instalēts NVIDIA CUDA Toolkit un cuDNN izpildlaika bibliotēkas.

## Prasības

- NVIDIA GPU ar CUDA atbalstu (ieteicams GeForce GTX 10. sērija vai jaunāka).
- Windows 10 vai 11 (64 bitu).
- Modeļu failiem jābūt jau lejupielādētiem. Skatiet [Modeļu lejupielāde](downloading_models.md).

## Instalēšanas soļi

### 1. Instalējiet CUDA Toolkit

Lejupielādējiet un palaidiet CUDA Toolkit instalētāju no NVIDIA izstrādātāju vietnes. Instalēšanas laikā akceptējiet noklusējuma ceļus. Instalētājs automātiski iestata `CUDA_PATH` vides mainīgo — Vernacula-Desktop izmanto šo mainīgo, lai atrastu CUDA bibliotēkas.

### 2. Instalējiet cuDNN

Lejupielādējiet cuDNN ZIP arhīvu savam instalētajam CUDA versijam no NVIDIA izstrādātāju vietnes. Izpakojiet arhīvu un nokopējiet tā `bin`, `include` un `lib` mapju saturu attiecīgajās mapēs CUDA Toolkit instalācijas direktorijā (ceļš, ko norāda `CUDA_PATH`).

Alternatīvi, instalējiet cuDNN, izmantojot NVIDIA cuDNN instalētāju, ja tāds ir pieejams jūsu CUDA versijai.

### 3. Restartējiet lietojumprogrammu

Aizveriet un atkārtoti atveriet Vernacula-Desktop pēc instalēšanas. Lietojumprogramma pārbauda CUDA klātbūtni startēšanas laikā.

## GPU statuss iestatījumos

Atveriet `Settings…` no izvēlnes joslas un aplūkojiet sadaļu **Hardware & Performance**. Katrs komponents rāda atzīmi (✓), kad tas tiek konstatēts:

| Elements | Ko tas nozīmē |
|---|---|
| GPU nosaukums un VRAM | Jūsu NVIDIA GPU tika atrasts |
| CUDA Toolkit ✓ | CUDA bibliotēkas atrastas, izmantojot `CUDA_PATH` |
| cuDNN ✓ | cuDNN izpildlaika DLL faili atrasti |
| CUDA Acceleration ✓ | ONNX Runtime ielādēja CUDA izpildes nodrošinātāju |

Ja kāds elements pēc instalēšanas nav redzams, noklikšķiniet uz `Re-check`, lai atkārtoti palaistu aparatūras noteikšanu, nerestartējot lietojumprogrammu.

Iestatījumu logs nodrošina arī tiešas lejupielādes saites CUDA Toolkit un cuDNN, ja tie vēl nav instalēti.

### Problēmu novēršana

Ja `CUDA Acceleration` nerāda atzīmi, pārbaudiet, vai:

- `CUDA_PATH` vides mainīgais ir iestatīts (pārbaudiet `System > Advanced system settings > Environment Variables`).
- cuDNN DLL faili atrodas direktorijā, kas norādīts sistēmas `PATH`, vai CUDA `bin` mapē.
- Jūsu GPU draiveris ir atjaunināts.

### Paketes lielums

Kad CUDA ir aktīvs, sadaļa **Hardware & Performance** rāda arī pašreizējo dinamisko pakešu maksimumu — maksimālo audio sekunžu skaitu, kas tiek apstrādāts vienā GPU izpildes reizē. Šī vērtība tiek aprēķināta no brīvās VRAM pēc modeļu ielādes un automātiski pielāgojas, ja pieejamā atmiņa mainās.

## Darbība bez GPU

Ja CUDA nav pieejams, Vernacula-Desktop automātiski pārslēdzas uz CPU apstrādi. Transkripcija joprojām darbojas, taču būs lēnāka, īpaši gariem audio failiem.

---