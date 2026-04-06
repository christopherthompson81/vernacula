---
title: "Iestatījumi"
description: "Visu opciju pārskats logā Iestatījumi."
topic_id: first_steps_settings_window
---

# Iestatījumi

Logs **Iestatījumi** ļauj kontrolēt aparatūras konfigurāciju, modeļu pārvaldību, segmentēšanas režīmu, redaktora uzvedību, izskatu un valodu. Atveriet to no izvēlņu joslas: `Settings…`.

## Aparatūra un veiktspēja

Šajā sadaļā ir redzams jūsu NVIDIA GPU un CUDA programmatūras stāvoklis, kā arī tiek ziņots par dinamisko paketes ierobežojumu, kas tiek izmantots GPU transkripcijas laikā.

| Elements | Apraksts |
|---|---|
| GPU nosaukums un VRAM | Atklātais NVIDIA GPU un pieejamā video atmiņa. |
| CUDA Toolkit | Vai CUDA izpildlaika bibliotēkas tika atrastas, izmantojot `CUDA_PATH`. |
| cuDNN | Vai cuDNN izpildlaika DLL faili ir pieejami. |
| CUDA paātrinājums | Vai ONNX Runtime veiksmīgi ielādēja CUDA izpildes nodrošinātāju. |

Noklikšķiniet uz `Re-check`, lai atkārtoti veiktu aparatūras noteikšanu bez lietojumprogrammas restartēšanas — noderīgi pēc CUDA vai cuDNN instalēšanas.

Ja šie komponenti netiek atklāti, tiek parādītas tiešās lejupielādes saites CUDA Toolkit un cuDNN.

Ziņojums par **paketes ierobežojumu** norāda, cik sekundes audio tiek apstrādātas katrā GPU izpildes ciklā. Šī vērtība tiek aprēķināta, pamatojoties uz brīvo VRAM pēc modeļu ielādes, un tiek pielāgota automātiski.

Pilnus CUDA iestatīšanas norādījumus skatiet šeit: [CUDA un cuDNN instalēšana](cuda_installation.md).

## Modeļi

Šajā sadaļā tiek pārvaldīti AI modeļu faili, kas nepieciešami transkripcijas veikšanai.

- **Lejupielādēt trūkstošos modeļus** — lejupielādē visus modeļu failus, kas vēl nav pieejami diskā. Progresa josla un statusa rindiņa izseko katru failu lejupielādes laikā.
- **Pārbaudīt atjauninājumus** — pārbauda, vai ir pieejami jaunāki modeļu svari. Atjauninājuma paziņojums arī automātiski parādās sākuma ekrānā, kad tiek atklāti atjaunināti svari.

## Segmentēšanas režīms

Nosaka, kā audio tiek sadalīts segmentos pirms runas atpazīšanas.

| Režīms | Apraksts |
|---|---|
| **Runātāju diarizācija** | Izmanto SortFormer modeli, lai identificētu atsevišķus runātājus un marķētu katru segmentu. Vislabāk piemērots intervijām, sanāksmēm un ierakstiem ar vairākiem runātājiem. |
| **Balss aktivitātes noteikšana** | Izmanto Silero VAD, lai noteiktu tikai runas apgabalus — bez runātāju marķējumiem. Ātrāka par diarizāciju un labi piemērota audio ar vienu runātāju. |

## Transkripcijas redaktors

**Noklusējuma atskaņošanas režīms** — iestata atskaņošanas režīmu, kas tiek izmantots, atverot transkripcijas redaktoru. To var mainīt arī tieši redaktorā jebkurā laikā. Katras režīma aprakstu skatiet sadaļā [Transkripciju rediģēšana](../operations/editing_transcripts.md).

## Izskats

Izvēlieties **Tumšo** vai **Gaišo** motīvu. Izmaiņas tiek piemērotas nekavējoties. Skatiet [Motīva izvēle](theme.md).

## Valoda

Izvēlieties lietojumprogrammas saskarnes attēlošanas valodu. Izmaiņas tiek piemērotas nekavējoties. Skatiet [Valodas izvēle](language.md).

---