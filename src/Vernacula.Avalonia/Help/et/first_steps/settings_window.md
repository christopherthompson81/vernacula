---
title: "Seaded"
description: "Ülevaade kõigist valikutest aknas Seaded."
topic_id: first_steps_settings_window
---

# Seaded

Aken **Seaded** võimaldab hallata riistvara seadistust, mudeleid, segmenteerimisrežiimi, redaktori käitumist, välimust ja keelt. Avage see menüüribalt: `Settings…`.

## Riistvara ja jõudlus

See jaotis näitab teie NVIDIA GPU ja CUDA tarkvarapinu olekut ning kuvab GPU transkriptsioonil kasutatava dünaamilise partiimahu ülempiiri.

| Üksus | Kirjeldus |
|---|---|
| GPU nimi ja VRAM | Tuvastatud NVIDIA GPU ja saadaolev videomälu. |
| CUDA Toolkit | Kas CUDA käitusajateegid leiti `CUDA_PATH` kaudu. |
| cuDNN | Kas cuDNN käitusaja DLL-failid on saadaval. |
| CUDA kiirendus | Kas ONNX Runtime laadis CUDA täideviija edukalt. |

Klõpsake `Re-check`, et käivitada riistvara tuvastamine uuesti ilma rakendust taaskäivitamata — kasulik pärast CUDA või cuDNN installimist.

Kui vastavaid komponente ei tuvastata, kuvatakse otselingid CUDA Toolkiti ja cuDNN allalaadimiseks.

**Partiimahu ülempiiri** teade näitab, mitu sekundit heli töödeldakse ühes GPU käivituses. See väärtus arvutatakse mudelite laadimise järel vabaks jääva VRAM-i põhjal ning kohandub automaatselt.

Täielikud CUDA seadistusjuhised leiate siit: [CUDA ja cuDNN installimine](cuda_installation.md).

## Mudelid

See jaotis haldab transkriptsiooniks vajalikke AI-mudelite faile.

- **Mudeli täpsus** — valige `INT8 (smaller download)` või `FP32 (more accurate)`. Vt [Mudeli kaalu täpsuse valimine](model_precision.md).
- **Puuduvate mudelite allalaadimine** — laadib alla mudelite failid, mida kettal veel ei ole. Edenemisriba ja olekurida jälgivad iga faili allalaadimist.
- **Uuenduste kontrollimine** — kontrollib, kas saadaval on uuemad mudelikaalud. Uuenduste bänner ilmub ka automaatselt avakuval, kui uuendatud kaalud tuvastatakse.

## Segmenteerimisrežiim

Määrab, kuidas heli enne kõnetuvastust segmentideks jagatakse.

| Režiim | Kirjeldus |
|---|---|
| **Kõneleja diariseerimine** | Kasutab SortFormeri mudelit üksikute kõnelejate tuvastamiseks ja iga segmendi märgistamiseks. Sobib kõige paremini intervjuude, koosolekute ja mitme kõnelejaga salvestiste jaoks. |
| **Kõneaktiivsuse tuvastamine** | Kasutab Silero VAD-i kõnepiirkondade tuvastamiseks — kõnelejate silte ei lisata. Kiirem kui diariseerimine ja sobib hästi ühe kõnelejaga helisalvestiste puhul. |

## Transkriptsiooni redaktor

**Vaikimisi taasesitusrežiim** — määrab taasesitusrežiimi, mida kasutatakse transkriptsiooni redaktori avamisel. Saate seda igal ajal ka otse redaktoris muuta. Iga režiimi kirjeldust vt [Transkriptsioonide redigeerimine](../operations/editing_transcripts.md).

## Välimus

Valige **tume** või **hele** teema. Muudatus rakendub kohe. Vt [Teema valimine](theme.md).

## Keel

Valige rakenduse kasutajaliidese kuvakeel. Muudatus rakendub kohe. Vt [Keele valimine](language.md).

---