---
title: "Transkriptų redagavimas"
description: "Kaip peržiūrėti, taisyti ir patvirtinti transkribuotus segmentus transkriptų redaktoriuje."
topic_id: operations_editing_transcripts
---

# Transkriptų redagavimas

**Transkriptų redaktorius** leidžia peržiūrėti ASR išvestį, taisyti tekstą, pervardyti kalbėtojus tiesiogiai, koreguoti segmentų laiką ir žymėti segmentus kaip patikrintus — visa tai klausantis originalaus garso.

## Redaktoriaus atidarymas

1. Įkelkite užbaigtą užduotį (žr. [Užbaigtų užduočių įkėlimas](loading_completed_jobs.md)).
2. **Rezultatų** rodinyje spustelėkite `Edit Transcript`.

Redaktorius atsidaro kaip atskiras langas ir gali likti atidarytas šalia pagrindinės programos.

## Išdėstymas

Kiekvienas segmentas rodomas kaip kortelė su dviem greta esančiomis sritimis:

- **Kairė sritis** — originali ASR išvestis su kiekvieno žodžio patikimumo spalvinimu. Žodžiai, kuriais modelis buvo mažiau tikras, pažymimi raudonai; didelio patikimumo žodžiai rodomi įprastos teksto spalvos.
- **Dešinė sritis** — redaguojamas teksto laukas. Taisymus atlikite čia; skirtumai, palyginti su originalu, išryškinami rašant.

Kalbėtojo etiketė ir laiko intervalas rodomi virš kiekvienos kortelės. Spustelėkite kortelę, kad ją sufokusuotumėte ir atskleistumėte jos veiksmų piktogramas. Užveskite pelę ant bet kurios piktogramos, kad pamatytumėte aprašomąjį patarimo tekstą.

## Piktogramų paaiškinimas

### Atkūrimo juosta

| Piktograma | Veiksmas |
|------------|----------|
| ▶ | Leisti |
| ⏸ | Pristabdyti |
| ⏮ | Pereiti į ankstesnį segmentą |
| ⏭ | Pereiti į kitą segmentą |

### Segmento kortelės veiksmai

| Piktograma | Veiksmas |
|------------|----------|
| <mdl2 ch="E77B"/> | Priskirti segmentą kitam kalbėtojui |
| <mdl2 ch="E916"/> | Koreguoti segmento pradžios ir pabaigos laiką |
| <mdl2 ch="EA39"/> | Slėpti arba rodyti segmentą |
| <mdl2 ch="E72B"/> | Sujungti su ankstesniu segmentu |
| <mdl2 ch="E72A"/> | Sujungti su kitu segmentu |
| <mdl2 ch="E8C6"/> | Padalyti segmentą |
| <mdl2 ch="E72C"/> | Pakartotinai vykdyti ASR šiam segmentui |

## Garso atkūrimas

Atkūrimo juosta driekiasi per visą redaktoriaus lango viršų:

| Valdiklis | Veiksmas |
|-----------|----------|
| Leisti / Pristabdyti piktograma | Pradėti arba pristabdyti atkūrimą |
| Peršokimo juosta | Vilkite, kad peršoktumėte į bet kurią garso vietą |
| Greičio slankiklis | Koreguoti atkūrimo greitį (0,5× – 2×) |
| Ankstesnio / Kito piktogramos | Pereiti į ankstesnį arba kitą segmentą |
| Atkūrimo režimo išskleidžiamasis sąrašas | Pasirinkti vieną iš trijų atkūrimo režimų (žr. toliau) |
| Garsumo slankiklis | Koreguoti atkūrimo garsumą |

Leidžiant, šiuo metu tariamas žodis išryškinamas kairiajame skydelyje. Pristabdžius po peršokimo, išryškinimas atnaujinamas ties peršokimo padėties žodžiu.

### Atkūrimo režimai

| Režimas | Elgesys |
|---------|---------|
| `Single` | Leisti esamą segmentą vieną kartą, tada sustoti. |
| `Auto-advance` | Leisti esamą segmentą; jam pasibaigus, pažymėti kaip patikrintą ir pereiti prie kito. |
| `Continuous` | Leisti visus segmentus iš eilės nepažymint nė vieno kaip patikrinto. |

Aktyvų režimą pasirinkite iš atkūrimo juostos išskleidžiamojo sąrašo.

## Segmento redagavimas

1. Spustelėkite kortelę, kad ją sufokusuotumėte.
2. Redaguokite tekstą dešiniajame skydelyje. Pakeitimai išsaugomi automatiškai, kai perkeliate fokusą į kitą kortelę.

## Kalbėtojo pervadinimas

Spustelėkite kalbėtojo etiketę sufokusuotoje kortelėje ir įveskite naują vardą. Paspauskite `Enter` arba spustelėkite kitur, kad išsaugotumėte. Naujas vardas taikomas tik tai kortelei; norėdami pervardyti kalbėtoją visoje transkriptoje, naudokite [Kalbėtojų vardų redagavimas](editing_speaker_names.md) rezultatų rodinyje.

## Segmento patvirtinimas

Spustelėkite `Verified` žymimąjį langelį sufokusuotoje kortelėje, kad pažymėtumėte jį kaip peržiūrėtą. Patvirtinimo būsena išsaugoma duomenų bazėje ir matoma redaktoriuje vėliau įkeliant.

## Segmento slėpimas

Spustelėkite `Suppress` sufokusuotoje kortelėje, kad paslėptumėte segmentą nuo eksportų (naudinga triukšmui, muzikai ar kitiems nekalbiniams skyriams). Spustelėkite `Unsuppress`, kad atkurtumėte jį.

## Segmento laiko koregavimas

Spustelėkite `Adjust Times` sufokusuotoje kortelėje, kad atidarytumėte laiko koregavimo dialogą. Sukite pelės ratuką virš **Pradžios** arba **Pabaigos** lauko, kad keistumėte reikšmę 0,1 sekundės žingsniais, arba įveskite reikšmę tiesiogiai. Spustelėkite `Save`, kad pritaikytumėte.

## Segmentų sujungimas

- Spustelėkite `⟵ Merge`, kad sujungtumėte sufokusuotą segmentą su prieš jį esančiu segmentu.
- Spustelėkite `Merge ⟶`, kad sujungtumėte sufokusuotą segmentą su po jo esančiu segmentu.

Abiejų kortelių tekstas ir laiko intervalas sujungiami. Tai naudinga, kai vienas ištartas sakinys buvo padalytas į du segmentus.

## Segmento padalijimas

Spustelėkite `Split…` sufokusuotoje kortelėje, kad atidarytumėte padalijimo dialogą. Nustatykite padalijimo tašką tekste ir patvirtinkite. Sukuriami du nauji segmentai, apimantys originalų laiko intervalą. Tai naudinga, kai dvi atskiros ištartys buvo sujungtos į vieną segmentą.

## ASR pakartojimas

Spustelėkite `Redo ASR` sufokusuotoje kortelėje, kad iš naujo vykdytumėte kalbos atpažinimą to segmento garsui. Modelis apdoroja tik to segmento garso atkarpą ir sukuria naują, vieno šaltinio transkripciją.

Naudokite šią funkciją, kai:

- Segmentas atsirado po sujungimo ir negali būti padalytas (sujungti segmentai apima kelis ASR šaltinius; ASR pakartojimas sujungia juos į vieną, po kurio `Split…` tampa prieinamas).
- Originali transkripcija yra prasta ir norite gauti švarų antrą pasą neneredaguodami rankiniu būdu.

**Pastaba:** Visas tekstas, kurį jau įvedėte dešiniajame skydelyje, bus atmestas ir pakeistas nauja ASR išvestimi. Operacija reikalauja, kad garso failas būtų įkeltas; mygtukas yra neaktyvus, jei garsas nepasiekiamas.