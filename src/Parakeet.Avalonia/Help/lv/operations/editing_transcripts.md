---
title: "Transkriptu rediģēšana"
description: "Kā pārskatīt, labot un verificēt transkribētos segmentus transkriptu redaktorā."
topic_id: operations_editing_transcripts
---

# Transkriptu rediģēšana

**Transkriptu redaktors** ļauj pārskatīt ASR izvadi, labot tekstu, pārdēvēt runātājus tiešrindē, pielāgot segmentu laikus un atzīmēt segmentus kā verificētus — visu to darot, vienlaikus klausoties oriģinālo audio.

## Redaktora atvēršana

1. Ielādējiet pabeigtu uzdevumu (skatīt [Pabeigto uzdevumu ielāde](loading_completed_jobs.md)).
2. Skatījumā **Rezultāti** noklikšķiniet uz `Edit Transcript`.

Redaktors atveras kā atsevišķs logs un var palikt atvērts blakus galvenajai lietotnei.

## Izkārtojums

Katrs segments tiek attēlots kā karte ar divām blakus esošām rūtīm:

- **Kreisā rūts** — oriģinālā ASR izvade ar vārdu ticamības krāsojumu. Vārdi, par kuriem modelis bija mazāk pārliecināts, parādās sarkanā krāsā; vārdi ar augstu ticamību parādās parastajā teksta krāsā.
- **Labā rūts** — rediģējams teksta lauks. Veiciet labojumus šeit; rakstot tiek izceltas atšķirības salīdzinājumā ar oriģinālu.

Virs katras kartes tiek rādīta runātāja etiķete un laika diapazons. Noklikšķiniet uz kartes, lai fokusētu to un atklātu tās darbību ikonas. Virziet peli virs jebkuras ikonas, lai redzētu rīka padomu ar tās funkcijas aprakstu.

## Ikonu apraksts

### Atskaņošanas josla

| Ikona | Darbība |
|-------|---------|
| ▶ | Atskaņot |
| ⏸ | Pauzēt |
| ⏮ | Pārlēkt uz iepriekšējo segmentu |
| ⏭ | Pārlēkt uz nākamo segmentu |

### Segmenta kartes darbības

| Ikona | Darbība |
|-------|---------|
| <mdl2 ch="E77B"/> | Pārvietot segmentu citam runātājam |
| <mdl2 ch="E916"/> | Pielāgot segmenta sākuma un beigu laiku |
| <mdl2 ch="EA39"/> | Slēpt vai atslēpt segmentu |
| <mdl2 ch="E72B"/> | Apvienot ar iepriekšējo segmentu |
| <mdl2 ch="E72A"/> | Apvienot ar nākamo segmentu |
| <mdl2 ch="E8C6"/> | Sadalīt segmentu |
| <mdl2 ch="E72C"/> | Atkārtoti veikt ASR šim segmentam |

## Audio atskaņošana

Redaktora loga augšdaļā atrodas atskaņošanas josla:

| Vadīkla | Darbība |
|---------|---------|
| Atskaņošanas/pauzes ikona | Sākt vai pauzēt atskaņošanu |
| Meklēšanas josla | Velciet, lai pārietu uz jebkuru audio pozīciju |
| Ātruma slīdnis | Pielāgot atskaņošanas ātrumu (0,5× – 2×) |
| Iepriekšējā/nākamā ikonas | Pārlēkt uz iepriekšējo vai nākamo segmentu |
| Atskaņošanas režīma nolaižamā izvēlne | Izvēlēties vienu no trim atskaņošanas režīmiem (skatīt zemāk) |
| Skaļuma slīdnis | Pielāgot atskaņošanas skaļumu |

Atskaņošanas laikā pašlaik runātais vārds tiek izcelts kreisajā rūtī. Pēc pauzes apstāšanās meklēšanas pozīcijā izcēlums atjaunojas uz vārdu meklēšanas pozīcijā.

### Atskaņošanas režīmi

| Režīms | Darbība |
|--------|---------|
| `Single` | Atskaņot pašreizējo segmentu vienu reizi, pēc tam apstāties. |
| `Auto-advance` | Atskaņot pašreizējo segmentu; beidzoties atskaņošanai, atzīmēt to kā verificētu un pāriet uz nākamo. |
| `Continuous` | Atskaņot visus segmentus pēc kārtas, neatzīmējot nevienu kā verificētu. |

Aktīvo režīmu izvēlieties no nolaižamās izvēlnes atskaņošanas joslā.

## Segmenta rediģēšana

1. Noklikšķiniet uz kartes, lai to fokusētu.
2. Rediģējiet tekstu labajā rūtī. Izmaiņas tiek saglabātas automātiski, kad pārnesiet fokusu uz citu karti.

## Runātāja pārdēvēšana

Noklikšķiniet uz runātāja etiķetes fokusētajā kartē un ierakstiet jaunu nosaukumu. Nospiediet `Enter` vai noklikšķiniet citur, lai saglabātu. Jaunais nosaukums tiek piemērots tikai šai kartei; lai pārdēvētu runātāju globāli, izmantojiet [Runātāju vārdu rediģēšana](editing_speaker_names.md) no skatījuma Rezultāti.

## Segmenta verificēšana

Noklikšķiniet uz izvēles rūtiņas `Verified` fokusētajā kartē, lai atzīmētu to kā pārskatītu. Verificēšanas statuss tiek saglabāts datubāzē un ir redzams redaktorā turpmākajās ielādēšanas reizēs.

## Segmenta slēpšana

Noklikšķiniet uz `Suppress` fokusētajā kartē, lai paslēptu segmentu no eksportiem (noderīgi troksnim, mūzikai vai citām nerunas daļām). Noklikšķiniet uz `Unsuppress`, lai to atjaunotu.

## Segmenta laiku pielāgošana

Noklikšķiniet uz `Adjust Times` fokusētajā kartē, lai atvērtu laika pielāgošanas dialogu. Izmantojiet ritināšanas riteni virs lauka **Start** vai **End**, lai mainītu vērtību par 0,1 sekundes soļiem, vai ierakstiet vērtību tieši. Noklikšķiniet uz `Save`, lai lietotu.

## Segmentu apvienošana

- Noklikšķiniet uz `⟵ Merge`, lai apvienotu fokusēto segmentu ar tam tieši iepriekšējo segmentu.
- Noklikšķiniet uz `Merge ⟶`, lai apvienotu fokusēto segmentu ar tam tieši nākamo segmentu.

Abas kartes tiek apvienotas ar kopējo tekstu un laika diapazonu. Tas ir noderīgi, kad viena runāta izteikuma daļas tika sadalītas divos segmentos.

## Segmenta sadalīšana

Noklikšķiniet uz `Split…` fokusētajā kartē, lai atvērtu sadalīšanas dialogu. Novietojiet sadalīšanas punktu tekstā un apstipriniet. Tiek izveidoti divi jauni segmenti, kas aptver oriģinālo laika diapazonu. Tas ir noderīgi, kad divi atsevišķi izteikumi tika apvienoti vienā segmentā.

## ASR atkārtota izpilde

Noklikšķiniet uz `Redo ASR` fokusētajā kartē, lai atkārtoti veiktu runas atpazīšanu šī segmenta audio daļā. Modelis apstrādā tikai šā segmenta audio fragmentu un izveido jaunu, viena avota transkripciju.

Izmantojiet šo funkciju, kad:

- Segments radies apvienošanas rezultātā un nevar tikt sadalīts (apvienotie segmenti aptver vairākus ASR avotus; Redo ASR tos sakļauj vienā, pēc kā kļūst pieejama funkcija `Split…`).
- Oriģinālā transkripcija ir slikta un vēlaties veikt tīru otro apstrādi bez manuālas labošanas.

**Piezīme:** Viss teksts, ko jau esat ierakstījis labajā rūtī, tiks dzēsts un aizstāts ar jaunu ASR izvadi. Darbībai nepieciešams, lai audio fails būtu ielādēts; poga ir atspējota, ja audio nav pieejams.