---
title: "Transkriptsioonide redigeerimine"
description: "Kuidas vaadata üle, parandada ja kinnitada transkribeeritud segmente transkriptsiooni redigeerimisriistas."
topic_id: operations_editing_transcripts
---

# Transkriptsioonide redigeerimine

**Transkriptsiooni redigeerimisriist** võimaldab vaadata üle ASR-i väljundit, parandada teksti, nimetada kõnelejaid ümber otse redaktoris, kohandada segmentide ajastust ning märkida segmente kinnitatuks — kõike seda algset helifaili kuulates.

## Redaktori avamine

1. Laadi lõpetatud töö (vt [Lõpetatud tööde laadimine](loading_completed_jobs.md)).
2. Klõpsa **Tulemused** vaates nupul `Edit Transcript`.

Redaktor avaneb eraldi aknas ning võib jääda avatuks koos põhirakendusega.

## Paigutus

Iga segment kuvatakse kaardina, millel on kaks kõrvutipaiknevat paneeli:

- **Vasak paneel** — algne ASR-i väljund sõnapõhise usaldusväärsuse värvimisega. Sõnad, milles mudel oli vähem kindel, kuvatakse punaselt; suure usaldusväärsusega sõnad kuvatakse tavalises tekstivärvis.
- **Parem paneel** — redigeeritav tekstiväli. Tee parandused siia; tippimise ajal tõstetakse esile erinevused algse tekstiga võrreldes.

Kõneleja silt ja ajavahemik kuvatakse iga kaardi kohal. Klõpsa kaardil, et see fookusesse seada ja kuvada selle tegevusikoonid. Hõljuta kursorit ikooni kohal, et näha selle funktsiooni kirjeldavat kohtspikrit.

## Ikoonide legend

### Taasesitusriba

| Ikoon | Toiming |
|-------|---------|
| ▶ | Esita |
| ⏸ | Peata |
| ⏮ | Hüppa eelmisele segmendile |
| ⏭ | Hüppa järgmisele segmendile |

### Segmendikaardi toimingud

| Ikoon | Toiming |
|-------|---------|
| <mdl2 ch="E77B"/> | Määra segment ümber teisele kõnelejale |
| <mdl2 ch="E916"/> | Kohanda segmendi algus- ja lõpuaega |
| <mdl2 ch="EA39"/> | Peida segment või taasta selle nähtavus |
| <mdl2 ch="E72B"/> | Ühenda eelmise segmendiga |
| <mdl2 ch="E72A"/> | Ühenda järgmise segmendiga |
| <mdl2 ch="E8C6"/> | Tükelda segment |
| <mdl2 ch="E72C"/> | Käivita sellel segmendil ASR uuesti |

## Heli taasesitus

Taasesitusriba asub redaktori akna ülaosas:

| Juhtelement | Toiming |
|-------------|---------|
| Esita / Peata ikoon | Alusta või peata taasesitus |
| Otsimisriba | Lohista, et hüpata helifaili suvalisse kohta |
| Kiiruse liugur | Kohanda taasesituse kiirust (0,5× – 2×) |
| Eelmine / Järgmine ikoonid | Hüppa eelmisele või järgmisele segmendile |
| Taasesitusrežiimi rippmenüü | Vali kolme taasesitusrežiimi seast (vt allpool) |
| Helitugevuse liugur | Kohanda taasesituse helitugevust |

Taasesituse ajal tõstetakse vasakus paneelil esile parasjagu kõneldav sõna. Kui taasesitus on peatatud pärast otsimist, uuendatakse esiletõstus otsimisasendi järgi.

### Taasesitusrežiimid

| Režiim | Käitumine |
|--------|-----------|
| `Single` | Esita praegune segment üks kord, seejärel peata. |
| `Auto-advance` | Esita praegune segment; kui see lõpeb, märgi see kinnitatuks ja liigu järgmisele. |
| `Continuous` | Esita kõik segmendid järjest ilma ühtegi kinnitatuks märkimata. |

Vali aktiivne režiim taasesitusriba rippmenüüst.

## Segmendi redigeerimine

1. Klõpsa kaardil, et see fookusesse seada.
2. Redigeeri teksti paremas paneelis. Muudatused salvestatakse automaatselt, kui liigud fookuse mõnele teisele kaardile.

## Kõneleja ümbernimetamine

Klõpsa fookuses oleval kaardil kõneleja sildil ja sisesta uus nimi. Vajuta `Enter` või klõpsa mujale, et salvestada. Uus nimi rakendatakse ainult sellele kaardile; kõneleja globaalseks ümbernimetamiseks kasuta tulemuste vaates [Kõnelejate nimede redigeerimine](editing_speaker_names.md).

## Segmendi kinnitamine

Klõpsa fookuses oleval kaardil märkeruudul `Verified`, et märkida see üle vaadatuks. Kinnitamise olek salvestatakse andmebaasi ja on redaktoris nähtav ka järgmistel laadimiskordadel.

## Segmendi peitmine

Klõpsa fookuses oleval kaardil nupul `Suppress`, et peita segment ekspordist (kasulik müra, muusika või muude kõneväliste osade puhul). Klõpsa `Unsuppress`, et taastada selle nähtavus.

## Segmendi aegade kohandamine

Klõpsa fookuses oleval kaardil nupul `Adjust Times`, et avada aja kohandamise dialoog. Kasuta hiire kerimisratast **Start** või **End** välja kohal, et nihutada väärtust 0,1-sekundiste sammude kaupa, või sisesta väärtus otse. Klõpsa `Save`, et rakendada muudatused.

## Segmentide ühendamine

- Klõpsa `⟵ Merge`, et ühendada fookuses olev segment vahetult eelneva segmendiga.
- Klõpsa `Merge ⟶`, et ühendada fookuses olev segment vahetult järgneva segmendiga.

Mõlema kaardi tekst ja ajavahemik liidetakse kokku. See on kasulik, kui üks kõneütlus on jaotatud kahe segmendi vahel.

## Segmendi tükeldamine

Klõpsa fookuses oleval kaardil nupul `Split…`, et avada tükeldamise dialoog. Määra teksti sees tükelduskoht ja kinnita. Luuakse kaks uut segmenti, mis katavad algse ajavahemiku. See on kasulik, kui kaks eraldiseisvat kõneütlust on ühendatud üheks segmendiks.

## ASR-i uuesti käivitamine

Klõpsa fookuses oleval kaardil nupul `Redo ASR`, et käivitada kõnetuvastus selle segmendi heli jaoks uuesti. Mudel töötleb ainult selle segmendi helilõiku ja koostab uue, üheallikase transkriptsiooni.

Kasuta seda, kui:

- Segment on tekkinud ühendamisest ega ole tükeldatav (ühendatud segmendid hõlmavad mitut ASR-i allikat; ASR-i uuesti käivitamine ühendab need üheks, misjärel muutub `Split…` kättesaadavaks).
- Algne transkriptsioon on kehv ja soovid puhast teist läbimist ilma käsitsi redigeerimata.

**Märkus:** Kõik paremas paneelis juba sisestatud tekstid kustutatakse ja asendatakse uue ASR-i väljundiga. Toiming nõuab helifaili laadimist; nupp on keelatud, kui heli pole saadaval.