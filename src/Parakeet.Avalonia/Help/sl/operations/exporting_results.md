---
title: "Izvoz rezultatov ali prepisov"
description: "Kako shraniti prepis v datoteko v različnih formatih."
topic_id: operations_exporting_results
---

# Izvoz rezultatov ali prepisov

Dokončan prepis lahko izvozite v več formatih datotek za uporabo v drugih aplikacijah.

## Kako izvoziti

1. Odprite dokončano opravilo. Glejte [Nalaganje dokončanih opravil](loading_completed_jobs.md).
2. V pogledu **Rezultati** kliknite `Export Transcript`.
3. Odpre se pogovorno okno **Export Transcript**. V spustnem meniju **Format** izberite format.
4. Kliknite `Save`. Odpre se pogovorno okno za shranjevanje datoteke.
5. Izberite ciljno mapo in ime datoteke, nato kliknite **Save**.

Na dnu pogovornega okna se prikaže potrditveno sporočilo s polno potjo shranjene datoteke.

## Razpoložljivi formati

| Format | Končnica | Najprimernejše za |
|---|---|---|
| Excel | `.xlsx` | Analizo v preglednicah s stolpci za govorca, časovne žige in vsebino. |
| CSV | `.csv` | Uvoz v katero koli preglednico ali orodje za obdelavo podatkov. |
| JSON | `.json` | Programsko obdelavo. |
| SRT Subtitles | `.srt` | Nalaganje v video urejevalnike ali predvajalnike medijev kot podnapisi. |
| Markdown | `.md` | Berljive dokumente v obliki navadnega besedila. |
| Word Document | `.docx` | Deljenje z uporabniki programa Microsoft Word. |
| SQLite Database | `.db` | Celoten izvoz baze podatkov za poizvedbe po meri. |

## Imena govorcev pri izvozu

Če ste govorcem dodelili prikazna imena, so ta imena uporabljena v vseh formatih izvoza. Če želite posodobiti imena pred izvozom, najprej kliknite `Edit Speaker Names`. Glejte [Urejanje imen govorcev](editing_speaker_names.md).

---