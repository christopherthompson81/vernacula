---
title: "Kõnelejate nimede muutmine"
description: "Kuidas asendada üldised kõnelejate ID-d transkriptsioonis pärisnimetega."
topic_id: operations_editing_speaker_names
---

# Kõnelejate nimede muutmine

Transkriptsioonimootor märgistab iga kõneleja automaatselt üldise ID-ga (näiteks `speaker_0`, `speaker_1`). Saate need asendada pärisnimetega, mis kuvatakse kogu transkriptsioonis ja kõigis eksporditud failides.

## Kuidas kõnelejate nimesid muuta

1. Avage lõpetatud töö. Vt [Lõpetatud tööde laadimine](loading_completed_jobs.md).
2. Klõpsake vaates **Results** nuppu `Edit Speaker Names`.
3. Avaneb dialoogiaken **Edit Speaker Names** kahe veeruga:
   - **Speaker ID** — mudeli määratud algne silt (kirjutuskaitstud).
   - **Display Name** — transkriptsioonis kuvatav nimi (muudetav).
4. Klõpsake veerus **Display Name** mõnda lahtrit ja tippige kõneleja nimi.
5. Vajutage `Tab` või klõpsake järgmisele reale liikumiseks mõnda muud rida.
6. Klõpsake muudatuste rakendamiseks `Save` või nende tühistamiseks `Cancel`.

## Kus nimed kuvatakse

Uuendatud kuvanimedega asendatakse üldised ID-d järgmistes kohtades:

- Tulemuste vaate segmentide tabelis.
- Kõigis eksporditud failides (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Nimede uuesti muutmine

Saate dialoogiakna Edit Speaker Names uuesti avada igal ajal, kui töö on tulemuste vaates laaditud. Muudatused salvestatakse kohalikku andmebaasi ja säilivad seansside vahel.

---