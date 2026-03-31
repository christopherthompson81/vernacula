---
title: "Uređivanje naziva govornika"
description: "Kako zamijeniti generičke identifikatore govornika stvarnim imenima u transkriptu."
topic_id: operations_editing_speaker_names
---

# Uređivanje naziva govornika

Mehanizam za transkripciju automatski dodjeljuje svakom govorniku generički identifikator (na primjer, `speaker_0`, `speaker_1`). Te identifikatore možete zamijeniti stvarnim imenima koja će se prikazivati u cijelom transkriptu i u svim izvezenim datotekama.

## Kako urediti nazive govornika

1. Otvorite dovršeni posao. Pogledajte [Učitavanje dovršenih poslova](loading_completed_jobs.md).
2. U prikazu **Rezultati** kliknite `Edit Speaker Names`.
3. Otvara se dijaloški okvir **Edit Speaker Names** s dva stupca:
   - **Speaker ID** — izvorna oznaka koju je dodijelio model (samo za čitanje).
   - **Display Name** — ime prikazano u transkriptu (može se uređivati).
4. Kliknite ćeliju u stupcu **Display Name** i upišite ime govornika.
5. Pritisnite `Tab` ili kliknite drugi redak kako biste prešli na sljedećeg govornika.
6. Kliknite `Save` za primjenu promjena ili `Cancel` za odustajanje.

## Gdje se imena prikazuju

Ažurirani nazivi za prikaz zamjenjuju generičke identifikatore u:

- Tablici segmenata u prikazu Rezultati.
- Svim izvezenim datotekama (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Ponovno uređivanje imena

Dijaloški okvir za uređivanje naziva govornika možete ponovo otvoriti u bilo kojem trenutku dok je posao učitan u prikazu Rezultati. Promjene se spremaju u lokalnu bazu podataka i zadržavaju između sesija.

---