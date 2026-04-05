---
title: "Úprava mien rečníkov"
description: "Ako nahradiť generické identifikátory rečníkov skutočnými menami v prepise."
topic_id: operations_editing_speaker_names
---

# Úprava mien rečníkov

Prepísovací modul automaticky označí každého rečníka generickým identifikátorom (napríklad `speaker_0`, `speaker_1`). Tieto identifikátory môžete nahradiť skutočnými menami, ktoré sa budú zobrazovať v celom prepise aj vo všetkých exportovaných súboroch.

## Ako upraviť mená rečníkov

1. Otvorte dokončenú úlohu. Pozrite si [Načítanie dokončených úloh](loading_completed_jobs.md).
2. V zobrazení **Výsledky** kliknite na `Edit Speaker Names`.
3. Otvorí sa dialógové okno **Edit Speaker Names** s dvoma stĺpcami:
   - **Speaker ID** — pôvodný štítok priradený modelom (iba na čítanie).
   - **Display Name** — meno zobrazované v prepise (upraviteľné).
4. Kliknite na bunku v stĺpci **Display Name** a zadajte meno rečníka.
5. Stlačte `Tab` alebo kliknite na iný riadok, čím prejdete na ďalšieho rečníka.
6. Kliknite na `Save` pre uloženie zmien alebo na `Cancel` pre ich zrušenie.

## Kde sa mená zobrazujú

Aktualizované zobrazované mená nahrádzajú generické identifikátory v:

- Tabuľke segmentov v zobrazení Výsledky.
- Všetkých exportovaných súboroch (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Opätovná úprava mien

Dialógové okno Úprava mien rečníkov môžete kedykoľvek znovu otvoriť, pokiaľ je úloha načítaná v zobrazení Výsledky. Zmeny sa uložia do lokálnej databázy a zachovajú sa aj po ukončení a opätovnom spustení aplikácie.

---