---
title: "Úprava jmen mluvčích"
description: "Jak nahradit obecná ID mluvčích skutečnými jmény v přepisu."
topic_id: operations_editing_speaker_names
---

# Úprava jmen mluvčích

Přepisovací engine automaticky označí každého mluvčího obecným ID (například `speaker_0`, `speaker_1`). Tato ID můžete nahradit skutečnými jmény, která se zobrazí v celém přepisu i ve všech exportovaných souborech.

## Jak upravit jména mluvčích

1. Otevřete dokončenou úlohu. Viz [Načítání dokončených úloh](loading_completed_jobs.md).
2. V zobrazení **Výsledky** klikněte na `Edit Speaker Names`.
3. Otevře se dialog **Edit Speaker Names** se dvěma sloupci:
   - **Speaker ID** — původní označení přiřazené modelem (pouze pro čtení).
   - **Display Name** — jméno zobrazované v přepisu (upravitelné).
4. Klikněte na buňku ve sloupci **Display Name** a zadejte jméno mluvčího.
5. Stiskněte `Tab` nebo klikněte na jiný řádek pro přechod k dalšímu mluvčímu.
6. Kliknutím na `Save` změny uložíte, nebo na `Cancel` je zrušíte.

## Kde se jména zobrazují

Aktualizovaná zobrazovaná jména nahrazují obecná ID v:

- Tabulce segmentů v zobrazení Výsledky.
- Všech exportovaných souborech (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Opětovná úprava jmen

Dialog pro úpravu jmen mluvčích můžete znovu otevřít kdykoli, dokud je úloha načtena v zobrazení Výsledky. Změny jsou uloženy do lokální databáze a přetrvávají mezi jednotlivými relacemi.

---