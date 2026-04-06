---
title: "Stahování modelů"
description: "Jak stáhnout soubory modelů AI potřebné pro přepis."
topic_id: first_steps_downloading_models
---

# Stahování modelů

Vernacula-Desktop ke svému chodu vyžaduje soubory modelů AI. Ty nejsou součástí instalace aplikace a před prvním přepisem je nutné je stáhnout.

## Stav modelů (domovská obrazovka)

Tenký stavový řádek v horní části domovské obrazovky zobrazuje, zda jsou vaše modely připraveny. Pokud soubory chybí, zobrazí se také tlačítko `Open Settings`, které vás přesměruje přímo do správy modelů.

| Stav | Význam |
|---|---|
| `All N model file(s) present ✓` | Všechny požadované soubory jsou staženy a připraveny k použití. |
| `N model file(s) missing: …` | Jeden nebo více souborů chybí; otevřete Nastavení a stáhněte je. |

Jakmile jsou modely připraveny, stanou se aktivními tlačítka `New Transcription` a `Bulk Add Jobs`.

## Jak stáhnout modely

1. Na domovské obrazovce klikněte na `Open Settings` (nebo přejděte do `Settings… > Models`).
2. V části **Models** klikněte na `Download Missing Models`.
3. Zobrazí se ukazatel průběhu a stavový řádek s informacemi o aktuálním souboru, jeho pořadí ve frontě a velikosti stahování — například: `[1/3] encoder-model.onnx — 42 MB`.
4. Počkejte, dokud se stav nezmění na `Download complete.`

## Zrušení stahování

Chcete-li zastavit probíhající stahování, klikněte na `Cancel`. Stavový řádek zobrazí `Download cancelled.` Částečně stažené soubory jsou zachovány, takže při příštím kliknutí na `Download Missing Models` stahování pokračuje od místa, kde bylo přerušeno.

## Chyby při stahování

Pokud se stahování nezdaří, stavový řádek zobrazí `Download failed: <reason>`. Zkontrolujte připojení k internetu a opakujte pokus kliknutím na `Download Missing Models`. Aplikace pokračuje od posledního úspěšně staženého souboru.

## Změna přesnosti

Soubory modelů, které je třeba stáhnout, závisí na zvolené **Model Precision**. Chcete-li ji změnit, přejděte do `Settings… > Models > Model Precision`. Pokud přesnost změníte po stažení modelů, nová sada souborů musí být stažena samostatně. Viz [Výběr přesnosti vah modelu](model_precision.md).

---