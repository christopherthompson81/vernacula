---
title: "Úpravy přepisů"
description: "Jak procházet, opravovat a ověřovat přepsané segmenty v editoru přepisů."
topic_id: operations_editing_transcripts
---

# Úpravy přepisů

**Editor přepisů** umožňuje procházet výstup ASR, opravovat text, přejmenovávat mluvčí přímo v editoru, upravovat časování segmentů a označovat segmenty jako ověřené — to vše při poslechu původního zvuku.

## Otevření editoru

1. Načtěte dokončenou úlohu (viz [Načítání dokončených úloh](loading_completed_jobs.md)).
2. V zobrazení **Výsledky** klikněte na `Edit Transcript`.

Editor se otevře jako samostatné okno a může zůstat otevřené vedle hlavní aplikace.

## Rozložení

Každý segment je zobrazen jako karta se dvěma panely vedle sebe:

- **Levý panel** — původní výstup ASR s barevným zvýrazněním spolehlivosti jednotlivých slov. Slova, o nichž si model nebyl jistý, jsou zobrazena červeně; slova s vysokou spolehlivostí jsou zobrazena v normální barvě textu.
- **Pravý panel** — upravitelné textové pole. Opravy provádějte zde; při psaní je průběžně zvýrazněn rozdíl oproti původnímu textu.

Nad každou kartou jsou zobrazeny jméno mluvčího a časový rozsah. Kliknutím na kartu ji aktivujete a zobrazíte její akční ikony. Přesuňte ukazatel myši nad libovolnou ikonu, abyste zobrazili popis její funkce.

## Přehled ikon

### Panel přehrávání

| Ikona | Akce |
|-------|------|
| ▶ | Přehrát |
| ⏸ | Pozastavit |
| ⏮ | Přejít na předchozí segment |
| ⏭ | Přejít na další segment |

### Akce na kartě segmentu

| Ikona | Akce |
|-------|------|
| <mdl2 ch="E77B"/> | Přiřadit segment jinému mluvčímu |
| <mdl2 ch="E916"/> | Upravit čas začátku a konce segmentu |
| <mdl2 ch="EA39"/> | Potlačit nebo zrušit potlačení segmentu |
| <mdl2 ch="E72B"/> | Sloučit s předchozím segmentem |
| <mdl2 ch="E72A"/> | Sloučit s dalším segmentem |
| <mdl2 ch="E8C6"/> | Rozdělit segment |
| <mdl2 ch="E72C"/> | Znovu spustit ASR na tomto segmentu |

## Přehrávání zvuku

Panel přehrávání se nachází v horní části okna editoru:

| Ovládací prvek | Akce |
|----------------|------|
| Ikona přehrát / pozastavit | Spustit nebo pozastavit přehrávání |
| Posuvník polohy | Přetažením přejdete na libovolné místo ve zvuku |
| Posuvník rychlosti | Nastavení rychlosti přehrávání (0,5× – 2×) |
| Ikony předchozí / další | Přejít na předchozí nebo další segment |
| Rozevírací seznam režimu přehrávání | Vyberte jeden ze tří režimů přehrávání (viz níže) |
| Posuvník hlasitosti | Nastavení hlasitosti přehrávání |

Během přehrávání je aktuálně vyslovované slovo zvýrazněno v levém panelu. Po pozastavení při přetažení posuvníku se zvýraznění aktualizuje na slovo odpovídající zvolené poloze.

### Režimy přehrávání

| Režim | Chování |
|-------|---------|
| `Single` | Přehraje aktuální segment jednou a zastaví se. |
| `Auto-advance` | Přehraje aktuální segment; po jeho skončení jej označí jako ověřený a přejde na další. |
| `Continuous` | Přehraje všechny segmenty za sebou bez označení jakéhokoliv jako ověřeného. |

Aktivní režim vyberte z rozevíracího seznamu na panelu přehrávání.

## Úprava segmentu

1. Kliknutím na kartu ji aktivujte.
2. Upravte text v pravém panelu. Změny se automaticky uloží, jakmile přesunete pozornost na jinou kartu.

## Přejmenování mluvčího

Klikněte na jméno mluvčího v aktivované kartě a zadejte nové jméno. Stiskněte `Enter` nebo klikněte jinam pro uložení. Nové jméno se použije pouze pro danou kartu; chcete-li přejmenovat mluvčího globálně, použijte [Úprava jmen mluvčích](editing_speaker_names.md) ze zobrazení Výsledky.

## Ověření segmentu

Klikněte na zaškrtávací políčko `Verified` na aktivované kartě a označte segment jako zkontrolovaný. Stav ověření se uloží do databáze a bude viditelný v editoru při dalším načtení.

## Potlačení segmentu

Klikněte na `Suppress` na aktivované kartě, čímž skryjete segment z exportů (užitečné pro šum, hudbu nebo jiné části bez řeči). Kliknutím na `Unsuppress` segment obnovíte.

## Úprava časování segmentu

Klikněte na `Adjust Times` na aktivované kartě, čímž otevřete dialog pro úpravu časů. Pomocí kolečka myši nad polem **Start** nebo **End** měňte hodnotu po krocích 0,1 sekundy, nebo hodnotu zadejte přímo. Kliknutím na `Save` změny použijete.

## Slučování segmentů

- Klikněte na `⟵ Merge` pro sloučení aktivovaného segmentu s bezprostředně předcházejícím segmentem.
- Klikněte na `Merge ⟶` pro sloučení aktivovaného segmentu s bezprostředně následujícím segmentem.

Text a časový rozsah obou karet se spojí dohromady. To je užitečné v případě, kdy jedna vyslovená promluva byla rozdělena do dvou segmentů.

## Rozdělení segmentu

Klikněte na `Split…` na aktivované kartě, čímž otevřete dialog pro rozdělení. Umístěte bod rozdělení v textu a potvrďte. Vytvoří se dva nové segmenty pokrývající původní časový rozsah. To je užitečné v případě, kdy dvě různé promluvy byly sloučeny do jednoho segmentu.

## Opakování ASR

Klikněte na `Redo ASR` na aktivované kartě, čímž znovu spustíte rozpoznávání řeči na zvuku daného segmentu. Model zpracuje pouze zvukový úsek tohoto segmentu a vytvoří nový přepis z jediného zdroje.

Použijte tuto funkci v těchto případech:

- Segment vznikl sloučením a nelze jej rozdělit (sloučené segmenty zahrnují více zdrojů ASR; funkce Redo ASR je sloučí do jednoho, po čemž bude dostupná možnost `Split…`).
- Původní přepis je nekvalitní a chcete provést čistý druhý průchod bez ručních úprav.

**Poznámka:** Veškerý text, který jste již zadali v pravém panelu, bude zahozen a nahrazen novým výstupem ASR. Tato operace vyžaduje, aby byl zvukový soubor načten; tlačítko je deaktivováno, pokud zvuk není dostupný.