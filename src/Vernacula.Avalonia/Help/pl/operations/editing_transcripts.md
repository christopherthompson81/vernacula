---
title: "Edytowanie transkrypcji"
description: "Jak przeglądać, poprawiać i weryfikować transkrybowane segmenty w edytorze transkrypcji."
topic_id: operations_editing_transcripts
---

# Edytowanie transkrypcji

**Edytor transkrypcji** umożliwia przeglądanie wyników ASR, poprawianie tekstu, zmianę nazw mówców bezpośrednio w edytorze, dostosowywanie czasów segmentów oraz oznaczanie segmentów jako zweryfikowane — wszystko przy jednoczesnym odsłuchiwaniu oryginalnego nagrania.

## Otwieranie edytora

1. Wczytaj ukończone zadanie (zobacz [Wczytywanie ukończonych zadań](loading_completed_jobs.md)).
2. W widoku **Wyniki** kliknij `Edit Transcript`.

Edytor otwiera się jako oddzielne okno i może pozostawać otwarte równolegle z głównym oknem aplikacji.

## Układ

Każdy segment wyświetlany jest jako karta z dwoma panelami umieszczonymi obok siebie:

- **Panel lewy** — oryginalne wyjście ASR z kolorowaniem słów według poziomu pewności. Słowa, co do których model miał mniejszą pewność, są wyróżnione na czerwono; słowa o wysokim poziomie pewności wyświetlane są w standardowym kolorze tekstu.
- **Panel prawy** — edytowalne pole tekstowe. Poprawki wprowadzaj tutaj; różnice względem oryginału są podświetlane podczas pisania.

Etykieta mówcy oraz zakres czasowy widoczne są nad każdą kartą. Kliknij kartę, aby ją zaznaczyć i wyświetlić jej ikony akcji. Najedź kursorem na dowolną ikonę, aby zobaczyć etykietkę z opisem jej funkcji.

## Legenda ikon

### Pasek odtwarzania

| Ikona | Akcja |
|-------|-------|
| ▶ | Odtwórz |
| ⏸ | Wstrzymaj |
| ⏮ | Przejdź do poprzedniego segmentu |
| ⏭ | Przejdź do następnego segmentu |

### Akcje karty segmentu

| Ikona | Akcja |
|-------|-------|
| <mdl2 ch="E77B"/> | Przypisz segment do innego mówcy |
| <mdl2 ch="E916"/> | Dostosuj czas początku i końca segmentu |
| <mdl2 ch="EA39"/> | Wycisz lub przywróć segment |
| <mdl2 ch="E72B"/> | Scal z poprzednim segmentem |
| <mdl2 ch="E72A"/> | Scal z następnym segmentem |
| <mdl2 ch="E8C6"/> | Podziel segment |
| <mdl2 ch="E72C"/> | Ponów ASR dla tego segmentu |

## Odtwarzanie dźwięku

Pasek odtwarzania rozciąga się wzdłuż górnej części okna edytora:

| Kontrolka | Akcja |
|-----------|-------|
| Ikona Odtwórz / Wstrzymaj | Rozpocznij lub wstrzymaj odtwarzanie |
| Pasek przewijania | Przeciągnij, aby przejść do dowolnej pozycji w nagraniu |
| Suwak szybkości | Dostosuj prędkość odtwarzania (0,5× – 2×) |
| Ikony Poprzedni / Następny | Przejdź do poprzedniego lub następnego segmentu |
| Lista rozwijana trybu odtwarzania | Wybierz jeden z trzech trybów odtwarzania (patrz niżej) |
| Suwak głośności | Dostosuj głośność odtwarzania |

Podczas odtwarzania aktualnie wypowiadane słowo jest podświetlane w lewym panelu. Po wstrzymaniu odtwarzania po przewinięciu podświetlenie aktualizuje się do słowa znajdującego się w miejscu przewinięcia.

### Tryby odtwarzania

| Tryb | Zachowanie |
|------|------------|
| `Single` | Odtwarza bieżący segment raz, a następnie zatrzymuje się. |
| `Auto-advance` | Odtwarza bieżący segment; po jego zakończeniu oznacza go jako zweryfikowany i przechodzi do następnego. |
| `Continuous` | Odtwarza wszystkie segmenty kolejno bez oznaczania żadnego jako zweryfikowany. |

Aktywny tryb wybierz z listy rozwijanej na pasku odtwarzania.

## Edytowanie segmentu

1. Kliknij kartę, aby ją zaznaczyć.
2. Edytuj tekst w prawym panelu. Zmiany są zapisywane automatycznie po przeniesieniu fokusu na inną kartę.

## Zmiana nazwy mówcy

Kliknij etykietę mówcy w zaznaczonej karcie i wpisz nową nazwę. Naciśnij `Enter` lub kliknij poza polem, aby zapisać. Nowa nazwa zostaje zastosowana wyłącznie do tej karty; aby zmienić nazwę mówcy globalnie, skorzystaj z opcji [Edytuj nazwy mówców](editing_speaker_names.md) dostępnej w widoku Wyniki.

## Weryfikowanie segmentu

Kliknij pole wyboru `Verified` na zaznaczonej karcie, aby oznaczyć segment jako przejrzany. Status weryfikacji jest zapisywany w bazie danych i będzie widoczny w edytorze przy kolejnych wczytaniach.

## Wyciszanie segmentu

Kliknij `Suppress` na zaznaczonej karcie, aby ukryć segment w eksportach (przydatne w przypadku szumów, muzyki lub innych fragmentów bez mowy). Kliknij `Unsuppress`, aby przywrócić segment.

## Dostosowywanie czasów segmentu

Kliknij `Adjust Times` na zaznaczonej karcie, aby otworzyć okno dialogowe dostosowywania czasów. Użyj kółka myszy nad polem **Start** lub **End**, aby zmieniać wartość z krokiem 0,1 sekundy, lub wpisz wartość bezpośrednio. Kliknij `Save`, aby zastosować zmiany.

## Scalanie segmentów

- Kliknij `⟵ Merge`, aby scalić zaznaczony segment z segmentem bezpośrednio go poprzedzającym.
- Kliknij `Merge ⟶`, aby scalić zaznaczony segment z segmentem bezpośrednio po nim następującym.

Tekst oraz zakres czasowy obu kart zostają połączone. Jest to przydatne, gdy jedna wypowiedź została podzielona na dwa segmenty.

## Dzielenie segmentu

Kliknij `Split…` na zaznaczonej karcie, aby otworzyć okno dialogowe dzielenia. Ustaw punkt podziału w tekście i potwierdź. Zostaną utworzone dwa nowe segmenty obejmujące oryginalny zakres czasowy. Jest to przydatne, gdy dwie odrębne wypowiedzi zostały połączone w jeden segment.

## Ponowne ASR

Kliknij `Redo ASR` na zaznaczonej karcie, aby ponownie uruchomić rozpoznawanie mowy dla audio danego segmentu. Model przetwarza wyłącznie wycinek audio odpowiadający temu segmentowi i generuje nową transkrypcję z jednego źródła.

Użyj tej opcji, gdy:

- Segment powstał ze scalenia i nie można go podzielić (scalone segmenty obejmują wiele źródeł ASR; ponowne ASR łączy je w jedno, po czym opcja `Split…` staje się dostępna).
- Oryginalna transkrypcja jest niskiej jakości i chcesz uzyskać czystą drugą wersję bez ręcznej edycji.

**Uwaga:** Tekst już wpisany w prawym panelu zostanie odrzucony i zastąpiony nowym wynikiem ASR. Operacja wymaga wczytania pliku audio; przycisk jest nieaktywny, jeśli audio jest niedostępne.