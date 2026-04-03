---
title: "Wczytywanie ukończonych zadań"
description: "Jak otworzyć wyniki poprzednio ukończonego transkrypcji."
topic_id: operations_loading_completed_jobs
---

# Wczytywanie ukończonych zadań

Wszystkie ukończone zadania transkrypcji są zapisywane w lokalnej bazie danych i pozostają dostępne w tabeli **Historia transkrypcji** na ekranie głównym.

## Jak wczytać ukończone zadanie

1. Na ekranie głównym znajdź zadanie w tabeli **Historia transkrypcji**. Ukończone zadania wyświetlają znacznik statusu `complete`.
2. Kliknij `Load` w kolumnie **Akcje** danego zadania.
3. Aplikacja przełącza się do widoku **Wyniki**, pokazując wszystkie transkrybowane segmenty dla tego zadania.

## Widok Wyniki

Widok Wyniki wyświetla:

- Nazwę pliku audio jako nagłówek strony.
- Podtytuł z liczbą segmentów (na przykład `42 segment(s)`).
- Tabelę segmentów z kolumnami **Mówca**, **Początek**, **Koniec** i **Treść**.

Z poziomu widoku Wyniki możesz:

- [Edytować transkrypcję](editing_transcripts.md) — przeglądać i poprawiać tekst, dostosowywać czas, łączyć lub dzielić segmenty oraz weryfikować segmenty podczas odsłuchiwania nagrania.
- [Edytować nazwy mówców](editing_speaker_names.md) — zastępować ogólne identyfikatory, takie jak `speaker_0`, prawdziwymi imionami.
- [Eksportować transkrypcję](exporting_results.md) — zapisywać transkrypcję do formatu Excel, CSV, JSON, SRT, Markdown, Word lub SQLite.

Aby powrócić do listy historii, kliknij `← Back to History`.

---