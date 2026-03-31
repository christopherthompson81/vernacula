---
title: "Edytowanie nazw mówców"
description: "Jak zastąpić ogólne identyfikatory mówców prawdziwymi imionami w transkrypcji."
topic_id: operations_editing_speaker_names
---

# Edytowanie nazw mówców

Silnik transkrypcji automatycznie przypisuje każdemu mówcy ogólny identyfikator (na przykład `speaker_0`, `speaker_1`). Możesz zastąpić je prawdziwymi imionami lub nazwami, które będą wyświetlane w całej transkrypcji oraz we wszystkich wyeksportowanych plikach.

## Jak edytować nazwy mówców

1. Otwórz ukończone zadanie. Zobacz [Ładowanie ukończonych zadań](loading_completed_jobs.md).
2. W widoku **Wyniki** kliknij `Edit Speaker Names`.
3. Otworzy się okno dialogowe **Edit Speaker Names** z dwiema kolumnami:
   - **Speaker ID** — oryginalna etykieta przypisana przez model (tylko do odczytu).
   - **Display Name** — nazwa wyświetlana w transkrypcji (edytowalna).
4. Kliknij komórkę w kolumnie **Display Name** i wpisz imię lub nazwę mówcy.
5. Naciśnij `Tab` lub kliknij inny wiersz, aby przejść do następnego mówcy.
6. Kliknij `Save`, aby zastosować zmiany, lub `Cancel`, aby je odrzucić.

## Gdzie wyświetlają się nazwy

Zaktualizowane nazwy wyświetlane zastępują ogólne identyfikatory w:

- Tabeli segmentów w widoku Wyniki.
- Wszystkich wyeksportowanych plikach (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Ponowne edytowanie nazw

Okno dialogowe edytowania nazw mówców możesz otworzyć ponownie w dowolnym momencie, gdy zadanie jest załadowane w widoku Wyniki. Zmiany są zapisywane w lokalnej bazie danych i zachowywane między sesjami.

---