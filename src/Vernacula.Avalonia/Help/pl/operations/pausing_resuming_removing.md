---
title: "Wstrzymywanie, wznawianie i usuwanie zadań"
description: "Jak wstrzymać uruchomione zadanie, wznowić zatrzymane lub usunąć zadanie z historii."
topic_id: operations_pausing_resuming_removing
---

# Wstrzymywanie, wznawianie i usuwanie zadań

## Wstrzymywanie zadania

Uruchomione lub oczekujące zadanie można wstrzymać z dwóch miejsc:

- **Widok postępu** — kliknij `Pause` w prawym dolnym rogu podczas obserwowania aktywnego zadania.
- **Tabela historii transkrypcji** — kliknij `Pause` w kolumnie **Actions** w wierszu, którego status to `running` lub `queued`.

Po kliknięciu `Pause` w wierszu stanu wyświetlany jest napis `Pausing…`, podczas gdy aplikacja kończy przetwarzanie bieżącej jednostki. Następnie status zadania zmienia się na `cancelled` w tabeli historii.

> Wstrzymanie zadania zapisuje wszystkie dotychczas przetworzone segmenty. Zadanie można wznowić później bez utraty wykonanej pracy.

## Wznawianie zadania

Aby wznowić wstrzymane lub nieudane zadanie:

1. Na ekranie głównym znajdź zadanie w tabeli **Transcription History**. Jego status będzie wynosił `cancelled` lub `failed`.
2. Kliknij `Resume` w kolumnie **Actions**.
3. Aplikacja powróci do widoku **Progress** i kontynuuje przetwarzanie od miejsca, w którym zostało przerwane.

W wierszu stanu przez chwilę wyświetlany jest napis `Resuming…`, podczas gdy zadanie jest ponownie inicjalizowane.

## Usuwanie zadania

Aby trwale usunąć zadanie i jego transkrypcję z historii:

1. W tabeli **Transcription History** kliknij `Remove` w kolumnie **Actions** w wierszu zadania, które chcesz usunąć.

Zadanie zostanie usunięte z listy, a jego dane zostaną skasowane z lokalnej bazy danych. Tej operacji nie można cofnąć. Wyeksportowane pliki zapisane na dysku nie są objęte tą operacją.

---