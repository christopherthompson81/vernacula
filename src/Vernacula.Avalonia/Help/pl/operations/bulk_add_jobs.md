---
title: "Dodawanie wielu plików audio do kolejki"
description: "Jak dodać kilka plików audio do kolejki zadań jednocześnie."
topic_id: operations_bulk_add_jobs
---

# Dodawanie wielu plików audio do kolejki

Użyj funkcji **Bulk Add Jobs**, aby dodać wiele plików audio lub wideo do transkrypcji w jednym kroku. Aplikacja przetwarza je kolejno, jedno po drugim, w kolejności dodawania.

## Wymagania wstępne

- Wszystkie pliki modelu muszą być pobrane. Karta **Model Status** musi wyświetlać `All N model file(s) present ✓`. Zobacz [Pobieranie modeli](../first_steps/downloading_models.md).

## Jak dodać wiele zadań jednocześnie

1. Na ekranie głównym kliknij `Bulk Add Jobs`.
2. Otworzy się okno wyboru plików. Zaznacz jeden lub więcej plików audio lub wideo — przytrzymaj `Ctrl` lub `Shift`, aby zaznaczyć wiele plików.
3. Kliknij **Otwórz**. Każdy zaznaczony plik zostanie dodany do tabeli **Transcription History** jako osobne zadanie.

> **Pliki wideo z wieloma strumieniami audio:** Jeśli plik wideo zawiera więcej niż jeden strumień audio (na przykład kilka wersji językowych lub ścieżkę z komentarzem reżysera), aplikacja automatycznie tworzy jedno zadanie dla każdego strumienia.

## Nazwy zadań

Każde zadanie otrzymuje automatycznie nazwę odpowiadającą nazwie pliku audio. Możesz zmienić nazwę zadania w dowolnym momencie, klikając jego nazwę w kolumnie **Title** tabeli historii transkrypcji, edytując tekst, a następnie naciskając `Enter` lub klikając w innym miejscu.

## Zachowanie kolejki

- Jeśli żadne zadanie nie jest aktualnie uruchomione, pierwszy plik rozpoczyna przetwarzanie natychmiast, a pozostałe są wyświetlane jako `queued`.
- Jeśli jakieś zadanie jest już uruchomione, wszystkie nowo dodane pliki są wyświetlane jako `queued` i będą uruchamiane automatycznie po kolei.
- Aby monitorować aktywne zadanie, kliknij `Monitor` w jego kolumnie **Actions**. Zobacz [Monitorowanie zadań](monitoring_jobs.md).
- Aby wstrzymać lub usunąć zadanie z kolejki przed jego uruchomieniem, użyj przycisków `Pause` lub `Remove` w jego kolumnie **Actions**. Zobacz [Wstrzymywanie, wznawianie i usuwanie zadań](pausing_resuming_removing.md).

---