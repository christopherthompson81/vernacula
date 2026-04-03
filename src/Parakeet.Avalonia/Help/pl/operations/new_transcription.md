---
title: "Nowa transkrypcja – przebieg pracy"
description: "Przewodnik krok po kroku dotyczący transkrypcji pliku audio."
topic_id: operations_new_transcription
---

# Nowa transkrypcja – przebieg pracy

Użyj tego przepływu pracy, aby transkrybować pojedynczy plik audio.

## Wymagania wstępne

- Wszystkie pliki modeli muszą być pobrane. Karta **Stan modeli** musi wyświetlać `All N model file(s) present ✓`. Zobacz [Pobieranie modeli](../first_steps/downloading_models.md).

## Obsługiwane formaty

### Audio

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Wideo

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Pliki wideo są dekodowane za pomocą FFmpeg. Jeśli plik wideo zawiera **wiele strumieni audio** (np. różne języki lub ścieżki komentarza), dla każdego strumienia automatycznie tworzone jest osobne zadanie transkrypcji.

## Kroki

### 1. Otwórz formularz nowej transkrypcji

Kliknij `New Transcription` na ekranie głównym lub przejdź do `File > New Transcription`.

### 2. Wybierz plik multimedialny

Kliknij `Browse…` obok pola **Audio File**. Otworzy się selektor plików odfiltrowany do obsługiwanych formatów audio i wideo. Wybierz plik i kliknij **Open**. Ścieżka do pliku pojawi się w polu.

### 3. Nadaj nazwę zadaniu

Pole **Job Name** jest wstępnie wypełnione na podstawie nazwy pliku. Możesz je edytować, jeśli chcesz użyć innej etykiety — ta nazwa będzie widoczna w historii transkrypcji na ekranie głównym.

### 4. Uruchom transkrypcję

Kliknij `Start Transcription`. Aplikacja przełączy się do widoku **Progress**.

Aby wrócić bez uruchamiania, kliknij `← Back`.

## Co dzieje się dalej

Zadanie przechodzi przez dwie fazy widoczne na pasku postępu:

1. **Audio Analysis** — diaryzacja mówców: identyfikowanie, kto mówi i kiedy.
2. **Speech Recognition** — zamiana mowy na tekst, segment po segmencie.

Transkrybowane segmenty pojawiają się na żywo w tabeli w miarę ich przetwarzania. Po zakończeniu przetwarzania aplikacja automatycznie przechodzi do widoku **Results**.

Jeśli dodasz zadanie, gdy inne jest już uruchomione, nowe zadanie otrzyma status `queued` i zostanie uruchomione po zakończeniu bieżącego. Zobacz [Monitorowanie zadań](monitoring_jobs.md).

---