---
title: "Pobieranie modeli"
description: "Jak pobrać pliki modeli AI wymagane do transkrypcji."
topic_id: first_steps_downloading_models
---

# Pobieranie modeli

Parakeet Transcription wymaga plików modeli AI do działania. Nie są one dołączone do aplikacji i muszą zostać pobrane przed pierwszą transkrypcją.

## Stan modeli (ekran główny)

W górnej części ekranu głównego znajduje się wąski pasek stanu, który informuje, czy modele są gotowe do użycia. Gdy brakuje plików, wyświetlany jest również przycisk `Open Settings`, który przekierowuje bezpośrednio do zarządzania modelami.

| Stan | Znaczenie |
|---|---|
| `All N model file(s) present ✓` | Wszystkie wymagane pliki zostały pobrane i są gotowe. |
| `N model file(s) missing: …` | Brakuje jednego lub więcej plików; otwórz Ustawienia, aby je pobrać. |

Gdy modele są gotowe, przyciski `New Transcription` i `Bulk Add Jobs` stają się aktywne.

## Jak pobrać modele

1. Na ekranie głównym kliknij `Open Settings` (lub przejdź do `Settings… > Models`).
2. W sekcji **Models** kliknij `Download Missing Models`.
3. Pojawi się pasek postępu oraz wiersz stanu pokazujący bieżący plik, jego pozycję w kolejce i rozmiar pobierania — na przykład: `[1/3] encoder-model.onnx — 42 MB`.
4. Poczekaj, aż stan zmieni się na `Download complete.`

## Anulowanie pobierania

Aby zatrzymać trwające pobieranie, kliknij `Cancel`. Wiersz stanu wyświetli komunikat `Download cancelled.` Częściowo pobrane pliki są zachowywane, dzięki czemu pobieranie zostanie wznowione od miejsca, w którym zostało przerwane, po ponownym kliknięciu `Download Missing Models`.

## Błędy pobierania

Jeśli pobieranie nie powiedzie się, wiersz stanu wyświetli komunikat `Download failed: <reason>`. Sprawdź połączenie internetowe i kliknij ponownie `Download Missing Models`, aby ponowić próbę. Aplikacja wznawia pobieranie od ostatniego pomyślnie ukończonego pliku.

## Zmiana precyzji

Pliki modeli, które należy pobrać, zależą od wybranej opcji **Model Precision**. Aby ją zmienić, przejdź do `Settings… > Models > Model Precision`. Jeśli zmienisz precyzję po pobraniu modeli, nowy zestaw plików musi zostać pobrany osobno. Zobacz [Wybór precyzji wag modelu](model_precision.md).

---