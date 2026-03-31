---
title: "Wybór precyzji wag modelu"
description: "Jak wybrać między precyzją modelu INT8 a FP32 i jakie są związane z tym kompromisy."
topic_id: first_steps_model_precision
---

# Wybór precyzji wag modelu

Precyzja modelu określa format numeryczny używany przez wagi modelu AI. Wpływa ona na rozmiar pobieranych plików, zużycie pamięci oraz dokładność rozpoznawania.

## Opcje precyzji

### INT8 (mniejszy plik do pobrania)

- Mniejsze pliki modelu — szybsze pobieranie i mniejsze zapotrzebowanie na miejsce na dysku.
- Nieznacznie niższa dokładność w przypadku niektórych nagrań audio.
- Zalecana, gdy masz ograniczone miejsce na dysku lub wolniejsze połączenie internetowe.

### FP32 (większa dokładność)

- Większe pliki modelu.
- Wyższa dokładność, szczególnie w przypadku trudnych nagrań audio z akcentami lub szumem w tle.
- Zalecana, gdy dokładność jest priorytetem i dysponujesz wystarczającą ilością miejsca na dysku.
- Wymagana do akceleracji GPU CUDA — ścieżka GPU zawsze używa FP32, niezależnie od tego ustawienia.

## Jak zmienić precyzję

Otwórz `Settings…` z paska menu, a następnie przejdź do sekcji **Models** i wybierz opcję `INT8 (smaller download)` lub `FP32 (more accurate)`.

## Po zmianie precyzji

Zmiana precyzji wymaga innego zestawu plików modelu. Jeśli pliki dla nowej precyzji nie zostały jeszcze pobrane, kliknij `Download Missing Models` w Ustawieniach. Wcześniej pobrane pliki dla drugiej precyzji pozostają na dysku i nie trzeba ich ponownie pobierać w przypadku powrotu do poprzedniego ustawienia.

---