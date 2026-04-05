---
title: "Monitorowanie zadań"
description: "Jak śledzić postęp uruchomionego lub oczekującego zadania."
topic_id: operations_monitoring_jobs
---

# Monitorowanie zadań

Widok **Postęp** umożliwia śledzenie na żywo uruchomionego zadania transkrypcji.

## Otwieranie widoku Postęp

- Po uruchomieniu nowej transkrypcji aplikacja automatycznie przechodzi do widoku Postęp.
- W przypadku zadania już uruchomionego lub oczekującego w kolejce — znajdź je w tabeli **Historia transkrypcji** i kliknij `Monitor` w kolumnie **Akcje**.

## Odczytywanie widoku Postęp

| Element | Opis |
|---|---|
| Pasek postępu | Ogólny procent ukończenia. Nieokreślony (animowany) podczas uruchamiania lub wznawiania zadania. |
| Etykieta procentowa | Wartość procentowa wyświetlana po prawej stronie paska. |
| Komunikat stanu | Bieżące działanie — na przykład `Audio Analysis` lub `Speech Recognition`. Wyświetla `Waiting in queue…`, jeśli zadanie jeszcze się nie rozpoczęło. |
| Tabela segmentów | Aktualizowany na żywo zapis transkrybowanych segmentów z kolumnami **Speaker**, **Start**, **End** i **Content**. Przewija się automatycznie wraz z pojawianiem się nowych segmentów. |

## Fazy postępu

Wyświetlane fazy zależą od **trybu segmentacji** wybranego w ustawieniach.

**Tryb diaryzacji mówców** (domyślny):

1. **Audio Analysis** — diaryzacja SortFormer przetwarza cały plik w celu identyfikacji granic między mówcami. Pasek może pozostawać blisko 0% do momentu zakończenia tej fazy.
2. **Speech Recognition** — każdy segment mówcy jest transkrybowany. Wartość procentowa stopniowo rośnie podczas tej fazy.

**Tryb wykrywania aktywności głosowej**:

1. **Detecting speech segments** — Silero VAD skanuje plik w poszukiwaniu fragmentów zawierających mowę. Ta faza przebiega szybko.
2. **Speech Recognition** — każdy wykryty fragment mowy jest transkrybowany.

W obu trybach tabela segmentów na żywo wypełnia się w miarę postępu transkrypcji.

## Opuszczanie widoku

Kliknij `← Back to Home`, aby powrócić do ekranu głównego bez przerywania zadania. Zadanie nadal działa w tle, a jego stan jest aktualizowany w tabeli **Historia transkrypcji**. Kliknij `Monitor` ponownie w dowolnym momencie, aby wrócić do widoku Postęp.

---