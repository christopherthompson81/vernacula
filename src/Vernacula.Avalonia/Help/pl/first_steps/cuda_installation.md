---
title: "Instalacja CUDA i cuDNN do akceleracji GPU"
description: "Jak skonfigurować NVIDIA CUDA i cuDNN, aby Vernacula-Desktop mógł korzystać z GPU."
topic_id: first_steps_cuda_installation
---

# Instalacja CUDA i cuDNN do akceleracji GPU

Vernacula-Desktop może korzystać z GPU firmy NVIDIA, aby znacznie przyspieszyć transkrypcję. Akceleracja GPU wymaga zainstalowania w systemie zestawu narzędzi NVIDIA CUDA Toolkit oraz bibliotek środowiska uruchomieniowego cuDNN.

## Wymagania

- GPU firmy NVIDIA obsługujące CUDA (zalecana seria GeForce GTX 10 lub nowsza).
- Windows 10 lub 11 (64-bitowy).
- Pliki modeli muszą być już pobrane. Zobacz [Pobieranie modeli](downloading_models.md).

## Kroki instalacji

### 1. Zainstaluj CUDA Toolkit

Pobierz i uruchom instalator CUDA Toolkit ze strony dla deweloperów NVIDIA. Podczas instalacji zaakceptuj domyślne ścieżki. Instalator automatycznie ustawia zmienną środowiskową `CUDA_PATH` — Vernacula-Desktop używa tej zmiennej do lokalizowania bibliotek CUDA.

### 2. Zainstaluj cuDNN

Pobierz archiwum ZIP cuDNN odpowiednie dla zainstalowanej wersji CUDA ze strony dla deweloperów NVIDIA. Rozpakuj archiwum i skopiuj zawartość folderów `bin`, `include` oraz `lib` do odpowiadających im folderów w katalogu instalacyjnym CUDA Toolkit (ścieżka wskazana przez `CUDA_PATH`).

Alternatywnie możesz zainstalować cuDNN za pomocą instalatora NVIDIA cuDNN, jeśli jest dostępny dla Twojej wersji CUDA.

### 3. Uruchom ponownie aplikację

Po zakończeniu instalacji zamknij i ponownie otwórz Vernacula-Desktop. Aplikacja sprawdza obecność CUDA podczas uruchamiania.

## Stan GPU w ustawieniach

Otwórz `Settings…` z paska menu i przejdź do sekcji **Hardware & Performance**. Każdy składnik wyświetla znacznik (✓), gdy zostanie wykryty:

| Element | Znaczenie |
|---|---|
| Nazwa GPU i VRAM | Znaleziono GPU NVIDIA |
| CUDA Toolkit ✓ | Biblioteki CUDA zlokalizowane przez `CUDA_PATH` |
| cuDNN ✓ | Znaleziono biblioteki DLL środowiska uruchomieniowego cuDNN |
| CUDA Acceleration ✓ | ONNX Runtime załadował dostawcę wykonywania CUDA |

Jeśli po instalacji brakuje któregoś elementu, kliknij `Re-check`, aby ponownie uruchomić wykrywanie sprzętu bez konieczności restartowania aplikacji.

Okno ustawień zawiera również bezpośrednie linki do pobrania CUDA Toolkit i cuDNN, jeśli nie są jeszcze zainstalowane.

### Rozwiązywanie problemów

Jeśli `CUDA Acceleration` nie wyświetla znacznika, sprawdź, czy:

- Zmienna środowiskowa `CUDA_PATH` jest ustawiona (sprawdź `System > Advanced system settings > Environment Variables`).
- Biblioteki DLL cuDNN znajdują się w katalogu uwzględnionym w systemowej zmiennej `PATH` lub w folderze `bin` instalacji CUDA.
- Sterownik GPU jest aktualny.

### Rozmiar partii

Gdy CUDA jest aktywne, sekcja **Hardware & Performance** wyświetla również bieżący dynamiczny limit partii — maksymalną liczbę sekund dźwięku przetwarzanych w jednym przebiegu GPU. Wartość ta jest obliczana na podstawie wolnego VRAM po załadowaniu modeli i dostosowuje się automatycznie w razie zmiany dostępnej pamięci.

## Działanie bez GPU

Jeśli CUDA nie jest dostępne, Vernacula-Desktop automatycznie przełącza się na przetwarzanie za pomocą CPU. Transkrypcja nadal działa, jednak będzie wolniejsza, szczególnie w przypadku długich plików audio.

---