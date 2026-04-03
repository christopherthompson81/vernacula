---
title: "Ustawienia"
description: "Przegląd wszystkich opcji w oknie Ustawienia."
topic_id: first_steps_settings_window
---

# Ustawienia

Okno **Ustawienia** umożliwia kontrolę nad konfiguracją sprzętu, zarządzaniem modelami, trybem segmentacji, zachowaniem edytora, wyglądem aplikacji oraz językiem. Otwórz je z paska menu: `Settings…`.

## Sprzęt i wydajność

Ta sekcja pokazuje stan karty graficznej NVIDIA GPU oraz stosu oprogramowania CUDA, a także informuje o dynamicznym limicie wsadowym używanym podczas transkrypcji na GPU.

| Element | Opis |
|---|---|
| Nazwa GPU i VRAM | Wykryta karta NVIDIA GPU i dostępna pamięć wideo. |
| CUDA Toolkit | Informacja, czy biblioteki środowiska uruchomieniowego CUDA zostały znalezione za pośrednictwem `CUDA_PATH`. |
| cuDNN | Informacja, czy biblioteki DLL środowiska uruchomieniowego cuDNN są dostępne. |
| Akceleracja CUDA | Informacja, czy środowisko ONNX Runtime pomyślnie załadowało dostawcę wykonania CUDA. |

Kliknij `Re-check`, aby ponownie uruchomić wykrywanie sprzętu bez ponownego uruchamiania aplikacji — przydatne po zainstalowaniu CUDA lub cuDNN.

Gdy wymagane komponenty nie zostaną wykryte, wyświetlane są bezpośrednie linki do pobrania CUDA Toolkit i cuDNN.

Komunikat o **limicie wsadowym** informuje, ile sekund dźwięku jest przetwarzanych w każdym przebiegu GPU. Wartość ta jest obliczana na podstawie wolnej pamięci VRAM po załadowaniu modeli i dostosowuje się automatycznie.

Pełne instrukcje dotyczące konfiguracji CUDA znajdziesz w artykule [Instalacja CUDA i cuDNN](cuda_installation.md).

## Modele

Ta sekcja służy do zarządzania plikami modeli AI wymaganymi do transkrypcji.

- **Precyzja modelu** — wybierz `INT8 (smaller download)` lub `FP32 (more accurate)`. Zobacz [Wybór precyzji wag modelu](model_precision.md).
- **Pobieranie brakujących modeli** — pobiera pliki modeli, których nie ma jeszcze na dysku. Pasek postępu i wiersz stanu śledzą pobieranie każdego pliku.
- **Sprawdzanie aktualizacji** — sprawdza, czy dostępne są nowsze wagi modeli. Baner z informacją o aktualizacji pojawia się również automatycznie na ekranie głównym po wykryciu zaktualizowanych wag.

## Tryb segmentacji

Kontroluje sposób podziału dźwięku na segmenty przed rozpoznawaniem mowy.

| Tryb | Opis |
|---|---|
| **Diaryzacja mówców** | Używa modelu SortFormer do identyfikacji poszczególnych mówców i oznaczania każdego segmentu. Najlepszy do wywiadów, spotkań i nagrań z wieloma mówcami. |
| **Wykrywanie aktywności głosowej** | Używa Silero VAD do wykrywania fragmentów zawierających mowę — bez etykiet mówców. Szybszy od diaryzacji i dobrze dopasowany do nagrań z jednym mówcą. |

## Edytor transkrypcji

**Domyślny tryb odtwarzania** — ustawia tryb odtwarzania używany po otwarciu edytora transkrypcji. Możesz go również zmienić bezpośrednio w edytorze w dowolnym momencie. Opis każdego trybu znajdziesz w artykule [Edytowanie transkrypcji](../operations/editing_transcripts.md).

## Wygląd

Wybierz motyw **Ciemny** lub **Jasny**. Zmiana jest stosowana natychmiast. Zobacz [Wybór motywu](theme.md).

## Język

Wybierz język wyświetlania interfejsu aplikacji. Zmiana jest stosowana natychmiast. Zobacz [Wybór języka](language.md).

---