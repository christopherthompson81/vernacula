---
title: "Eksportowanie wyników lub transkrypcji"
description: "Jak zapisać transkrypcję do pliku w różnych formatach."
topic_id: operations_exporting_results
---

# Eksportowanie wyników lub transkrypcji

Ukończoną transkrypcję możesz wyeksportować do kilku formatów plików, aby wykorzystać ją w innych aplikacjach.

## Jak eksportować

1. Otwórz ukończone zadanie. Zobacz [Ładowanie ukończonych zadań](loading_completed_jobs.md).
2. W widoku **Wyniki** kliknij `Export Transcript`.
3. Otworzy się okno dialogowe **Export Transcript**. Wybierz format z listy rozwijanej **Format**.
4. Kliknij `Save`. Otworzy się okno dialogowe zapisu pliku.
5. Wybierz folder docelowy i nazwę pliku, a następnie kliknij **Zapisz**.

Na dole okna dialogowego pojawi się komunikat potwierdzający z pełną ścieżką zapisanego pliku.

## Dostępne formaty

| Format | Rozszerzenie | Najlepsze zastosowanie |
|---|---|---|
| Excel | `.xlsx` | Analiza w arkuszu kalkulacyjnym z kolumnami dla prelegenta, znaczników czasu i treści. |
| CSV | `.csv` | Import do dowolnego arkusza kalkulacyjnego lub narzędzia do analizy danych. |
| JSON | `.json` | Przetwarzanie programistyczne. |
| SRT Subtitles | `.srt` | Wczytywanie do edytorów wideo lub odtwarzaczy multimedialnych jako napisy. |
| Markdown | `.md` | Czytelne dokumenty w formacie zwykłego tekstu. |
| Word Document | `.docx` | Udostępnianie użytkownikom programu Microsoft Word. |
| SQLite Database | `.db` | Pełny eksport bazy danych do niestandardowych zapytań. |

## Nazwy prelegentów w eksportach

Jeśli przypisałeś prelegentom nazwy wyświetlane, są one używane we wszystkich formatach eksportu. Aby zaktualizować nazwy przed eksportem, najpierw kliknij `Edit Speaker Names`. Zobacz [Edytowanie nazw prelegentów](editing_speaker_names.md).

---