---
title: "Taken bewaken"
description: "Hoe u de voortgang van een actieve of wachtende taak kunt volgen."
topic_id: operations_monitoring_jobs
---

# Taken bewaken

De weergave **Voortgang** biedt u een live overzicht van een actieve transcriptietaak.

## De voortgangsweergave openen

- Wanneer u een nieuwe transcriptie start, gaat de applicatie automatisch naar de voortgangsweergave.
- Voor een taak die al actief is of in de wachtrij staat, zoekt u deze op in de tabel **Transcriptiegeschiedenis** en klikt u op `Monitor` in de kolom **Acties**.

## De voortgangsweergave begrijpen

| Element | Beschrijving |
|---|---|
| Voortgangsbalk | Algeheel voltooiingspercentage. Onbepaald (geanimeerd) terwijl de taak wordt gestart of hervat. |
| Percentagelabel | Numeriek percentage weergegeven rechts van de balk. |
| Statusbericht | Huidige activiteit — bijvoorbeeld `Audio Analysis` of `Speech Recognition`. Toont `Waiting in queue…` als de taak nog niet is gestart. |
| Segmententabel | Live overzicht van getranscribeerde segmenten met de kolommen **Spreker**, **Begin**, **Einde** en **Inhoud**. Schuift automatisch mee naarmate nieuwe segmenten binnenkomen. |

## Voortgangsfasen

Welke fasen worden weergegeven, hangt af van de **Segmentatiemodus** die is geselecteerd in de instellingen.

**Modus Sprekersdiarisering** (standaard):

1. **Audio Analysis** — SortFormer-diarisering wordt uitgevoerd over het hele bestand om sprekersgrenzen te bepalen. De balk blijft mogelijk dicht bij 0% totdat deze fase is voltooid.
2. **Speech Recognition** — elk sprekersegment wordt getranscribeerd. Het percentage stijgt gestaag tijdens deze fase.

**Modus Stemactiviteitsdetectie**:

1. **Detecting speech segments** — Silero VAD scant het bestand om spraakfragmenten te vinden. Deze fase verloopt snel.
2. **Speech Recognition** — elk gedetecteerd spraakfragment wordt getranscribeerd.

In beide modi wordt de live segmententabel gevuld naarmate de transcriptie vordert.

## Navigeren naar een andere weergave

Klik op `← Back to Home` om terug te keren naar het startscherm zonder de taak te onderbreken. De taak blijft op de achtergrond actief en de status wordt bijgewerkt in de tabel **Transcriptiegeschiedenis**. Klik op elk gewenst moment opnieuw op `Monitor` om terug te keren naar de voortgangsweergave.

---