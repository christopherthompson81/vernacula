---
title: "Mitme helifaili järjekorda lisamine"
description: "Kuidas lisada mitu helifaili korraga tööjärjekorda."
topic_id: operations_bulk_add_jobs
---

# Mitme helifaili järjekorda lisamine

Kasuta funktsiooni **Bulk Add Jobs**, et lisada mitu heli- või videofaili transkribeerimise järjekorda ühe sammuga. Rakendus töötleb neid ükshaaval lisamise järjekorras.

## Eeltingimused

- Kõik mudelifailid peavad olema alla laaditud. **Model Status** kaardil peab olema kuvatud `All N model file(s) present ✓`. Vaata [Mudelite allalaadimine](../first_steps/downloading_models.md).

## Kuidas kasutada funktsiooni Bulk Add Jobs

1. Avakuval klõpsa `Bulk Add Jobs`.
2. Avaneb failiválija. Vali üks või mitu heli- või videofaili — mitme faili valimiseks hoia all `Ctrl` või `Shift`.
3. Klõpsa **Open**. Iga valitud fail lisatakse **Transcription History** tabelisse eraldi tööna.

> **Mitme helivooguga videofailid:** Kui videofail sisaldab rohkem kui ühte helivoogu (näiteks mitu keelt või režissööri kommentaaride rada), loob rakendus automaatselt ühe töö iga helivoo kohta.

## Tööde nimed

Iga töö saab nime automaatselt vastava helifaili nime põhjal. Töö nime saab igal ajal muuta, klõpsates selle nimel **Title** veerus Transcription History tabelis, muutes teksti ning vajutades `Enter` või klõpsates mujale.

## Järjekorra käitumine

- Kui ükski töö ei ole parasjagu käimas, algab esimene fail kohe ja ülejäänud kuvatakse olekuga `queued`.
- Kui mõni töö on juba käimas, kuvatakse kõik äsja lisatud failid olekuga `queued` ning need alustavad automaatselt järjest.
- Aktiivse töö jälgimiseks klõpsa selle **Actions** veerus `Monitor`. Vaata [Tööde jälgimine](monitoring_jobs.md).
- Järjekorras oleva töö peatamiseks või eemaldamiseks enne selle algust kasuta selle **Actions** veerus olevaid nuppe `Pause` või `Remove`. Vaata [Tööde peatamine, jätkamine või eemaldamine](pausing_resuming_removing.md).

---