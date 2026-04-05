---
title: "Tööde jälgimine"
description: "Kuidas jälgida töötava või järjekorras oleva töö edenemist."
topic_id: operations_monitoring_jobs
---

# Tööde jälgimine

**Edenemise** vaade annab teile reaalajas ülevaate käimasolevast transkriptsioonitööst.

## Edenemise vaate avamine

- Kui alustate uut transkriptsiooni, läheb rakendus automaatselt edenemise vaatesse.
- Juba töötava või järjekorras oleva töö puhul leidke see **Transkriptsiooni ajaloo** tabelist ja klõpsake selle **Toimingute** veerus nuppu `Monitor`.

## Edenemise vaate lugemine

| Element | Kirjeldus |
|---|---|
| Edenemisriba | Üldine lõpetamise protsent. Määramatu (animeeritud) kuni töö käivitub või jätkub. |
| Protsendi silt | Arvuline protsent, mis kuvatakse riba paremal pool. |
| Olekuteade | Praegune tegevus — näiteks `Audio Analysis` või `Speech Recognition`. Näitab `Waiting in queue…`, kui töö pole veel alanud. |
| Segmentide tabel | Transkribeeritud segmentide reaalajas voog veergudega **Speaker**, **Start**, **End** ja **Content**. Kerib automaatselt, kui uued segmendid saabuvad. |

## Edenemise faasid

Kuvatavad faasid sõltuvad seadistustes valitud **segmenteerimisrežiimist**.

**Kõneleja diariseerimine** (vaikimisi):

1. **Audio Analysis** — SortFormer diariseerimine käib kogu faili üle, et tuvastada kõnelejate piirid. Riba võib jääda 0% lähedale, kuni see faas lõpeb.
2. **Speech Recognition** — iga kõneleja segment transkribeeritakse. Protsent tõuseb selle faasi jooksul ühtlaselt.

**Hääleaktiivsuse tuvastamise režiim**:

1. **Detecting speech segments** — Silero VAD skannib faili kõnepiirkondade leidmiseks. See faas on kiire.
2. **Speech Recognition** — iga tuvastatud kõnepiirkond transkribeeritakse.

Mõlemas režiimis täitub reaalajas segmentide tabel transkriptsiooni edenedes.

## Vaate sulgemine

Klõpsake `← Back to Home`, et naasta avalehele ilma tööd katkestamata. Töö jätkub taustal ja selle olek uueneb **Transkriptsiooni ajaloo** tabelis. Klõpsake igal ajal uuesti `Monitor`, et naasta edenemise vaatesse.

---