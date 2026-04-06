---
title: "Inställningar"
description: "Översikt över alla alternativ i fönstret Inställningar."
topic_id: first_steps_settings_window
---

# Inställningar

Fönstret **Inställningar** ger dig kontroll över maskinvarukonfiguration, modellhantering, segmenteringsläge, redigeringsbeteende, utseende och språk. Öppna det från menyraden: `Settings…`.

## Maskinvara och prestanda

Det här avsnittet visar status för ditt NVIDIA GPU och CUDA-programvarustacken, samt rapporterar den dynamiska batchgränsen som används vid GPU-transkription.

| Post | Beskrivning |
|---|---|
| GPU-namn och VRAM | Identifierat NVIDIA GPU och tillgängligt videominne. |
| CUDA Toolkit | Om CUDA-körtidsbiblioteken hittades via `CUDA_PATH`. |
| cuDNN | Om cuDNN-körtids-DLL-filerna är tillgängliga. |
| CUDA-acceleration | Om ONNX Runtime lyckades läsa in CUDA-exekveringsleverantören. |

Klicka på `Re-check` för att köra om maskinvaruidentifieringen utan att starta om programmet — användbart efter installation av CUDA eller cuDNN.

Direkta nedladdningslänkar för CUDA Toolkit och cuDNN visas när dessa komponenter inte identifieras.

Meddelandet om **batchgräns** rapporterar hur många sekunders ljud som bearbetas i varje GPU-körning. Detta värde beräknas utifrån ledigt VRAM efter att modellerna har lästs in och justeras automatiskt.

För fullständiga installationsanvisningar för CUDA, se [Installera CUDA och cuDNN](cuda_installation.md).

## Modeller

Det här avsnittet hanterar de AI-modellfiler som krävs för transkription.

- **Ladda ned saknade modeller** — laddar ned modellfiler som ännu inte finns på disk. Ett förloppsindikator och en statusrad visar varje fil allt eftersom den laddas ned.
- **Sök efter uppdateringar** — kontrollerar om nyare modellvikter finns tillgängliga. En uppdateringsbanner visas även automatiskt på startskärmen när uppdaterade vikter identifieras.

## Segmenteringsläge

Styr hur ljudet delas upp i segment innan taligenkänning.

| Läge | Beskrivning |
|---|---|
| **Talaridentifiering** | Använder SortFormer-modellen för att identifiera enskilda talare och märka varje segment. Bäst lämpad för intervjuer, möten och inspelningar med flera talare. |
| **Röstaktivitetsidentifiering** | Använder Silero VAD för att enbart identifiera talregioner — utan talaretiketter. Snabbare än talaridentifiering och väl lämpad för ljud med en enda talare. |

## Transkriptionsredigerare

**Standarduppspelningsläge** — anger det uppspelningsläge som används när du öppnar transkriptionsredigeraren. Du kan även ändra det direkt i redigeraren när som helst. Se [Redigera transkript](../operations/editing_transcripts.md) för en beskrivning av varje läge.

## Utseende

Välj temat **Mörkt** eller **Ljust**. Ändringen tillämpas omedelbart. Se [Välja ett tema](theme.md).

## Språk

Välj visningsspråk för programgränssnittet. Ändringen tillämpas omedelbart. Se [Välja ett språk](language.md).

---