---
title: "Indstillinger"
description: "Oversigt over alle muligheder i vinduet Indstillinger."
topic_id: first_steps_settings_window
---

# Indstillinger

Vinduet **Indstillinger** giver dig kontrol over hardwarekonfiguration, modelstyring, segmenteringstilstand, editoradfærd, udseende og sprog. Åbn det fra menulinjen: `Settings…`.

## Hardware og ydeevne

Dette afsnit viser status for din NVIDIA GPU og CUDA-softwarestakken samt rapporterer det dynamiske batchloft, der bruges under GPU-transskription.

| Element | Beskrivelse |
|---|---|
| GPU-navn og VRAM | Registreret NVIDIA GPU og tilgængeligt videohukommelse. |
| CUDA Toolkit | Om CUDA-kørselstidsbibliotekerne blev fundet via `CUDA_PATH`. |
| cuDNN | Om cuDNN-kørselstids-DLL'erne er tilgængelige. |
| CUDA-acceleration | Om ONNX Runtime har indlæst CUDA-udførelsesudbyder korrekt. |

Klik på `Re-check` for at køre hardwareregistreringen igen uden at genstarte programmet — nyttigt efter installation af CUDA eller cuDNN.

Direkte downloadlinks til CUDA Toolkit og cuDNN vises, når disse komponenter ikke registreres.

Meddelelsen om **batchloftet** rapporterer, hvor mange sekunders lyd der behandles i hver GPU-kørsel. Denne værdi beregnes ud fra ledig VRAM efter indlæsning af modeller og justeres automatisk.

For en fuldstændig vejledning til CUDA-opsætning, se [Installation af CUDA og cuDNN](cuda_installation.md).

## Modeller

Dette afsnit administrerer de AI-modelfiler, der kræves til transskription.

- **Download Missing Models** — downloader eventuelle modelfiler, der endnu ikke er til stede på disken. En statuslinje og et statusfelt viser fremgangen for hver fil under download.
- **Check for Updates** — kontrollerer, om der er nyere modelvægte tilgængelige. Et opdateringsbanner vises også automatisk på startskærmen, når opdaterede vægte registreres.

## Segmenteringstilstand

Styrer, hvordan lyden opdeles i segmenter inden talegenkendelse.

| Tilstand | Beskrivelse |
|---|---|
| **Talerdiarisering** | Bruger SortFormer-modellen til at identificere individuelle talere og mærke hvert segment. Bedst egnet til interviews, møder og optagelser med flere talere. |
| **Stemmeaktivitetsregistrering** | Bruger Silero VAD til kun at registrere taleregioner — ingen talermærker. Hurtigere end diarisering og velegnet til lyd med én taler. |

## Transskriptionseditor

**Standardafspilningstilstand** — angiver den afspilningstilstand, der bruges, når du åbner transskriptionseditoren. Du kan også ændre den direkte i editoren når som helst. Se [Redigering af transskriptioner](../operations/editing_transcripts.md) for en beskrivelse af hver tilstand.

## Udseende

Vælg **Mørkt** eller **Lyst** tema. Ændringen træder i kraft med det samme. Se [Valg af tema](theme.md).

## Sprog

Vælg det viste sprog for programgrænsefladen. Ændringen træder i kraft med det samme. Se [Valg af sprog](language.md).

---