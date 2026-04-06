---
title: "L-Installazzjoni ta' CUDA u cuDNN għall-Aċċelerazzjoni tal-GPU"
description: "Kif twaqqaf NVIDIA CUDA u cuDNN sabiex Vernacula-Desktop ikun jista' juża l-GPU tiegħek."
topic_id: first_steps_cuda_installation
---

# L-Installazzjoni ta' CUDA u cuDNN għall-Aċċelerazzjoni tal-GPU

Vernacula-Desktop jista' juża GPU ta' NVIDIA biex jaċċelera t-traskrizzjoni b'mod sinifikanti. L-aċċelerazzjoni tal-GPU teħtieġ li l-NVIDIA CUDA Toolkit u l-libreriji runtime ta' cuDNN ikunu installati fuq is-sistema tiegħek.

## Rekwiżiti

- GPU ta' NVIDIA li jappoġġa CUDA (GeForce GTX 10-series jew aktar reċenti huwa rakkomandat).
- Windows 10 jew 11 (64-bit).
- Il-fajls tal-mudell iridu jkunu diġà mniżżla. Ara [Il-Niżżil tal-Mudelli](downloading_models.md).

## Passi tal-Installazzjoni

### 1. Installa l-CUDA Toolkit

Niżżel u mexxi l-installer tal-CUDA Toolkit mis-sit web tal-iżviluppaturi ta' NVIDIA. Matul l-installazzjoni, aċċetta l-paths default. L-installer jistabbilixxi l-varjabbli ambjentali `CUDA_PATH` awtomatikament — Vernacula-Desktop juża din il-varjabbli biex isib il-libreriji tal-CUDA.

### 2. Installa cuDNN

Niżżel l-arkivju ZIP ta' cuDNN għall-verżjoni tal-CUDA installata tiegħek mis-sit web tal-iżviluppaturi ta' NVIDIA. Estradi l-arkivju u kkopja l-kontenut tal-folders `bin`, `include`, u `lib` tiegħu fil-folders korrispondenti fil-directory tal-installazzjoni tal-CUDA Toolkit tiegħek (il-path muri minn `CUDA_PATH`).

Inkella, installa cuDNN billi tuża l-installer ta' NVIDIA cuDNN jekk ikun disponibbli għall-verżjoni tal-CUDA tiegħek.

### 3. Erġa' Ibda l-Applikazzjoni

Agħlaq u erġa' iftaħ Vernacula-Desktop wara l-installazzjoni. L-applikazzjoni tiċċekkja għal CUDA meta tibda.

## L-Istat tal-GPU fis-Settings

Iftaħ `Settings…` mill-menu bar u ħares fit-taqsima **Hardware & Performance**. Kull komponent juri checkmark (✓) meta jinstabar:

| Oġġett | X'ifisser |
|---|---|
| Isem tal-GPU u l-VRAM | Il-GPU ta' NVIDIA tiegħek instabet |
| CUDA Toolkit ✓ | Il-libreriji tal-CUDA nstabu permezz ta' `CUDA_PATH` |
| cuDNN ✓ | Id-DLLs runtime ta' cuDNN nstabu |
| CUDA Acceleration ✓ | ONNX Runtime lload il-CUDA execution provider |

Jekk xi oġġett ikun nieqes wara l-installazzjoni, ikklikkja `Re-check` biex terġa' tgħaddi mid-detezzjoni tal-hardware mingħajr ma terġa' tibda l-applikazzjoni.

It-tieqa tas-Settings tipprovdi wkoll links diretti għan-niżżil tal-CUDA Toolkit u cuDNN jekk dawn ma jkunux installati għadhom.

### Soluzzjoni tal-Problemi

Jekk `CUDA Acceleration` ma jurux checkmark, ivverifika li:

- Il-varjabbli ambjentali `CUDA_PATH` tkun stabbilita (iċċekkja `System > Advanced system settings > Environment Variables`).
- Id-DLLs ta' cuDNN ikunu f'directory fuq il-`PATH` tas-sistema tiegħek jew ġewwa l-folder `bin` tal-CUDA.
- Id-driver tal-GPU tiegħek ikun aġġornat.

### Batch Sizing

Meta CUDA ikun attiv, it-taqsima **Hardware & Performance** turi wkoll il-limitu dinamiku attwali tal-batch — il-massimum ta' sekondi ta' awdjo pproċessati f'ġirja waħda tal-GPU. Dan il-valur jiġi kkalkolat mill-VRAM libera wara li l-mudelli jitgħabbew u jaġġusta awtomatikament jekk il-memorja disponibbli tiegħek tinbidel.

## It-Tħaddim Mingħajr GPU

Jekk CUDA ma jkunx disponibbli, Vernacula-Desktop jaqa' lura awtomatikament għall-ipproċessar tal-CPU. It-traskrizzjoni xorta taħdem iżda tkun aktar bil-mod, speċjalment għal fajls ta' awdjo twal.

---