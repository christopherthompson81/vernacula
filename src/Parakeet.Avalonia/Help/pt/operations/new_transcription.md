---
title: "Novo Fluxo de Transcrição"
description: "Guia passo a passo para transcrever um arquivo de áudio."
topic_id: operations_new_transcription
---

# Novo Fluxo de Transcrição

Use este fluxo de trabalho para transcrever um único arquivo de áudio.

## Pré-requisitos

- Todos os arquivos de modelo devem estar baixados. O cartão **Status do Modelo** deve exibir `All N model file(s) present ✓`. Consulte [Baixando Modelos](../first_steps/downloading_models.md).

## Formatos Suportados

### Áudio

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Vídeo

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Arquivos de vídeo são decodificados via FFmpeg. Se um arquivo de vídeo contiver **múltiplos fluxos de áudio** (por exemplo, múltiplos idiomas ou faixas de comentários), um trabalho de transcrição é criado automaticamente para cada fluxo.

## Etapas

### 1. Abrir o formulário de Nova Transcrição

Clique em `New Transcription` na tela inicial ou acesse `File > New Transcription`.

### 2. Selecionar um arquivo de mídia

Clique em `Browse…` ao lado do campo **Audio File**. Um seletor de arquivos é aberto filtrado para os formatos de áudio e vídeo suportados. Selecione seu arquivo e clique em **Open**. O caminho do arquivo aparece no campo.

### 3. Nomear o trabalho

O campo **Job Name** é preenchido automaticamente com o nome do arquivo. Edite-o se desejar um rótulo diferente — esse nome aparece no Histórico de Transcrições na tela inicial.

### 4. Iniciar a transcrição

Clique em `Start Transcription`. O aplicativo muda para a visualização de **Progress**.

Para voltar sem iniciar, clique em `← Back`.

## O Que Acontece a Seguir

O trabalho percorre duas fases exibidas na barra de progresso:

1. **Audio Analysis** — diarização de locutores: identificação de quem está falando e quando.
2. **Speech Recognition** — conversão de fala em texto, segmento por segmento.

Os segmentos transcritos aparecem na tabela ao vivo conforme são produzidos. Quando o processamento é concluído, o aplicativo move automaticamente para a visualização de **Results**.

Se você adicionar um trabalho enquanto outro já estiver em execução, o novo trabalho exibirá o status `queued` e iniciará quando o trabalho atual terminar. Consulte [Monitorando Trabalhos](monitoring_jobs.md).

---