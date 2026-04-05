---
title: "Pausar, Retomar ou Remover Trabalhos"
description: "Como pausar um trabalho em execução, retomar um trabalho parado ou excluir um trabalho do histórico."
topic_id: operations_pausing_resuming_removing
---

# Pausar, Retomar ou Remover Trabalhos

## Pausar um Trabalho

Você pode pausar um trabalho em execução ou na fila a partir de dois lugares:

- **Visualização de progresso** — clique em `Pause` no canto inferior direito enquanto acompanha o trabalho ativo.
- **Tabela de Histórico de Transcrições** — clique em `Pause` na coluna **Actions** de qualquer linha cujo status seja `running` ou `queued`.

Após clicar em `Pause`, a linha de status exibe `Pausing…` enquanto o aplicativo conclui a unidade de processamento atual. O status do trabalho então muda para `cancelled` na tabela de histórico.

> Pausar salva todos os segmentos transcritos até o momento. Você pode retomar o trabalho posteriormente sem perder esse progresso.

## Retomar um Trabalho

Para retomar um trabalho pausado ou com falha:

1. Na tela inicial, localize o trabalho na tabela **Transcription History**. Seu status será `cancelled` ou `failed`.
2. Clique em `Resume` na coluna **Actions**.
3. O aplicativo retorna à visualização de **Progress** e continua a partir do ponto onde o processamento foi interrompido.

A linha de status exibe `Resuming…` brevemente enquanto o trabalho é reinicializado.

## Remover um Trabalho

Para excluir permanentemente um trabalho e sua transcrição do histórico:

1. Na tabela **Transcription History**, clique em `Remove` na coluna **Actions** do trabalho que deseja excluir.

O trabalho é removido da lista e seus dados são excluídos do banco de dados local. Esta ação não pode ser desfeita. Os arquivos exportados salvos no disco não são afetados.

---