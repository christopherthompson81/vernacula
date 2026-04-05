---
title: "Enfileirar Vários Arquivos de Áudio"
description: "Como adicionar vários arquivos de áudio à fila de trabalhos de uma só vez."
topic_id: operations_bulk_add_jobs
---

# Enfileirar Vários Arquivos de Áudio

Use **Adicionar Trabalhos em Lote** para enfileirar vários arquivos de áudio ou vídeo para transcrição em uma única etapa. O aplicativo os processa um por vez, na ordem em que foram adicionados.

## Pré-requisitos

- Todos os arquivos de modelo devem estar baixados. O cartão **Status do Modelo** deve exibir `All N model file(s) present ✓`. Consulte [Baixar Modelos](../first_steps/downloading_models.md).

## Como Adicionar Trabalhos em Lote

1. Na tela inicial, clique em `Bulk Add Jobs`.
2. Um seletor de arquivos será aberto. Selecione um ou mais arquivos de áudio ou vídeo — mantenha `Ctrl` ou `Shift` pressionado para selecionar vários arquivos.
3. Clique em **Abrir**. Cada arquivo selecionado é adicionado à tabela **Histórico de Transcrições** como um trabalho separado.

> **Arquivos de vídeo com múltiplas faixas de áudio:** Se um arquivo de vídeo contiver mais de uma faixa de áudio (por exemplo, vários idiomas ou uma faixa de comentários do diretor), o aplicativo criará um trabalho por faixa automaticamente.

## Nomes dos Trabalhos

Cada trabalho recebe automaticamente o nome do arquivo de áudio correspondente. Você pode renomear um trabalho a qualquer momento clicando em seu nome na coluna **Título** da tabela Histórico de Transcrições, editando o texto e pressionando `Enter` ou clicando em outro lugar.

## Comportamento da Fila

- Se nenhum trabalho estiver em execução no momento, o primeiro arquivo é iniciado imediatamente e os demais são exibidos como `queued`.
- Se um trabalho já estiver em execução, todos os arquivos recém-adicionados serão exibidos como `queued` e serão iniciados automaticamente em sequência.
- Para monitorar o trabalho ativo, clique em `Monitor` na coluna **Ações** correspondente. Consulte [Monitorar Trabalhos](monitoring_jobs.md).
- Para pausar ou remover um trabalho enfileirado antes que ele seja iniciado, use os botões `Pause` ou `Remove` na coluna **Ações** correspondente. Consulte [Pausar, Retomar ou Remover Trabalhos](pausing_resuming_removing.md).

---