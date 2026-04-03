---
title: "Carregando Trabalhos Concluídos"
description: "Como abrir os resultados de uma transcrição concluída anteriormente."
topic_id: operations_loading_completed_jobs
---

# Carregando Trabalhos Concluídos

Todos os trabalhos de transcrição concluídos são salvos no banco de dados local e permanecem acessíveis na tabela **Histórico de Transcrições** na tela inicial.

## Como Carregar um Trabalho Concluído

1. Na tela inicial, localize o trabalho na tabela **Histórico de Transcrições**. Os trabalhos concluídos exibem um selo de status `complete`.
2. Clique em `Load` na coluna **Ações** do trabalho.
3. O aplicativo muda para a visualização de **Resultados**, exibindo todos os segmentos transcritos daquele trabalho.

## Visualização de Resultados

A visualização de Resultados exibe:

- O nome do arquivo de áudio como título da página.
- Um subtítulo com a contagem de segmentos (por exemplo, `42 segment(s)`).
- Uma tabela de segmentos com as colunas **Falante**, **Início**, **Fim** e **Conteúdo**.

Na visualização de Resultados você pode:

- [Editar a transcrição](editing_transcripts.md) — revisar e corrigir o texto, ajustar o tempo, mesclar ou dividir segmentos e verificar os segmentos enquanto ouve o áudio.
- [Editar nomes de falantes](editing_speaker_names.md) — substituir IDs genéricos, como `speaker_0`, por nomes reais.
- [Exportar a transcrição](exporting_results.md) — salvar a transcrição em Excel, CSV, JSON, SRT, Markdown, Word ou SQLite.

Para retornar à lista do histórico, clique em `← Back to History`.

---