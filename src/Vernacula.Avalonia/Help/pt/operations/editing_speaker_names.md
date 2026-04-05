---
title: "Editando Nomes de Falantes"
description: "Como substituir IDs de falantes genéricos por nomes reais em uma transcrição."
topic_id: operations_editing_speaker_names
---

# Editando Nomes de Falantes

O mecanismo de transcrição rotula automaticamente cada falante com um ID genérico (por exemplo, `speaker_0`, `speaker_1`). Você pode substituí-los por nomes reais que aparecerão em toda a transcrição e nos arquivos exportados.

## Como Editar Nomes de Falantes

1. Abra um trabalho concluído. Consulte [Carregando Trabalhos Concluídos](loading_completed_jobs.md).
2. Na visualização **Resultados**, clique em `Edit Speaker Names`.
3. O diálogo **Edit Speaker Names** será aberto com duas colunas:
   - **Speaker ID** — o rótulo original atribuído pelo modelo (somente leitura).
   - **Display Name** — o nome exibido na transcrição (editável).
4. Clique em uma célula na coluna **Display Name** e digite o nome do falante.
5. Pressione `Tab` ou clique em outra linha para ir ao próximo falante.
6. Clique em `Save` para aplicar as alterações ou em `Cancel` para descartá-las.

## Onde os Nomes Aparecem

Os nomes de exibição atualizados substituem os IDs genéricos em:

- A tabela de segmentos na visualização de Resultados.
- Todos os arquivos exportados (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Editando Nomes Novamente

Você pode reabrir o diálogo Edit Speaker Names a qualquer momento enquanto o trabalho estiver carregado na visualização de Resultados. As alterações são salvas no banco de dados local e persistem entre as sessões.

---