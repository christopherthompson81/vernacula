---
title: "Baixando Modelos"
description: "Como baixar os arquivos de modelo de IA necessários para a transcrição."
topic_id: first_steps_downloading_models
---

# Baixando Modelos

O Parakeet Transcription requer arquivos de modelo de IA para funcionar. Esses arquivos não estão incluídos no aplicativo e precisam ser baixados antes da sua primeira transcrição.

## Status do Modelo (Tela Inicial)

Uma linha de status compacta na parte superior da tela inicial indica se os seus modelos estão prontos. Quando algum arquivo estiver ausente, ela também exibe um botão `Open Settings` que leva diretamente ao gerenciamento de modelos.

| Status | Significado |
|---|---|
| `All N model file(s) present ✓` | Todos os arquivos necessários foram baixados e estão prontos. |
| `N model file(s) missing: …` | Um ou mais arquivos estão ausentes; abra as Configurações para baixá-los. |

Quando os modelos estiverem prontos, os botões `New Transcription` e `Bulk Add Jobs` ficam ativos.

## Como Baixar os Modelos

1. Na tela inicial, clique em `Open Settings` (ou vá para `Settings… > Models`).
2. Na seção **Models**, clique em `Download Missing Models`.
3. Uma barra de progresso e uma linha de status são exibidas, mostrando o arquivo atual, sua posição na fila e o tamanho do download — por exemplo: `[1/3] encoder-model.onnx — 42 MB`.
4. Aguarde até que o status exiba `Download complete.`

## Cancelando um Download

Para interromper um download em andamento, clique em `Cancel`. A linha de status exibirá `Download cancelled.` Os arquivos parcialmente baixados são preservados, de modo que o download é retomado do ponto onde parou na próxima vez que você clicar em `Download Missing Models`.

## Erros de Download

Se um download falhar, a linha de status exibirá `Download failed: <reason>`. Verifique sua conexão com a internet e clique em `Download Missing Models` novamente para tentar de novo. O aplicativo retoma a partir do último arquivo concluído com sucesso.

## Alterando a Precisão

Os arquivos de modelo que precisam ser baixados dependem da **Model Precision** selecionada. Para alterá-la, vá para `Settings… > Models > Model Precision`. Se você mudar a precisão após o download, o novo conjunto de arquivos precisará ser baixado separadamente. Consulte [Escolhendo a Precisão dos Pesos do Modelo](model_precision.md).

---