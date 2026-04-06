---
title: "Configurações"
description: "Visão geral de todas as opções na janela de Configurações."
topic_id: first_steps_settings_window
---

# Configurações

A janela de **Configurações** oferece controle sobre a configuração de hardware, gerenciamento de modelos, modo de segmentação, comportamento do editor, aparência e idioma. Abra-a pela barra de menus: `Settings…`.

## Hardware e Desempenho

Esta seção exibe o status da sua GPU NVIDIA e da pilha de software CUDA, além de informar o limite dinâmico de lote utilizado durante a transcrição por GPU.

| Item | Descrição |
|---|---|
| Nome da GPU e VRAM | GPU NVIDIA detectada e memória de vídeo disponível. |
| CUDA Toolkit | Indica se as bibliotecas de tempo de execução do CUDA foram encontradas via `CUDA_PATH`. |
| cuDNN | Indica se as DLLs de tempo de execução do cuDNN estão disponíveis. |
| Aceleração CUDA | Indica se o ONNX Runtime carregou com êxito o provedor de execução CUDA. |

Clique em `Re-check` para executar novamente a detecção de hardware sem reiniciar o aplicativo — útil após instalar o CUDA ou o cuDNN.

Links de download direto para o CUDA Toolkit e o cuDNN são exibidos quando esses componentes não são detectados.

A mensagem de **limite de lote** informa quantos segundos de áudio são processados em cada execução na GPU. Esse valor é calculado com base na VRAM livre após o carregamento dos modelos e é ajustado automaticamente.

Para instruções completas de configuração do CUDA, consulte [Instalando CUDA e cuDNN](cuda_installation.md).

## Modelos

Esta seção gerencia os arquivos de modelo de IA necessários para a transcrição.

- **Baixar Modelos Ausentes** — baixa os arquivos de modelo que ainda não estão presentes no disco. Uma barra de progresso e uma linha de status acompanham cada arquivo durante o download.
- **Verificar Atualizações** — verifica se há versões mais recentes dos pesos dos modelos disponíveis. Um banner de atualização também é exibido automaticamente na tela inicial quando pesos atualizados são detectados.

## Modo de Segmentação

Controla como o áudio é dividido em segmentos antes do reconhecimento de fala.

| Modo | Descrição |
|---|---|
| **Diarização de Falantes** | Usa o modelo SortFormer para identificar falantes individuais e rotular cada segmento. Ideal para entrevistas, reuniões e gravações com múltiplos falantes. |
| **Detecção de Atividade de Voz** | Usa o Silero VAD para detectar apenas as regiões com fala — sem rótulos de falantes. Mais rápido do que a diarização e adequado para áudios com um único falante. |

## Editor de Transcrição

**Modo de Reprodução Padrão** — define o modo de reprodução utilizado ao abrir o editor de transcrição. Você também pode alterá-lo diretamente no editor a qualquer momento. Consulte [Editando Transcrições](../operations/editing_transcripts.md) para uma descrição de cada modo.

## Aparência

Selecione o tema **Escuro** ou **Claro**. A alteração é aplicada imediatamente. Consulte [Escolhendo um Tema](theme.md).

## Idioma

Selecione o idioma de exibição da interface do aplicativo. A alteração é aplicada imediatamente. Consulte [Escolhendo um Idioma](language.md).

---