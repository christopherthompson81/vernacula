---
title: "Instalando CUDA e cuDNN para Aceleração por GPU"
description: "Como configurar o NVIDIA CUDA e o cuDNN para que o Vernacula-Desktop possa usar sua GPU."
topic_id: first_steps_cuda_installation
---

# Instalando CUDA e cuDNN para Aceleração por GPU

O Vernacula-Desktop pode usar uma GPU NVIDIA para acelerar significativamente a transcrição. A aceleração por GPU requer que o NVIDIA CUDA Toolkit e as bibliotecas de tempo de execução do cuDNN estejam instalados no seu sistema.

## Requisitos

- Uma GPU NVIDIA compatível com CUDA (recomenda-se a GeForce GTX série 10 ou mais recente).
- Windows 10 ou 11 (64 bits).
- Os arquivos de modelo já devem ter sido baixados. Consulte [Baixando Modelos](downloading_models.md).

## Etapas de Instalação

### 1. Instalar o CUDA Toolkit

Baixe e execute o instalador do CUDA Toolkit no site de desenvolvedores da NVIDIA. Durante a instalação, aceite os caminhos padrão. O instalador define automaticamente a variável de ambiente `CUDA_PATH` — o Vernacula-Desktop usa essa variável para localizar as bibliotecas CUDA.

### 2. Instalar o cuDNN

Baixe o arquivo ZIP do cuDNN correspondente à versão do CUDA instalada no site de desenvolvedores da NVIDIA. Extraia o arquivo e copie o conteúdo das pastas `bin`, `include` e `lib` para as pastas correspondentes dentro do diretório de instalação do CUDA Toolkit (o caminho indicado por `CUDA_PATH`).

Como alternativa, instale o cuDNN usando o instalador oficial da NVIDIA, caso ele esteja disponível para a sua versão do CUDA.

### 3. Reiniciar o Aplicativo

Feche e reabra o Vernacula-Desktop após a instalação. O aplicativo verifica a presença do CUDA na inicialização.

## Status da GPU nas Configurações

Abra `Settings…` na barra de menus e localize a seção **Hardware & Performance**. Cada componente exibe uma marca de verificação (✓) quando detectado:

| Item | O que significa |
|---|---|
| Nome da GPU e VRAM | Sua GPU NVIDIA foi encontrada |
| CUDA Toolkit ✓ | Bibliotecas CUDA localizadas via `CUDA_PATH` |
| cuDNN ✓ | DLLs de tempo de execução do cuDNN encontradas |
| CUDA Acceleration ✓ | O ONNX Runtime carregou o provedor de execução CUDA |

Se algum item estiver ausente após a instalação, clique em `Re-check` para executar novamente a detecção de hardware sem reiniciar o aplicativo.

A janela de Configurações também fornece links diretos para download do CUDA Toolkit e do cuDNN, caso ainda não estejam instalados.

### Solução de Problemas

Se `CUDA Acceleration` não exibir uma marca de verificação, verifique se:

- A variável de ambiente `CUDA_PATH` está definida (verifique em `System > Advanced system settings > Environment Variables`).
- As DLLs do cuDNN estão em um diretório presente no `PATH` do sistema ou dentro da pasta `bin` do CUDA.
- O driver da sua GPU está atualizado.

### Dimensionamento de Lote

Quando o CUDA está ativo, a seção **Hardware & Performance** também exibe o limite dinâmico de lote atual — o número máximo de segundos de áudio processados em uma única execução na GPU. Esse valor é calculado com base na VRAM disponível após o carregamento dos modelos e é ajustado automaticamente caso a memória disponível mude.

## Executando Sem uma GPU

Se o CUDA não estiver disponível, o Vernacula-Desktop retorna automaticamente ao processamento por CPU. A transcrição continuará funcionando, porém de forma mais lenta, especialmente para arquivos de áudio longos.

---