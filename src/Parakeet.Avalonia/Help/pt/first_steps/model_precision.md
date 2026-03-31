---
title: "Escolhendo a Precisão dos Pesos do Modelo"
description: "Como escolher entre a precisão INT8 e FP32 do modelo e quais são as compensações envolvidas."
topic_id: first_steps_model_precision
---

# Escolhendo a Precisão dos Pesos do Modelo

A precisão do modelo controla o formato numérico utilizado pelos pesos do modelo de IA. Ela afeta o tamanho do download, o uso de memória e a exatidão dos resultados.

## Opções de Precisão

### INT8 (download menor)

- Arquivos de modelo menores — download mais rápido e menos espaço em disco necessário.
- Precisão ligeiramente inferior em alguns áudios.
- Recomendado se você tiver espaço em disco limitado ou uma conexão de internet mais lenta.

### FP32 (mais preciso)

- Arquivos de modelo maiores.
- Maior precisão, especialmente em áudios difíceis com sotaques ou ruído de fundo.
- Recomendado quando a precisão é a prioridade e você tem espaço em disco suficiente.
- Obrigatório para aceleração por GPU CUDA — o caminho de GPU sempre usa FP32, independentemente desta configuração.

## Como Alterar a Precisão

Abra `Settings…` na barra de menus, vá até a seção **Models** e selecione `INT8 (smaller download)` ou `FP32 (more accurate)`.

## Após Alterar a Precisão

Alterar a precisão requer um conjunto diferente de arquivos de modelo. Se os modelos da nova precisão ainda não foram baixados, clique em `Download Missing Models` nas Configurações. Os arquivos baixados anteriormente para a outra precisão são mantidos no disco e não precisam ser baixados novamente caso você queira reverter a escolha.

---