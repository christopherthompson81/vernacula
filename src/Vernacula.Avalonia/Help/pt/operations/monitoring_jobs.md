---
title: "Monitorando Trabalhos"
description: "Como acompanhar o progresso de um trabalho em execução ou na fila."
topic_id: operations_monitoring_jobs
---

# Monitorando Trabalhos

A visualização de **Progresso** oferece uma visão em tempo real de um trabalho de transcrição em execução.

## Abrindo a Visualização de Progresso

- Ao iniciar uma nova transcrição, o aplicativo vai automaticamente para a visualização de Progresso.
- Para um trabalho já em execução ou na fila, localize-o na tabela de **Histórico de Transcrições** e clique em `Monitor` na coluna **Ações**.

## Lendo a Visualização de Progresso

| Elemento | Descrição |
|---|---|
| Barra de progresso | Percentual de conclusão geral. Indeterminada (animada) enquanto o trabalho está sendo iniciado ou retomado. |
| Rótulo de percentual | Percentual numérico exibido à direita da barra. |
| Mensagem de status | Atividade atual — por exemplo, `Audio Analysis` ou `Speech Recognition`. Exibe `Waiting in queue…` se o trabalho ainda não tiver começado. |
| Tabela de segmentos | Feed ao vivo dos segmentos transcritos com as colunas **Speaker**, **Start**, **End** e **Content**. Rola automaticamente conforme novos segmentos chegam. |

## Fases do Progresso

As fases exibidas dependem do **Modo de Segmentação** selecionado nas Configurações.

**Modo de Diarização de Locutor** (padrão):

1. **Audio Analysis** — a diarização do SortFormer é executada sobre o arquivo inteiro para identificar as fronteiras entre locutores. A barra pode permanecer próxima de 0% até que esta fase seja concluída.
2. **Speech Recognition** — cada segmento de locutor é transcrito. O percentual sobe de forma constante durante esta fase.

**Modo de Detecção de Atividade de Voz**:

1. **Detecting speech segments** — o Silero VAD percorre o arquivo para encontrar regiões com fala. Esta fase é rápida.
2. **Speech Recognition** — cada região de fala detectada é transcrita.

Em ambos os modos, a tabela de segmentos ao vivo é preenchida conforme a transcrição avança.

## Navegando para Outro Local

Clique em `← Back to Home` para retornar à tela inicial sem interromper o trabalho. O trabalho continua sendo executado em segundo plano e seu status é atualizado na tabela de **Histórico de Transcrições**. Clique em `Monitor` novamente a qualquer momento para retornar à visualização de Progresso.

---