---
title: "Editando Transcrições"
description: "Como revisar, corrigir e verificar segmentos transcritos no editor de transcrições."
topic_id: operations_editing_transcripts
---

# Editando Transcrições

O **Editor de Transcrições** permite revisar a saída do ASR, corrigir texto, renomear locutores diretamente no editor, ajustar o tempo dos segmentos e marcar segmentos como verificados — tudo isso enquanto ouve o áudio original.

## Abrindo o Editor

1. Carregue um trabalho concluído (consulte [Carregando Trabalhos Concluídos](loading_completed_jobs.md)).
2. Na visualização **Resultados**, clique em `Edit Transcript`.

O editor abre em uma janela separada e pode permanecer aberto ao lado da aplicação principal.

## Layout

Cada segmento é exibido como um cartão com dois painéis lado a lado:

- **Painel esquerdo** — a saída original do ASR com coloração de confiança por palavra. Palavras sobre as quais o modelo tinha menos certeza aparecem em vermelho; palavras com alta confiança aparecem na cor de texto normal.
- **Painel direito** — uma caixa de texto editável. Faça as correções aqui; a diferença em relação ao original é destacada conforme você digita.

O rótulo do locutor e o intervalo de tempo aparecem acima de cada cartão. Clique em um cartão para focalizá-lo e revelar seus ícones de ação. Passe o cursor sobre qualquer ícone para ver uma dica de ferramenta descrevendo sua função.

## Legenda de Ícones

### Barra de Reprodução

| Ícone | Ação |
|-------|------|
| ▶ | Reproduzir |
| ⏸ | Pausar |
| ⏮ | Ir para o segmento anterior |
| ⏭ | Ir para o próximo segmento |

### Ações do Cartão de Segmento

| Ícone | Ação |
|-------|------|
| <mdl2 ch="E77B"/> | Reatribuir o segmento a um locutor diferente |
| <mdl2 ch="E916"/> | Ajustar os tempos de início e fim do segmento |
| <mdl2 ch="EA39"/> | Suprimir ou remover a supressão do segmento |
| <mdl2 ch="E72B"/> | Mesclar com o segmento anterior |
| <mdl2 ch="E72A"/> | Mesclar com o próximo segmento |
| <mdl2 ch="E8C6"/> | Dividir o segmento |
| <mdl2 ch="E72C"/> | Refazer o ASR neste segmento |

## Reprodução de Áudio

Uma barra de reprodução percorre a parte superior da janela do editor:

| Controle | Ação |
|----------|------|
| Ícone Reproduzir / Pausar | Iniciar ou pausar a reprodução |
| Barra de busca | Arraste para ir a qualquer posição no áudio |
| Controle deslizante de velocidade | Ajustar a velocidade de reprodução (0,5× – 2×) |
| Ícones Anterior / Próximo | Ir para o segmento anterior ou seguinte |
| Menu suspenso de modo de reprodução | Selecionar um dos três modos de reprodução (veja abaixo) |
| Controle deslizante de volume | Ajustar o volume de reprodução |

Durante a reprodução, a palavra sendo falada no momento é destacada no painel esquerdo. Quando pausado após uma busca, o destaque é atualizado para a palavra na posição buscada.

### Modos de Reprodução

| Modo | Comportamento |
|------|---------------|
| `Single` | Reproduz o segmento atual uma vez e para. |
| `Auto-advance` | Reproduz o segmento atual; ao terminar, marca-o como verificado e avança para o próximo. |
| `Continuous` | Reproduz todos os segmentos em sequência sem marcar nenhum como verificado. |

Selecione o modo ativo no menu suspenso da barra de reprodução.

## Editando um Segmento

1. Clique em um cartão para focalizá-lo.
2. Edite o texto no painel direito. As alterações são salvas automaticamente quando você move o foco para outro cartão.

## Renomeando um Locutor

Clique no rótulo do locutor dentro do cartão focalizado e digite um novo nome. Pressione `Enter` ou clique fora para salvar. O novo nome é aplicado somente a esse cartão; para renomear um locutor globalmente, use [Editar Nomes de Locutores](editing_speaker_names.md) na visualização Resultados.

## Verificando um Segmento

Clique na caixa de seleção `Verified` em um cartão focalizado para marcá-lo como revisado. O status de verificação é salvo no banco de dados e fica visível no editor em carregamentos futuros.

## Suprimindo um Segmento

Clique em `Suppress` em um cartão focalizado para ocultar o segmento das exportações (útil para ruídos, música ou outras seções sem fala). Clique em `Unsuppress` para restaurá-lo.

## Ajustando os Tempos do Segmento

Clique em `Adjust Times` em um cartão focalizado para abrir o diálogo de ajuste de tempo. Use a roda do mouse sobre o campo **Start** ou **End** para ajustar o valor em incrementos de 0,1 segundo, ou digite um valor diretamente. Clique em `Save` para aplicar.

## Mesclando Segmentos

- Clique em `⟵ Merge` para mesclar o segmento focalizado com o segmento imediatamente anterior.
- Clique em `Merge ⟶` para mesclar o segmento focalizado com o segmento imediatamente seguinte.

O texto combinado e o intervalo de tempo de ambos os cartões são unidos. Isso é útil quando uma única fala foi dividida em dois segmentos.

## Dividindo um Segmento

Clique em `Split…` em um cartão focalizado para abrir o diálogo de divisão. Posicione o ponto de divisão dentro do texto e confirme. Dois novos segmentos são criados cobrindo o intervalo de tempo original. Isso é útil quando duas falas distintas foram mescladas em um único segmento.

## Refazer ASR

Clique em `Redo ASR` em um cartão focalizado para reprocessar o reconhecimento de fala no áudio desse segmento. O modelo processa apenas o trecho de áudio daquele segmento e produz uma nova transcrição de fonte única.

Use este recurso quando:

- Um segmento foi criado por mesclagem e não pode ser dividido (segmentos mesclados abrangem múltiplas fontes de ASR; Refazer ASR os consolida em um único, após o qual `Split…` fica disponível).
- A transcrição original é de baixa qualidade e você deseja uma segunda passagem limpa sem editar manualmente.

**Nota:** Qualquer texto que você já tenha digitado no painel direito será descartado e substituído pela nova saída do ASR. A operação requer que o arquivo de áudio esteja carregado; o botão fica desativado se o áudio não estiver disponível.