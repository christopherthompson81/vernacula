---
title: "Modifica delle Trascrizioni"
description: "Come rivedere, correggere e verificare i segmenti trascritti nell'editor delle trascrizioni."
topic_id: operations_editing_transcripts
---

# Modifica delle Trascrizioni

L'**Editor delle Trascrizioni** consente di rivedere l'output ASR, correggere il testo, rinominare gli speaker direttamente, regolare i tempi dei segmenti e contrassegnare i segmenti come verificati — il tutto mentre si ascolta l'audio originale.

## Apertura dell'Editor

1. Carica un job completato (vedi [Caricamento dei Job Completati](loading_completed_jobs.md)).
2. Nella vista **Risultati**, fai clic su `Edit Transcript`.

L'editor si apre come finestra separata e può rimanere aperto accanto all'applicazione principale.

## Layout

Ogni segmento è visualizzato come una scheda con due pannelli affiancati:

- **Pannello sinistro** — l'output ASR originale con la colorazione della confidenza per singola parola. Le parole di cui il modello era meno sicuro appaiono in rosso; le parole ad alta confidenza appaiono nel colore del testo normale.
- **Pannello destro** — una casella di testo modificabile. Apporta le correzioni qui; la differenza rispetto all'originale viene evidenziata mentre digiti.

L'etichetta dello speaker e l'intervallo di tempo appaiono sopra ogni scheda. Fai clic su una scheda per metterla a fuoco e mostrarne le icone delle azioni. Passa il cursore su qualsiasi icona per visualizzare un suggerimento che ne descrive la funzione.

## Legenda delle Icone

### Barra di Riproduzione

| Icona | Azione |
|-------|--------|
| ▶ | Riproduci |
| ⏸ | Pausa |
| ⏮ | Vai al segmento precedente |
| ⏭ | Vai al segmento successivo |

### Azioni della Scheda Segmento

| Icona | Azione |
|-------|--------|
| <mdl2 ch="E77B"/> | Riassegna il segmento a uno speaker diverso |
| <mdl2 ch="E916"/> | Regola i tempi di inizio e fine del segmento |
| <mdl2 ch="EA39"/> | Sopprimi o ripristina il segmento |
| <mdl2 ch="E72B"/> | Unisci con il segmento precedente |
| <mdl2 ch="E72A"/> | Unisci con il segmento successivo |
| <mdl2 ch="E8C6"/> | Dividi il segmento |
| <mdl2 ch="E72C"/> | Riesegui l'ASR su questo segmento |

## Riproduzione Audio

Una barra di riproduzione si estende nella parte superiore della finestra dell'editor:

| Controllo | Azione |
|-----------|--------|
| Icona Riproduci / Pausa | Avvia o mette in pausa la riproduzione |
| Barra di avanzamento | Trascina per spostarti in qualsiasi posizione nell'audio |
| Cursore velocità | Regola la velocità di riproduzione (0,5× – 2×) |
| Icone Precedente / Successivo | Vai al segmento precedente o successivo |
| Menu a discesa modalità di riproduzione | Seleziona una delle tre modalità di riproduzione (vedi sotto) |
| Cursore volume | Regola il volume di riproduzione |

Durante la riproduzione, la parola attualmente pronunciata viene evidenziata nel pannello sinistro. Quando la riproduzione è in pausa dopo uno spostamento, l'evidenziazione si aggiorna alla parola nella posizione raggiunta.

### Modalità di Riproduzione

| Modalità | Comportamento |
|----------|---------------|
| `Single` | Riproduce il segmento corrente una volta, poi si ferma. |
| `Auto-advance` | Riproduce il segmento corrente; al termine, lo contrassegna come verificato e passa al successivo. |
| `Continuous` | Riproduce tutti i segmenti in sequenza senza contrassegnarne nessuno come verificato. |

Seleziona la modalità attiva dal menu a discesa nella barra di riproduzione.

## Modifica di un Segmento

1. Fai clic su una scheda per metterla a fuoco.
2. Modifica il testo nel pannello destro. Le modifiche vengono salvate automaticamente quando si sposta il fuoco su un'altra scheda.

## Rinominare uno Speaker

Fai clic sull'etichetta dello speaker nella scheda attiva e digita un nuovo nome. Premi `Enter` o fai clic altrove per salvare. Il nuovo nome viene applicato solo a quella scheda; per rinominare uno speaker globalmente, utilizza [Modifica Nomi Speaker](editing_speaker_names.md) dalla vista Risultati.

## Verificare un Segmento

Fai clic sulla casella di controllo `Verified` su una scheda attiva per contrassegnarla come revisionata. Lo stato di verifica viene salvato nel database ed è visibile nell'editor ai successivi caricamenti.

## Sopprimere un Segmento

Fai clic su `Suppress` su una scheda attiva per nascondere il segmento dalle esportazioni (utile per rumore, musica o altre sezioni non vocali). Fai clic su `Unsuppress` per ripristinarlo.

## Regolazione dei Tempi del Segmento

Fai clic su `Adjust Times` su una scheda attiva per aprire la finestra di dialogo di regolazione dei tempi. Utilizza la rotella del mouse sui campi **Start** o **End** per modificare il valore a incrementi di 0,1 secondi, oppure digita direttamente un valore. Fai clic su `Save` per applicare.

## Unione dei Segmenti

- Fai clic su `⟵ Merge` per unire il segmento attivo con il segmento immediatamente precedente.
- Fai clic su `Merge ⟶` per unire il segmento attivo con il segmento immediatamente successivo.

Il testo combinato e l'intervallo di tempo di entrambe le schede vengono uniti. Questa funzione è utile quando un'unica battuta pronunciata è stata suddivisa in due segmenti.

## Divisione di un Segmento

Fai clic su `Split…` su una scheda attiva per aprire la finestra di dialogo di divisione. Posiziona il punto di divisione all'interno del testo e conferma. Vengono creati due nuovi segmenti che coprono l'intervallo di tempo originale. Questa funzione è utile quando due enunciati distinti sono stati uniti in un unico segmento.

## Rieseguire l'ASR

Fai clic su `Redo ASR` su una scheda attiva per rieseguire il riconoscimento vocale sull'audio di quel segmento. Il modello elabora solo la porzione audio relativa a quel segmento e produce una nuova trascrizione da singola sorgente.

Utilizza questa funzione quando:

- Un segmento proviene da un'unione e non può essere diviso (i segmenti uniti comprendono più sorgenti ASR; Redo ASR le riduce a una sola, dopodiché `Split…` diventa disponibile).
- La trascrizione originale è di scarsa qualità e si desidera un secondo passaggio pulito senza modificare manualmente.

**Nota:** Qualsiasi testo già digitato nel pannello destro viene scartato e sostituito con il nuovo output ASR. L'operazione richiede che il file audio sia caricato; il pulsante è disabilitato se l'audio non è disponibile.