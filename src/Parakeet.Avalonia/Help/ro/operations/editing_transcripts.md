---
title: "Editarea Transcrierilor"
description: "Cum să revizuiți, corectați și verificați segmentele transcrise în editorul de transcrieri."
topic_id: operations_editing_transcripts
---

# Editarea Transcrierilor

**Editorul de Transcrieri** vă permite să revizuiți rezultatele ASR, să corectați textul, să redenumiți vorbitorii direct în editor, să ajustați durata segmentelor și să marcați segmentele ca verificate — totul în timp ce ascultați înregistrarea audio originală.

## Deschiderea Editorului

1. Încărcați o lucrare finalizată (consultați [Încărcarea Lucrărilor Finalizate](loading_completed_jobs.md)).
2. În vizualizarea **Rezultate**, faceți clic pe `Edit Transcript`.

Editorul se deschide ca fereastră separată și poate rămâne deschis alături de aplicația principală.

## Aspect

Fiecare segment este afișat sub formă de card cu două panouri alăturate:

- **Panoul stâng** — rezultatul original ASR cu colorare a încrederii per cuvânt. Cuvintele pentru care modelul a avut mai puțină certitudine apar în roșu; cuvintele cu încredere ridicată apar în culoarea normală a textului.
- **Panoul drept** — o casetă de text editabilă. Efectuați corecțiile aici; diferența față de original este evidențiată pe măsură ce scrieți.

Eticheta vorbitorului și intervalul de timp apar deasupra fiecărui card. Faceți clic pe un card pentru a-l focaliza și a afișa pictogramele de acțiune. Treceți cu cursorul peste orice pictogramă pentru a vedea un tooltip care descrie funcția acesteia.

## Legenda Pictogramelor

### Bara de Redare

| Pictogramă | Acțiune |
|------------|---------|
| ▶ | Redare |
| ⏸ | Pauză |
| ⏮ | Salt la segmentul anterior |
| ⏭ | Salt la segmentul următor |

### Acțiunile Cardului de Segment

| Pictogramă | Acțiune |
|------------|---------|
| <mdl2 ch="E77B"/> | Reatribuirea segmentului unui alt vorbitor |
| <mdl2 ch="E916"/> | Ajustarea timpilor de început și de sfârșit ai segmentului |
| <mdl2 ch="EA39"/> | Suprimarea sau anularea suprimării segmentului |
| <mdl2 ch="E72B"/> | Fuzionare cu segmentul anterior |
| <mdl2 ch="E72A"/> | Fuzionare cu segmentul următor |
| <mdl2 ch="E8C6"/> | Împărțirea segmentului |
| <mdl2 ch="E72C"/> | Reluarea ASR pentru acest segment |

## Redarea Audio

O bară de redare se întinde în partea de sus a ferestrei editorului:

| Control | Acțiune |
|---------|---------|
| Pictograma Redare / Pauză | Pornirea sau oprirea temporară a redării |
| Bara de derulare | Trageți pentru a sări la orice poziție în înregistrarea audio |
| Glisorul de viteză | Ajustarea vitezei de redare (0.5× – 2×) |
| Pictogramele Anterior / Următor | Salt la segmentul anterior sau următor |
| Meniul derulant pentru modul de redare | Selectați unul dintre cele trei moduri de redare (vezi mai jos) |
| Glisorul de volum | Ajustarea volumului de redare |

În timpul redării, cuvântul rostit în acel moment este evidențiat în panoul stâng. Când redarea este oprită după o derulare, evidențierea se actualizează la cuvântul corespunzător poziției selectate.

### Moduri de Redare

| Mod | Comportament |
|-----|-------------|
| `Single` | Redă segmentul curent o singură dată, apoi se oprește. |
| `Auto-advance` | Redă segmentul curent; la final, îl marchează ca verificat și avansează la următorul. |
| `Continuous` | Redă toate segmentele în ordine fără a marca niciunul ca verificat. |

Selectați modul activ din meniul derulant din bara de redare.

## Editarea unui Segment

1. Faceți clic pe un card pentru a-l focaliza.
2. Editați textul în panoul drept. Modificările sunt salvate automat când mutați focalizarea pe un alt card.

## Redenumirea unui Vorbitor

Faceți clic pe eticheta vorbitorului din cardul focalizat și introduceți un nume nou. Apăsați `Enter` sau faceți clic în altă parte pentru a salva. Noul nume se aplică doar acelui card; pentru a redenumi un vorbitor la nivel global, utilizați [Editarea Numelor Vorbitorilor](editing_speaker_names.md) din vizualizarea Rezultate.

## Verificarea unui Segment

Faceți clic pe caseta de selectare `Verified` de pe un card focalizat pentru a-l marca ca revizuit. Starea de verificare este salvată în baza de date și este vizibilă în editor la încărcările viitoare.

## Suprimarea unui Segment

Faceți clic pe `Suppress` de pe un card focalizat pentru a ascunde segmentul din exporturi (util pentru zgomot, muzică sau alte secțiuni fără vorbire). Faceți clic pe `Unsuppress` pentru a-l restaura.

## Ajustarea Timpilor Segmentului

Faceți clic pe `Adjust Times` de pe un card focalizat pentru a deschide dialogul de ajustare a timpilor. Folosiți rotița mouse-ului deasupra câmpului **Start** sau **End** pentru a modifica valoarea în incremente de 0,1 secunde, sau introduceți o valoare direct. Faceți clic pe `Save` pentru a aplica.

## Fuzionarea Segmentelor

- Faceți clic pe `⟵ Merge` pentru a fuziona segmentul focalizat cu segmentul imediat anterior.
- Faceți clic pe `Merge ⟶` pentru a fuziona segmentul focalizat cu segmentul imediat următor.

Textul combinat și intervalul de timp al ambelor carduri sunt unite. Aceasta este utilă când o singură rostire a fost împărțită în două segmente.

## Împărțirea unui Segment

Faceți clic pe `Split…` de pe un card focalizat pentru a deschide dialogul de împărțire. Poziționați punctul de împărțire în interiorul textului și confirmați. Vor fi create două segmente noi care acoperă intervalul de timp original. Aceasta este utilă când două rostiri distincte au fost unite într-un singur segment.

## Reluarea ASR

Faceți clic pe `Redo ASR` de pe un card focalizat pentru a relua recunoașterea vorbirii pe înregistrarea audio a acelui segment. Modelul procesează doar fragmentul audio corespunzător segmentului și produce o transcriere nouă dintr-o singură sursă.

Utilizați această funcție când:

- Un segment provine dintr-o fuzionare și nu poate fi împărțit (segmentele fuzionate acoperă mai multe surse ASR; Redo ASR le combină într-una singură, după care `Split…` devine disponibil).
- Transcrierea originală este de slabă calitate și doriți o a doua procesare curată fără editare manuală.

**Notă:** Orice text pe care l-ați introdus deja în panoul drept va fi eliminat și înlocuit cu noul rezultat ASR. Operațiunea necesită ca fișierul audio să fie încărcat; butonul este dezactivat dacă înregistrarea audio nu este disponibilă.