---
title: "Modeļu lejupielāde"
description: "Kā lejupielādēt AI modeļu failus, kas nepieciešami transkripcijai."
topic_id: first_steps_downloading_models
---

# Modeļu lejupielāde

Parakeet Transcription darbībai nepieciešami AI modeļu faili. Tie nav iekļauti lietojumprogrammā un ir jālejupielādē pirms pirmās transkripcijas.

## Modeļu statuss (sākuma ekrāns)

Sākuma ekrāna augšdaļā ir redzama šaura statusa josla, kas norāda, vai modeļi ir gatavi lietošanai. Ja faili trūkst, tajā parādās arī poga `Open Settings`, kas tieši aizved uz modeļu pārvaldību.

| Statuss | Nozīme |
|---|---|
| `All N model file(s) present ✓` | Visi nepieciešamie faili ir lejupielādēti un gatavi lietošanai. |
| `N model file(s) missing: …` | Viens vai vairāki faili trūkst; atveriet Iestatījumus, lai tos lejupielādētu. |

Kad modeļi ir gatavi, pogas `New Transcription` un `Bulk Add Jobs` kļūst aktīvas.

## Kā lejupielādēt modeļus

1. Sākuma ekrānā noklikšķiniet uz `Open Settings` (vai dodieties uz `Settings… > Models`).
2. Sadaļā **Models** noklikšķiniet uz `Download Missing Models`.
3. Parādās progresa josla un statusa rindiņa, kurā redzams pašreizējais fails, tā pozīcija rindā un lejupielādes lielums — piemēram: `[1/3] encoder-model.onnx — 42 MB`.
4. Pagaidiet, līdz statuss kļūst `Download complete.`

## Lejupielādes atcelšana

Lai apturētu lejupielādi, noklikšķiniet uz `Cancel`. Statusa rindiņā tiks parādīts `Download cancelled.` Daļēji lejupielādētie faili tiek saglabāti, tāpēc nākamreiz, noklikšķinot uz `Download Missing Models`, lejupielāde atsāksies no pārtrauktās vietas.

## Lejupielādes kļūdas

Ja lejupielāde neizdodas, statusa rindiņā tiek parādīts `Download failed: <reason>`. Pārbaudiet interneta savienojumu un vēlreiz noklikšķiniet uz `Download Missing Models`, lai mēģinātu vēlreiz. Lietojumprogramma atsāk lejupielādi no pēdējā veiksmīgi pabeigtā faila.

## Precizitātes maiņa

Lejupielādējamie modeļu faili ir atkarīgi no izvēlētās **Model Precision**. Lai to mainītu, dodieties uz `Settings… > Models > Model Precision`. Ja precizitāti maināt pēc lejupielādes, jaunais failu komplekts ir jālejupielādē atsevišķi. Skatiet [Modeļu svara precizitātes izvēle](model_precision.md).

---