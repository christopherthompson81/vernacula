---
title: "Lägga till flera ljudfiler i kön"
description: "Hur du lägger till flera ljudfiler i jobbkön på en gång."
topic_id: operations_bulk_add_jobs
---

# Lägga till flera ljudfiler i kön

Använd **Bulk Add Jobs** för att köa flera ljud- eller videofiler för transkription i ett enda steg. Programmet bearbetar dem en i taget i den ordning de lades till.

## Förutsättningar

- Alla modellfiler måste vara nedladdade. Kortet **Model Status** måste visa `All N model file(s) present ✓`. Se [Ladda ned modeller](../first_steps/downloading_models.md).

## Så här lägger du till flera jobb på en gång

1. På startskärmen klickar du på `Bulk Add Jobs`.
2. En filväljare öppnas. Välj en eller flera ljud- eller videofiler — håll ned `Ctrl` eller `Shift` för att markera flera filer.
3. Klicka på **Open**. Varje vald fil läggs till i tabellen **Transcription History** som ett separat jobb.

> **Videofiler med flera ljudströmmar:** Om en videofil innehåller mer än en ljudström (till exempel flera språk eller ett regissörskommentarspår) skapar programmet automatiskt ett jobb per ström.

## Jobbnamn

Varje jobb namnges automatiskt efter sitt ljudfilsnamn. Du kan byta namn på ett jobb när som helst genom att klicka på dess namn i kolumnen **Title** i tabellen Transcription History, redigera texten och trycka på `Enter` eller klicka någon annanstans.

## Köbeteende

- Om inget jobb körs just nu startar den första filen omedelbart och de övriga visas som `queued`.
- Om ett jobb redan körs visas alla nyligen tillagda filer som `queued` och startar automatiskt i tur och ordning.
- För att övervaka det aktiva jobbet klickar du på `Monitor` i dess kolumn **Actions**. Se [Övervaka jobb](monitoring_jobs.md).
- För att pausa eller ta bort ett köat jobb innan det startar använder du knapparna `Pause` eller `Remove` i dess kolumn **Actions**. Se [Pausa, återuppta eller ta bort jobb](pausing_resuming_removing.md).

---