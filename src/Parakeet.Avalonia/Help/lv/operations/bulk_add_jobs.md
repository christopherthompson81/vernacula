---
title: "Vairāku audio failu pievienošana rindai"
description: "Kā vienlaicīgi pievienot vairākus audio failus darbu rindai."
topic_id: operations_bulk_add_jobs
---

# Vairāku audio failu pievienošana rindai

Izmantojiet **Bulk Add Jobs**, lai vienā solī pievienotu rindai vairākus audio vai video failus transkribēšanai. Lietojumprogramma apstrādā tos pa vienam tādā secībā, kādā tie tika pievienoti.

## Priekšnosacījumi

- Visiem modeļu failiem jābūt lejupielādētiem. **Model Status** kartē jāparādās `All N model file(s) present ✓`. Skatiet [Modeļu lejupielāde](../first_steps/downloading_models.md).

## Kā pievienot vairākus darbus rindai

1. Sākuma ekrānā noklikšķiniet uz `Bulk Add Jobs`.
2. Atveras failu atlasītājs. Atlasiet vienu vai vairākus audio vai video failus — turiet `Ctrl` vai `Shift`, lai atlasītu vairākus failus.
3. Noklikšķiniet uz **Open**. Katrs atlasītais fails tiek pievienots **Transcription History** tabulai kā atsevišķs darbs.

> **Video faili ar vairākiem audio straumēm:** ja video fails satur vairāk nekā vienu audio straumi (piemēram, vairākas valodas vai režisora komentāru celiņu), lietojumprogramma automātiski izveido vienu darbu katrai straumei.

## Darbu nosaukumi

Katram darbam nosaukums tiek piešķirts automātiski no attiecīgā audio faila nosaukuma. Darba nosaukumu var mainīt jebkurā laikā, noklikšķinot uz tā **Title** kolonnā Transcription History tabulā, rediģējot tekstu un nospiežot `Enter` vai noklikšķinot citur.

## Rindas darbība

- Ja šobrīd neviens darbs nedarbojas, pirmais fails sāk apstrādi nekavējoties, bet pārējie tiek parādīti kā `queued`.
- Ja kāds darbs jau darbojas, visi jaunizveidotie darbi tiek parādīti kā `queued` un tiks sākti automātiski secīgi.
- Lai uzraudzītu aktīvo darbu, noklikšķiniet uz `Monitor` tā **Actions** kolonnā. Skatiet [Darbu uzraudzība](monitoring_jobs.md).
- Lai apturētu vai noņemtu darbu no rindas pirms tā sākšanas, izmantojiet pogas `Pause` vai `Remove` tā **Actions** kolonnā. Skatiet [Darbu apturēšana, atsākšana vai noņemšana](pausing_resuming_removing.md).

---