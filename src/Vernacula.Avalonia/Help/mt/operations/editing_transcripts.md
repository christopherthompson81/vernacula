---
title: "L-Editjar tat-Traskrizzjonijiet"
description: "Kif tirrevedi, tikkoreġi, u tivverifika segmenti traskritti fl-editur tat-traskrizzjonijiet."
topic_id: operations_editing_transcripts
---

# L-Editjar tat-Traskrizzjonijiet

L-**Editur tat-Traskrizzjonijiet** jippermettilek tirrevedi l-output tal-ASR, tikkoreġi t-test, issemmi mill-ġdid l-ispeaker inline, taġġusta l-ħin tas-segmenti, u timmarkja s-segmenti bħala vverifikati — kollox waqt li tisma' l-awdjo oriġinali.

## Kif Tiftaħ l-Editur

1. Iċċarġja xogħol komplut (ara [Il-Ġbid ta' Xogħlijiet Kompluti](loading_completed_jobs.md)).
2. Fil-veduta tar-**Riżultati**, ikklikkja `Edit Transcript`.

L-editur jiftaħ bħala tieqa separata u jista' jibqa' miftuħ ħdejn l-applikazzjoni ewlenija.

## Il-Qasma

Kull segment jidher bħala karta b'żewġ panewlijiet ħdejn xulxin:

- **Panell tax-xellug** — l-output oriġinali tal-ASR bil-kulur tal-kunfidenza għal kull kelma. Il-kliem li dwaru il-mudell kien inqas ċert jidher f'aħmar; il-kliem ta' kunfidenza għolja jidher fil-kulur normali tat-test.
- **Panell tal-lemin** — kaxxa tat-test li tista' tiġi editjata. Agħmel il-korrezzjonijiet hawn; id-differenza mill-oriġinali tiġi enfasizzata waqt li tikteb.

It-tikketta tal-ispeaker u l-medda taż-żmien jidhru fuq kull karta. Ikklikkja karta biex tiffukassaha u turi l-ikoni tal-azzjonijiet tagħha. Żomm il-cursor fuq kwalunkwe ikona biex tara tooltip li jiddeskrivi l-funzjoni tagħha.

## Leġġenda tal-Ikoni

### Barra tal-Playback

| Ikona | Azzjon |
|-------|--------|
| ▶ | Pplejja |
| ⏸ | Pawsa |
| ⏮ | Aqbeż lejn is-segment ta' qabel |
| ⏭ | Aqbeż lejn is-segment li jmiss |

### Azzjonijiet tal-Karta tas-Segment

| Ikona | Azzjon |
|-------|--------|
| <mdl2 ch="E77B"/> | Assenja mill-ġdid is-segment lil speaker differenti |
| <mdl2 ch="E916"/> | Aġġusta l-ħinijiet tal-bidu u t-tmiem tas-segment |
| <mdl2 ch="EA39"/> | Soppriemi jew neħħi s-soppressjoni tas-segment |
| <mdl2 ch="E72B"/> | Ingħaqad mas-segment ta' qabel |
| <mdl2 ch="E72A"/> | Ingħaqad mas-segment li jmiss |
| <mdl2 ch="E8C6"/> | Qassam is-segment |
| <mdl2 ch="E72C"/> | Erġa' agħmel l-ASR fuq dan is-segment |

## Il-Playback tal-Awdjo

Barra tal-playback tgħaddi fil-quċċata tat-tieqa tal-editur:

| Kontroll | Azzjon |
|----------|--------|
| Ikona Play / Pause | Ibda jew waqqaf il-playback |
| Seek bar | Iġbed biex taqbeż għal kwalunkwe pożizzjoni fl-awdjo |
| Slider tal-veloċità | Aġġusta l-veloċità tal-playback (0.5× – 2×) |
| Ikoni Prev / Next | Aqbeż lejn is-segment ta' qabel jew li jmiss |
| Dropdown tal-modalità tal-playback | Agħżel waħda mit-tliet modalitajiet tal-playback (ara hawn taħt) |
| Slider tal-volum | Aġġusta l-volum tal-playback |

Waqt il-playback, il-kelma li qed tiġi ddikjarata bħalissa tiġi enfasizzata fil-panell tax-xellug. Meta jkun hemm pawsa wara seek, l-enfasi taġġorna għall-kelma fil-pożizzjoni tas-seek.

### Modalitajiet tal-Playback

| Modalità | Imġiba |
|----------|--------|
| `Single` | Pplejja s-segment attwali darba, imbagħad ieqaf. |
| `Auto-advance` | Pplejja s-segment attwali; meta jintemm, immarkjah bħala vverifikat u avvanza għall-li jmiss. |
| `Continuous` | Pplejja s-segmenti kollha f'sekwenza mingħajr ma timmarkja l-ebda wieħed bħala vverifikat. |

Agħżel il-modalità attiva mid-dropdown fil-barra tal-playback.

## L-Editjar ta' Segment

1. Ikklikkja karta biex tiffukassaha.
2. Editja t-test fil-panell tal-lemin. Il-bidliet jissalvaw awtomatikament meta tċaqlaq il-fokus lejn karta oħra.

## Issemmi mill-Ġdid Speaker

Ikklikkja t-tikketta tal-ispeaker ġewwa l-karta ffukata u ikteb isem ġdid. Agħfas `Enter` jew ikklikkja barra biex issalva. L-isem il-ġdid jiġi applikat lil dik il-karta biss; biex issemmi speaker globalment, uża [Editja l-Ismijiet tal-Speakers](editing_speaker_names.md) mill-veduta tar-Riżultati.

## Il-Verifika ta' Segment

Ikklikkja l-checkbox `Verified` fuq karta ffukata biex timmarkjaha bħala riveduta. L-istat tal-verifika jissalva fid-database u jkun viżibbli fl-editur f'tagħbiet futuri.

## Is-Soppressjoni ta' Segment

Ikklikkja `Suppress` fuq karta ffukata biex taħbi s-segment mill-esportazzjonijiet (utli għal storbju, mużika, jew sezzjonijiet oħra li mhumiex diskors). Ikklikkja `Unsuppress` biex terġa' tistabbilixxi.

## L-Aġġustament tal-Ħinijiet tas-Segment

Ikklikkja `Adjust Times` fuq karta ffukata biex tiftaħ id-djalogu tal-aġġustament tal-ħin. Uża r-rota tal-iskroll fuq il-qasam **Start** jew **End** biex tbiddel il-valur f'inkrementijiet ta' 0.1 sekonda, jew ikteb valur direttament. Ikklikkja `Save` biex tapplika.

## L-Għaqda ta' Segmenti

- Ikklikkja `⟵ Merge` biex tgħaqqad is-segment iffukat mas-segment immedjatament ta' qablu.
- Ikklikkja `Merge ⟶` biex tgħaqqad is-segment iffukat mas-segment immedjatament ta' warajh.

It-test u l-medda taż-żmien kombinati taż-żewġ karti jingħaqdu. Dan huwa utli meta dikjarazzjoni waħda mitkellma nqasmet fuq żewġ segmenti.

## Il-Qsim ta' Segment

Ikklikkja `Split…` fuq karta ffukata biex tiftaħ id-djalogu tal-qsim. Poġġi l-punt tal-qsim fit-test u kkonfermah. Jinħolqu żewġ segmenti ġodda li jkopru l-medda oriġinali taż-żmien. Dan huwa utli meta żewġ dikjarazzjonijiet distinti ngħaqdu f'segment wieħed.

## Erġa' Agħmel l-ASR

Ikklikkja `Redo ASR` fuq karta ffukata biex terġa' tħaddem ir-rikonoxximent tal-kelma fuq l-awdjo ta' dak is-segment. Il-mudell jipproċessa biss il-fetta tal-awdjo ta' dak is-segment u jipproduċi traskrizzjoni ġdida minn sors wieħed.

Uża dan meta:

- Segment ġie minn għaqda u ma jistax jinqasam (segmenti magħqudin jikopru sorsi multipli tal-ASR; Redo ASR jiġborhom f'wieħed, wara li `Split…` isir disponibbli).
- It-traskrizzjoni oriġinali hija fqira u trid pass ieħor nadif mingħajr ma teditja manwalment.

**Nota:** Kwalunkwe test li diġà iktbt fil-panell tal-lemin jiġi skartat u jinbidel bl-output ġdid tal-ASR. L-operazzjoni teħtieġ li l-fajl tal-awdjo jkun imċarġjat; il-buttuna tkun diżattivata jekk l-awdjo ma jkunx disponibbli.