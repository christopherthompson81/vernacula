/**
 * Model manifest and the language list the demo offers.
 *
 * ⚠ The language list is NOT "every language the phonemizer routes" (192). It is the set the v6
 * fine-tune actually trained on, because those are the ones whose audio we can stand behind. The
 * phonemizer will happily produce IPA for the rest and the model will render it from phones it
 * already holds — that is the whole premise — but the result is extrapolated, so anything outside
 * this list belongs behind an "experimental" affordance rather than in the default picker.
 */
export interface LanguageOption {
  /** vernacula-phonemizer language code. */
  code: string;
  name: string;
  /** Approximate size of this language's data files, in MB — shown before download. */
  dataMb: number;
  sample: string;
}

/** The 28 languages of the v6 coverage set, with the phonemizer data each one pulls. */
export const LANGUAGES: LanguageOption[] = [
  { code: "en",  name: "English",    dataMb: 14,  sample: "The quick brown fox jumps over the lazy dog." },
  { code: "es",  name: "Spanish",    dataMb: 0.1, sample: "El tiempo está muy agradable hoy en la costa." },
  { code: "de",  name: "German",     dataMb: 2.0, sample: "Das Wetter ist heute sehr schön an der Küste." },
  { code: "fr",  name: "French",     dataMb: 4.8, sample: "Le temps est très agréable aujourd'hui sur la côte." },
  { code: "pt",  name: "Portuguese", dataMb: 0.1, sample: "O tempo está muito agradável hoje na costa." },
  { code: "ca",  name: "Catalan",    dataMb: 0.2, sample: "Avui fa molt bon temps a la costa." },
  { code: "cs",  name: "Czech",      dataMb: 0.1, sample: "Dnes je velmi příjemné počasí na pobřeží." },
  { code: "sv",  name: "Swedish",    dataMb: 0.6, sample: "Vädret är mycket trevligt idag vid kusten." },
  { code: "ru",  name: "Russian",    dataMb: 8.4, sample: "Сегодня очень приятная погода на побережье." },
  { code: "tr",  name: "Turkish",    dataMb: 0.1, sample: "Bugün sahilde hava çok güzel." },
  { code: "cy",  name: "Welsh",      dataMb: 0.1, sample: "Mae'r tywydd yn braf heddiw ar hyd y traeth." },
  { code: "ga",  name: "Irish",      dataMb: 0.3, sample: "Tá an aimsir go hálainn inniu cois cósta." },
  { code: "cmn", name: "Mandarin",   dataMb: 1.7, sample: "今天海邊的天氣非常宜人。" },
  { code: "ja",  name: "Japanese",   dataMb: 8.9, sample: "今日は海岸の天気がとても快適です。" },
  { code: "ko",  name: "Korean",     dataMb: 0.1, sample: "오늘 해안의 날씨가 매우 좋습니다." },
  { code: "hi",  name: "Hindi",      dataMb: 0.1, sample: "आज तट पर मौसम बहुत सुहावना है।" },
  { code: "ta",  name: "Tamil",      dataMb: 0.1, sample: "இன்று கடற்கரையில் வானிலை மிக இனிமையாக உள்ளது." },
  { code: "th",  name: "Thai",       dataMb: 1.7, sample: "วันนี้อากาศริมชายฝั่งดีมาก" },
  { code: "vi",  name: "Vietnamese", dataMb: 0.1, sample: "Hôm nay thời tiết ven biển rất dễ chịu." },
  { code: "ar",  name: "Arabic",     dataMb: 35,  sample: "الطقس اليوم لطيف جدا على الساحل." },
  { code: "am",  name: "Amharic",    dataMb: 0.1, sample: "ዛሬ በባህር ዳርቻ የአየሩ ሁኔታ በጣም ደስ የሚል ነው።" },
  { code: "om",  name: "Oromo",      dataMb: 0.1, sample: "Har'a qilleensi qarqara galaanaa baay'ee gaarii dha." },
  { code: "ha",  name: "Hausa",      dataMb: 0.1, sample: "Yanayi a bakin teku yana da kyau sosai yau." },
  { code: "ff",  name: "Fula",       dataMb: 0.1, sample: "Hannde weeyo maayo ngoo no welti sanne." },
  { code: "zu",  name: "Zulu",       dataMb: 0.1, sample: "Isimo sezulu sihle kakhulu namuhla ogwini." },
  { code: "xh",  name: "Xhosa",      dataMb: 0.1, sample: "Imozulu intle kakhulu namhlanje elunxwemeni." },
  { code: "kk",  name: "Kazakh",     dataMb: 0.1, sample: "Бүгін жағалауда ауа райы өте жағымды." },
  { code: "sd",  name: "Sindhi",     dataMb: 2.7, sample: "اڄ ساحل تي موسم تمام سٺي آهي." },
];

/**
 * Where the ONNX bundle lives. Netlify serves /models/* with immutable caching (netlify.toml);
 * the filename carries the precision so a cached older build cannot be mistaken for a newer one.
 *
 * ⚠ PRECISION IS THE OPEN QUESTION, not a preference. The diffusion loop is precision-sensitive:
 * TF32 produced incoherent noise and fp16 a different-but-valid rendering, both measured in
 * docs/omnivoice_onnx_investigation.md. int8 is 617 MB and pending a listening test; fp16 is
 * ~1.2 GB and was previously listen-confirmed good. Do not treat the small one as the default
 * until it has been heard.
 */
export const MODELS = {
  transformer: "/models/omnivoice_transformer_ipa_v6.int8.onnx",
  decoder: "/models/higgs_decoder.int8.onnx",
  /** Reference voice as pre-encoded codec codes — a few KB, so the 654 MB Higgs ENCODER never
   *  ships. Also means every generation runs in clone mode, which is what keeps short input
   *  stable (see README "Short input"). */
  voices: "/models/voices.json",
} as const;

/** Diffusion steps. 32 is the desktop default; 16 halves browser latency at some quality cost. */
export const NUM_STEPS = 16;
