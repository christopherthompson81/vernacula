"""Builds the VoxLingua107 language map: class index → ISO code + English name.

SpeechBrain's VoxLingua107 checkpoint stores its label list in a YAML-loaded
attribute on the EncoderClassifier. Each label is an ISO 639-1 code (with a
handful of 639-3 codes for languages without 639-1 equivalents). This helper
enriches the raw label into a structured record consumable by the C# side.
"""

from __future__ import annotations

import json
from pathlib import Path

from speechbrain.inference.classifiers import EncoderClassifier


# Minimal ISO code → English name table for the 107 VoxLingua107 languages.
# Sourced from SpeechBrain's label list. Where a label is already 639-3
# (e.g. long-tail languages without a 639-1 equivalent) it is stored as-is.
LANGUAGE_NAMES: dict[str, str] = {
    "ab": "Abkhazian",      "af": "Afrikaans",      "am": "Amharic",
    "ar": "Arabic",         "as": "Assamese",       "az": "Azerbaijani",
    "ba": "Bashkir",        "be": "Belarusian",     "bg": "Bulgarian",
    "bn": "Bengali",        "bo": "Tibetan",        "br": "Breton",
    "bs": "Bosnian",        "ca": "Catalan",        "ceb": "Cebuano",
    "cs": "Czech",          "cy": "Welsh",          "da": "Danish",
    "de": "German",         "el": "Greek",          "en": "English",
    "eo": "Esperanto",      "es": "Spanish",        "et": "Estonian",
    "eu": "Basque",         "fa": "Persian",        "fi": "Finnish",
    "fo": "Faroese",        "fr": "French",         "gl": "Galician",
    "gn": "Guarani",        "gu": "Gujarati",       "gv": "Manx",
    "ha": "Hausa",          "haw": "Hawaiian",      "hi": "Hindi",
    "hr": "Croatian",       "ht": "Haitian",        "hu": "Hungarian",
    "hy": "Armenian",       "ia": "Interlingua",    "id": "Indonesian",
    "is": "Icelandic",      "it": "Italian",        "iw": "Hebrew",
    "ja": "Japanese",       "jw": "Javanese",       "ka": "Georgian",
    "kk": "Kazakh",         "km": "Khmer",          "kn": "Kannada",
    "ko": "Korean",         "la": "Latin",          "lb": "Luxembourgish",
    "ln": "Lingala",        "lo": "Lao",            "lt": "Lithuanian",
    "lv": "Latvian",        "mg": "Malagasy",       "mi": "Maori",
    "mk": "Macedonian",     "ml": "Malayalam",      "mn": "Mongolian",
    "mr": "Marathi",        "ms": "Malay",          "mt": "Maltese",
    "my": "Burmese",        "ne": "Nepali",         "nl": "Dutch",
    "nn": "Norwegian Nynorsk", "no": "Norwegian",   "oc": "Occitan",
    "pa": "Punjabi",        "pl": "Polish",         "ps": "Pashto",
    "pt": "Portuguese",     "ro": "Romanian",       "ru": "Russian",
    "sa": "Sanskrit",       "sco": "Scots",         "sd": "Sindhi",
    "si": "Sinhala",        "sk": "Slovak",         "sl": "Slovenian",
    "sn": "Shona",          "so": "Somali",         "sq": "Albanian",
    "sr": "Serbian",        "su": "Sundanese",      "sv": "Swedish",
    "sw": "Swahili",        "ta": "Tamil",          "te": "Telugu",
    "tg": "Tajik",          "th": "Thai",           "tk": "Turkmen",
    "tl": "Tagalog",        "tr": "Turkish",        "tt": "Tatar",
    "uk": "Ukrainian",      "ur": "Urdu",           "uz": "Uzbek",
    "vi": "Vietnamese",     "war": "Waray",         "yi": "Yiddish",
    "yo": "Yoruba",         "zh": "Chinese",
}


def read_labels(classifier: EncoderClassifier) -> list[str]:
    """Extract the ordered label list from a loaded SpeechBrain classifier."""
    labels = classifier.hparams.label_encoder.ind2lab
    # ind2lab is a dict {index_int: label_str}; sort by index for stability.
    return [labels[i] for i in sorted(labels)]


def build_lang_map(classifier: EncoderClassifier) -> dict[str, dict[str, str]]:
    """Returns {index_str: {"iso": code, "name": english_name}} for all 107 classes."""
    out: dict[str, dict[str, str]] = {}
    for idx, raw_label in enumerate(read_labels(classifier)):
        # SpeechBrain labels are formatted like "en: English" — split defensively.
        iso = raw_label.split(":", 1)[0].strip()
        name = LANGUAGE_NAMES.get(iso, raw_label)
        out[str(idx)] = {"iso": iso, "name": name}
    return out


def write_lang_map(classifier: EncoderClassifier, out_path: Path) -> None:
    lang_map = build_lang_map(classifier)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(lang_map, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
