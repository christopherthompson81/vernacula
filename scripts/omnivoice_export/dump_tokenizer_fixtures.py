#!/usr/bin/env python3
"""Dump Qwen3 tokenizer fixtures for the C# Qwen3Tokenizer parity test.

For each test string, record the token-id sequence the OmniVoice text tokenizer produces
with add_special_tokens=False (the core byte-level BPE the C# port must match exactly),
plus a few full sequences as OmniVoice actually builds them (style text, wrapped text via
the nonverbal-tag tokenizer). The C# xUnit test loads the model's tokenizer.json and asserts
exact equality on these.

Output: tests/fixtures/omnivoice_tokenizer_fixtures.json (committed; small).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

# Cover the byte-level BPE corner cases: ASCII words, contractions, leading spaces,
# multiple spaces, newlines/tabs, digits, punctuation runs, CJK (no spaces), mixed
# scripts, emoji (surrogate pairs / multi-byte), accented Latin, and every OmniVoice
# special token + nonverbal tag.
PLAIN = [
    "Hello, this is a test of the OmniVoice O N N X export pipeline.",
    "Hello world",
    "  leading and   multiple   spaces  ",
    "don't can't I've we'll they're it's",
    "Line one\nline two\ttabbed",
    "Numbers: 1234567890 and 3.14159 and -42",
    "Punctuation!!! ??? ... ,,, ;;; (parens) [brackets] {braces}",
    "你好，世界。这是一个测试。",
    "Mixed 中文 and English 123 text",
    "Café naïve résumé Zürich",
    "emoji 😀 and 🎉 test",
    "日本語のテスト",
    "한국어 테스트입니다",
    "Здравствуй мир",
    "ALLCAPS and lowercase MiXeD",
    "trailing space ",
    " leading space",
    "",
    "a",
    " ",
]

# OmniVoice special tokens (must split out as single ids).
SPECIALS = [
    "<|denoise|>",
    "<|lang_start|>en<|lang_end|>",
    "<|instruct_start|>None<|instruct_end|>",
    "<|denoise|><|lang_start|>en<|lang_end|><|instruct_start|>female, american accent<|instruct_end|>",
    "<|text_start|>Hello there.<|text_end|>",
    "<|endoftext|><|im_start|><|im_end|>",
]

# Nonverbal tags handled by _tokenize_with_nonverbal_tags.
NONVERBAL = [
    "Hello [laughter] world",
    "Really? [surprise-ah] amazing [sigh]",
    "[laughter]",
    "no tags here",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/data/models/omnivoice/k2-fsa-OmniVoice")
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2]
                    / "tests/fixtures/omnivoice_tokenizer_fixtures.json"),
    )
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    # Does the tokenizer add specials by default? Record both so the C# side knows.
    probe = "Hello"
    add_default = tok(probe).input_ids
    add_false = tok(probe, add_special_tokens=False).input_ids

    def enc(s, add_special):
        return tok(s, add_special_tokens=add_special).input_ids

    cases = []
    for s in PLAIN + SPECIALS + NONVERBAL:
        cases.append({"text": s, "add_special_tokens": False, "ids": enc(s, False)})

    # Reproduce OmniVoice's exact nonverbal-tag tokenization (the standalone-tag path).
    from omnivoice.models.omnivoice import _tokenize_with_nonverbal_tags
    nonverbal_cases = []
    for s in NONVERBAL:
        ids = _tokenize_with_nonverbal_tags(s, tok)[0].tolist()
        nonverbal_cases.append({"text": s, "ids": ids})

    out = {
        "model": args.model,
        "tokenizer_default_adds_specials": add_default != add_false,
        "probe": {"text": probe, "add_special_true": add_default, "add_special_false": add_false},
        "cases": cases,
        "nonverbal_tag_cases": nonverbal_cases,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(cases)} cases + {len(nonverbal_cases)} nonverbal cases -> {out_path}")
    print(f"tokenizer default adds specials: {out['tokenizer_default_adds_specials']}")
    print(f"probe add_special_false: {add_false}")


if __name__ == "__main__":
    main()
