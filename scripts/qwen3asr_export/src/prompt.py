"""
Prompt helpers and token IDs for Qwen3-ASR.

The export uses these token IDs to build config metadata and verify the
upstream tokenizer still matches what the runtime expects.
"""

from __future__ import annotations

ENDOFTEXT_TOKEN_ID = 151643
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
AUDIO_START_TOKEN_ID = 151669
AUDIO_END_TOKEN_ID = 151670
AUDIO_PAD_TOKEN_ID = 151676
ASR_TEXT_TOKEN_ID = 151704

EOS_TOKEN_IDS = [ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID]
NEWLINE_TOKEN_ID = 198


def get_feat_extract_output_lengths(input_lengths: int) -> int:
    """Compute the encoder output token count from a mel frame count."""
    from src.encoder_wrapper import _get_feat_extract_output_lengths

    return _get_feat_extract_output_lengths(input_lengths)


def build_prompt_ids(audio_token_count: int, language: str | None = None) -> list[int]:
    """Build the standard ASR chat-template prompt IDs."""
    ids = [
        IM_START_TOKEN_ID,
        9125,
        NEWLINE_TOKEN_ID,
        IM_END_TOKEN_ID,
        NEWLINE_TOKEN_ID,
        IM_START_TOKEN_ID,
        882,
        NEWLINE_TOKEN_ID,
        AUDIO_START_TOKEN_ID,
    ]

    ids.extend([AUDIO_PAD_TOKEN_ID] * audio_token_count)
    ids.extend(
        [
            AUDIO_END_TOKEN_ID,
            IM_END_TOKEN_ID,
            NEWLINE_TOKEN_ID,
            IM_START_TOKEN_ID,
            77091,
            NEWLINE_TOKEN_ID,
        ]
    )

    if language is not None:
        raise NotImplementedError(
            "Language forcing requires tokenizer access. Use the processor chat template when we wire that in."
        )

    return ids


def get_audio_pad_range(prompt_ids: list[int]) -> tuple[int, int]:
    """Return the [start, end) indices covered by audio pad tokens."""
    start = None
    end = None
    for index, token_id in enumerate(prompt_ids):
        if token_id == AUDIO_PAD_TOKEN_ID:
            if start is None:
                start = index
            end = index + 1
    if start is None or end is None:
        raise ValueError("No <|audio_pad|> tokens found in prompt")
    return start, end
