#!/usr/bin/env python3
"""Diagnostic: does OmniVoice actually condition its audio-token predictions on the
IPA text, or is eval loss just the (LR-invariant, text-blind) audio-prediction floor?

Paired test on the dev set: for each eval batch, compute loss twice with IDENTICAL
masking — once with the real IPA, once with each utterance's IPA-token ORDER permuted
in place (same tokens, same count, same positions, same audio mask). The only thing
that changes is phoneme sequence. If the model uses the phonemes,
    mean loss(real)  <  mean loss(shuffled).
A near-zero gap means the model ignores the text -> eval loss is a poor axis for
judging the IPA adapter, and we should judge by generation instead.

Runs on the base checkpoint (no adapter) by default; pass --adapter <dir> to test a
tuned LoRA. Reuses OmniVoice's own dataloader/collator/forward so masking matches
training exactly.
"""
import argparse
import torch

from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig

TEXT_START, TEXT_END = 151674, 151675


def shuffle_text_inplace(input_ids, gen):
    """Permute token ids strictly between <|text_start|> and <|text_end|> per sample,
    across all codebook channels identically (text is broadcast across channels).
    input_ids: [B, C, L]. Returns a garbled clone."""
    g = input_ids.clone()
    B, C, L = g.shape
    for b in range(B):
        row = g[b, 0]
        starts = (row == TEXT_START).nonzero(as_tuple=True)[0]
        ends = (row == TEXT_END).nonzero(as_tuple=True)[0]
        if len(starts) == 0 or len(ends) == 0:
            continue
        s = int(starts[0]) + 1
        e = int(ends[0])
        if e - s < 2:
            continue
        perm = torch.randperm(e - s, generator=gen)
        g[b, :, s:e] = g[b, :, s:e][:, perm]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", required=True)
    ap.add_argument("--data_config", required=True)
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    ap.add_argument("--batches", type=int, default=40)
    a = ap.parse_args()

    config = TrainingConfig.from_json(a.train_config)
    config.data_config = a.data_config
    model, tokenizer = build_model_and_tokenizer(config)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model = model.to("cuda").eval()

    _, eval_loader = build_dataloaders(config, tokenizer)
    gen = torch.Generator().manual_seed(0)

    real_sum = shuf_sum = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= a.batches:
                break
            batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
            real_loss = model(**batch).loss.item()
            garbled = dict(batch)
            garbled["input_ids"] = shuffle_text_inplace(batch["input_ids"], gen)
            shuf_loss = model(**garbled).loss.item()
            real_sum += real_loss
            shuf_sum += shuf_loss
            n += 1
            print(f"batch {i}: real={real_loss:.4f} shuffled={shuf_loss:.4f} "
                  f"delta={shuf_loss - real_loss:+.4f}")

    rm, sm = real_sum / n, shuf_sum / n
    print(f"\n=== {n} batches ===")
    print(f"mean real IPA loss     : {rm:.4f}")
    print(f"mean shuffled IPA loss : {sm:.4f}")
    print(f"delta (shuffled-real)  : {sm - rm:+.4f}  "
          f"({'USES phoneme order' if sm - rm > 0.02 else 'IGNORES text (loss is a poor axis)'})")


if __name__ == "__main__":
    main()
