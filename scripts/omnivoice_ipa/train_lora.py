#!/usr/bin/env python3
"""LoRA fine-tune launcher for OmniVoice IPA-input adaptation.

Fork of omnivoice.cli.train with a peft.get_peft_model() wrap inserted between
model construction and the trainer (the package ships no native LoRA support —
confirmed no `peft`/`lora` references anywhere in omnivoice==0.1.5).

Scope (per user decision): exclusively IPA input, no orthographic-text mixing.
The webdataset label JSONLs (build_webdataset.py) already put the phonemized IPA
string directly in "text" — no data-side substitution logic needed.

LoRA targets q/k/v/o_proj + gate/up/down_proj across all 28 Qwen3 layers (the
sequence/attention path that has to learn to interpret IPA-token context
differently). `embed_tokens` is fully unfrozen via modules_to_save (not low-rank)
since it's the sole entry point for the new modality and full fine-tune of an
embedding table is cheap. peft freezes everything else by default, including
audio_embeddings/audio_heads — exactly right, since this adapter shouldn't touch
audio-generation quality, only text comprehension.

Usage:
  accelerate launch -m scripts.omnivoice_ipa.train_lora \\
      --train_config /mnt/data/omnivoice_ipa/train/train_config.json \\
      --data_config /mnt/data/omnivoice_ipa/train/data_config.json \\
      --output_dir /mnt/data/omnivoice_ipa/train/checkpoints
"""
import argparse

from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig
from omnivoice.training.trainer import OmniTrainer
from peft import LoraConfig, get_peft_model

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def main():
    parser = argparse.ArgumentParser(description="OmniVoice IPA LoRA fine-tune")
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_config", type=str, required=True)
    args = parser.parse_args()

    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config

    model, tokenizer = build_model_and_tokenizer(config)
    train_loader, eval_loader = build_dataloaders(config, tokenizer)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=["embed_tokens"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = OmniTrainer(
        model=model, config=config,
        train_dataloader=train_loader, eval_dataloader=eval_loader,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
