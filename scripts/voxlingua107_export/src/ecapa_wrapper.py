"""A plain nn.Module that composes SpeechBrain's VoxLingua107 pipeline
(FBANK → mean-variance norm → ECAPA-TDNN → classifier) for ONNX export.

SpeechBrain ships the model as an `EncoderClassifier` with YAML-configured
modules. That wrapper is not exportable directly — it has inference-only
helpers, sample-rate coercion, and dynamic device moves. This module
unwraps the submodules we need and exposes a single forward pass whose
graph ONNX can consume.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from speechbrain.inference.classifiers import EncoderClassifier


class VoxLinguaONNX(nn.Module):
    """Wraps SpeechBrain's VoxLingua107 ECAPA-TDNN for ONNX export.

    Forward input:
        audio: [batch, samples] float32 at 16 kHz, mono.

    Forward outputs:
        logits:    [batch, 107] float32 — language classification logits.
        embedding: [batch, 256] float32 — the pooled speaker/language embedding.
    """

    def __init__(self, classifier: EncoderClassifier):
        super().__init__()
        # Modules live under `classifier.mods` in SpeechBrain >=1.0.
        self.compute_features = classifier.mods.compute_features
        self.mean_var_norm = classifier.mods.mean_var_norm
        self.embedding_model = classifier.mods.embedding_model
        self.classifier = classifier.mods.classifier

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # audio: [B, T]
        feats = self.compute_features(audio)  # [B, T', 80]

        # SpeechBrain's InputNormalization expects per-utterance length fractions;
        # for full-clip inference all samples are valid so the fraction is 1.0.
        lens = torch.ones(audio.size(0), dtype=audio.dtype, device=audio.device)
        feats = self.mean_var_norm(feats, lens)

        # embedding_model returns [B, 1, D]; collapse the middle singleton.
        # Use explicit indexing rather than .squeeze() so the dynamo exporter
        # can't mis-record which axis to squeeze.
        embedding = self.embedding_model(feats)[:, 0, :]  # [B, 256]
        # classifier returns [B, 1, 107]; same treatment.
        logits = self.classifier(embedding[:, None, :])[:, 0, :]  # [B, 107]

        return logits, embedding
