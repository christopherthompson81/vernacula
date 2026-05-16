#!/usr/bin/env python3
"""Chatterbox export wrappers.

Originally vendored from VladOS95-cyber's MIT-licensed conversion
script (https://github.com/VladOS95-cyber/onnx_conversion_scripts/tree/main/chatterbox).
The ~600 lines of vendored S3Tokenizer / FSMN / FSQ / AudioEncoderV2 model
code are gone — replaced by direct use of upstream `chatterbox.s3gen.tokenizer`
plus four scoped patches in `_export_patches.py` that are mathematically
verified equivalent (parity test passes bit-perfect; see Run 6 in
`docs/chatterbox_investigation.md`).

What remains here is the *orchestration* — wrappers around upstream
chatterbox submodules that present an ONNX-export-friendly interface:

  * `SafeDenseLayer` — kept as a STUB only; the original substitution
    drifted speaker_embeddings by 93% per parity (Run 5). The export
    script raises if anyone tries to apply it.
  * `PrepareConditionalsModel` — speech encoder + S3 tokenizer + cond
    prep pipeline, returns the four outputs the speech_encoder.onnx
    graph emits. `self.s3` is a direct reference to upstream
    `chatterbox.s3gen.tokenizer` (no vendoring).
  * `InputsEmbeds` — flat-tensor dispatch over text / speech /
    exaggeration tokens via bool masks (export-friendly).
  * `ISTFT` + `make_pad_mask` + `mask_to_bias` — vocoder helpers used
    by `ConditionalDecoder`. ISTFT uses scatter_add for window_sumsquare
    (Run 3 fix #12); not yet replaceable with upstream.
  * `ConditionalDecoder` — orchestrates `chatterbox.s3gen.flow` +
    `mel2wav` for speech-tokens → waveform. Several upstream-trace
    workarounds inline.

Notes:

  * The global `torch.Tensor.item = lambda x: x` monkeypatch from Vlad's
    script is INTENTIONALLY OMITTED. If the export raises on a `.item()`
    call, apply the patch via the scoped context manager in
    `export_chatterbox_to_onnx.py::item_no_op_patch` rather than as a
    global mutation.
  * `EXAGGERATION_TOKEN = 6563` verified against upstream constants
    (`chatterbox.models.t3.t3`).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio as ta
from torchaudio.compliance.kaldi import get_mel_banks

import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
# Sampling rate of the inputs to S3TokenizerV2
S3GEN_SR = 24000
S3_SR = 16_000
S3_HOP = 160
S3_TOKEN_HOP = 640
S3_TOKEN_RATE = 25 # 25 tokens/sec
SPEECH_VOCAB_SIZE = 6561
MILLISECONDS_TO_SECONDS = 0.001

START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
EXAGGERATION_TOKEN = 6563

ENC_COND_LEN = 6 * S3_SR
DEC_COND_LEN = 10 * S3GEN_SR

CFM_PARAMS = {
    "sigma_min": 1e-06,
    "solver": "euler",
    "t_scheduler": "cosine",
    "training_cfg_rate": 0.2,
    "inference_cfg_rate": 0.7,
    "reg_loss_type": "l1"
}
ISTFT_PARAMS = {"n_fft": 16, "hop_len": 4}

class SafeDenseLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(SafeDenseLayer, self).__init__()
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = torch.nn.Sequential()
        self.nonlinear.add_module("layernorm", torch.nn.LayerNorm(out_channels))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.linear(x)
        if x.size(-1) == 1:
            x = x[:, :, 0]
        x = self.nonlinear(x)
        return x


class PrepareConditionalsModel(torch.nn.Module):

    speech_cond_prompt_len = 150
    speaker_embed_size = 256

    def __init__(self, chatterbox):
        super().__init__()

        # TODO: Move loading elsewhere
        # Use upstream chatterbox.s3gen.tokenizer directly. The vendored
        # S3Tokenizer chain (verified mathematically identical via
        # parity test, see Run 6 in docs/chatterbox_investigation.md)
        # was replaced by four scoped patches in `_export_patches.py`
        # (`patched_s3tokenizer_for_export` + `patched_rotary_for_export`).
        # Active during the speech_encoder ONNX export only — eager
        # PyTorch usage is unaffected.
        self.s3 = chatterbox.s3gen.tokenizer

        self.speaker_encoder = chatterbox.s3gen.speaker_encoder
        self.flow = chatterbox.s3gen.flow

        self.cond_enc = chatterbox.t3.cond_enc

        self.resampler = ta.transforms.Resample(S3GEN_SR, S3_SR)
        self.eps = torch.tensor(torch.finfo(torch.float).eps)
        self.n_fft = 400
        _mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=128
        )
        self.register_buffer(
            "mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

        self.speech_emb = chatterbox.t3.speech_emb
        self.speech_pos_emb = chatterbox.t3.speech_pos_emb

        # Speaker embedding projection
        # NOTE: From testing, randomly/zero initializing speaker embedding seems to work fine
        # speaker_emb = torch.randn(batch_size, self.speaker_embed_size)
        # Pre-computed in __init__ so the export graph sees it as a
        # constant. detach() drops the autograd graph so torch.jit tracing
        # won't refuse to inline it (Vlad's reference works because
        # @torch.no_grad() wraps the whole export — we use scoped no_grad
        # + detach instead, to avoid the global decorator and its
        # inference-tensor footgun). Device pulled from spkr_enc so
        # __init__ works whether the chatterbox model was loaded on
        # cpu or cuda.
        spkr_device = next(self.cond_enc.spkr_enc.parameters()).device
        speaker_emb = torch.zeros(1, self.speaker_embed_size, device=spkr_device)
        with torch.no_grad():
            cond_spkr_init = self.cond_enc.spkr_enc(
                speaker_emb.view(-1, self.speaker_embed_size)
            )[:, None].detach()  # (B, 1, dim)
        # register_buffer so .to(device) / .cpu() actually move it.
        # Plain-attribute assignment (Vlad's original) leaves it on its
        # init-time device when the wrapper is moved — broke the
        # speech_encoder export-on-cpu workaround.
        self.register_buffer("cond_spkr", cond_spkr_init)

    def mel_spectrogram(self, y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920,
                    fmin=0, fmax=8000, center=False):
        y = F.pad(
            y.unsqueeze(1),
            ((n_fft - hop_size) // 2, (n_fft - hop_size) // 2),
            mode="reflect",
        )
        y = y.squeeze(1)
        # device must match `y`; Vlad's CPU-only flow didn't need this.
        hann_window = torch.hann_window(win_size, device=y.device)
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel = torch.from_numpy(mel).float().to(y.device)
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )
        # real = spec[..., 0]
        # imag = spec[..., 1]
        # spec = torch.sqrt(real**2 + imag**2 + 1e-9)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(mel, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5) * 1) # spectral_normalize_torch

        return spec
    
    def _next_power_of_2(self, x: int) -> int:
        r"""Returns the smallest power of 2 that is greater than x"""
        return 1 if x == 0 else 2 ** (x - 1).bit_length()
    
    def _get_strided(self, waveform: torch.Tensor, window_size: int, window_shift: int) -> torch.Tensor:
        r"""Given a waveform (1D tensor of size ``num_samples``), it returns a 2D tensor (m, ``window_size``)
        representing how the window is shifted along the waveform. Each row is a frame.

        Args:
            waveform (Tensor): Tensor of size ``num_samples``
            window_size (int): Frame length
            window_shift (int): Frame shift
            snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
                in the file, and the number of frames depends on the frame_length.  If False, the number of frames
                depends only on the frame_shift, and we reflect the data at the ends.

        Returns:
            Tensor: 2D tensor of size (m, ``window_size``) where each row is a frame
        """
        num_samples = waveform.size(0)
        strides = (window_shift * waveform.stride(0), waveform.stride(0))

        if num_samples < window_size:
            return torch.empty((0, 0), dtype=waveform.dtype, device=waveform.device)
        else:
            m = 1 + (num_samples - window_size) // window_shift

        sizes = (m, window_size)
        return waveform.as_strided(sizes, strides)

    def _get_log_energy(self, strided_input: torch.Tensor, epsilon: torch.Tensor, energy_floor: float) -> torch.Tensor:
        r"""Returns the log energy of size (m) for a strided_input (m,*)"""
        device, dtype = strided_input.device, strided_input.dtype
        log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()  # size (m)
        if energy_floor == 0.0:
            return log_energy
        return torch.max(log_energy, torch.tensor(math.log(energy_floor), device=device, dtype=dtype))

    def _get_window(
        self,
        waveform: torch.Tensor,
        padded_window_size: int,
        window_size: int,
        window_shift: int,
        preemphasis_coefficient: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Gets a window and its log energy

        Returns:
            (Tensor, Tensor): strided_input of size (m, ``padded_window_size``) and signal_log_energy of size (m)
        """
        device, dtype = waveform.device, waveform.dtype
        # size (m, window_size)
        strided_input = self._get_strided(waveform, window_size, window_shift)

        # Subtract each row/frame by its mean
        row_means = torch.mean(strided_input, dim=1).unsqueeze(1)  # size (m, 1)
        strided_input = strided_input - row_means

        if preemphasis_coefficient != 0.0:
            # strided_input[i,j] -= preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
            offset_strided_input = F.pad(strided_input.unsqueeze(0), (1, 0), mode="replicate").squeeze(
                0
            )  # size (m, window_size + 1)
            strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]

        # Apply window_function to each row/frame
        window_function = torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85).unsqueeze(0)  # size (1, window_size)
        strided_input = strided_input * window_function  # size (m, window_size)

        # Pad columns with zero until we reach size (m, padded_window_size)
        if padded_window_size != window_size:
            padding_right = padded_window_size - window_size
            strided_input = F.pad(
                strided_input.unsqueeze(0), (0, padding_right), mode="constant", value=0
            ).squeeze(0)

        return strided_input

    def _get_waveform_and_window_properties(
        self,
        waveform: torch.Tensor,
        channel: int,
        sample_frequency: float,
        frame_shift: float,
        frame_length: float,
    ) -> Tuple[torch.Tensor, int, int, int]:
        r"""Gets the waveform and window properties"""
        channel = max(channel, 0)
        waveform = waveform[channel, :]  # size (n)
        window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
        window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
        padded_window_size = self._next_power_of_2(window_size)
        return waveform, window_shift, window_size, padded_window_size

    def extract_feature(self, waveform: torch.Tensor,
        channel: int = -1,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        high_freq: float = 0.0,
        low_freq: float = 20.0,
        num_mel_bins: int = 23,
        preemphasis_coefficient: float = 0.97,
        sample_frequency: float = 16000.0,
        vtln_high: float = -500.0,
        vtln_low: float = 100.0,
        vtln_warp: float = 1.0):

        device, dtype = waveform.device, waveform.dtype
        waveform, window_shift, window_size, padded_window_size = self._get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length)

        # strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
        strided_input = self._get_window(
            waveform,
            padded_window_size,
            window_size,
            window_shift,
            preemphasis_coefficient,
        )

        # size (m, padded_window_size // 2 + 1)
        spec = torch.stft(
            strided_input,
            n_fft=512,
            hop_length=512,
            center=False,
            window=None,
            return_complex=False
        )   # shape: [..., freq, 2]  (last dim = [real, imag])

        # Compute magnitude manually
        real = spec[..., 0]
        imag = spec[..., 1]
        spectrum = torch.sqrt(real**2 + imag**2).squeeze(-1)
        spectrum = spectrum.pow(2.0)

        # size (num_mel_bins, padded_window_size // 2)
        mel_energies, _ = get_mel_banks(
            num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq, vtln_low, vtln_high, vtln_warp
        )
        mel_energies = mel_energies.to(device=device, dtype=dtype)

        # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
        mel_energies = F.pad(mel_energies, (0, 1), mode="constant", value=0)

        # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
        mel_energies = torch.matmul(spectrum, mel_energies.T)

        # avoid log of zero (which should be prevented anyway by dithering)
        mel_energies = torch.max(mel_energies, self.eps).log()
        return mel_energies

    def prepare_conditions_from_audio(self, audio_values):
        batch_size = audio_values.shape[0]

        # Compute embed_ref
        ref_wav_24 = audio_values[..., :DEC_COND_LEN]
        speaker_features = self.mel_spectrogram(ref_wav_24).transpose(1, 2)

        # Resample to 16kHz
        ref_wav_16 = self.resampler(audio_values) # resample uncropped audio

        # Speech cond prompt tokens
        # TODO START REMOVE
        # -- AT EXPORT, WE MUST SWAP THIS WITH self.resampler(audio_values)
        # ref_wav_16 = librosa.resample(audio_values.cpu().numpy(), orig_sr=S3GEN_SR, target_sr=S3_SR)
        # ref_wav_16 = torch.from_numpy(ref_wav_16).to(audio_values.device)
        # TODO END REMOVE

        feature = self.extract_feature(ref_wav_16, num_mel_bins=80) # == Kaldi.fbank(ref_wav_16, num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        speaker_embeddings = self.speaker_encoder(feature.unsqueeze(0))

        # Upstream chatterbox.s3gen.tokenizer returns (tokens, lens);
        # Vlad's vendored S3Tokenizer returned just tokens. Take [0]
        # to drop the lens — they're not consumed downstream.
        t3_cond_prompt_tokens = self.s3(ref_wav_16[..., :ENC_COND_LEN], max_len=self.speech_cond_prompt_len)[0]

        resampled_wav_16 = self.resampler(ref_wav_24) # resample uncropped audio

        # NOTE: For some reason, we do two passes of the s3 tokenizer
        # TODO: Try reduce this?
        # Tokenize 16khz reference
        prompt_token = self.s3(resampled_wav_16, max_len=None)[0]

        cond_prompt_speech_emb = self.speech_emb(t3_cond_prompt_tokens) + \
                     self.speech_pos_emb(t3_cond_prompt_tokens)

        # Cond prompt
        cond_prompt_speech_emb = self.cond_enc.perceiver(cond_prompt_speech_emb)

        expanded_cond_spkr = self.cond_spkr.expand(batch_size, -1, -1)  # (B, 1, dim)

        # Concat and return
        cond_emb = torch.cat((
            expanded_cond_spkr,
            cond_prompt_speech_emb,
        ), dim=1)  # (B, len_cond, dim)
        # assert cond_emb.dim() == 3
        return cond_emb, prompt_token, speaker_embeddings, speaker_features

    def forward(
        self,
        audio_values: torch.Tensor, # NOTE: Must have sample rate of S3GEN_SR=24000
    ):
        cond_emb, prompt_token, speaker_embeddings, speaker_features = self.prepare_conditions_from_audio(audio_values)
        return cond_emb, prompt_token, speaker_embeddings, speaker_features


class InputsEmbeds(nn.Module):
    def __init__(self, chatterbox):
        super().__init__()
        self.text_emb = chatterbox.t3.text_emb
        self.text_pos_emb = chatterbox.t3.text_pos_emb.emb

        self.speech_emb = chatterbox.t3.speech_emb
        self.speech_pos_emb = chatterbox.t3.speech_pos_emb.emb

        self.emotion_adv_fc = chatterbox.t3.cond_enc.emotion_adv_fc

    def forward(self, input_ids, position_ids, exaggeration):
        assert position_ids.shape == input_ids.shape
        batch_size, seq_len = input_ids.shape

        x = input_ids
        idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Detect first zero
        is_zero = (x == 0)
        has_zero = is_zero.any(dim=1)
        zero_pos = torch.where(
            has_zero,
            is_zero.float().argmax(dim=1),
            torch.full((batch_size,), -1, device=x.device)  # placeholder
        )

        # Masks
        exaggeration_mask = (x == EXAGGERATION_TOKEN)
        base_text_mask = (idx <= zero_pos.unsqueeze(1)) & has_zero.unsqueeze(1)
        
        text_mask = base_text_mask & ~exaggeration_mask
        speech_mask = ~base_text_mask & ~exaggeration_mask

        # Compute relative positions by multiplying with the masks
        text_pos_ids = position_ids * text_mask
        speech_pos_ids = position_ids * speech_mask

        # Flatten
        flat_x = x.view(-1)
        flat_text_mask = text_mask.view(-1)
        flat_speech_mask = speech_mask.view(-1)
        flat_exaggeration_mask = exaggeration_mask.view(-1)
        flat_text_pos = text_pos_ids.view(-1)
        flat_speech_pos = speech_pos_ids.view(-1)

        # Replace invalid indices with 0 (safe padding idx)
        safe_text_idx = torch.where(flat_text_mask, flat_x, torch.zeros_like(flat_x))
        safe_text_pos = torch.where(flat_text_mask, flat_text_pos, torch.zeros_like(flat_text_pos))

        safe_speech_idx = torch.where(flat_speech_mask, flat_x, torch.zeros_like(flat_x))
        safe_speech_pos = torch.where(flat_speech_mask, flat_speech_pos, torch.zeros_like(flat_speech_pos))

        # Embed everything, but irrelevant positions will become "padding" embeddings
        all_text_emb = self.text_emb(safe_text_idx) + self.text_pos_emb(safe_text_pos)
        all_speech_emb = self.speech_emb(safe_speech_idx) + self.speech_pos_emb(safe_speech_pos)

        # Finally, mask out the padding positions to zero them
        text_emb = all_text_emb * flat_text_mask.unsqueeze(-1)
        speech_emb = all_speech_emb * flat_speech_mask.unsqueeze(-1)

        # Emotion Adv: must provide a value if this model uses emotion conditioning
        emotion_adv = exaggeration.view(-1, 1, 1)
        cond_emotion_adv = self.emotion_adv_fc(emotion_adv)

        # Reshape to [B*L, D] to match masks
        embed_dim = text_emb.size(-1)
        text_emb_full   = text_emb
        speech_emb_full = speech_emb

        # Start with zeros
        out = torch.zeros(batch_size * seq_len, embed_dim, device=x.device, dtype=text_emb.dtype)

        # Where text mask is True → take text_emb, else keep current out
        out = torch.where(flat_text_mask.unsqueeze(-1), text_emb_full, out)

        # Where speech mask is True → take speech_emb, else keep current out
        out = torch.where(flat_speech_mask.unsqueeze(-1), speech_emb_full, out)
        
        # Handle exaggeration tokens
        # We need to expand cond_emotion_adv to match the number of exaggeration tokens
        # This assumes cond_emotion_adv is (batch_size, 1, dim) and we need to map it correctly
        # to the flattened positions.
        # We can create an index mapping from the flattened index to the batch index.
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, seq_len).reshape(-1)
        exaggeration_emb_full = cond_emotion_adv[batch_indices].transpose(0, 1) 

        # Zero out positions where mask is False
        exaggeration_emb = exaggeration_emb_full * flat_exaggeration_mask.unsqueeze(-1)

        out = out + exaggeration_emb
        out = out.view(batch_size, seq_len, embed_dim)
        return out


class ISTFT(torch.nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        assert n_fft >= win_length
        super().__init__()

        self.filter_length = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = self.filter_length // 2 + 1
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])


        # Register as a buffer so .to(device) moves it. Plain-attribute
        # assignment (Vlad's original) leaves it on CPU when the module
        # is moved to CUDA, causing device-mismatch failures in
        # window_sumsquare() during the cond-decoder forward.
        self.register_buffer('window', torch.hann_window(win_length))

        # Center pad the window to the size of n_fft
        pad_length = n_fft - self.window.size(0)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left

        torch_fft_window = F.pad(self.window, (pad_left, pad_right), mode='constant', value=0)
        inverse_basis *= torch_fft_window

        self.register_buffer('inverse_basis', inverse_basis.float())

    @staticmethod
    def window_sumsquare(
        window,
        n_frames,
        hop_length,
        win_length,
        n_fft,
    ):
        """Overlap-add of window**2 over n_frames, stride hop_length.

        Vlad's original used `F.conv_transpose1d` driven by
        `torch.ones(1, 1, n_frames)`. That hits a hard ONNX symbolic
        limit at opset 17 and 18 ("ONNX export of convolution for kernel
        of unknown shape") — the trace produces a ConstantOfShape input
        whose dynamic last dim defeats the conv symbolic. We replace it
        with a scatter_add formulation that ONNX handles cleanly because
        the operation is index-based instead of conv-based.
        """
        if win_length is None:
            win_length = n_fft

        n = n_fft + hop_length * (n_frames - 1)

        # Squared window padded to n_fft (no-op when win_length == n_fft).
        win_sq = window ** 2
        pad_length = n_fft - win_sq.size(0)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        win_sq = F.pad(win_sq, (pad_left, pad_right), mode='constant', value=0)
        # win_sq is now (n_fft,)

        # Output position indices: for frame i, sample k → i*hop_length + k.
        device = window.device
        frame_idx = torch.arange(n_frames, device=device).unsqueeze(1)  # (n_frames, 1)
        sample_idx = torch.arange(n_fft, device=device).unsqueeze(0)    # (1, n_fft)
        out_idx = (frame_idx * hop_length + sample_idx).flatten()       # (n_frames*n_fft,)

        values = win_sq.unsqueeze(0).expand(n_frames, n_fft).reshape(-1)  # (n_frames*n_fft,)

        x = torch.zeros(n, dtype=window.dtype, device=device)
        x = x.scatter_add(0, out_idx, values)
        return x

    def forward(self, recombine_magnitude_phase):
        assert recombine_magnitude_phase.dim() == 3, 'must be [B, 2 * N, T]'
        num_frames = recombine_magnitude_phase.size(-1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )

        window_sum = self.window_sumsquare(
            self.window,
            n_frames=num_frames,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.filter_length,
        )

        tiny_value = torch.finfo(window_sum.dtype).tiny

        denom = torch.where(
            window_sum > tiny_value,
            window_sum,
            torch.tensor(1.0, dtype=window_sum.dtype, device=window_sum.device),
        )
        # Apply the transformation
        inverse_transform /= denom

        # scale by hop ratio
        inverse_transform *= self.filter_length / self.hop_length

        q = self.filter_length // 2
        inverse_transform = inverse_transform[:, 0, q:-q]
        return inverse_transform

istft = ISTFT(ISTFT_PARAMS["n_fft"], ISTFT_PARAMS["hop_len"], ISTFT_PARAMS["n_fft"])


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max()
    seq_range = torch.arange(0,
                            max_len,
                            dtype=torch.int64,
                            device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -1.0e+10
    return mask


class ConditionalDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.output_size = model.s3gen.flow.output_size
        self.input_embedding = model.s3gen.flow.input_embedding
        self.spk_embed_affine_layer = model.s3gen.flow.spk_embed_affine_layer
        self.encoder = model.s3gen.flow.encoder
        self.encoder_proj = model.s3gen.flow.encoder_proj
        self.time_embeddings = model.s3gen.flow.decoder.estimator.time_embeddings
        self.time_mlp = model.s3gen.flow.decoder.estimator.time_mlp
        self.up_blocks = model.s3gen.flow.decoder.estimator.up_blocks
        self.static_chunk_size = model.s3gen.flow.decoder.estimator.static_chunk_size
        self.mid_blocks = model.s3gen.flow.decoder.estimator.mid_blocks
        self.down_blocks = model.s3gen.flow.decoder.estimator.down_blocks
        self.final_block = model.s3gen.flow.decoder.estimator.final_block
        self.final_proj = model.s3gen.flow.decoder.estimator.final_proj
        self.n_fft = ISTFT_PARAMS["n_fft"]
        self.hop_len = ISTFT_PARAMS["hop_len"]
        self.n_trim = S3GEN_SR // 50
        self.stft_window = model.s3gen.mel2wav.stft_window
        self.f0_predictor = model.s3gen.mel2wav.f0_predictor
        self.f0_upsamp = model.s3gen.mel2wav.f0_upsamp
        self.m_source = model.s3gen.mel2wav.m_source
        self.inference_cfg_rate = 0.7
        self.conv_pre = model.s3gen.mel2wav.conv_pre
        self.lrelu_slope = model.s3gen.mel2wav.lrelu_slope
        self.reflection_pad = model.s3gen.mel2wav.reflection_pad
        self.ups = model.s3gen.mel2wav.ups
        self.source_downs = model.s3gen.mel2wav.source_downs
        self.source_resblocks = model.s3gen.mel2wav.source_resblocks
        self.resblocks = model.s3gen.mel2wav.resblocks
        self.conv_post = model.s3gen.mel2wav.conv_post
        self.istft = istft
    
    def cond_forward(self, x, mask, mu, t, spks, cond) -> torch.Tensor:
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.
        """

        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        x = torch.cat([x, mu], dim=1)
        spks = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.cat([x, spks], dim=1)
        x = torch.cat([x, cond], dim=1)

        # Cast mask to x.dtype here, before it enters the upstream Block1D
        # blocks. Block1D.forward does `x * mask` internally — eager mode
        # handles bool→float promotion, but torch.onnx.export's trace puts
        # the promoted intermediate on cpu while x stays on cuda, producing
        # a device-mismatch error. Pre-casting to float keeps everything
        # on the same device through the trace.
        mask = mask.to(x.dtype)

        masks = [mask]
        resnet, transformer_blocks, downsample = self.down_blocks[0]
        mask_down = masks[-1]
        x = resnet(x, mask_down, t)
        x = x.permute(0, 2, 1).contiguous()
        attn_mask = mask_to_bias(mask_down.bool() == 1, x.dtype)
        for transformer_block in transformer_blocks:
            x = transformer_block(
                hidden_states=x,
                attention_mask=attn_mask,
                timestep=t,
            )
        x = x.permute(0, 2, 1).contiguous()
        residual = x  # Save hidden states for skip connections
        x = downsample(x * mask_down)
        masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = x.permute(0, 2, 1).contiguous()
            attn_mask = mask_to_bias(mask_mid.bool() == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = x.permute(0, 2, 1).contiguous() 

        resnet, transformer_blocks, upsample = self.up_blocks[0]
        mask_up = masks.pop()
        x = torch.cat([x[:, :, :residual.shape[-1]], residual], dim=1)
        x = resnet(x, mask_up, t)
        x = x.permute(0, 2, 1).contiguous()
        attn_mask = mask_to_bias(mask_up.bool() == 1, x.dtype)
        for transformer_block in transformer_blocks:
            x = transformer_block(
                hidden_states=x,
                attention_mask=attn_mask,
                timestep=t,
            )
        x = x.permute(0, 2, 1).contiguous()
        x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output

    def flow_forward(self, speech_tokens, token_len, mask, embedding, prompt_feat):
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        speech_tokens = self.input_embedding(torch.clamp(speech_tokens, min=0))
        speech_tokens = speech_tokens * mask

        # text encode
        text_encoded, _ = self.encoder(speech_tokens, token_len)
        mel_len1, mel_len2 = prompt_feat.shape[1], text_encoded.shape[1] - prompt_feat.shape[1]
        text_encoded = self.encoder_proj(text_encoded)

        # get conditions
        conds = torch.zeros(
            [1, mel_len1 + mel_len2, self.output_size],
            device=text_encoded.device, dtype=text_encoded.dtype,
        )
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mu = text_encoded
        spks = embedding
        if not isinstance(mel_len1, torch.Tensor):
            mel_len1 = torch.tensor(mel_len1, device=speech_tokens.device)
        if not isinstance(mel_len2, torch.Tensor):
            mel_len2 = torch.tensor(mel_len2, device=speech_tokens.device)
        return mel_len1, mel_len2, mu, spks, conds

    def decode(self, x: torch.Tensor, s_stft: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)

        # ---- Upsample 0 ----
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[0](x)

        si = self.source_downs[0](s_stft)
        si = self.source_resblocks[0](si)
        x = x + si

        xs0 = self.resblocks[0](x) + self.resblocks[1](x) + self.resblocks[2](x)
        x = xs0 / 3

        # ---- Upsample 1 ----
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[1](x)

        si = self.source_downs[1](s_stft)
        si = self.source_resblocks[1](si)
        x = x + si

        xs1 = self.resblocks[3](x) + self.resblocks[4](x) + self.resblocks[5](x)
        x = xs1 / 3

        # ---- Upsample 2 ----
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[2](x)
        x = self.reflection_pad(x)

        si = self.source_downs[2](s_stft)
        si = self.source_resblocks[2](si)
        x = x + si

        xs2 = self.resblocks[6](x) + self.resblocks[7](x) + self.resblocks[8](x)
        x = xs2 / 3

        # ---- Final layers ----
        x = F.leaky_relu(x)
        x = self.conv_post(x)

        magnitude = torch.exp(x[:, :self.n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.n_fft // 2 + 1:, :])

        return magnitude, phase

    def forward(self, speech_tokens, speaker_embeddings, speaker_features):
        token_len = torch.full((speech_tokens.size(0),), speech_tokens.size(1), dtype=torch.long, device=speech_tokens.device)
        mask = (~make_pad_mask(token_len)).unsqueeze(-1)
        mel_len1, mel_len2, mu, spks, cond = self.flow_forward(speech_tokens, token_len, mask, speaker_embeddings, speaker_features)
        mu = mu.transpose(1, 2).contiguous()
        total_len = mel_len1.add(mel_len2).unsqueeze(0)
        mask = (~make_pad_mask(total_len)).unsqueeze(0)
        n_timesteps = 10
        temperature = 1.0
        x = torch.randn_like(mu, dtype=mu.dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps+1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        dt_all = t_span[1:] - t_span[:-1]

        t = t_span[0:1]
        dt = dt_all[0:1]

        x_in = torch.cat([x, torch.zeros_like(x)], dim=0) 
        mask_in = torch.cat([mask, torch.zeros_like(mask)], dim=0) 
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0) 
        t_in = torch.cat([t, torch.zeros_like(t)], dim=0) 
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0) 
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        ## Classifier-Free Guidance inference introduced in VoiceBox
        # step 1
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[1 + 1] - t

        # step 2
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[2 + 1] - t

        # step 3
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[3 + 1] - t

        # step 4
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[4 + 1] - t

        # step 5
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[5 + 1] - t

        # step 6
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[6 + 1] - t

        # step 7
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[7 + 1] - t

        # step 8
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[8 + 1] - t

        # step 9
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[9 + 1] - t

        # step 10
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        output = x.float()
        speech_feat = torch.narrow(output, dim=2, start=mel_len1, length=output.size(2) - mel_len1)
        #mel->f0
        f0 = self.f0_predictor(speech_feat)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        output_sources = s.transpose(1, 2).squeeze(1)
        spec = torch.stft(
            output_sources,
            self.n_fft, 
            self.hop_len, 
            self.n_fft, 
            window=self.stft_window.to(output_sources.device),
            return_complex=False)
        s_stft_real, s_stft_imag = spec[..., 0], spec[..., 1]
        output_sources = torch.cat([s_stft_real, s_stft_imag], dim=1)
        magnitude, phase = self.decode(x=speech_feat, s_stft=output_sources)
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        recombine_magnitude_phase = torch.cat([real, img], dim=1)
        output_wavs = self.istft(recombine_magnitude_phase)
        trim_fade = torch.zeros(2 * self.n_trim, device=output_wavs.device)
        cosine_window = (torch.cos(
            torch.linspace(torch.pi, 0, self.n_trim, device=output_wavs.device)
        ) + 1) / 2
        trim_fade[self.n_trim:] = cosine_window
        output_wavs[:, :trim_fade.size(0)] *= trim_fade
        return output_wavs

