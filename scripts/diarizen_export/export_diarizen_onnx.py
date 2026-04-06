#!/usr/bin/env python3
"""
Export DiariZen segmentation to ONNX format.

This script exports the neural network components of DiariZen (segmentation model)
to ONNX format for use in Parakeet C# implementation. The clustering step is not
exported as it's algorithmic (not a neural net).

Usage:
    python scripts/export_diarizen_onnx.py \
        --model-repo BUT-FIT/diarizen-wavlm-large-s80-md \
        --output-dir ./models/diarizen_onnx

Dependencies:
    - torch, torchaudio
    - onnxruntime
    - huggingface_hub
    - toml
    - local DiariZen checkout with pyannote-audio available on disk

Notes:
    - This script loads weights/config from HuggingFace, but it constructs the
      model class from the local DiariZen checkout.
    - In practice, the local DiariZen repo is required for export.
    - For a tested parity environment, prefer Python 3.10 with
      torch/torchaudio/torchvision 2.1.1/2.1.1/0.16.1.
"""

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import onnx
from huggingface_hub import snapshot_download, hf_hub_download


def load_diarizen_segmentation_model(model_dir: Path):
    """Load the DiariZen segmentation model from HuggingFace checkpoint.
    
    The model uses WavLM + Conformer architecture for EEND (Encoder-Enhanced 
    Neural Diarization) segmentation.
    """
    # Import after path setup
    import os
    
    # Add temporary paths for imports - try multiple locations
    possible_roots = [
        Path(__file__).resolve().parents[2] / "DiariZen",  # ../DiariZen from scripts/
        Path("/home/chris/Programming/DiariZen"),  # Absolute path
    ]
    
    diarizen_root = None
    for root in possible_roots:
        if root.exists() and (root / "diarizen").exists():
            diarizen_root = root
            break
    
    if diarizen_root is None:
        raise FileNotFoundError(
            f"DiariZen repository not found. Searched:\n" + 
            "\n".join(f"  - {r}" for r in possible_roots)
        )
    
    print(f"Using DiariZen from: {diarizen_root}")
    sys.path.insert(0, str(diarizen_root))
    sys.path.insert(0, str(diarizen_root / "pyannote-audio"))
    
    from diarizen.models.eend.model_wavlm_conformer import Model as WavLMConformerModel
    
    # Load config.toml to get model parameters  
    import toml
    config_path = model_dir / "config.toml"
    if config_path.exists():
        config = toml.load(config_path)
        print(f"Found config: {config_path}")
        
        model_args = dict(config.get("model", {}).get("args", {}))
    else:
        print("No config.toml found, using defaults")
        model_args = {
            "wavlm_src": "wavlm_large_s80_md",
            "wavlm_layer_num": 25,
            "wavlm_feat_dim": 1024,
            "chunk_size": 16,
            "attention_in": 256,
            "ffn_hidden": 1024,
            "num_head": 4,
            "num_layer": 4,
        }
    
    # Preserve repo defaults for arguments that are part of the constructor
    # but may be omitted by the HuggingFace config.
    model_args.setdefault("max_speakers_per_chunk", 4)
    model_args.setdefault("max_speakers_per_frame", 2)
    
    print(f"\nLoading segmentation model...")
    print(f"  WavLM source: {model_args.get('wavlm_src')}")
    print(f"  WavLM layers: {model_args.get('wavlm_layer_num')}, feat_dim: {model_args.get('wavlm_feat_dim')}")
    print(f"  Chunk size: {model_args.get('chunk_size')}s")
    print(f"  Max speakers per chunk: {model_args.get('max_speakers_per_chunk')}")
    
    # Initialize the model with the actual config args rather than manually
    # reconstructing a subset. That keeps export aligned with the HF snapshot
    # and reduces the chance of silent semantic drift.
    try:
        signature = inspect.signature(WavLMConformerModel.__init__)
        supported_args = {
            key: value
            for key, value in model_args.items()
            if key in signature.parameters
        }
        ignored_args = sorted(set(model_args) - set(supported_args))
        if ignored_args:
            print(f"  Ignoring unsupported config args: {ignored_args}")

        model = WavLMConformerModel(**supported_args)
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise
    
    # Load checkpoint
    ckpt_path = model_dir / "pytorch_model.bin"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"  Loading weights from {ckpt_path.name}...")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    
    # Handle potential key mismatches (e.g., 'module.' prefix from DataParallel)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        cleaned_state_dict[new_key] = value
    
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch during DiariZen export.\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}"
        )
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {num_params:,} parameters ({num_params/1e6:.1f}M)")
    
    return model


def export_segmentation_model(model, output_path: Path, chunk_duration: float = 16.0):
    """Export the segmentation model to ONNX."""
    print(f"\nExporting segmentation model to {output_path}...")
    
    # Create dummy input: (batch=1, channels=1, samples)
    sample_rate = 16000
    num_samples = int(sample_rate * chunk_duration)
    
    dummy_input = torch.randn(1, 1, num_samples)
    
    # Test forward pass first
    print("  Testing PyTorch forward pass...")
    with torch.no_grad():
        test_output = model(dummy_input)
    print(f"    Output shape: {test_output.shape}")
    
    # Export to ONNX
    print("  Running ONNX export...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["waveform"],
        output_names=["scores"],
        dynamic_axes={
            "waveform": {0: "batch", 2: "samples"},
            "scores": {0: "batch", 1: "frames", 2: "speakers"}
        },
        opset_version=18,  # Updated to match PyTorch's recommended version
        do_constant_folding=True,
        # Use the legacy exporter for better compatibility with complex models
        _exported_model=None,
    )
    
    # Validate the ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    print(f"  ✓ Exported successfully!")
    print(f"  Input shape: (batch, 1, samples)")
    print(f"  Output shape: (batch, frames, powerset_classes)")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def test_onnx_model(onnx_path: Path, torch_model=None):
    """Test the exported ONNX model with ONNX Runtime."""
    import onnxruntime as ort
    
    print(f"\nTesting ONNX model: {onnx_path}")
    
    # Create session
    session = ort.InferenceSession(str(onnx_path))
    
    # Get input info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {input_shape}")
    
    # Create test input (8 seconds at 16kHz)
    batch_size = 1
    num_samples = int(16000 * 8)
    test_input = np.random.randn(batch_size, 1, num_samples).astype(np.float32)
    
    # Run inference
    inputs = {input_name: test_input}
    outputs = session.run(None, inputs)
    
    print(f"  Output count: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"    Output {i}: shape={output.shape}, dtype={output.dtype}")
    
    # Optional: compare with PyTorch if model provided
    if torch_model is not None:
        torch_input = torch.from_numpy(test_input)
        with torch.no_grad():
            torch_output = torch_model(torch_input)
        
        np.testing.assert_allclose(
            outputs[0],
            torch_output.numpy(),
            rtol=1e-4,
            atol=1e-5
        )
        print("  ✓ Output matches PyTorch model!")


def main():
    parser = argparse.ArgumentParser(description="Export DiariZen models to ONNX")
    parser.add_argument(
        "--model-repo",
        type=str,
        default="BUT-FIT/diarizen-wavlm-large-s80-md",
        help="HuggingFace repo ID for DiariZen model"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models/diarizen_onnx"),
        help="Output directory for ONNX models"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test exported model with ONNX Runtime"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download DiariZen model
    print(f"Downloading DiariZen model: {args.model_repo}")
    model_dir = Path(snapshot_download(repo_id=args.model_repo))
    print(f"  Model downloaded to: {model_dir}\n")
    
    # Export segmentation model
    seg_model = load_diarizen_segmentation_model(model_dir)
    seg_output = args.output_dir / "diarizen_segmentation.onnx"
    
    # Get chunk duration from config for export
    import toml
    config_path = model_dir / "config.toml"
    if config_path.exists():
        config = toml.load(config_path)
        chunk_duration = config.get("model", {}).get("args", {}).get("chunk_size", 16.0)
    else:
        chunk_duration = 16.0
    
    export_segmentation_model(seg_model, seg_output, chunk_duration)
    
    if args.test:
        test_onnx_model(seg_output, seg_model)
    
    # Write metadata
    import json
    metadata = {
        "model_repo": args.model_repo,
        "sample_rate": 16000,
        "segmentation_model": str(seg_output.relative_to(args.output_dir.parent)),
        "notes": [
            "Segmentation model outputs powerset scores (frame x speaker_combinations)",
            "Model uses WavLM Large + Conformer architecture",
            "Clustering logic must be reimplemented separately in C#",
            "Input: 16kHz mono audio, Output: per-frame speaker activity scores"
        ]
    }
    
    metadata_path = args.output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Export complete!")
    print(f"  Models saved to: {args.output_dir}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
