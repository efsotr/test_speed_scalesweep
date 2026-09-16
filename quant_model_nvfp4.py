"""Quantize local Hugging Face models to NVFP4 W4A4 with calibration data.

Weights use round-to-nearest (RTN); NVFP4 activation scales are calibrated by
LLM Compressor and then applied dynamically at inference time.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

# Set this before importing torch/transformers/llmcompressor.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


DEFAULT_MODELS = ("Llama-3.2-1B-Instruct", "Llama-3.1-8B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        default=list(DEFAULT_MODELS),
        help="Model directory names under --models-dir",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--dataset",
        default="open_platypus",
        help=(
            "LLM Compressor dataset name, Hugging Face dataset name, or local "
            "JSON/CSV path (default: open_platypus)"
        ),
    )
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--num-calibration-samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow trusted local model repositories to execute custom code",
    )
    return parser.parse_args()


def quantize(source: Path, destination: Path, args: argparse.Namespace) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source model is not a directory: {source}")
    if not list(source.glob("*.safetensors")):
        raise FileNotFoundError(f"No safetensors checkpoint found in: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=["lm_head"],
    )
    print(f"Quantizing {source} -> {destination}", flush=True)
    oneshot(
        model=str(source),
        tokenizer=str(source),
        recipe=recipe,
        dataset=args.dataset,
        dataset_config_name=args.dataset_config,
        splits=args.split,
        text_column=args.text_column,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        pad_to_max_length=False,
        concatenate_data=True,
        precision="bfloat16",
        trust_remote_code_model=args.trust_remote_code,
        output_dir=str(destination),
    )


def main() -> None:
    args = parse_args()
    if args.num_calibration_samples < 1:
        raise ValueError("--num-calibration-samples must be at least 1")
    if args.max_seq_length < 1:
        raise ValueError("--max-seq-length must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    models_dir = args.models_dir.expanduser()
    for model_name in args.models:
        source = models_dir / model_name
        destination = models_dir / f"{Path(model_name).name}-NVFP4-RTN"
        quantize(source, destination, args)

        # Release the first model before loading the next one.
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


if __name__ == "__main__":
    main()
