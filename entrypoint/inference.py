"""Entry point for model inference.

Usage:
    python -m entrypoint.inference \
        --checkpoint ./outputs/best.ckpt \
        --input ./data/test_v2 \
        --output ./predictions
"""

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained SegFormer ship-detection model",
    )

    # ── model ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="HuggingFace model identifier (must match the trained model)",
    )

    # ── I/O ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory with input images or path to a single image",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions"),
        help="Directory for output masks / submission CSV",
    )

    # ── inference settings ────────────────────────────────────────────
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
    )

    # ── output format ─────────────────────────────────────────────────
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save predicted masks as PNG images alongside the submission CSV",
    )
    parser.add_argument(
        "--submission-file",
        type=str,
        default="submission.csv",
        help="Filename for the Kaggle submission CSV",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # TODO: load model from checkpoint
    # TODO: build inference dataset & data-loader
    # TODO: run prediction loop
    # TODO: post-process masks (threshold, RLE encode)
    # TODO: save submission CSV / mask images
    raise NotImplementedError("Inference pipeline is not yet implemented")


if __name__ == "__main__":
    main()
