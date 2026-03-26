"""Entry point for model inference.

Usage:
    python -m entrypoint.inference \
        --checkpoint ./outputs/best.ckpt \
        --input ./data/test_v2 \
        --output ./predictions
"""

import argparse
from pathlib import Path

from src.inference.predictor import ShipPredictor


def _resolve_device(accelerator: str) -> str:
    """Map CLI accelerator choice to a torch device string."""
    if accelerator == "auto":
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if accelerator == "gpu":
        return "cuda"
    return accelerator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained FPN ship-detection model",
    )

    # ── model ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="resnet34",
        help="Encoder backbone used during training (default: resnet34)",
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
        "--image-size",
        type=int,
        default=768,
        help="Spatial resolution (must match the size the model was trained on)",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
    )
    parser.add_argument(
        "--use-tta",
        action="store_true",
        help="Enable test-time augmentation (4 flip variants)",
    )
    parser.add_argument(
        "--min-ship-pixels",
        type=int,
        default=10,
        help="Minimum connected component size to count as a ship (default: 10)",
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

    device = _resolve_device(args.accelerator)

    predictor = ShipPredictor(
        checkpoint_path=args.checkpoint,
        device=device,
        threshold=args.threshold,
        image_size=args.image_size,
        encoder_name=args.encoder_name,
        use_tta=args.use_tta,
        min_ship_pixels=args.min_ship_pixels,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    # ── submission CSV ────────────────────────────────────────────────
    submission = predictor.generate_submission(
        image_dir=args.input,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    csv_path = args.output / args.submission_file
    submission.to_csv(csv_path, index=False)
    print(f"Submission saved to {csv_path} ({len(submission)} rows)")

    # ── optional mask PNGs ────────────────────────────────────────────
    if args.save_masks:
        mask_dir = args.output / "masks"
        predictor.save_masks(
            image_dir=args.input,
            output_dir=mask_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"Masks saved to {mask_dir}")


if __name__ == "__main__":
    main()
