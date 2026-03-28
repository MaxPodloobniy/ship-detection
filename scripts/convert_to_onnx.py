"""Convert a Lightning .ckpt checkpoint to ONNX format.

Usage:
    python scripts/convert_to_onnx.py \
        --checkpoint path/to/model.ckpt \
        --output model.onnx \
        --image-size 768 \
        --fp16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import segmentation_models_pytorch as smp
import torch
from onnxruntime.transformers import float16


def load_pytorch_model(
    checkpoint_path: Path,
    encoder_name: str = "resnet34",
) -> torch.nn.Module:
    """Load an FPN model from a Lightning checkpoint."""
    model = smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    cleaned = {
        k.removeprefix("model."): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    model.load_state_dict(cleaned)
    model.eval()
    return model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    image_size: int = 768,
) -> None:
    """Export a PyTorch model to ONNX format."""
    dummy_input = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=17,
    )


def convert_to_fp16(input_path: Path, output_path: Path) -> None:
    """Convert ONNX model from FP32 to FP16 (graph + weights)."""
    model = onnx.load(str(input_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(output_path))


def validate_export(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    image_size: int = 768,
) -> float:
    """Compare PyTorch and ONNX outputs, return max absolute difference."""
    import onnxruntime as ort

    dummy_input = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

    return float(np.max(np.abs(pytorch_output - onnx_output)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Lightning .ckpt to ONNX format",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to Lightning .ckpt checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model.onnx"),
        help="Output ONNX file path (default: model.onnx)",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="resnet34",
        help="Encoder backbone (default: resnet34)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=768,
        help="Input image size (default: 768)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert weights to FP16 for smaller model size",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_pytorch_model(args.checkpoint, args.encoder_name)

    fp32_path = args.output if not args.fp16 else args.output.with_suffix(".fp32.onnx")

    print(f"Exporting to ONNX (opset 17, image size {args.image_size})...")
    export_to_onnx(model, fp32_path, args.image_size)

    max_diff = validate_export(model, fp32_path, args.image_size)
    print(f"Validation: max diff between PyTorch and ONNX = {max_diff:.6e}")

    if args.fp16:
        print("Converting to FP16...")
        convert_to_fp16(fp32_path, args.output)
        fp32_path.unlink()
        print(f"FP16 model saved: {args.output}")
    else:
        print(f"FP32 model saved: {args.output}")

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
