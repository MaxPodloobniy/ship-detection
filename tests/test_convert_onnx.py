import lightning
import numpy as np
import onnxruntime as ort
import pytest
import torch

from scripts.convert_to_onnx import (
    convert_to_fp16,
    export_to_onnx,
    load_pytorch_model,
    validate_export,
)
from src.training.trainer import ShipSegmentationModule


@pytest.fixture()
def dummy_checkpoint(tmp_path):
    """Create a Lightning checkpoint for testing."""
    model = ShipSegmentationModule(encoder_name="resnet34", lr=1e-4)
    trainer = lightning.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.strategy.connect(model)
    ckpt_path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt_path)
    return ckpt_path


class TestLoadPytorchModel:
    def test_loads_model(self, dummy_checkpoint):
        model = load_pytorch_model(dummy_checkpoint, encoder_name="resnet34")
        assert not model.training

    def test_forward_pass(self, dummy_checkpoint):
        model = load_pytorch_model(dummy_checkpoint, encoder_name="resnet34")
        out = model(torch.randn(1, 3, 256, 256))
        assert out.shape == (1, 1, 256, 256)


class TestExportToOnnx:
    def test_creates_onnx_file(self, dummy_checkpoint, tmp_path):
        model = load_pytorch_model(dummy_checkpoint)
        onnx_path = tmp_path / "model.onnx"
        export_to_onnx(model, onnx_path, image_size=256)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_onnx_runs_inference(self, dummy_checkpoint, tmp_path):
        model = load_pytorch_model(dummy_checkpoint)
        onnx_path = tmp_path / "model.onnx"
        export_to_onnx(model, onnx_path, image_size=256)

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
        output = session.run(None, {"input": dummy})
        assert output[0].shape == (1, 1, 256, 256)

    def test_dynamic_batch(self, dummy_checkpoint, tmp_path):
        model = load_pytorch_model(dummy_checkpoint)
        onnx_path = tmp_path / "model.onnx"
        export_to_onnx(model, onnx_path, image_size=256)

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        dummy = np.random.randn(2, 3, 256, 256).astype(np.float32)
        output = session.run(None, {"input": dummy})
        assert output[0].shape == (2, 1, 256, 256)


class TestConvertToFp16:
    def test_produces_valid_fp16_model(self, dummy_checkpoint, tmp_path):
        model = load_pytorch_model(dummy_checkpoint)
        fp32_path = tmp_path / "model_fp32.onnx"
        fp16_path = tmp_path / "model_fp16.onnx"

        export_to_onnx(model, fp32_path, image_size=256)
        convert_to_fp16(fp32_path, fp16_path)

        assert fp16_path.exists()
        assert fp16_path.stat().st_size > 0

    def test_fp16_runs_inference(self, dummy_checkpoint, tmp_path):
        model = load_pytorch_model(dummy_checkpoint)
        fp32_path = tmp_path / "model_fp32.onnx"
        fp16_path = tmp_path / "model_fp16.onnx"

        export_to_onnx(model, fp32_path, image_size=256)
        convert_to_fp16(fp32_path, fp16_path)

        session = ort.InferenceSession(
            str(fp16_path), providers=["CPUExecutionProvider"]
        )
        dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
        output = session.run(None, {"input": dummy})
        assert output[0].shape == (1, 1, 256, 256)


class TestValidateExport:
    def test_outputs_close(self, dummy_checkpoint, tmp_path):
        model = load_pytorch_model(dummy_checkpoint)
        onnx_path = tmp_path / "model.onnx"
        export_to_onnx(model, onnx_path, image_size=256)

        max_diff = validate_export(model, onnx_path, image_size=256)
        assert max_diff < 1e-4
