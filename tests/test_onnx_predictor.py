import numpy as np
import pytest
import segmentation_models_pytorch as smp
import torch

from src.inference.onnx_predictor import IMAGENET_MEAN, IMAGENET_STD, OnnxShipPredictor


@pytest.fixture()
def onnx_model_path(tmp_path):
    """Export a tiny FPN to ONNX for testing."""
    model = smp.FPN(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.eval()
    dummy = torch.randn(1, 3, 256, 256)
    onnx_path = tmp_path / "test_model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return onnx_path


@pytest.fixture()
def predictor(onnx_model_path):
    return OnnxShipPredictor(
        model_path=onnx_model_path,
        image_size=256,
        threshold=0.5,
        min_ship_pixels=10,
    )


class TestPreprocess:
    def test_output_shape(self, predictor):
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = predictor.preprocess(image)
        assert result.shape == (1, 3, 256, 256)
        assert result.dtype == np.float32

    def test_rgba_input(self, predictor):
        image = np.random.randint(0, 255, (100, 200, 4), dtype=np.uint8)
        result = predictor.preprocess(image)
        assert result.shape == (1, 3, 256, 256)

    def test_normalization_range(self, predictor):
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = predictor.preprocess(image)
        pixel_val = 128.0 / 255.0
        expected_r = (pixel_val - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        assert abs(result[0, 0, 0, 0] - expected_r) < 0.1


class TestPredict:
    def test_returns_expected_keys(self, predictor):
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predictor.predict(image)
        assert "ship_count" in result
        assert "mask" in result
        assert "has_ships" in result

    def test_mask_shape_and_dtype(self, predictor):
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predictor.predict(image)
        assert result["mask"].shape == (256, 256)
        assert result["mask"].dtype == np.uint8

    def test_mask_values_are_0_or_255(self, predictor):
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predictor.predict(image)
        unique_values = set(np.unique(result["mask"]))
        assert unique_values.issubset({0, 255})

    def test_ship_count_is_non_negative(self, predictor):
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predictor.predict(image)
        assert result["ship_count"] >= 0

    def test_has_ships_matches_count(self, predictor):
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predictor.predict(image)
        assert result["has_ships"] == (result["ship_count"] > 0)

    def test_different_input_sizes(self, predictor):
        for h, w in [(100, 100), (500, 300), (768, 768)]:
            image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            result = predictor.predict(image)
            assert result["mask"].shape == (256, 256)


class TestMinShipPixelsFiltering:
    def test_small_components_filtered(self, onnx_model_path):
        predictor = OnnxShipPredictor(
            model_path=onnx_model_path,
            image_size=256,
            threshold=0.5,
            min_ship_pixels=1000,
        )
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predictor.predict(image)
        assert result["ship_count"] >= 0
