import io

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import torch
from PIL import Image

from src.web.app import create_app


@pytest.fixture()
def onnx_model_path(tmp_path):
    """Export a tiny FPN to ONNX for the web app tests."""
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
def client(onnx_model_path):
    """Flask test client with a dummy ONNX model."""
    app = create_app(model_path=onnx_model_path, image_size=256)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _make_image_bytes(fmt: str = "JPEG", size: tuple[int, int] = (100, 100)) -> bytes:
    """Create a test image as bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


class TestIndexRoute:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_contains_html(self, client):
        response = client.get("/")
        assert b"Ship Detection" in response.data


class TestHealthRoute:
    def test_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.get_json()["status"] == "ok"


class TestPredictRoute:
    def test_no_file_returns_400(self, client):
        response = client.post("/predict")
        assert response.status_code == 400

    def test_invalid_extension_returns_400(self, client):
        data = {"file": (io.BytesIO(b"not an image"), "test.txt")}
        response = client.post(
            "/predict", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 400

    def test_valid_jpeg_returns_result(self, client):
        img_bytes = _make_image_bytes("JPEG")
        data = {"file": (io.BytesIO(img_bytes), "test.jpg")}
        response = client.post(
            "/predict", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 200
        json_data = response.get_json()
        assert "ship_count" in json_data
        assert "has_ships" in json_data
        assert "overlay" in json_data
        assert isinstance(json_data["ship_count"], int)

    def test_valid_png_returns_result(self, client):
        img_bytes = _make_image_bytes("PNG")
        data = {"file": (io.BytesIO(img_bytes), "test.png")}
        response = client.post(
            "/predict", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 200
        json_data = response.get_json()
        assert "ship_count" in json_data

    def test_overlay_is_base64(self, client):
        import base64

        img_bytes = _make_image_bytes("JPEG")
        data = {"file": (io.BytesIO(img_bytes), "test.jpg")}
        response = client.post(
            "/predict", data=data, content_type="multipart/form-data"
        )
        json_data = response.get_json()
        decoded = base64.b64decode(json_data["overlay"])
        assert len(decoded) > 0
