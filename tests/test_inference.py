import lightning
import numpy as np
import pytest
import torch
from PIL import Image

from src.inference.predictor import (
    InferenceImageDataset,
    ShipPredictor,
    mask_to_submission_rows,
)
from src.training.trainer import ShipSegmentationModule


@pytest.fixture()
def dummy_checkpoint(tmp_path):
    """Create a proper Lightning checkpoint via the Trainer."""
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


class TestInferenceImageDataset:
    def test_getitem_shape_and_image_id(self, tmp_path):
        Image.new("RGB", (768, 768)).save(tmp_path / "a.jpg")
        Image.new("RGB", (768, 768)).save(tmp_path / "b.png")

        ds = InferenceImageDataset(tmp_path)
        assert len(ds) == 2

        sample = ds[0]
        assert sample["pixel_values"].shape == (3, 768, 768)
        assert sample["image_id"].endswith((".jpg", ".png"))


class TestShipPredictor:
    def test_predict_batch_shapes(self, dummy_checkpoint):
        predictor = ShipPredictor(
            dummy_checkpoint, device="cpu", threshold=0.5, encoder_name="resnet34"
        )
        probs, masks = predictor.predict_batch(torch.randn(2, 3, 768, 768))

        assert probs.shape == (2, 1, 768, 768)
        assert masks.shape == (2, 1, 768, 768)
        assert masks.dtype == torch.uint8

    def test_tta_predict_shapes(self, dummy_checkpoint):
        predictor = ShipPredictor(
            dummy_checkpoint,
            device="cpu",
            threshold=0.5,
            encoder_name="resnet34",
            use_tta=True,
        )
        probs, masks = predictor.predict_batch(torch.randn(2, 3, 256, 256))

        assert probs.shape == (2, 1, 256, 256)
        assert masks.shape == (2, 1, 256, 256)

    def test_generate_submission(self, tmp_path, dummy_checkpoint):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for name in ["img1.jpg", "img2.jpg"]:
            Image.new("RGB", (768, 768)).save(img_dir / name)

        predictor = ShipPredictor(
            dummy_checkpoint, device="cpu", encoder_name="resnet34"
        )
        df = predictor.generate_submission(img_dir, batch_size=2, num_workers=0)

        assert list(df.columns) == ["ImageId", "EncodedPixels"]
        assert set(df["ImageId"].unique()) == {"img1.jpg", "img2.jpg"}


class TestMaskToSubmissionRows:
    def test_empty_mask_returns_nan(self):
        mask = np.zeros((768, 768), dtype=np.uint8)
        rows = mask_to_submission_rows("test.jpg", mask)
        assert len(rows) == 1
        assert rows[0]["ImageId"] == "test.jpg"
        assert np.isnan(rows[0]["EncodedPixels"])

    def test_single_ship(self):
        mask = np.zeros((768, 768), dtype=np.uint8)
        mask[100:120, 200:220] = 1
        rows = mask_to_submission_rows("test.jpg", mask)
        assert len(rows) == 1
        assert rows[0]["ImageId"] == "test.jpg"
        assert isinstance(rows[0]["EncodedPixels"], str)

    def test_two_ships(self):
        mask = np.zeros((768, 768), dtype=np.uint8)
        mask[100:120, 200:220] = 1
        mask[400:420, 500:520] = 1
        rows = mask_to_submission_rows("test.jpg", mask)
        assert len(rows) == 2

    def test_small_component_filtered(self):
        mask = np.zeros((768, 768), dtype=np.uint8)
        mask[100:102, 200:202] = 1  # 4 pixels
        rows = mask_to_submission_rows("test.jpg", mask, min_pixels=10)
        assert len(rows) == 1
        assert np.isnan(rows[0]["EncodedPixels"])
