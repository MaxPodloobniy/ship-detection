import lightning
import pytest
import torch
from PIL import Image

from src.inference.predictor import InferenceImageDataset, ShipPredictor
from src.training.trainer import ShipSegmentationModule


@pytest.fixture()
def dummy_checkpoint(tmp_path):
    """Create a proper Lightning checkpoint via the Trainer."""
    model = ShipSegmentationModule(
        model_name="nvidia/segformer-b0-finetuned-ade-512-512",
        lr=1e-4,
    )
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
        predictor = ShipPredictor(dummy_checkpoint, device="cpu", threshold=0.5)
        probs, masks = predictor.predict_batch(torch.randn(2, 3, 768, 768))

        assert probs.shape == (2, 1, 768, 768)
        assert masks.shape == (2, 1, 768, 768)
        assert masks.dtype == torch.uint8

    def test_generate_submission(self, tmp_path, dummy_checkpoint):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for name in ["img1.jpg", "img2.jpg"]:
            Image.new("RGB", (768, 768)).save(img_dir / name)

        predictor = ShipPredictor(dummy_checkpoint, device="cpu")
        df = predictor.generate_submission(img_dir, batch_size=2, num_workers=0)

        assert len(df) == 2
        assert list(df.columns) == ["ImageId", "EncodedPixels"]
