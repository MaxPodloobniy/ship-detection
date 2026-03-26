"""Entry point for model training.

Usage:
    python -m entrypoint.training \
        --data-dir ./data \
        --epochs 30 \
        --batch-size 16 \
        --lr 1e-4
"""

import argparse
from pathlib import Path

import lightning
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from src.training.dataset import ShipDataModule
from src.training.trainer import ShipSegmentationModule


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FPN for Airbus Ship Detection",
    )

    # ── data ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing train_v2/ images and train_ship_segmentations_v2.csv",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=768,
        help="Spatial resolution for images and masks (default: 768, native Kaggle size)",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Fraction of ship-free images to keep (e.g. 0.1 to downsample negatives)",
    )

    # ── model ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="resnet34",
        help="Encoder backbone name (default: resnet34)",
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Pretrained weights identifier (default: imagenet)",
    )

    # ── training hyper-parameters ─────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--lovasz-weight", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument(
        "--loss-type",
        type=str,
        default="bce_dice",
        choices=["bce_dice", "bce_lovasz"],
        help="Loss function to use (default: bce_dice)",
    )
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="plateau",
        choices=["plateau", "cosine"],
        help="LR scheduler type (default: plateau)",
    )

    # ── hardware ──────────────────────────────────────────────────────
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--num-workers", type=int, default=4)

    # ── checkpointing / logging ───────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for checkpoints, logs, and artifacts",
    )
    parser.add_argument("--experiment-name", type=str, default="ship_fpn")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    lightning.seed_everything(args.seed, workers=True)

    # ── data ──────────────────────────────────────────────────────────
    datamodule = ShipDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )

    # ── model ─────────────────────────────────────────────────────────
    model = ShipSegmentationModule(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        lr=args.lr,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        lovasz_weight=args.lovasz_weight,
        pos_weight=args.pos_weight,
        loss_type=args.loss_type,
        scheduler_type=args.scheduler_type,
    )

    # ── callbacks ─────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir / args.experiment_name / "checkpoints",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")

    # ── trainer ───────────────────────────────────────────────────────
    trainer = lightning.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor_cb],
        logger=CSVLogger(save_dir=args.output_dir, name=args.experiment_name),
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
