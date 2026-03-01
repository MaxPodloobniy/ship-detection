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
        description="Train SegFormer for Airbus Ship Detection",
    )

    # ── data ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing train_v2/ images and train_ship_segmentations_v2.csv",
    )

    # ── model ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--model-name",
        type=str,
        default="nvidia/segformer-b1-finetuned-ade-512-512",
        help="HuggingFace model identifier for SegFormer backbone",
    )

    # ── training hyper-parameters ─────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=None)

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
    parser.add_argument("--experiment-name", type=str, default="ship_segformer")
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
        seed=args.seed,
    )

    # ── model ─────────────────────────────────────────────────────────
    model = ShipSegmentationModule(
        model_name=args.model_name,
        lr=args.lr,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=args.pos_weight,
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
