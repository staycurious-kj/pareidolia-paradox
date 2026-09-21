#!/usr/bin/env python3
"""The Pareidolia Paradox — original v3.3 training pipeline."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


SEED = 42
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def rotate_normalize(image, azimuth):
    """v3.3 preprocessing: grayscale -> reflect pad -> rotate -> crop -> resize."""
    gray = ImageOps.grayscale(image)

    w, h = gray.size

    # Original v3.3 used 25% reflective padding.
    pad = max(w, h) // 4

    arr = np.array(gray)
    arr_padded = np.pad(
        arr,
        pad_width=pad,
        mode="reflect"
    )

    padded = Image.fromarray(arr_padded)

    rotated = padded.rotate(
        -float(azimuth),
        resample=Image.BICUBIC,
        expand=False
    )

    cropped = rotated.crop(
        (pad, pad, pad + w, pad + h)
    )

    resized = cropped.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.BICUBIC
    )

    return resized


class PareidoliaDataset(Dataset):

    def __init__(
        self,
        df,
        image_dir,
        train_mode=False
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.train_mode = train_mode

        aug = []

        if train_mode:
            aug.append(
                transforms.RandomHorizontalFlip(p=0.5)
            )

        aug.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])

        self.transform = transforms.Compose(aug)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = self.image_dir / row["image_id"]

        image = Image.open(image_path).convert("RGB")

        image = rotate_normalize(
            image,
            float(row["sun_azimuth_angle"])
        )

        image = image.convert("RGB")

        tensor = self.transform(image)

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32
        )

        return tensor, label


def create_model():

    model = resnet34(
        weights=ResNet34_Weights.IMAGENET1K_V1
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        1
    )

    return model


def predict_with_tta(
    model,
    loader,
    device
):

    model.eval()

    all_probs = []

    use_amp = device.type == "cuda"

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            flipped = torch.flip(
                images,
                dims=[3]
            )

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):

                p1 = torch.sigmoid(
                    model(images).view(-1)
                )

                p2 = torch.sigmoid(
                    model(flipped).view(-1)
                )

            probs = (
                (p1 + p2) / 2.0
            ).float().cpu().numpy()

            all_probs.append(probs)

    return np.concatenate(all_probs)


def train_one_fold(
    train_df,
    val_df,
    image_dir,
    checkpoint_path,
    fold,
    epochs,
    batch_size,
    lr,
    weight_decay,
    num_workers
):

    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_dataset = PareidoliaDataset(
        train_df,
        image_dir,
        train_mode=True
    )

    val_dataset = PareidoliaDataset(
        val_df,
        image_dir,
        train_mode=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = create_model().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    use_amp = device.type == "cuda"

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    best_ba = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0
        n_seen = 0

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):

                logits = model(
                    images
                ).view(-1)

                loss = criterion(
                    logits,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            running_loss += (
                loss.item() * images.size(0)
            )

            n_seen += images.size(0)

        scheduler.step()

        train_loss = (
            running_loss / n_seen
        )

        val_probs = predict_with_tta(
            model,
            val_loader,
            device
        )

        val_targets = (
            val_df["label"]
            .values
            .astype(int)
        )

        val_ba = balanced_accuracy_score(
            val_targets,
            (val_probs >= 0.5).astype(int)
        )

        print(
            f"[fold {fold}] "
            f"epoch {epoch}/{epochs} "
            f"loss={train_loss:.4f} "
            f"val_BA={val_ba:.6f}"
        )

        if val_ba > best_ba:

            best_ba = float(val_ba)

            best_state = {
                "model_state": model.state_dict(),
                "best_ba": best_ba,
                "seed": SEED,
                "fold": fold,
                "completed": True
            }

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        best_state,
        checkpoint_path
    )

    print(
        f"[fold {fold}] saved -> "
        f"{checkpoint_path} "
        f"| best_BA={best_ba:.6f}"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Pareidolia Paradox v3.3 training"
    )

    parser.add_argument(
        "--train-metadata",
        required=True
    )

    parser.add_argument(
        "--train-images",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        default="checkpoints_run"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=8
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2
    )

    args = parser.parse_args()

    seed_everything(SEED)

    metadata_path = Path(
        args.train_metadata
    )

    image_dir = Path(
        args.train_images
    )

    output_dir = Path(
        args.output_dir
    )

    checkpoint_dir = (
        output_dir / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        metadata_path
    )

    required_columns = {
        "image_id",
        "sun_azimuth_angle",
        "label"
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df["image_id"].duplicated().any():
        raise ValueError(
            "Duplicate image_id values found."
        )

    print(
        f"Training rows: {len(df)}"
    )

    # Original v3.3 split strategy.
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        skf.split(
            df,
            df["label"]
        ),
        start=1
    ):

        checkpoint_path = (
            checkpoint_dir
            / f"seed_{SEED}_fold_{fold}.pt"
        )

        if checkpoint_path.exists():

            print(
                f"[fold {fold}] "
                f"checkpoint already exists: "
                f"{checkpoint_path}"
            )

            continue

        train_df = (
            df.iloc[train_idx]
            .reset_index(drop=True)
        )

        val_df = (
            df.iloc[val_idx]
            .reset_index(drop=True)
        )

        train_one_fold(
            train_df=train_df,
            val_df=val_df,
            image_dir=image_dir,
            checkpoint_path=checkpoint_path,
            fold=fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers
        )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()