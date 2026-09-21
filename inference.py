#!/usr/bin/env python3
"""Inference for the original Pareidolia Paradox v3.3 model."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet34


SEED = 42
N_FOLDS = 5
IMG_SIZE = 224
DEFAULT_THRESHOLD = 0.555

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def rotate_normalize(image, azimuth):

    gray = ImageOps.grayscale(image)

    w, h = gray.size

    pad = max(w, h) // 4

    arr = np.array(gray)

    arr_padded = np.pad(
        arr,
        pad_width=pad,
        mode="reflect"
    )

    padded = Image.fromarray(
        arr_padded
    )

    rotated = padded.rotate(
        -float(azimuth),
        resample=Image.BICUBIC,
        expand=False
    )

    cropped = rotated.crop(
        (
            pad,
            pad,
            pad + w,
            pad + h
        )
    )

    return cropped.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.BICUBIC
    )


class PareidoliaTestDataset(Dataset):

    def __init__(
        self,
        df,
        image_dir
    ):

        self.df = (
            df.reset_index(drop=True)
        )

        self.image_dir = Path(
            image_dir
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = (
            self.image_dir
            / row["image_id"]
        )

        image = (
            Image.open(image_path)
            .convert("RGB")
        )

        image = rotate_normalize(
            image,
            float(row["sun_azimuth_angle"])
        )

        image = image.convert("RGB")

        return self.transform(image)


def create_model():

    model = resnet34(
        weights=None
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

        for images in loader:

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

    return np.concatenate(
        all_probs
    )


def validate_metadata(df):

    required = {
        "image_id",
        "sun_azimuth_angle"
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df["image_id"].isna().any():
        raise ValueError(
            "image_id contains missing values"
        )

    if df["image_id"].duplicated().any():
        raise ValueError(
            "image_id contains duplicates"
        )


def validate_submission(
    submission,
    metadata
):

    assert list(
        submission.columns
    ) == [
        "image_id",
        "label"
    ]

    assert (
        len(submission)
        == len(metadata)
    )

    assert (
        submission["image_id"].is_unique
    )

    assert (
        set(submission["image_id"])
        == set(metadata["image_id"])
    )

    assert (
        submission["image_id"]
        .equals(metadata["image_id"])
    )

    assert (
        submission.isnull()
        .sum()
        .sum()
        == 0
    )

    assert set(
        submission["label"].unique()
    ).issubset({0, 1})


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Pareidolia Paradox "
            "v3.3 inference"
        )
    )

    parser.add_argument(
        "--test-metadata",
        required=True
    )

    parser.add_argument(
        "--test-images",
        required=True
    )

    parser.add_argument(
        "--checkpoints-dir",
        required=True
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--output",
        default="submission.csv"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    metadata = pd.read_csv(
        args.test_metadata
    )

    validate_metadata(
        metadata
    )

    print(
        f"Test rows: {len(metadata)}"
    )

    dataset = PareidoliaTestDataset(
        metadata,
        args.test_images
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        )
    )

    checkpoint_dir = Path(
        args.checkpoints_dir
    )

    fold_probabilities = []

    for fold in range(
        1,
        N_FOLDS + 1
    ):

        checkpoint_path = (
            checkpoint_dir
            / f"seed_{SEED}_fold_{fold}.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: "
                f"{checkpoint_path}"
            )

        state = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        model = (
            create_model()
            .to(device)
        )

        # Original v3.3 checkpoint format.
        model.load_state_dict(
            state["model_state"]
        )

        probabilities = (
            predict_with_tta(
                model,
                loader,
                device
            )
        )

        assert len(
            probabilities
        ) == len(metadata)

        fold_probabilities.append(
            probabilities
        )

        print(
            f"Fold {fold}: "
            f"best_BA="
            f"{state.get('best_ba', 'unknown'):.6f}"
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensemble_probabilities = np.mean(
        np.vstack(
            fold_probabilities
        ),
        axis=0
    )

    predictions = (
        ensemble_probabilities
        >= args.threshold
    ).astype(int)

    submission = pd.DataFrame({
        "image_id":
            metadata["image_id"].values,
        "label":
            predictions
    })

    validate_submission(
        submission,
        metadata
    )

    submission.to_csv(
        args.output,
        index=False
    )

    print("\n" + "=" * 60)
    print("SUBMISSION CREATED")
    print("=" * 60)

    print(
        f"Rows: {len(submission)}"
    )

    print(
        f"Threshold: "
        f"{args.threshold:.3f}"
    )

    print(
        "Models: "
        f"{N_FOLDS} seed-42 folds"
    )

    print(
        f"Output: {args.output}"
    )

    print(
        "\nPrediction counts:"
    )

    print(
        submission[
            "label"
        ].value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()
