#!/usr/bin/env python3
"""Visualize similar image pairs detected by the CUDA auditor."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def load_pairs(csv_path):
    pairs = {}

    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            first = row["image_1"]
            second = row["image_2"]
            distance = int(row["hamming_distance"])
            pairs[frozenset((first, second))] = distance

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CUDA-detected similar image pairs."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--similarity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", type=int, default=0)
    args = parser.parse_args()

    pairs = load_pairs(args.similarity)

    prefix = f"group_{args.group:03d}"
    original_name = f"{prefix}_original.ppm"

    transformations = [
        ("exact", "Exact duplicate"),
        ("bright", "Brightness change"),
        ("blur", "Blur"),
        ("resized", "Resized"),
    ]

    selected = []

    for suffix, label in transformations:
        transformed_name = f"{prefix}_{suffix}.ppm"
        key = frozenset((original_name, transformed_name))

        if key not in pairs:
            print(f"Skipping {transformed_name}: pair not in report")
            continue

        selected.append(
            (transformed_name, label, pairs[key])
        )

    if not selected:
        parser.error("No matching similarity pairs found")

    fig, axes = plt.subplots(
        len(selected),
        2,
        figsize=(9, 3 * len(selected)),
        squeeze=False,
    )

    for row_index, (filename, label, distance) in enumerate(selected):
        original_path = args.images / original_name
        transformed_path = args.images / filename

        with Image.open(original_path) as image:
            original = image.convert("RGB").copy()

        with Image.open(transformed_path) as image:
            transformed = image.convert("RGB").copy()

        axes[row_index][0].imshow(original)
        axes[row_index][0].set_title("Original image")
        axes[row_index][0].axis("off")

        axes[row_index][1].imshow(transformed)
        axes[row_index][1].set_title(
            f"{label} | Hamming distance: {distance}"
        )
        axes[row_index][1].axis("off")

    fig.suptitle(
        f"GPU Image Similarity Detection — Group {args.group:03d}",
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Similarity examples generated: {args.output}")
    print(f"Pairs visualized: {len(selected)}")


if __name__ == "__main__":
    main()
