#!/usr/bin/env python3
"""Generate a dataset audit summary from CUDA output CSV files."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize(quality_rows, similarity_rows, threshold):
    brightness = [float(row["mean_brightness"]) for row in quality_rows]
    contrast = [float(row["contrast"]) for row in quality_rows]
    blur = [float(row["blur_score"]) for row in quality_rows]

    low_brightness = [
        row["filename"] for row in quality_rows
        if float(row["mean_brightness"]) < 60
    ]
    high_brightness = [
        row["filename"] for row in quality_rows
        if float(row["mean_brightness"]) > 200
    ]

    sorted_blur = sorted(blur)
    blur_cutoff = sorted_blur[max(0, len(sorted_blur) // 10 - 1)]

    potentially_blurry = [
        row["filename"] for row in quality_rows
        if float(row["blur_score"]) <= blur_cutoff
    ]

    similar_pairs = [
        {
            "image_1": row["image_1"],
            "image_2": row["image_2"],
            "hamming_distance": int(row["hamming_distance"]),
        }
        for row in similarity_rows
        if int(row["hamming_distance"]) <= threshold
    ]

    return {
        "images_analyzed": len(quality_rows),
        "reported_pairs": len(similarity_rows),
        "similarity_threshold": threshold,
        "similar_pairs": len(similar_pairs),
        "average_brightness": mean(brightness),
        "average_contrast": mean(contrast),
        "average_blur_score": mean(blur),
        "low_brightness_images": low_brightness,
        "high_brightness_images": high_brightness,
        "potentially_blurry_images": potentially_blurry,
        "similarity_examples": similar_pairs[:20],
    }


def write_markdown(summary, output_path):
    lines = [
        "# GPU Visual Dataset Audit Summary",
        "",
        "Generated from CUDA image quality and similarity reports.",
        "",
        f"- Images analyzed: {summary['images_analyzed']}",
        f"- Pairs in similarity report: {summary['reported_pairs']}",
        f"- Similarity threshold: {summary['similarity_threshold']}",
        f"- Similar image pairs: {summary['similar_pairs']}",
        f"- Average brightness: {summary['average_brightness']:.2f}",
        f"- Average contrast: {summary['average_contrast']:.2f}",
        f"- Average blur score: {summary['average_blur_score']:.2f}",
        "",
        "## Quality observations",
        "",
        f"- Low-brightness images (<60): "
        f"{len(summary['low_brightness_images'])}",
        f"- High-brightness images (>200): "
        f"{len(summary['high_brightness_images'])}",
        f"- Images in the lowest 10% of blur scores: "
        f"{len(summary['potentially_blurry_images'])}",
        "",
        "Low blur scores may indicate blurred images, but the score "
        "depends on image content and resolution.",
        "",
        "## Similarity examples",
        "",
        "| Image 1 | Image 2 | Hamming distance |",
        "|---|---|---:|",
    ]

    for pair in summary["similarity_examples"]:
        lines.append(
            f"| {pair['image_1']} | {pair['image_2']} | "
            f"{pair['hamming_distance']} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize GPU-generated image audit reports."
    )
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--similarity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=5)
    args = parser.parse_args()

    if not 0 <= args.threshold <= 64:
        parser.error("--threshold must be between 0 and 64")

    quality_rows = read_csv(args.quality)
    similarity_rows = read_csv(args.similarity)

    if not quality_rows:
        parser.error("Quality report contains no image rows")

    summary = summarize(
        quality_rows, similarity_rows, args.threshold
    )

    args.output.mkdir(parents=True, exist_ok=True)

    json_path = args.output / "audit_summary.json"
    md_path = args.output / "audit_summary.md"

    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, md_path)

    print("Dataset audit summary generated")
    print(f"Images analyzed: {summary['images_analyzed']}")
    print(f"Pairs in similarity report: {summary['reported_pairs']}")
    print(f"Similar pairs: {summary['similar_pairs']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
