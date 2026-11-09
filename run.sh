#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== GPU Visual Dataset Auditor: Capstone ==="

mkdir -p results/processed
mkdir -p results/optimized_validation
mkdir -p results/benchmarks
mkdir -p results/capstone_audit

echo "[1/6] Building CUDA application..."
make

echo "[2/6] Generating datasets..."
python3 scripts/generate_dataset.py
python3 scripts/generate_validation.py
python3 scripts/generate_robustness_dataset.py

echo "[3/6] Running GPU image processing..."
./visual_auditor \
  --input data/images \
  --output results/processed \
  --threshold 5

./visual_auditor \
  --input data/robustness \
  --output results/optimized_validation \
  --threshold 5

echo "[4/6] Evaluating similarity detection..."
python3 scripts/evaluate_similarity.py \
  --input data/robustness \
  --report results/optimized_validation/similarity_report.csv

echo "[5/6] Generating audit summary..."
python3 scripts/generate_audit_summary.py \
  --quality results/optimized_validation/quality_report.csv \
  --similarity results/optimized_validation/similarity_report.csv \
  --output results/capstone_audit \
  --threshold 5

echo "[6/6] Generating visual reports..."
python3 scripts/generate_audit_dashboard.py \
  --quality results/optimized_validation/quality_report.csv \
  --output results/capstone_audit/quality_distribution.png

python3 scripts/generate_similarity_examples.py \
  --images data/robustness \
  --similarity results/optimized_validation/similarity_report.csv \
  --output results/capstone_audit/similarity_examples.png \
  --group 0

echo "========================================"
echo "Capstone execution completed!"
echo "========================================"
