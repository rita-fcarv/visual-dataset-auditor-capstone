# GPU-Accelerated Visual Dataset Auditor

**A CUDA/C++ application for evaluating image dataset quality and detecting visually similar images using GPU-accelerated processing.**

## Overview

Image datasets are fundamental to computer vision and machine learning applications. However, building and maintaining high-quality datasets can be challenging, particularly when working with large collections of images.

Datasets may contain duplicate or near-duplicate images, blurry samples, or images with poor brightness and contrast. These issues can affect downstream analysis, introduce redundancy into training datasets, and make data preparation more time-consuming.

The **GPU-Accelerated Visual Dataset Auditor** was developed to help address these challenges by providing an automated way to analyse image collections and generate reports highlighting potential data quality issues.

The project combines traditional image processing techniques with custom CUDA kernels to explore how GPU parallelism can improve the efficiency of dataset auditing.

Rather than automatically deciding which images should be removed, the application provides objective measurements and similarity information to support informed dataset curation.

## Project Goals

The project has three main goals:

1. **Assess image quality:** Extract meaningful metrics that help identify potentially blurry, poorly exposed, or low-contrast images.
2. **Detect duplicate and similar images:** Use perceptual hashing to identify image pairs that may contain redundant visual information.
3. **Evaluate GPU acceleration:** Implement image processing and comparison algorithms using CUDA, validate their correctness, and compare their performance against CPU implementations.

Together, these objectives demonstrate how GPU programming can support practical computer vision data preparation workflows.

## How It Works

The application processes a directory of images through a sequence of GPU-accelerated operations.

### 1. Image Loading and Preprocessing

Images are loaded using OpenCV and converted from BGR to grayscale using a custom CUDA kernel.

Grayscale conversion simplifies subsequent analysis by representing each pixel with a single intensity value.

### 2. Image Quality Assessment

For each image, the application computes three quality metrics:

| Metric | Description | Purpose |
|---|---|---|
| Mean brightness | Average grayscale pixel intensity | Identify potentially underexposed or overexposed images |
| Contrast | Standard deviation of pixel intensities | Identify images with limited intensity variation |
| Blur score | Variance of the Laplacian response | Estimate sharpness and identify potentially blurry images |

These metrics are calculated using custom CUDA kernels and parallel reductions.

The resulting values are saved to a CSV report for further inspection.

Importantly, these measurements are indicators rather than definitive quality classifications. For example, a naturally dark image may have low brightness without being incorrectly exposed.

### 3. Perceptual Image Hashing

To detect visually similar images, the application computes a 64-bit difference hash (*dHash*) for each image.

The process involves resizing the grayscale image to 9×8 pixels, comparing adjacent pixel intensities, and encoding the comparisons into a compact binary hash.

Unlike cryptographic hashing, perceptual hashing is designed to capture aspects of visual structure. Images with similar visual content may therefore produce similar hashes even when their pixel values are not identical.

### 4. GPU Similarity Detection

After generating the hashes, the application compares image pairs using their Hamming distance, which counts the number of differing bits between two hashes.

Pairs with distances below or equal to a configurable threshold are reported as potentially similar.

The comparison is accelerated using CUDA, allowing multiple image pairs to be evaluated in parallel.

The optimised implementation processes unique image pairs and collects matching results on the GPU, avoiding the need to transfer a complete pairwise distance matrix back to the CPU.

### 5. Report Generation

The application generates two main reports:

- **Quality report:** Contains the image quality metrics for each successfully processed image.
- **Similarity report:** Contains pairs of potentially similar images and their Hamming distances.

These reports can be used to inspect image collections, investigate potential duplicates, and support decisions about dataset cleaning.

## Technologies

| Technology | Role |
|---|---|
| CUDA | GPU kernels, parallel processing and reductions |
| C++14 | Main application and processing logic |
| OpenCV | Image loading, resizing and output |
| NVIDIA GPU | Hardware acceleration |
| Python 3 | Synthetic dataset generation and evaluation |
| GNU Make | Building the application |

The project was developed and tested in the following environment:

- GPU: NVIDIA L4
- CUDA Toolkit: 11.3
- GCC: 9.3
- OpenCV: 4.2

The supplied Makefile uses the `sm_86` CUDA architecture target, which is compatible with the tested environment. Other GPU/toolkit combinations may require adjusting this flag.

## Installation and Build

### Prerequisites

A Linux environment with the following dependencies is required:

- An NVIDIA GPU with compatible drivers
- CUDA Toolkit and `nvcc`
- A C++14-compatible compiler
- OpenCV 4 development libraries
- GNU Make
- Python 3

### Build the application

From the project root:

```bash
make clean
make
```

This generates the `visual_auditor` executable.

## Usage

### Run the auditor

```bash
./visual_auditor \
  --input data/images \
  --output results/processed \
  --threshold 5
```

The application accepts the following arguments:

| Argument | Description |
|---|---|
| `--input` | Directory containing the input images |
| `--output` | Directory where reports and processed outputs are saved |
| `--threshold` | Maximum Hamming distance used to identify similar images (0–64) |

The threshold is configurable because the appropriate level of similarity depends on the characteristics of the dataset.

A threshold of 5 was selected for the project's synthetic robustness evaluation. It should not be treated as universally optimal.

### Generate the test datasets

The repository includes Python scripts for generating synthetic datasets:

```bash
python3 scripts/generate_dataset.py
python3 scripts/generate_validation.py
python3 scripts/generate_robustness_dataset.py
```

These scripts create reproducible image collections for testing the application, including controlled variations such as brightness changes, blur and resizing.

### Run the evaluation

```bash
./visual_auditor \
  --input data/robustness \
  --output results/optimized_validation \
  --threshold 5

python3 scripts/evaluate_similarity.py \
  --input data/robustness \
  --report results/optimized_validation/similarity_report.csv
```

The evaluation compares detected image pairs against the expected relationships in the synthetic dataset.

### Run the automated workflow

A convenience script is also provided:

```bash
./run.sh
```

It builds the application, generates the test datasets, runs the auditor and evaluates similarity detection.

## Validation and Experimental Results

Correctness and performance were evaluated separately to ensure that GPU acceleration did not compromise the expected results.

### Similarity Detection Accuracy

A synthetic robustness dataset was created with 120 images organised into groups containing controlled image variations.

The evaluation includes exact copies, brightness adjustments, blur, resizing and visually different images.

At a Hamming distance threshold of 5, the results were:

| Metric | Result |
|---|---:|
| Images | 120 |
| Possible image pairs | 7,140 |
| Expected similar pairs | 200 |
| Detected similar pairs | 200 |
| True positives | 200 |
| False positives | 0 |
| False negatives | 0 |
| True negatives | 6,940 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 score | 1.0000 |

These results demonstrate correct detection against the defined synthetic ground truth. They do not imply perfect accuracy on arbitrary real-world image datasets.

### CPU vs GPU: Image Processing

Grayscale conversion and image quality metrics were benchmarked at three image resolutions.

| Resolution | Grayscale speedup | Quality metrics speedup |
|---|---:|---:|
| 128×128 | 0.15× | 0.21× |
| 512×512 | 1.52× | 1.18× |
| 1024×1024 | **2.36×** | **2.79×** |

For 1024×1024 images:

| Operation | CPU | GPU |
|---|---:|---:|
| Grayscale conversion | 2.2104 ms | 0.9384 ms |
| Quality metrics | 3.2144 ms | 1.1507 ms |

The GPU provides measurable benefits for larger images, while the CPU remains faster for smaller images because GPU allocation, kernel launches and data transfers introduce additional overhead.

Grayscale outputs were also compared for correctness. At 1024×1024 resolution, only 64 pixels differed between the CPU and GPU implementations, with a maximum difference of one intensity level.

### CPU vs GPU: Similarity Detection

The similarity detection algorithm was benchmarked using synthetic 64-bit image hashes.

| Images | CPU | Original GPU | Optimised GPU | Optimised speedup vs CPU |
|---|---:|---:|---:|---:|
| 120 | 0.0233 ms | 0.2418 ms | 0.2420 ms | 0.10× |
| 500 | 0.3883 ms | 0.5496 ms | 0.2443 ms | 1.59× |
| 1,000 | 1.5476 ms | 1.5796 ms | 0.2625 ms | **5.89×** |

The original GPU implementation transferred the full pairwise distance matrix back to the CPU.

The optimised implementation reduces unnecessary data transfers by collecting only matching pairs, resulting in a substantial improvement for larger datasets.

At 1,000 images, the optimised GPU implementation achieved approximately a **5.89× speedup over the CPU baseline** and reduced measured execution time by approximately 83.4% compared with the original GPU implementation.

All benchmarks were performed in the NVIDIA L4 environment using synthetic workloads. The GPU timings include memory allocation and transfer overhead, and the CPU reference implementations are single-threaded. Results may differ with other hardware, datasets and implementations.

## Implementation Challenges and Lessons Learned

### GPU Overhead and Workload Size

One important observation was that GPU acceleration is not automatically beneficial for every workload.

For small images and datasets, the cost of allocating GPU memory, transferring data and launching kernels can exceed the benefits of parallel execution.

As the workload increases, GPU parallelism becomes more effective.

### Memory Transfer Optimisation

The initial similarity implementation transferred a complete matrix of pairwise distances to the CPU.

Although the comparisons were performed in parallel, transferring unnecessary results introduced substantial overhead.

The optimised implementation instead evaluates unique image pairs and compacts matching results on the GPU.

This change demonstrated how memory movement and data representation can be as important as kernel execution speed.

### CPU and GPU Numerical Consistency

CPU and GPU implementations can produce slightly different floating-point results because of differences in arithmetic evaluation and rounding.

During grayscale validation, a small number of pixels differed by one intensity level.

This highlighted the importance of using appropriate numerical tolerances when validating equivalent implementations rather than requiring exact bit-for-bit equality in every case.

### Correctness Before Performance

Performance improvements were evaluated only after checking that the implementation produced the expected results.

The project uses controlled synthetic datasets and explicit similarity ground truth to distinguish correctness from execution speed.

This approach helped ensure that optimisation did not silently change the detector's behaviour.

## Limitations and Future Improvements

Although the application demonstrates useful functionality and measurable acceleration, several limitations remain.

**Perceptual hashing:** dHash is a lightweight similarity technique, but it is not robust to every visual transformation. Certain changes may lead to missed matches, while unrelated images may occasionally produce similar hashes.

**Scalability:** Pairwise comparison has quadratic computational complexity. Very large datasets would benefit from candidate filtering, indexing or approximate nearest-neighbour techniques.

**GPU memory management:** Reusing device buffers and processing images in batches could further reduce allocation and transfer overhead.

**Quality assessment:** Brightness, contrast and Laplacian variance provide useful indicators but do not capture all aspects of image quality.

**Dataset diversity:** The current validation relies primarily on synthetic datasets. Evaluation using real-world image collections would provide a more representative assessment of practical performance.

Potential future extensions include more advanced perceptual similarity methods, configurable quality filters, interactive report visualisation and support for larger datasets through streaming or batched processing.

## Project Structure

```text
visual-dataset-auditor/
├── src/
│   ├── main.cu
│   ├── image_processor.cu
│   ├── image_processor.cuh
│   ├── image_metrics.cu
│   ├── image_metrics.cuh
│   ├── image_hash.cu
│   ├── image_hash.cuh
│   ├── similarity.cu
│   └── similarity.cuh
├── benchmarks/
│   ├── benchmark_image.cu
│   └── benchmark_similarity.cu
├── scripts/
│   ├── generate_dataset.py
│   ├── generate_validation.py
│   ├── generate_robustness_dataset.py
│   └── evaluate_similarity.py
├── data/
├── results/
│   └── benchmarks/
├── submission/
├── Makefile
├── run.sh
└── README.md
```

Generated datasets, reports and benchmark outputs are stored in their respective directories. The `submission/` directory contains supporting material for the Coursera capstone submission.

## Conclusion

The GPU-Accelerated Visual Dataset Auditor demonstrates how CUDA can be applied to a practical computer vision data preparation problem.

By combining image quality assessment, perceptual hashing and parallel similarity detection, the project provides a foundation for inspecting image collections and identifying potential issues before they are used in downstream workflows.

The experimental results show that GPU acceleration can offer meaningful performance improvements, particularly for larger images and datasets. At the same time, the project illustrates that efficient GPU programming requires careful consideration of workload size, memory transfers, numerical correctness and algorithm design.

Overall, the project connects GPU programming concepts with a real-world dataset quality challenge while providing a reproducible framework for further experimentation and optimisation.
---

## Capstone Extension: GPU Dataset Audit Reporting

### Overview

This capstone extends the GPU-Accelerated Visual Dataset Auditor with a reproducible dataset auditing and visualization workflow.

The project combines custom CUDA/C++ kernels for image processing and similarity detection with Python scripts for summarizing and visualizing the GPU-generated results.

The goal is to demonstrate a practical GPU application for computer vision dataset preparation, including image quality assessment, duplicate detection, correctness validation, performance benchmarking, and visual reporting.

### Capstone Features

The complete application provides:

1. GPU-accelerated grayscale conversion using CUDA.
2. Image quality assessment using brightness, contrast, and Laplacian variance.
3. Perceptual image hashing using 64-bit dHash.
4. GPU-based image similarity detection using Hamming distance.
5. Synthetic dataset generation and robustness evaluation.
6. CPU versus GPU performance benchmarking.
7. Consolidated dataset audit reports in JSON and Markdown.
8. Quality metric distribution visualizations.
9. Visual comparisons of original and transformed images.

The CUDA/C++ application performs the core image processing and similarity computations. Python is used for dataset generation, evaluation, reporting, and visualization.

### Development Environment

The project was developed and tested in the Coursera GPU laboratory using:

| Component | Configuration |
|---|---|
| GPU | NVIDIA L4 (approximately 23 GB VRAM) |
| CUDA Toolkit | 11.3 |
| GPU architecture | sm_86 |
| Programming language | CUDA C++14 |
| Image library | OpenCV C++ |
| Reporting language | Python 3 |
| Visualization libraries | Matplotlib, Pillow |
| Build system | GNU Make |

The Makefile targets NVIDIA GPUs supporting the `sm_86` architecture. For other compatible NVIDIA GPUs, the `-arch` setting may need to be adjusted.

### Prerequisites and Installation

The CUDA application requires:

- A compatible NVIDIA GPU and driver.
- NVIDIA CUDA Toolkit, including `nvcc`.
- A C++ compiler supported by the installed CUDA Toolkit.
- OpenCV development libraries discoverable through `pkg-config opencv4`.
- GNU Make.
- Python 3 with Matplotlib and Pillow.

On an Ubuntu-based environment, the system build dependencies can typically be installed with:

```bash
sudo apt-get update
sudo apt-get install -y build-essential make pkg-config libopencv-dev python3 python3-pip
```

Install the NVIDIA driver and CUDA Toolkit separately according to the GPU and operating system. CUDA 11.3 was used for the recorded results.

Install the Python reporting dependencies:

```bash
python3 -m pip install --user -r requirements.txt
```

Check the GPU and CUDA compiler:

```bash
nvidia-smi
nvcc --version
```

Compile the CUDA application:

```bash
make
```

The resulting executable is named `visual_auditor`.

### Command-Line Usage

The main application accepts an input directory, an output directory, and a similarity threshold.

Example:

```bash
./visual_auditor \
  --input data/robustness \
  --output results/optimized_validation \
  --threshold 5
```

Arguments:

| Argument | Description |
|---|---|
| `--input` | Directory containing images to process |
| `--output` | Directory where the generated results are saved |
| `--threshold` | Maximum dHash Hamming distance for reporting similar image pairs |

A lower threshold is more restrictive, while a higher threshold allows more image pairs to be reported as potentially similar.

The application generates image quality and similarity reports as CSV files.

### Running the Complete Capstone

Run the automated workflow from the project root:

```bash
./run.sh
```

This workflow:

1. Builds the CUDA application.
2. Generates synthetic image datasets.
3. Processes the main dataset of 500 images.
4. Processes the robustness dataset of 120 images.
5. Evaluates similarity detection against known synthetic transformations.
6. Generates consolidated audit reports.
7. Generates quality metric visualizations.
8. Generates visual examples of detected similar images.

To save the complete execution log:

```bash
set -o pipefail
./run.sh 2>&1 | tee results/capstone_execution.log
```

The `results/` directory is created automatically by the workflow.

### Capstone Reporting Scripts

#### Dataset Audit Summary

The script `scripts/generate_audit_summary.py` reads CUDA-generated CSV reports and produces a consolidated dataset summary.

Example:

```bash
python3 scripts/generate_audit_summary.py \
  --quality results/optimized_validation/quality_report.csv \
  --similarity results/optimized_validation/similarity_report.csv \
  --output results/capstone_audit \
  --threshold 5
```

Outputs:

- `audit_summary.json`: structured audit statistics.
- `audit_summary.md`: human-readable audit report.

The summary includes image counts, average quality metrics, brightness observations, images with relatively low Laplacian variance, and reported similar image pairs.

The low-blur-score category uses the lowest 10% of scores in the evaluated dataset. It is a relative indicator, not a validated blur classification threshold.

The similarity CSV contains reported pairs meeting the application's threshold, rather than every possible image pair.

#### Quality Distribution Dashboard

The script `scripts/generate_audit_dashboard.py` generates histograms for brightness, contrast, and Laplacian variance.

Example:

```bash
python3 scripts/generate_audit_dashboard.py \
  --quality results/optimized_validation/quality_report.csv \
  --output results/capstone_audit/quality_distribution.png
```

#### Similarity Visualization

The script `scripts/generate_similarity_examples.py` displays original images alongside transformations such as exact duplication, brightness adjustment, blurring, and resizing.

The Hamming distances displayed in the figure are obtained from the CUDA-generated similarity report.

Example:

```bash
python3 scripts/generate_similarity_examples.py \
  --images data/robustness \
  --similarity results/optimized_validation/similarity_report.csv \
  --output results/capstone_audit/similarity_examples.png \
  --group 0
```

### Capstone Execution Results

The complete pipeline was executed successfully using the NVIDIA L4 GPU.

| Measurement | Result |
|---|---:|
| Main dataset images processed | 500 |
| Main dataset processing failures | 0 |
| Main dataset similar pairs reported | 6,113 |
| Main dataset processing time | 17.539 seconds |
| Robustness dataset images processed | 120 |
| Robustness dataset processing failures | 0 |
| Robustness dataset similar pairs reported | 200 |
| Robustness dataset processing time | 3.063 seconds |

#### Robustness Evaluation

The robustness dataset consists of 20 groups of 6 images, including controlled transformations and visually different examples.

For 120 images, there are 7,140 possible unique image pairs.

At a Hamming distance threshold of 5, the evaluation produced:

| Metric | Result |
|---|---:|
| Expected similar pairs | 200 |
| Detected similar pairs | 200 |
| True positives | 200 |
| False positives | 0 |
| False negatives | 0 |
| True negatives | 6,940 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 score | 1.0000 |

These results demonstrate correct detection under the controlled synthetic transformations used in this evaluation. They do not establish equivalent performance on arbitrary real-world datasets.

#### GPU Performance

The original project includes CPU-versus-GPU benchmarks for grayscale conversion, image quality calculations, and image similarity detection.

Previously recorded benchmark results include:

| Operation | Recorded GPU speedup |
|---|---:|
| Grayscale conversion (1024×1024) | Approximately 2.36× |
| Image quality metrics (1024×1024) | Approximately 2.79× |
| Optimized similarity comparison (1,000 hashes) | Approximately 5.89× |

These measurements are workload-specific and should not be interpreted as universal GPU speedups.

GPU acceleration introduces overhead from memory transfers and kernel launches. Consequently, smaller workloads may execute faster on the CPU.

### Generated Outputs

The complete workflow generates the following reporting artifacts:

```text
results/
├── processed/
│   └── GPU-generated outputs for the main dataset
├── optimized_validation/
│   ├── quality_report.csv
│   └── similarity_report.csv
└── capstone_audit/
    ├── audit_summary.json
    ├── audit_summary.md
    ├── quality_distribution.png
    └── similarity_examples.png
```

Additional generated datasets, validation outputs, and benchmark files may also be present.

### Execution Evidence for Reviewers

The `evidence/` directory contains representative outputs from a successful capstone execution.

```text
evidence/
├── capstone_execution.log
├── audit_reports/
│   ├── audit_summary.json
│   ├── audit_summary.md
│   ├── quality_distribution.png
│   └── similarity_examples.png
├── cuda_reports/
│   ├── quality_report.csv
│   └── similarity_report.csv
└── benchmark_results/
    ├── image_benchmark.log
    ├── similarity_benchmark.csv
    └── similarity_optimized.csv
```

The execution log demonstrates that the application processed multiple images in a single execution and completed the robustness evaluation.

The CSV files provide machine-readable results, while the PNG figures illustrate the image quality distributions and similarity detection examples.

Generated datasets and intermediate results are excluded from Git to avoid unnecessarily large repository contents. The dataset generators and execution scripts allow these results to be reproduced.

### Challenges and Lessons Learned

A major challenge in GPU programming is identifying workloads that benefit from parallel execution.

Although GPUs can process large amounts of data concurrently, memory transfers, kernel launches, and synchronization introduce overhead.

The project demonstrated that GPU acceleration is most beneficial when the amount of parallel work is sufficient to offset these costs.

Another challenge was validating similarity detection. Using synthetic datasets with known transformations made it possible to evaluate precision, recall, and false detections systematically.

The capstone extension also highlighted the importance of presenting technical results in an accessible format. Consolidated reports and visual comparisons make the CUDA-generated measurements easier to interpret.

### Limitations and Future Work

The current implementation has several limitations:

- dHash captures relatively simple image structure and may not reliably detect all types of perceptual similarity.
- Image quality metrics are indicators rather than definitive assessments.
- The robustness evaluation uses synthetic data rather than a representative real-world benchmark.
- The current GPU implementation and build configuration are optimized for the tested CUDA environment.
- Performance gains depend on dataset size, image resolution, and hardware characteristics.

Future improvements could include larger real-world datasets, more advanced perceptual hashing or image embeddings, additional quality metrics, and improved GPU memory management and batching.

### Capstone Presentation

A recorded 5–10 minute presentation will demonstrate the project goals, CUDA implementation, execution workflow, results, challenges, and potential future improvements.

The presentation URL will be added when the recording is available.
