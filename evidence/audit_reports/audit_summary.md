# GPU Visual Dataset Audit Summary

Generated from CUDA image quality and similarity reports.

- Images analyzed: 120
- Pairs in similarity report: 200
- Similarity threshold: 5
- Similar image pairs: 200
- Average brightness: 149.03
- Average contrast: 76.59
- Average blur score: 3937.85

## Quality observations

- Low-brightness images (<60): 0
- High-brightness images (>200): 0
- Images in the lowest 10% of blur scores: 12

Low blur scores may indicate blurred images, but the score depends on image content and resolution.

## Similarity examples

| Image 1 | Image 2 | Hamming distance |
|---|---|---:|
| group_000_blur.ppm | group_000_bright.ppm | 0 |
| group_000_blur.ppm | group_000_exact.ppm | 0 |
| group_000_blur.ppm | group_000_original.ppm | 0 |
| group_000_blur.ppm | group_000_resized.ppm | 0 |
| group_000_bright.ppm | group_000_exact.ppm | 0 |
| group_000_bright.ppm | group_000_original.ppm | 0 |
| group_000_bright.ppm | group_000_resized.ppm | 0 |
| group_000_exact.ppm | group_000_original.ppm | 0 |
| group_000_exact.ppm | group_000_resized.ppm | 0 |
| group_000_original.ppm | group_000_resized.ppm | 0 |
| group_001_blur.ppm | group_001_bright.ppm | 1 |
| group_001_blur.ppm | group_001_exact.ppm | 0 |
| group_001_blur.ppm | group_001_original.ppm | 0 |
| group_001_blur.ppm | group_001_resized.ppm | 0 |
| group_001_bright.ppm | group_001_exact.ppm | 1 |
| group_001_bright.ppm | group_001_original.ppm | 1 |
| group_001_bright.ppm | group_001_resized.ppm | 1 |
| group_001_exact.ppm | group_001_original.ppm | 0 |
| group_001_exact.ppm | group_001_resized.ppm | 0 |
| group_001_original.ppm | group_001_resized.ppm | 0 |
