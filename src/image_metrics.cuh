#ifndef IMAGE_METRICS_CUH_
#define IMAGE_METRICS_CUH_

#include <opencv2/core.hpp>

struct ImageMetrics {
  double mean_brightness = 0.0;
  double contrast = 0.0;
  double blur_score = 0.0;
};

// Computes image quality metrics using CUDA.
// Input must be a grayscale CV_8UC1 image.
bool ComputeImageMetricsGpu(const cv::Mat& grayscale,
                            ImageMetrics* metrics);

#endif  // IMAGE_METRICS_CUH_
