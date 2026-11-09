#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

#include "image_metrics.cuh"
#include "image_processor.cuh"

namespace {

using Clock = std::chrono::steady_clock;

cv::Mat MakeTestImage(int width, int height) {
  cv::Mat image(height, width, CV_8UC3);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image.at<cv::Vec3b>(y, x) = cv::Vec3b(
          (x * 3 + y) % 256,
          (x + y * 5) % 256,
          (x * 7 + y * 2) % 256);
    }
  }

  return image;
}

cv::Mat ConvertToGrayscaleCpu(const cv::Mat& input) {
  cv::Mat output(input.rows, input.cols, CV_8UC1);

  for (int y = 0; y < input.rows; ++y) {
    for (int x = 0; x < input.cols; ++x) {
      const cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);

      const float value =
          0.114f * pixel[0] +
          0.587f * pixel[1] +
          0.299f * pixel[2];

      output.at<unsigned char>(y, x) =
          static_cast<unsigned char>(value);
    }
  }

  return output;
}

ImageMetrics ComputeMetricsCpu(const cv::Mat& grayscale) {
  double sum = 0.0;
  double sum_squared = 0.0;
  double lap_sum = 0.0;
  double lap_squared = 0.0;

  const int width = grayscale.cols;
  const int height = grayscale.rows;
  const double count = static_cast<double>(width) * height;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const double center = grayscale.at<unsigned char>(y, x);

      sum += center;
      sum_squared += center * center;

      double laplacian = 0.0;

      if (x > 0 && x < width - 1 &&
          y > 0 && y < height - 1) {
        const double left =
            grayscale.at<unsigned char>(y, x - 1);
        const double right =
            grayscale.at<unsigned char>(y, x + 1);
        const double top =
            grayscale.at<unsigned char>(y - 1, x);
        const double bottom =
            grayscale.at<unsigned char>(y + 1, x);

        laplacian = left + right + top + bottom -
                    4.0 * center;
      }

      lap_sum += laplacian;
      lap_squared += laplacian * laplacian;
    }
  }

  ImageMetrics metrics;

  const double mean = sum / count;
  const double lap_mean = lap_sum / count;

  metrics.mean_brightness = mean;
  metrics.contrast =
      std::sqrt(std::max(0.0, sum_squared / count - mean * mean));
  metrics.blur_score =
      std::max(0.0, lap_squared / count - lap_mean * lap_mean);

  return metrics;
}

double ElapsedMs(Clock::time_point start,
                 Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(
      end - start).count();
}

bool RunBenchmark(int width, int height, int repetitions) {
  const cv::Mat input = MakeTestImage(width, height);

  cv::Mat gpu_grayscale;

  if (!ConvertToGrayscaleGpu(input, &gpu_grayscale)) {
    std::cerr << "GPU grayscale conversion failed.\n";
    return false;
  }

  const cv::Mat cpu_grayscale = ConvertToGrayscaleCpu(input);

  cv::Mat difference;
  cv::absdiff(cpu_grayscale, gpu_grayscale, difference);

  double maximum_difference = 0.0;
  cv::minMaxLoc(difference, nullptr, &maximum_difference);

  const int differing_pixels = cv::countNonZero(difference);

  std::cerr << "Resolution: " << width << "x" << height
            << " | Differing pixels: " << differing_pixels
            << " | Maximum difference: " << maximum_difference
            << "\n";

  if (maximum_difference > 1.0) {
    std::cerr << "CPU/GPU grayscale mismatch.\n";
    return false;
  }

  ImageMetrics gpu_metrics;

  if (!ComputeImageMetricsGpu(gpu_grayscale, &gpu_metrics)) {
    std::cerr << "GPU metrics calculation failed.\n";
    return false;
  }

  const ImageMetrics cpu_metrics =
      ComputeMetricsCpu(cpu_grayscale);

  if (std::abs(cpu_metrics.mean_brightness -
               gpu_metrics.mean_brightness) > 0.001 ||
      std::abs(cpu_metrics.contrast -
               gpu_metrics.contrast) > 0.001 ||
      std::abs(cpu_metrics.blur_score -
               gpu_metrics.blur_score) > 0.01) {
    std::cerr << "CPU/GPU metrics mismatch.\n";
    return false;
  }

  double cpu_gray_ms = 0.0;
  double gpu_gray_ms = 0.0;
  double cpu_metrics_ms = 0.0;
  double gpu_metrics_ms = 0.0;

  // Volatile sink prevents benchmark-only CPU calculations
  // from being discarded by the compiler.
  volatile double sink = 0.0;

  for (int i = 0; i < repetitions; ++i) {
    auto start = Clock::now();
    const cv::Mat gray = ConvertToGrayscaleCpu(input);
    auto end = Clock::now();
    cpu_gray_ms += ElapsedMs(start, end);
    sink += gray.at<unsigned char>(0, 0);

    start = Clock::now();
    if (!ConvertToGrayscaleGpu(input, &gpu_grayscale)) {
      return false;
    }
    end = Clock::now();
    gpu_gray_ms += ElapsedMs(start, end);

    start = Clock::now();
    const ImageMetrics metrics = ComputeMetricsCpu(cpu_grayscale);
    end = Clock::now();
    cpu_metrics_ms += ElapsedMs(start, end);
    sink += metrics.blur_score;

    start = Clock::now();
    if (!ComputeImageMetricsGpu(gpu_grayscale, &gpu_metrics)) {
      return false;
    }
    end = Clock::now();
    gpu_metrics_ms += ElapsedMs(start, end);
  }

  const double cpu_gray = cpu_gray_ms / repetitions;
  const double gpu_gray = gpu_gray_ms / repetitions;
  const double cpu_metric = cpu_metrics_ms / repetitions;
  const double gpu_metric = gpu_metrics_ms / repetitions;

  std::cout << width << "x" << height << ","
            << std::fixed << std::setprecision(4)
            << cpu_gray << ","
            << gpu_gray << ","
            << cpu_gray / gpu_gray << ","
            << cpu_metric << ","
            << gpu_metric << ","
            << cpu_metric / gpu_metric << "\n";

  (void)sink;
  return true;
}

}  // namespace

int main() {
  const int repetitions = 20;

  std::cout
      << "resolution,cpu_gray_ms,gpu_gray_ms,gray_speedup,"
      << "cpu_metrics_ms,gpu_metrics_ms,metrics_speedup\n";

  for (int size : {128, 512, 1024}) {
    if (!RunBenchmark(size, size, repetitions)) {
      return 1;
    }
  }

  return 0;
}
