#include "image_metrics.cuh"

#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace {

__global__ void ComputePixelMetrics(
    const unsigned char* image,
    int width,
    int height,
    double* intensity,
    double* laplacian) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_pixels = width * height;

  if (index >= num_pixels) {
    return;
  }

  int x = index % width;
  int y = index / width;

  double center = image[index];
  intensity[index] = center;

  // Use zero Laplacian at image boundaries.
  if (x == 0 || y == 0 ||
      x == width - 1 || y == height - 1) {
    laplacian[index] = 0.0;
    return;
  }

  double left = image[index - 1];
  double right = image[index + 1];
  double top = image[index - width];
  double bottom = image[index + width];

  laplacian[index] =
      left + right + top + bottom - 4.0 * center;
}

__global__ void ReduceStatistics(
    const double* intensity,
    const double* laplacian,
    int num_pixels,
    double* partial_sums) {
  __shared__ double shared_intensity[256];
  __shared__ double shared_intensity_squared[256];
  __shared__ double shared_laplacian[256];
  __shared__ double shared_laplacian_squared[256];

  int thread = threadIdx.x;
  int index = blockIdx.x * blockDim.x + thread;

  double value = 0.0;
  double lap = 0.0;

  if (index < num_pixels) {
    value = intensity[index];
    lap = laplacian[index];
  }

  shared_intensity[thread] = value;
  shared_intensity_squared[thread] = value * value;
  shared_laplacian[thread] = lap;
  shared_laplacian_squared[thread] = lap * lap;

  __syncthreads();

  for (int stride = blockDim.x / 2;
       stride > 0; stride /= 2) {
    if (thread < stride) {
      shared_intensity[thread] +=
          shared_intensity[thread + stride];
      shared_intensity_squared[thread] +=
          shared_intensity_squared[thread + stride];
      shared_laplacian[thread] +=
          shared_laplacian[thread + stride];
      shared_laplacian_squared[thread] +=
          shared_laplacian_squared[thread + stride];
    }
    __syncthreads();
  }

  if (thread == 0) {
    int offset = blockIdx.x * 4;
    partial_sums[offset] = shared_intensity[0];
    partial_sums[offset + 1] =
        shared_intensity_squared[0];
    partial_sums[offset + 2] = shared_laplacian[0];
    partial_sums[offset + 3] =
        shared_laplacian_squared[0];
  }
}

bool CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    std::cerr << operation << ": "
              << cudaGetErrorString(error) << "\n";
    return false;
  }
  return true;
}

}  // namespace

bool ComputeImageMetricsGpu(const cv::Mat& grayscale,
                            ImageMetrics* metrics) {
  if (metrics == nullptr || grayscale.empty() ||
      grayscale.type() != CV_8UC1) {
    return false;
  }

  cv::Mat input = grayscale.isContinuous()
                      ? grayscale
                      : grayscale.clone();

  const int width = input.cols;
  const int height = input.rows;
  const int num_pixels = width * height;
  const int threads = 256;
  const int blocks = (num_pixels + threads - 1) / threads;

  unsigned char* device_image = nullptr;
  double* device_intensity = nullptr;
  double* device_laplacian = nullptr;
  double* device_sums = nullptr;

  bool success = false;
  std::vector<double> sums(blocks * 4);

  do {
    if (!CheckCuda(cudaMalloc(
            reinterpret_cast<void**>(&device_image),
            num_pixels), "Allocate image")) {
      break;
    }

    if (!CheckCuda(cudaMalloc(
            reinterpret_cast<void**>(&device_intensity),
            num_pixels * sizeof(double)), "Allocate intensity")) {
      break;
    }

    if (!CheckCuda(cudaMalloc(
            reinterpret_cast<void**>(&device_laplacian),
            num_pixels * sizeof(double)), "Allocate Laplacian")) {
      break;
    }

    if (!CheckCuda(cudaMalloc(
            reinterpret_cast<void**>(&device_sums),
            sums.size() * sizeof(double)), "Allocate sums")) {
      break;
    }

    if (!CheckCuda(cudaMemcpy(
            device_image, input.data, num_pixels,
            cudaMemcpyHostToDevice), "Copy image")) {
      break;
    }

    ComputePixelMetrics<<<blocks, threads>>>(
        device_image, width, height,
        device_intensity, device_laplacian);

    if (!CheckCuda(cudaGetLastError(), "Pixel kernel")) {
      break;
    }

    ReduceStatistics<<<blocks, threads>>>(
        device_intensity, device_laplacian,
        num_pixels, device_sums);

    if (!CheckCuda(cudaGetLastError(), "Reduction kernel") ||
        !CheckCuda(cudaDeviceSynchronize(), "Kernel execution")) {
      break;
    }

    if (!CheckCuda(cudaMemcpy(
            sums.data(), device_sums,
            sums.size() * sizeof(double),
            cudaMemcpyDeviceToHost), "Copy sums")) {
      break;
    }

    double sum = 0.0;
    double sum_squared = 0.0;
    double lap_sum = 0.0;
    double lap_sum_squared = 0.0;

    for (int block = 0; block < blocks; ++block) {
      int offset = block * 4;
      sum += sums[offset];
      sum_squared += sums[offset + 1];
      lap_sum += sums[offset + 2];
      lap_sum_squared += sums[offset + 3];
    }

    const double count = static_cast<double>(num_pixels);
    const double mean = sum / count;
    const double lap_mean = lap_sum / count;

    metrics->mean_brightness = mean;
    metrics->contrast = std::sqrt(
        std::max(0.0, sum_squared / count - mean * mean));
    metrics->blur_score = std::max(
        0.0, lap_sum_squared / count - lap_mean * lap_mean);

    success = true;
  } while (false);

  if (device_image != nullptr) cudaFree(device_image);
  if (device_intensity != nullptr) cudaFree(device_intensity);
  if (device_laplacian != nullptr) cudaFree(device_laplacian);
  if (device_sums != nullptr) cudaFree(device_sums);

  return success;
}
