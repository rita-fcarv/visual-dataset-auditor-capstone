#include "image_hash.cuh"

#include <iostream>

#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>

namespace {

__global__ void DHashKernel(const unsigned char* pixels,
                            unsigned long long* hash) {
  const int index = threadIdx.x;

  if (index >= 64) {
    return;
  }

  const int row = index / 8;
  const int col = index % 8;

  const unsigned char left = pixels[row * 9 + col];
  const unsigned char right = pixels[row * 9 + col + 1];

  if (left > right) {
    atomicOr(hash, 1ULL << index);
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

bool ComputeDHashGpu(const cv::Mat& grayscale, uint64_t* hash) {
  if (hash == nullptr || grayscale.empty() ||
      grayscale.type() != CV_8UC1) {
    return false;
  }

  cv::Mat resized;
  cv::resize(grayscale, resized, cv::Size(9, 8),
             0, 0, cv::INTER_AREA);

  unsigned char* device_pixels = nullptr;
  unsigned long long* device_hash = nullptr;
  unsigned long long result = 0;
  bool success = false;

  do {
    if (!CheckCuda(cudaMalloc(
            reinterpret_cast<void**>(&device_pixels),
            9 * 8), "Allocate dHash pixels")) {
      break;
    }

    if (!CheckCuda(cudaMalloc(
            reinterpret_cast<void**>(&device_hash),
            sizeof(unsigned long long)), "Allocate dHash")) {
      break;
    }

    if (!CheckCuda(cudaMemcpy(
            device_pixels, resized.data, 9 * 8,
            cudaMemcpyHostToDevice), "Copy dHash pixels")) {
      break;
    }

    if (!CheckCuda(cudaMemset(
            device_hash, 0, sizeof(unsigned long long)),
            "Initialize dHash")) {
      break;
    }

    DHashKernel<<<1, 64>>>(device_pixels, device_hash);

    if (!CheckCuda(cudaGetLastError(), "dHash kernel") ||
        !CheckCuda(cudaDeviceSynchronize(), "dHash execution")) {
      break;
    }

    if (!CheckCuda(cudaMemcpy(
            &result, device_hash, sizeof(result),
            cudaMemcpyDeviceToHost), "Copy dHash")) {
      break;
    }

    *hash = static_cast<uint64_t>(result);
    success = true;
  } while (false);

  if (device_pixels != nullptr) cudaFree(device_pixels);
  if (device_hash != nullptr) cudaFree(device_hash);

  return success;
}
