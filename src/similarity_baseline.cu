#include "similarity.cuh"

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace {

__global__ void CompareHashesKernel(
    const unsigned long long* hashes,
    int count,
    int threshold,
    int* distances) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = count * count;

  if (index >= total) {
    return;
  }

  const int first = index / count;
  const int second = index % count;

  if (first >= second) {
    distances[index] = -1;
    return;
  }

  const unsigned long long difference =
      hashes[first] ^ hashes[second];

  const int distance = __popcll(difference);

  distances[index] = distance <= threshold ? distance : -1;
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

bool FindSimilarPairsGpu(const std::vector<uint64_t>& hashes,
                         int threshold,
                         std::vector<SimilarPair>* pairs) {
  if (pairs == nullptr || threshold < 0 || threshold > 64) {
    return false;
  }

  pairs->clear();

  const int count = static_cast<int>(hashes.size());
  if (count < 2) {
    return true;
  }

  std::vector<unsigned long long> host_hashes(
      hashes.begin(), hashes.end());

  std::vector<int> distances(
      static_cast<size_t>(count) * count);

  unsigned long long* device_hashes = nullptr;
  int* device_distances = nullptr;
  bool success = false;

  do {
    if (!CheckCuda(cudaMalloc(
            reinterpret_cast<void**>(&device_hashes),
            host_hashes.size() * sizeof(unsigned long long)),
            "Allocate hashes")) {
      break;
    }

    if (!CheckCuda(cudaMalloc(
            reinterpret_cast<void**>(&device_distances),
            distances.size() * sizeof(int)),
            "Allocate distances")) {
      break;
    }

    if (!CheckCuda(cudaMemcpy(
            device_hashes, host_hashes.data(),
            host_hashes.size() * sizeof(unsigned long long),
            cudaMemcpyHostToDevice), "Copy hashes")) {
      break;
    }

    const int threads = 256;
    const int blocks = (count * count + threads - 1) / threads;

    CompareHashesKernel<<<blocks, threads>>>(
        device_hashes, count, threshold, device_distances);

    if (!CheckCuda(cudaGetLastError(), "Comparison kernel") ||
        !CheckCuda(cudaDeviceSynchronize(), "Comparison execution")) {
      break;
    }

    if (!CheckCuda(cudaMemcpy(
            distances.data(), device_distances,
            distances.size() * sizeof(int),
            cudaMemcpyDeviceToHost), "Copy distances")) {
      break;
    }

    for (int first = 0; first < count; ++first) {
      for (int second = first + 1; second < count; ++second) {
        const int distance = distances[first * count + second];
        if (distance >= 0) {
          pairs->push_back({first, second, distance});
        }
      }
    }

    success = true;
  } while (false);

  if (device_hashes != nullptr) cudaFree(device_hashes);
  if (device_distances != nullptr) cudaFree(device_distances);

  return success;
}
