#include "similarity.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kInitialPairCapacity = 1024;

// Convert a flattened index into a unique image pair.
//
// Each row contains comparisons with later images only:
// row 0: (0,1), (0,2), ..., (0,n-1)
// row 1: (1,2), (1,3), ..., (1,n-1)
// etc.
//
// Binary search avoids launching threads for duplicate or
// self-comparisons.
__device__ unsigned long long RowStart(
    unsigned long long row,
    unsigned long long count) {
  return row * (2ULL * count - row - 1ULL) / 2ULL;
}

__global__ void CompareHashesKernel(
    const unsigned long long* hashes,
    int count,
    unsigned long long total_pairs,
    int threshold,
    SimilarPair* matches,
    unsigned int capacity,
    unsigned int* match_count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) *
          blockDim.x +
      threadIdx.x;

  if (index >= total_pairs) {
    return;
  }

  // Locate the row containing this unique pair.
  unsigned long long low = 0;
  unsigned long long high =
      static_cast<unsigned long long>(count - 1);

  while (low < high) {
    const unsigned long long mid =
        low + (high - low + 1ULL) / 2ULL;

    if (RowStart(mid, count) <= index) {
      low = mid;
    } else {
      high = mid - 1ULL;
    }
  }

  const int first = static_cast<int>(low);

  const unsigned long long offset =
      index - RowStart(low, count);

  const int second =
      first + 1 + static_cast<int>(offset);

  const unsigned long long difference =
      hashes[first] ^ hashes[second];

  const int distance = __popcll(difference);

  if (distance <= threshold) {
    const unsigned int position =
        atomicAdd(match_count, 1U);

    if (position < capacity) {
      matches[position] = {first, second, distance};
    }
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

bool PairLess(const SimilarPair& a, const SimilarPair& b) {
  if (a.first_index != b.first_index) {
    return a.first_index < b.first_index;
  }

  return a.second_index < b.second_index;
}

}  // namespace

bool FindSimilarPairsGpu(
    const std::vector<uint64_t>& hashes,
    int threshold,
    std::vector<SimilarPair>* pairs) {
  if (pairs == nullptr ||
      threshold < 0 ||
      threshold > 64) {
    return false;
  }

  pairs->clear();

  if (hashes.size() < 2) {
    return true;
  }

  if (hashes.size() >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    std::cerr << "Too many images.\n";
    return false;
  }

  const int count = static_cast<int>(hashes.size());

  const unsigned long long total_pairs =
      static_cast<unsigned long long>(count) *
      static_cast<unsigned long long>(count - 1) / 2ULL;

  const unsigned long long blocks_needed =
      (total_pairs + kThreadsPerBlock - 1ULL) /
      kThreadsPerBlock;

  if (blocks_needed >
      static_cast<unsigned long long>(
          std::numeric_limits<unsigned int>::max())) {
    std::cerr << "Too many comparisons for one launch.\n";
    return false;
  }

  std::vector<unsigned long long> host_hashes(
      hashes.begin(), hashes.end());

  unsigned long long* device_hashes = nullptr;
  SimilarPair* device_matches = nullptr;
  unsigned int* device_match_count = nullptr;

  bool success = false;

  do {
    if (!CheckCuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_hashes),
                host_hashes.size() *
                    sizeof(unsigned long long)),
            "Allocate hashes")) {
      break;
    }

    if (!CheckCuda(
            cudaMemcpy(
                device_hashes,
                host_hashes.data(),
                host_hashes.size() *
                    sizeof(unsigned long long),
                cudaMemcpyHostToDevice),
            "Copy hashes")) {
      break;
    }

    if (!CheckCuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_match_count),
                sizeof(unsigned int)),
            "Allocate match counter")) {
      break;
    }

    unsigned int capacity =
        static_cast<unsigned int>(
            std::min<unsigned long long>(
                total_pairs,
                kInitialPairCapacity));

    // Retry with a larger buffer if the number of
    // matching pairs exceeds the current capacity.
    while (true) {
      if (device_matches != nullptr) {
        cudaFree(device_matches);
        device_matches = nullptr;
      }

      if (!CheckCuda(
              cudaMalloc(
                  reinterpret_cast<void**>(&device_matches),
                  static_cast<size_t>(capacity) *
                      sizeof(SimilarPair)),
              "Allocate matching pairs")) {
        break;
      }

      if (!CheckCuda(
              cudaMemset(
                  device_match_count,
                  0,
                  sizeof(unsigned int)),
              "Reset match counter")) {
        break;
      }

      CompareHashesKernel<<<
          static_cast<unsigned int>(blocks_needed),
          kThreadsPerBlock>>>(
          device_hashes,
          count,
          total_pairs,
          threshold,
          device_matches,
          capacity,
          device_match_count);

      if (!CheckCuda(
              cudaGetLastError(),
              "Comparison kernel launch") ||
          !CheckCuda(
              cudaDeviceSynchronize(),
              "Comparison kernel execution")) {
        break;
      }

      unsigned int actual_count = 0;

      if (!CheckCuda(
              cudaMemcpy(
                  &actual_count,
                  device_match_count,
                  sizeof(unsigned int),
                  cudaMemcpyDeviceToHost),
              "Copy match count")) {
        break;
      }

      if (actual_count > capacity) {
        if (total_pairs >
            std::numeric_limits<unsigned int>::max()) {
          std::cerr
              << "Match count exceeds supported capacity.\n";
          break;
        }

        capacity = actual_count;
        continue;
      }

      pairs->resize(actual_count);

      if (actual_count > 0) {
        if (!CheckCuda(
                cudaMemcpy(
                    pairs->data(),
                    device_matches,
                    static_cast<size_t>(actual_count) *
                        sizeof(SimilarPair),
                    cudaMemcpyDeviceToHost),
                "Copy matching pairs")) {
          pairs->clear();
          break;
        }
      }

      std::sort(pairs->begin(), pairs->end(), PairLess);

      success = true;
      break;
    }
  } while (false);

  if (device_hashes != nullptr) {
    cudaFree(device_hashes);
  }

  if (device_matches != nullptr) {
    cudaFree(device_matches);
  }

  if (device_match_count != nullptr) {
    cudaFree(device_match_count);
  }

  if (!success) {
    pairs->clear();
  }

  return success;
}
