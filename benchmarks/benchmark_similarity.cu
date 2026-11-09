#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "similarity.cuh"

namespace {

using Clock = std::chrono::steady_clock;

std::vector<SimilarPair> FindSimilarPairsCpu(
    const std::vector<uint64_t>& hashes,
    int threshold) {
  std::vector<SimilarPair> pairs;

  for (size_t first = 0; first < hashes.size(); ++first) {
    for (size_t second = first + 1;
         second < hashes.size(); ++second) {
      const uint64_t difference =
          hashes[first] ^ hashes[second];

      const int distance =
          __builtin_popcountll(difference);

      if (distance <= threshold) {
        pairs.push_back({
            static_cast<int>(first),
            static_cast<int>(second),
            distance
        });
      }
    }
  }

  return pairs;
}

bool SamePairs(const std::vector<SimilarPair>& first,
               const std::vector<SimilarPair>& second) {
  if (first.size() != second.size()) {
    return false;
  }

  for (size_t i = 0; i < first.size(); ++i) {
    if (first[i].first_index != second[i].first_index ||
        first[i].second_index != second[i].second_index ||
        first[i].distance != second[i].distance) {
      return false;
    }
  }

  return true;
}

double Milliseconds(Clock::time_point start,
                    Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(
      end - start).count();
}

bool RunBenchmark(int count, int threshold, int repetitions) {
  std::mt19937_64 generator(42);

  std::vector<uint64_t> hashes(count);

  for (uint64_t& hash : hashes) {
    hash = generator();
  }

  // Add known matching hashes.
  for (int i = 1; i < count; i += 10) {
    hashes[i] = hashes[i - 1];
  }

  // Warm up CPU and CUDA before timing.
  const auto expected =
      FindSimilarPairsCpu(hashes, threshold);

  std::vector<SimilarPair> gpu_pairs;

  if (!FindSimilarPairsGpu(hashes, threshold, &gpu_pairs)) {
    std::cerr << "GPU warmup failed.\n";
    return false;
  }

  if (!SamePairs(expected, gpu_pairs)) {
    std::cerr << "CPU/GPU correctness mismatch.\n";
    return false;
  }

  double cpu_total_ms = 0.0;
  double gpu_total_ms = 0.0;

  for (int iteration = 0;
       iteration < repetitions; ++iteration) {
    const auto cpu_start = Clock::now();

    const auto cpu_result =
        FindSimilarPairsCpu(hashes, threshold);

    const auto cpu_end = Clock::now();

    const auto gpu_start = Clock::now();

    if (!FindSimilarPairsGpu(
            hashes, threshold, &gpu_pairs)) {
      std::cerr << "GPU benchmark failed.\n";
      return false;
    }

    const auto gpu_end = Clock::now();

    if (!SamePairs(cpu_result, gpu_pairs)) {
      std::cerr << "Correctness mismatch during benchmark.\n";
      return false;
    }

    cpu_total_ms += Milliseconds(cpu_start, cpu_end);
    gpu_total_ms += Milliseconds(gpu_start, gpu_end);
  }

  const double cpu_ms = cpu_total_ms / repetitions;
  const double gpu_ms = gpu_total_ms / repetitions;

  const double speedup =
      gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;

  std::cout << std::fixed << std::setprecision(4);

  std::cout << count << ","
            << expected.size() << ","
            << cpu_ms << ","
            << gpu_ms << ","
            << speedup << "\n";

  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  int repetitions = 20;

  if (argc > 2) {
    std::cerr << "Usage: " << argv[0]
              << " [repetitions]\n";
    return 1;
  }

  if (argc == 2) {
    char* end = nullptr;
    const long value = std::strtol(argv[1], &end, 10);

    if (*argv[1] == '\0' || *end != '\0' ||
        value < 1 || value > 10000) {
      std::cerr << "Repetitions must be between 1 and 10000.\n";
      return 1;
    }

    repetitions = static_cast<int>(value);
  }

  const int threshold = 5;

  std::cout
      << "images,similar_pairs,cpu_ms,gpu_ms,cpu_over_gpu\n";

  for (int count : {120, 500, 1000}) {
    if (!RunBenchmark(count, threshold, repetitions)) {
      return 1;
    }
  }

  return 0;
}
