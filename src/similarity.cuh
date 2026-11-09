#ifndef SIMILARITY_CUH_
#define SIMILARITY_CUH_

#include <cstdint>
#include <vector>

struct SimilarPair {
  int first_index;
  int second_index;
  int distance;
};

bool FindSimilarPairsGpu(const std::vector<uint64_t>& hashes,
                         int threshold,
                         std::vector<SimilarPair>* pairs);

#endif  // SIMILARITY_CUH_
