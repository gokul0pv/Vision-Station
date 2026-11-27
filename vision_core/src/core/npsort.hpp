// numpy's argsort (default kind) for float32 / float64 / int64 keys, ties and
// all: on a CPU with AVX-512 (SKX) or AVX2 numpy 2.1 sorts with Intel's
// x86-simd-sort -- whose order among equal keys is not the input order -- and
// otherwise with its own introsort. Vision Core does the same, with the same
// x86-simd-sort commit, chosen by the same CPU test.
#pragma once

#include <cstdint>
#include <vector>

namespace vc::npy {

std::vector<int64_t> argsort(const float* a, size_t n);
std::vector<int64_t> argsort(const double* a, size_t n);
std::vector<int64_t> argsort(const int64_t* a, size_t n);
const char* argsort_path();   // "avx512", "avx2" or "introsort"

}  // namespace vc::npy
