// numpy's arithmetic, reproduced where Vision Station's results depend on it.
// Each function states which numpy call it stands for. Verified behaviour
// (numpy 2.1.3, scipy-openblas 0.3.28 on this PC):
//  - reductions run over chunks of 8192 elements, each chunk summed pairwise
//    (8 accumulators, blocks of 128), the chunks added from zero;
//  - float32 dot products are OpenBLAS sdot (Haswell kernel);
//  - a Python float meeting a float32 array is cast to float32 (NEP 50).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vc::npy {

// arr.sum() for float32 data -> float32
float sum_f32(const float* a, size_t n);
// arr.sum() for float64 data (or uint8 data summed with dtype float64) -> float64
double sum_f64(const double* a, size_t n);
double sum_u8_as_f64(const uint8_t* a, size_t n);

// arr.mean() of a uint8 array -> float64
double mean_u8(const uint8_t* a, size_t n);
// arr.std() of a uint8 array -> float64 (ddof 0)
double std_u8(const uint8_t* a, size_t n);
// arr.mean() of a float32 array -> float32
float mean_f32(const float* a, size_t n);
// arr.std() of a float32 array -> float32
float std_f32(const float* a, size_t n);

// np.dot(x, y) / x.dot(y) for float32 vectors (OpenBLAS sdot)
float dot_f32(const float* x, const float* y, size_t n);
// np.linalg.norm of a float32 array
float norm_f32(const float* a, size_t n);

// np.percentile(a, q) (linear) of float32 data -> float32 result as numpy gives it
float percentile_f32(std::vector<float> a, double q);

// np.percentile(a, q) of uint8 data (linear, float64 index and result)
double percentile_u8(std::vector<uint8_t> a, double q);
// np.median of uint8 data -> float64
double median_u8(std::vector<uint8_t> a);
// np.gradient of float32 data (edge_order 1) -> float32
std::vector<float> gradient_f32(const std::vector<float>& f);

}  // namespace vc::npy
