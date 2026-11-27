// one ISA's x86-simd-sort argsort; included by npsort_avx2.cpp / npsort_avx512.cpp
#include "../../third_party/x86-simd-sort/src/x86simdsort-static-incl.h"

#include <cstdint>

namespace vc::npy::NPSORT_NS {
void argsort_f32(float* a, size_t* idx, size_t n) { x86simdsortStatic::argsort(a, idx, n, true); }
void argsort_f64(double* a, size_t* idx, size_t n) { x86simdsortStatic::argsort(a, idx, n, true); }
void argsort_i64(int64_t* a, size_t* idx, size_t n) { x86simdsortStatic::argsort(a, idx, n, true); }
}  // namespace vc::npy::NPSORT_NS
