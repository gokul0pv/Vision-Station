#include "npsort.hpp"

#include <intrin.h>

#include <cmath>
#include <numeric>

namespace vc::npy {

namespace avx2 {
void argsort_f32(float*, size_t*, size_t);
void argsort_f64(double*, size_t*, size_t);
void argsort_i64(int64_t*, size_t*, size_t);
}  // namespace avx2
namespace avx512 {
void argsort_f32(float*, size_t*, size_t);
void argsort_f64(double*, size_t*, size_t);
void argsort_i64(int64_t*, size_t*, size_t);
}  // namespace avx512

namespace {

enum class Isa { none, avx2, avx512 };

// numpy's CPU test (npy_cpu_features): AVX2 needs AVX2 + FMA3 + F16C and the OS saving YMM;
// AVX512_SKX needs F, CD, BW, DQ, VL and the OS saving ZMM
Isa detect() {
    int r[4];
    __cpuid(r, 0);
    int maxleaf = r[0];
    if (maxleaf < 7) return Isa::none;
    __cpuid(r, 1);
    bool osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1, fma = (r[2] >> 12) & 1, f16c = (r[2] >> 29) & 1;
    if (!osxsave || !avx) return Isa::none;
    unsigned long long xcr0 = _xgetbv(0);
    bool ymm = (xcr0 & 0x6) == 0x6, zmm = (xcr0 & 0xe6) == 0xe6;
    __cpuidex(r, 7, 0);
    bool avx2 = (r[1] >> 5) & 1;
    bool f = (r[1] >> 16) & 1, dq = (r[1] >> 17) & 1, cd = (r[1] >> 28) & 1, bw = (r[1] >> 30) & 1, vl = (r[1] >> 31) & 1;
    if (zmm && f && cd && bw && dq && vl && avx2 && fma && f16c) return Isa::avx512;
    if (ymm && avx2 && fma && f16c) return Isa::avx2;
    return Isa::none;
}

Isa isa() {
    static Isa v = detect();
    return v;
}

// numpy/_core/src/npysort/quicksort.cpp aquicksort_ / heapsort.cpp aheapsort_
template <class T>
bool lt(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) return a < b || (b != b && a == a);
    else return a < b;
}

template <class T>
void aheapsort(const T* v, int64_t* tosort, int64_t n) {
    int64_t* a = tosort - 1;
    int64_t i, j, l, tmp;
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && lt(v[a[j]], v[a[j + 1]])) j += 1;
            if (lt(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            } else break;
        }
        a[i] = tmp;
    }
    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && lt(v[a[j]], v[a[j + 1]])) j++;
            if (lt(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            } else break;
        }
        a[i] = tmp;
    }
}

template <class T>
void aquicksort(const T* v, int64_t* tosort, int64_t num) {
    const int SMALL = 15, STACK = 128;
    int64_t* pl = tosort;
    int64_t* pr = tosort + num - 1;
    int64_t* stack[STACK];
    int64_t** sptr = stack;
    int depth[STACK];
    int* psdepth = depth;
    int msb = 0;
    for (int64_t k = num; k > 1; k >>= 1) msb++;
    int cdepth = msb * 2;
    for (;;) {
        if (cdepth < 0) {
            aheapsort(v, pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL) {
            int64_t* pm = pl + ((pr - pl) >> 1);
            if (lt(v[*pm], v[*pl])) std::swap(*pm, *pl);
            if (lt(v[*pr], v[*pm])) std::swap(*pr, *pm);
            if (lt(v[*pm], v[*pl])) std::swap(*pm, *pl);
            T vp = v[*pm];
            int64_t* pi = pl;
            int64_t* pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do ++pi; while (lt(v[*pi], vp));
                do --pj; while (lt(vp, v[*pj]));
                if (pi >= pj) break;
                std::swap(*pi, *pj);
            }
            int64_t* pk = pr - 1;
            std::swap(*pi, *pk);
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            } else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }
        for (int64_t* pi = pl + 1; pi <= pr; ++pi) {
            int64_t vi = *pi;
            T vp = v[vi];
            int64_t* pj = pi;
            int64_t* pk = pi - 1;
            while (pj > pl && lt(vp, v[*pk])) *pj-- = *pk--;
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) break;
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }
}

template <class T, class F2, class F5>
std::vector<int64_t> run(const T* a, size_t n, F2 f2, F5 f5) {
    std::vector<int64_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if (n == 0) return idx;
    Isa i = isa();
    if (i != Isa::none) {
        std::vector<T> copy(a, a + n);   // x86-simd-sort keeps keys intact, but never risk the caller's data
        static_assert(sizeof(size_t) == sizeof(int64_t), "64-bit indices");
        (i == Isa::avx512 ? f5 : f2)(copy.data(), reinterpret_cast<size_t*>(idx.data()), n);
        return idx;
    }
    aquicksort(a, idx.data(), (int64_t)n);
    return idx;
}

}  // namespace

std::vector<int64_t> argsort(const float* a, size_t n) { return run(a, n, avx2::argsort_f32, avx512::argsort_f32); }
std::vector<int64_t> argsort(const double* a, size_t n) { return run(a, n, avx2::argsort_f64, avx512::argsort_f64); }
std::vector<int64_t> argsort(const int64_t* a, size_t n) { return run(a, n, avx2::argsort_i64, avx512::argsort_i64); }

const char* argsort_path() {
    Isa i = isa();
    return i == Isa::avx512 ? "avx512" : i == Isa::avx2 ? "avx2" : "introsort";
}

}  // namespace vc::npy
