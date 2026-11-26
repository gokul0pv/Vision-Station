#include "npy.hpp"

#include <algorithm>
#include <cmath>

namespace vc::npy {

namespace {

template <typename T>
T pairwise(const T* a, size_t n) {
    if (n < 8) {
        T r = 0;
        for (size_t i = 0; i < n; i++) r += a[i];
        return r;
    }
    if (n <= 128) {
        T r[8];
        for (int k = 0; k < 8; k++) r[k] = a[k];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; k++) r[k] += a[i + k];
        T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

constexpr size_t kBuf = 8192;

template <typename T>
T chunked(const T* a, size_t n) {
    T r = 0;
    for (size_t i = 0; i < n; i += kBuf) r += pairwise(a + i, std::min(kBuf, n - i));
    return r;
}

inline float fma32(float a, float b, float c) {
    // an exact fused multiply-add: the product of two floats is exact in double
    return (float)((double)a * (double)b + (double)c);
}

}  // namespace

float sum_f32(const float* a, size_t n) { return chunked(a, n); }
double sum_f64(const double* a, size_t n) { return chunked(a, n); }

double sum_u8_as_f64(const uint8_t* a, size_t n) {
    std::vector<double> d(a, a + n);
    return chunked(d.data(), n);
}

double mean_u8(const uint8_t* a, size_t n) { return n ? sum_u8_as_f64(a, n) / (double)n : NAN; }

double std_u8(const uint8_t* a, size_t n) {
    if (!n) return NAN;
    double m = mean_u8(a, n);
    std::vector<double> d(n);
    for (size_t i = 0; i < n; i++) {
        double t = (double)a[i] - m;
        d[i] = t * t;
    }
    return std::sqrt(chunked(d.data(), n) / (double)n);
}

float mean_f32(const float* a, size_t n) { return n ? sum_f32(a, n) / (float)n : NAN; }

float std_f32(const float* a, size_t n) {
    if (!n) return NAN;
    float m = mean_f32(a, n);
    std::vector<float> d(n);
    for (size_t i = 0; i < n; i++) {
        float t = a[i] - m;
        d[i] = t * t;
    }
    return std::sqrt(sum_f32(d.data(), n) / (float)n);
}

float dot_f32(const float* x, const float* y, size_t n) {
    // OpenBLAS 0.3.28 sdot, Haswell kernel: 32 at a time in 4 x 8 fused
    // accumulators, folded in a fixed order; the rest added in double.
    size_t n1 = n & ~(size_t)31;
    float acc[4][8] = {};
    for (size_t i = 0; i < n1; i += 32)
        for (int r = 0; r < 4; r++)
            for (int l = 0; l < 8; l++) {
                size_t k = i + 8 * r + l;
                acc[r][l] = fma32(x[k], y[k], acc[r][l]);
            }
    float mydot = 0.0f;
    if (n1) {
        float h[4][4];
        for (int r = 0; r < 4; r++)
            for (int l = 0; l < 4; l++) h[r][l] = acc[r][l] + acc[r][l + 4];
        float s[4];
        for (int l = 0; l < 4; l++) s[l] = (h[0][l] + h[1][l]) + (h[2][l] + h[3][l]);
        mydot = (s[0] + s[1]) + (s[2] + s[3]);
    }
    double dot = 0.0;
    for (size_t k = n1; k < n; k++) dot += (double)(y[k] * x[k]);
    dot += (double)mydot;
    return (float)dot;
}

float norm_f32(const float* a, size_t n) { return std::sqrt(dot_f32(a, a, n)); }

float percentile_f32(std::vector<float> a, double q) {
    if (a.empty()) return NAN;
    std::sort(a.begin(), a.end());
    size_t n = a.size();
    // numpy 2.1 linear method, in the array's float32
    float qq = (float)(q / 100.0);
    float vi = ((float)n - 1.0f) * qq;
    float fl = std::floor(vi);
    float t = vi - fl;
    size_t i = std::min((size_t)fl, n - 1);
    float lo = a[i], hi = a[std::min(i + 1, n - 1)];
    float d = hi - lo;
    return t >= 0.5f ? hi - d * (1.0f - t) : lo + d * t;
}

double percentile_u8(std::vector<uint8_t> a, double q) {
    if (a.empty()) return NAN;
    std::sort(a.begin(), a.end());
    size_t n = a.size();
    double vi = ((double)n - 1.0) * (q / 100.0);
    double fl = std::floor(vi);
    double t = vi - fl;
    size_t i = std::min((size_t)fl, n - 1);
    double lo = a[i], hi = a[std::min(i + 1, n - 1)];
    double d = hi - lo;
    return t >= 0.5 ? hi - d * (1.0 - t) : lo + d * t;
}

double median_u8(std::vector<uint8_t> a) {
    if (a.empty()) return NAN;
    std::sort(a.begin(), a.end());
    size_t n = a.size();
    if (n % 2) return (double)a[n / 2];
    return ((double)a[n / 2 - 1] + (double)a[n / 2]) / 2.0;
}

std::vector<float> gradient_f32(const std::vector<float>& f) {
    size_t n = f.size();
    std::vector<float> o(n, 0.0f);
    if (n < 2) return o;
    for (size_t i = 1; i + 1 < n; i++) o[i] = (f[i + 1] - f[i - 1]) / 2.0f;
    o[0] = (f[1] - f[0]) / 1.0f;
    o[n - 1] = (f[n - 1] - f[n - 2]) / 1.0f;
    return o;
}

}  // namespace vc::npy
