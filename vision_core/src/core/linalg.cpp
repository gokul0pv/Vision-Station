#include "linalg.hpp"
#include "npy.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <mutex>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace vc::la {

namespace {

using i64 = int64_t;
// Fortran LAPACK, ILP64; the hidden string lengths gfortran expects come last
using dgesdd_t = void (*)(const char*, const i64*, const i64*, double*, const i64*, double*, double*, const i64*,
                          double*, const i64*, double*, const i64*, i64*, i64*, size_t);
using dgelsd_t = void (*)(const i64*, const i64*, const i64*, double*, const i64*, double*, const i64*, double*,
                          const double*, i64*, double*, const i64*, i64*, i64*);
// CBLAS, ILP64
enum { RowMajor = 101, ColMajor = 102, NoTrans = 111, Trans = 112 };
using dgemv_t = void (*)(int, int, i64, i64, double, const double*, i64, const double*, i64, double, double*, i64);
using sgemv_t = void (*)(int, int, i64, i64, float, const float*, i64, const float*, i64, float, float*, i64);
using ddot_t = double (*)(i64, const double*, i64, const double*, i64);
using threads_t = void (*)(int);

struct Lib {
    dgesdd_t dgesdd = nullptr;
    dgelsd_t dgelsd = nullptr;
    dgemv_t dgemv = nullptr;
    sgemv_t sgemv = nullptr;
    ddot_t ddot = nullptr;
    std::string where;
    bool ok = false;
};

std::wstring exe_dir() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return std::filesystem::path(std::wstring(buf, n)).parent_path().wstring();
}

const Lib& lib() {
    static Lib L;
    static std::once_flag once;
    std::call_once(once, [] {
        namespace fs = std::filesystem;
        std::vector<fs::path> tries;
        if (const wchar_t* e = _wgetenv(L"VC_OPENBLAS")) tries.emplace_back(e);
        std::error_code ec;
        for (auto& d : {fs::path(exe_dir()), fs::path(exe_dir()) / "openblas"})
            for (auto& f : fs::directory_iterator(d, ec))
                if (f.path().filename().wstring().rfind(L"libscipy_openblas64_", 0) == 0 && f.path().extension() == ".dll")
                    tries.push_back(f.path());
        for (auto& p : tries) {
            HMODULE h = LoadLibraryExW(p.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (!h) continue;
            L.dgesdd = (dgesdd_t)GetProcAddress(h, "scipy_dgesdd_64_");
            L.dgelsd = (dgelsd_t)GetProcAddress(h, "scipy_dgelsd_64_");
            L.dgemv = (dgemv_t)GetProcAddress(h, "scipy_cblas_dgemv64_");
            L.sgemv = (sgemv_t)GetProcAddress(h, "scipy_cblas_sgemv64_");
            L.ddot = (ddot_t)GetProcAddress(h, "scipy_cblas_ddot64_");
            if (L.dgesdd && L.dgelsd && L.dgemv && L.sgemv && L.ddot) {
                if (auto t = (threads_t)GetProcAddress(h, "scipy_openblas_set_num_threads64_")) t(1);
                L.ok = true;
                L.where = p.u8string();
                return;
            }
            FreeLibrary(h);
        }
        L.where = "numpy's OpenBLAS (libscipy_openblas64_*.dll) not found; using built-in arithmetic";
    });
    return L;
}

// row-major m x n -> Fortran (column-major)
std::vector<double> fortran(const double* a, int m, int n) {
    std::vector<double> f((size_t)m * n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) f[(size_t)j * m + i] = a[(size_t)i * n + j];
    return f;
}

}  // namespace

bool available() { return lib().ok; }
std::string source() { return lib().where; }

std::vector<double> svd_vh(const double* a, int m, int n) {
    const Lib& L = lib();
    if (L.ok && m > 0 && n > 0) {
        // numpy/linalg/umath_linalg.cpp init_gesdd + call_gesdd, jobz 'A'
        i64 M = m, N = n, lda = std::max(m, 1), ldu = std::max(m, 1), ldvt = n, info = 0;
        std::vector<double> A = fortran(a, m, n), S(std::min(m, n)), U((size_t)m * m), VT((size_t)n * n);
        std::vector<i64> iwork(8 * (size_t)std::min(m, n));
        double q = 0;
        i64 lwork = -1;
        L.dgesdd("A", &M, &N, A.data(), &lda, S.data(), U.data(), &ldu, VT.data(), &ldvt, &q, &lwork, iwork.data(), &info, 1);
        if (info == 0) {
            lwork = (i64)q;
            if (lwork == 0) lwork = 1;
            std::vector<double> work((size_t)lwork);
            L.dgesdd("A", &M, &N, A.data(), &lda, S.data(), U.data(), &ldu, VT.data(), &ldvt, work.data(), &lwork, iwork.data(), &info, 1);
            if (info == 0) {
                std::vector<double> vh((size_t)n * n);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++) vh[(size_t)i * n + j] = VT[(size_t)j * n + i];
                return vh;
            }
        }
    }
    cv::Mat A(m, n, CV_64F, const_cast<double*>(a)), w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::FULL_UV);
    return std::vector<double>((double*)vt.data, (double*)vt.data + (size_t)n * n);
}

std::vector<double> lstsq(const double* Ain, const double* b, int m, int n) {
    const Lib& L = lib();
    if (L.ok && m > 0 && n > 0) {
        // init_gelsd + call_gelsd; rcond = finfo(float64).eps * max(m, n)
        i64 M = m, N = n, nrhs = 1, lda = std::max(m, 1), ldb = std::max(std::max(m, n), 1), rank = 0, info = 0;
        double rcond = 2.220446049250313e-16 * std::max(m, n);
        std::vector<double> A = fortran(Ain, m, n), B((size_t)ldb, 0.0), S(std::min(m, n));
        for (int i = 0; i < m; i++) B[i] = b[i];
        double wq = 0;
        i64 iwq = 0, lwork = -1;
        L.dgelsd(&M, &N, &nrhs, A.data(), &lda, B.data(), &ldb, S.data(), &rcond, &rank, &wq, &lwork, &iwq, &info);
        if (info == 0) {
            lwork = (i64)wq;
            std::vector<double> work((size_t)std::max<i64>(lwork, 1));
            std::vector<i64> iwork((size_t)std::max<i64>(iwq, 1));
            L.dgelsd(&M, &N, &nrhs, A.data(), &lda, B.data(), &ldb, S.data(), &rcond, &rank, work.data(), &lwork, iwork.data(), &info);
            if (info == 0) return std::vector<double>(B.begin(), B.begin() + n);
        }
    }
    cv::Mat A(m, n, CV_64F, const_cast<double*>(Ain)), B(m, 1, CV_64F, const_cast<double*>(b)), x;
    cv::solve(A, B, x, cv::DECOMP_SVD);
    return std::vector<double>((double*)x.data, (double*)x.data + n);
}

template <class T, class F>
static std::vector<T> matvec_impl(F gemv, const T* a, const T* x, int m, int n) {
    std::vector<T> y((size_t)m, (T)0);
    if (m == 0) return y;
    if (n == 0) return y;
    if (gemv) {
        // numpy/_core/src/umath/matmul.cpp.src @TYPE@_gemv: column-major, transposed
        gemv(ColMajor, Trans, (i64)n, (i64)m, (T)1, a, (i64)n, x, 1, (T)0, y.data(), 1);
        return y;
    }
    for (int i = 0; i < m; i++) {
        T s = 0;
        for (int j = 0; j < n; j++) s += a[(size_t)i * n + j] * x[j];
        y[i] = s;
    }
    return y;
}

std::vector<double> matvec(const double* a, const double* x, int m, int n) {
    if (m == 1) return {dot(a, x, n)};             // numpy takes the dot path for a single row
    return matvec_impl<double>(lib().ok ? lib().dgemv : nullptr, a, x, m, n);
}

std::vector<float> matvec(const float* a, const float* x, int m, int n) {
    if (m == 1) return {npy::dot_f32(a, x, (size_t)n)};   // numpy takes the sdot path for a single row
    return matvec_impl<float>(lib().ok ? lib().sgemv : nullptr, a, x, m, n);
}

double dot(const double* x, const double* y, int n) {
    if (lib().ok) return lib().ddot((i64)n, x, 1, y, 1);
    double s = 0;
    for (int i = 0; i < n; i++) s += x[i] * y[i];
    return s;
}

double norm(const double* x, int n) { return std::sqrt(dot(x, x, n)); }

}  // namespace vc::la
