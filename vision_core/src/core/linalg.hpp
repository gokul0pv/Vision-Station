// numpy.linalg and numpy's matmul, computed the way numpy computes them: by
// calling the same OpenBLAS / LAPACK routines in the same OpenBLAS build numpy
// ships (scipy-openblas64, ILP64). The DLL is looked for next to the program
// (libscipy_openblas64_*.dll) or at %VC_OPENBLAS%. Without it every function
// still works, with plain C++ / OpenCV arithmetic that agrees to rounding.
#pragma once

#include <string>
#include <vector>

namespace vc::la {

bool available();              // numpy's OpenBLAS was found and loaded
std::string source();          // the DLL path, or why it is not loaded

// np.linalg.svd(a)[2] (vh, full_matrices=True) of an m x n row-major matrix,
// computed in float64 as numpy does; n*n row-major
std::vector<double> svd_vh(const double* a, int m, int n);
// np.linalg.lstsq(A, b, rcond=None)[0] for an m x n row-major A and one right-hand side
std::vector<double> lstsq(const double* A, const double* b, int m, int n);

// the same with several right-hand sides: B is m x k row-major, the result n x k row-major
std::vector<double> lstsq(const double* A, const double* B, int m, int n, int k);
// np.linalg.det of an n x n row-major matrix (LU, then sign * exp(sum log|u_ii|), as numpy)
double det(const double* a, int n);
// numpy matmul of two matrices as cblas_dgemm(RowMajor, ta, tb, M, N, K, A, lda, B, ldb)
std::vector<double> gemm(bool transA, bool transB, int M, int N, int K, const double* A, int lda, const double* B, int ldb);

// numpy matmul of a C-contiguous (m, n) matrix with an (n,) vector
std::vector<double> matvec(const double* a, const double* x, int m, int n);
std::vector<float> matvec(const float* a, const float* x, int m, int n);
// numpy matmul / dot of two 1-D float64 vectors
double dot(const double* x, const double* y, int n);
// np.linalg.norm of a 1-D float64 vector
double norm(const double* x, int n);

}  // namespace vc::la
