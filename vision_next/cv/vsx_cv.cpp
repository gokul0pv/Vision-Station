// vsx_cv: the OpenCV calls Vision Station makes, behind a plain C interface,
// so Vision Next (Rust) gets OpenCV's own results -- same library, same
// Intel IPP paths -- instead of re-implementing them.
//
// Images are passed as (data, width, height, channels, row stride in
// bytes), 8-bit unless the name says f32. Output buffers are allocated by
// the caller unless noted; functions return 0 on success, negative on error
// (vsx_last_error explains).
//
// Build (MSVC, x64):  see build.bat

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <cstring>
#include <string>
#include <vector>

#define VSX_API extern "C" __declspec(dllexport)

static thread_local std::string g_err;

static int fail(const cv::Exception& e) {
    g_err = e.what();
    return -1;
}

static cv::Mat wrap(const void* data, int w, int h, int cn, int stride, int depth = CV_8U) {
    return cv::Mat(h, w, CV_MAKETYPE(depth, cn), const_cast<void*>(data), (size_t)stride);
}

VSX_API const char* vsx_last_error() { return g_err.c_str(); }

VSX_API const char* vsx_version() { return CV_VERSION; }

// cv::ipp::useIPP() as this process has it (1 / 0)
VSX_API int vsx_use_ipp() { return cv::ipp::useIPP() ? 1 : 0; }

// cv::resize(src, (dw, dh) or fx/fy, interp). dw/dh 0 -> computed from fx, fy as OpenCV does;
// the size used is written back.
VSX_API int vsx_resize(const uint8_t* src, int w, int h, int cn, int stride,
                       uint8_t* dst, int dst_cap, int* dw, int* dh, double fx, double fy, int interp) {
    try {
        cv::Mat s = wrap(src, w, h, cn, stride), d;
        if (*dw > 0 && *dh > 0)
            cv::resize(s, d, cv::Size(*dw, *dh), 0, 0, interp);
        else
            cv::resize(s, d, cv::Size(), fx, fy, interp);
        *dw = d.cols; *dh = d.rows;
        size_t need = d.total() * d.elemSize();
        if ((size_t)dst_cap < need) { g_err = "dst too small"; return -2; }
        std::memcpy(dst, d.ptr(), need);
        return 0;
    } catch (const cv::Exception& e) { return fail(e); }
}

// cv::warpAffine / cv::warpPerspective (m: 6 or 9 doubles), 8-bit or float (depth 0 / 5)
VSX_API int vsx_warp(const void* src, int w, int h, int cn, int stride, int depth,
                     void* dst, int dw, int dh, const double* m, int perspective, int flags,
                     int border, double border_value) {
    try {
        cv::Mat s = wrap(src, w, h, cn, stride, depth);
        cv::Mat d(dh, dw, CV_MAKETYPE(depth, cn), dst);
        if (perspective)
            cv::warpPerspective(s, d, cv::Mat(3, 3, CV_64F, const_cast<double*>(m)), d.size(), flags, border, cv::Scalar::all(border_value));
        else
            cv::warpAffine(s, d, cv::Mat(2, 3, CV_64F, const_cast<double*>(m)), d.size(), flags, border, cv::Scalar::all(border_value));
        if (d.data != dst) { g_err = "reallocated"; return -2; }
        return 0;
    } catch (const cv::Exception& e) { return fail(e); }
}

// cv::GaussianBlur, 8-bit or float, default border
VSX_API int vsx_gaussian(const void* src, int w, int h, int cn, int stride, int depth,
                         void* dst, int kw, int kh, double sx, double sy) {
    try {
        cv::Mat s = wrap(src, w, h, cn, stride, depth);
        cv::Mat d(h, w, CV_MAKETYPE(depth, cn), dst);
        cv::GaussianBlur(s, d, cv::Size(kw, kh), sx, sy);
        return 0;
    } catch (const cv::Exception& e) { return fail(e); }
}

// cv::cvtColor(BGR or RGB -> grey): code as OpenCV's
VSX_API int vsx_cvt(const uint8_t* src, int w, int h, int cn, int stride, uint8_t* dst, int dcn, int code) {
    try {
        cv::Mat s = wrap(src, w, h, cn, stride);
        cv::Mat d(h, w, CV_MAKETYPE(CV_8U, dcn), dst);
        cv::cvtColor(s, d, code);
        return 0;
    } catch (const cv::Exception& e) { return fail(e); }
}

// cv::matchTemplate(img, tpl, method) on 8-bit grey; result (w-tw+1) x (h-th+1) floats
VSX_API int vsx_match_template(const uint8_t* img, int w, int h, int stride,
                               const uint8_t* tpl, int tw, int th, int tstride, int method, float* out) {
    try {
        cv::Mat r(h - th + 1, w - tw + 1, CV_32F, out);
        cv::matchTemplate(wrap(img, w, h, 1, stride), wrap(tpl, tw, th, 1, tstride), r, method);
        return 0;
    } catch (const cv::Exception& e) { return fail(e); }
}

// SIFT_create(nfeatures).detectAndCompute(gray, None). Results kept in a
// handle: read them with vsx_sift_get, then vsx_sift_free.
struct SiftResult {
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
};

VSX_API void* vsx_sift(const uint8_t* gray, int w, int h, int stride, int nfeatures, int* count) {
    try {
        auto* r = new SiftResult();
        cv::SIFT::create(nfeatures)->detectAndCompute(wrap(gray, w, h, 1, stride), cv::noArray(), r->kps, r->desc);
        *count = (int)r->kps.size();
        return r;
    } catch (const cv::Exception& e) { fail(e); *count = -1; return nullptr; }
}

// per keypoint 6 floats: x, y, size, angle, response, octave (as float bits of int);
// descriptors count x 128 floats
VSX_API void vsx_sift_get(void* handle, float* kp6, float* desc) {
    auto* r = (SiftResult*)handle;
    for (size_t i = 0; i < r->kps.size(); i++) {
        const auto& k = r->kps[i];
        kp6[i * 6 + 0] = k.pt.x; kp6[i * 6 + 1] = k.pt.y; kp6[i * 6 + 2] = k.size;
        kp6[i * 6 + 3] = k.angle; kp6[i * 6 + 4] = k.response;
        std::memcpy(&kp6[i * 6 + 5], &k.octave, 4);
    }
    if (!r->desc.empty())
        std::memcpy(desc, r->desc.ptr(), r->kps.size() * 128 * sizeof(float));
}

VSX_API void vsx_sift_free(void* handle) { delete (SiftResult*)handle; }

// BFMatcher(NORM_L2).knnMatch(q, t, k=2): per query 4 values: train index of
// the best, its distance, train index of the second (-1 if none), its distance
VSX_API int vsx_knn2(const float* q, int nq, const float* t, int nt, float* out4) {
    try {
        cv::Mat qm(nq, 128, CV_32F, const_cast<float*>(q)), tm(nt, 128, CV_32F, const_cast<float*>(t));
        std::vector<std::vector<cv::DMatch>> m;
        cv::BFMatcher(cv::NORM_L2).knnMatch(qm, tm, m, 2);
        for (int i = 0; i < nq; i++) {
            float* o = out4 + i * 4;
            o[0] = -1; o[1] = 0; o[2] = -1; o[3] = 0;
            if (i < (int)m.size()) {
                if (m[i].size() > 0) { o[0] = (float)m[i][0].trainIdx; o[1] = m[i][0].distance; }
                if (m[i].size() > 1) { o[2] = (float)m[i][1].trainIdx; o[3] = m[i][1].distance; }
            }
        }
        return 0;
    } catch (const cv::Exception& e) { return fail(e); }
}

// cv::estimateAffinePartial2D(src, dst, RANSAC, thr): m6 out (doubles), mask out;
// returns 1 found, 0 not found
VSX_API int vsx_affine_partial(const float* src2, const float* dst2, int n, double thr, double* m6, uint8_t* mask) {
    try {
        cv::Mat s(n, 1, CV_32FC2, const_cast<float*>(src2)), d(n, 1, CV_32FC2, const_cast<float*>(dst2));
        cv::Mat inl;
        cv::Mat M = cv::estimateAffinePartial2D(s, d, inl, cv::RANSAC, thr);
        if (M.empty()) return 0;
        for (int i = 0; i < 6; i++) m6[i] = M.at<double>(i / 3, i % 3);
        for (int i = 0; i < n; i++) mask[i] = inl.empty() ? 0 : inl.at<uchar>(i);
        return 1;
    } catch (const cv::Exception& e) { return fail(e); }
}

// cv::findTransformECC(templ, input, warp (2x3 float, in/out), MOTION_EUCLIDEAN,
// (COUNT|EPS, iters, eps), no mask, gauss): returns 0 and rho, or -1 (OpenCV raised)
VSX_API int vsx_ecc(const uint8_t* templ, int tw, int th, int tstride,
                    const uint8_t* input, int iw, int ih, int istride,
                    float* warp6, int iters, double eps, int gauss, double* rho) {
    try {
        cv::Mat w(2, 3, CV_32F, warp6);
        *rho = cv::findTransformECC(wrap(templ, tw, th, 1, tstride), wrap(input, iw, ih, 1, istride), w,
                                    cv::MOTION_EUCLIDEAN,
                                    cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, iters, eps),
                                    cv::noArray(), gauss);
        if (w.data != (uchar*)warp6) { g_err = "reallocated"; return -2; }
        return 0;
    } catch (const cv::Exception& e) { return fail(e); }
}
