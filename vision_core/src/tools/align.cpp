// Locate part (vision_station/tools/align.py): features (SIFT + ratio test +
// estimateAffinePartial2D RANSAC, ECC polish kept when the region fits
// better), rotated template matching, or features then template ("auto").
#include "../core/npy.hpp"
#include "../core/tool.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <cmath>
#include <mutex>

namespace vc {
namespace {

constexpr double kPi = 3.14159265358979323846;
using Win = std::array<int, 4>;   // x0, y0, x1, y1

cv::Mat slice(const cv::Mat& m, int x0, int y0, int x1, int y1) {
    x0 = std::clamp(x0, 0, m.cols); y0 = std::clamp(y0, 0, m.rows);
    x1 = std::clamp(x1, x0, m.cols); y1 = std::clamp(y1, y0, m.rows);
    return m(cv::Rect(x0, y0, x1 - x0, y1 - y0));
}

cv::Mat resize_fx(const cv::Mat& m, double s) {
    cv::Mat o;
    cv::resize(m, o, cv::Size(), s, s, cv::INTER_AREA);
    return o;
}

// numpy matmul of 3 x 3 doubles
cv::Matx33d mm(const cv::Matx33d& a, const cv::Matx33d& b) {
    cv::Matx33d o;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) o(i, j) = a(i, 0) * b(0, j) + a(i, 1) * b(1, j) + a(i, 2) * b(2, j);
    return o;
}

cv::Matx33d m33(const cv::Mat& M32) {   // 3x3 CV_32F -> double
    cv::Matx33d o;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) o(i, j) = (double)M32.at<float>(i, j);
    return o;
}

cv::Mat to_f32_3x3(const cv::Matx33d& d) {
    cv::Mat o(3, 3, CV_32F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) o.at<float>(i, j) = (float)d(i, j);
    return o;
}

bool all_finite(const cv::Mat& m) {
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            if (!std::isfinite(m.at<float>(i, j))) return false;
    return true;
}

struct RefFeatures {
    std::vector<std::pair<double, double>> pts;
    cv::Mat desc;
};
std::mutex g_cache_mx;
std::map<std::string, RefFeatures> g_ref_cache;

}  // namespace

class Align : public Tool {
public:
    using Tool::Tool;
    bool is_fixture() const override { return false; }

    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        if (ctx.reference.empty())
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "no reference image loaded for this project", now_ms() - t0);
        std::string method = ps("method", "template");
        double min_score = pf("min_score", 0.6);
        last_margin_.reset();
        last_inliers_ = 0;
        cv::Mat M;
        double score = 0.0;
        std::string err;
        try {
            cv::Mat rg = to_gray(ctx.reference), ig = to_gray(image);
            if (method == "auto") {
                std::tie(M, score) = features(rg, ig);
                method = "features";
                if (M.empty() || score < min_score) {
                    int inl = last_inliers_;
                    auto [M2, score2] = templ(rg, ig);
                    if (!M2.empty() && score2 > (!M.empty() ? score : -1.0)) {
                        M = M2;
                        score = score2;
                        method = "template";
                    } else {
                        last_margin_.reset();
                        last_inliers_ = inl;
                    }
                }
            } else if (method == "features") {
                std::tie(M, score) = features(rg, ig);
            } else {
                std::tie(M, score) = templ(rg, ig);
            }
        } catch (const cv::Exception& e) {
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, e.what(), now_ms() - t0);
        } catch (const std::exception& e) {
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, e.what(), now_ms() - t0);
        }

        bool ok = !M.empty() && score >= min_score;
        int min_inl = pi("min_inliers", 12);
        std::string why;
        char buf[512];
        if (M.empty() && method == "features" && last_inliers_ < min_inl) {
            std::snprintf(buf, sizeof buf, "part not found - only %d matching points agree on a position (needs %d). "
                          "Move or enlarge the region onto a part of the product with more detail.", last_inliers_, min_inl);
            why = buf;
        } else if (!ok) {
            std::snprintf(buf, sizeof buf, "no match above %.2f (best %.2f)", min_score, score);
            why = buf;
        }
        double min_margin = pf("min_uniqueness", 0.06);
        if (ok && last_margin_ && *last_margin_ < min_margin) {
            ok = false;
            std::snprintf(buf, sizeof buf, "match is not unique - the fixture feature appears more than once (margin %.3f, needs %.3f). "
                          "Enlarge the region or move it onto something that appears only once on the part.", *last_margin_, min_margin);
            why = buf;
        }
        if (ok) ctx.fixture = M.clone();

        double dx = 0, dy = 0, ang = 0;
        if (!M.empty()) {
            dx = M.at<float>(0, 2);
            dy = M.at<float>(1, 2);
            // np.degrees(np.arctan2(float32, float32)): all in float32
            float a = std::atan2(M.at<float>(1, 0), M.at<float>(0, 0));
            ang = (double)(a * (float)(180.0 / kPi));
        }
        std::vector<Graphic> g;
        if (!M.empty()) {
            float w = (float)ctx.reference.cols, h = (float)ctx.reference.rows;
            std::vector<cv::Point2f> box = {{0, 0}, {w, 0}, {w, h}, {0, h}}, out;
            cv::transform(box, out, M.rowRange(0, 2));
            Graphic gr("poly", {}, ok ? "pass" : "fail", "fixture");
            for (auto& q : out) gr.points.emplace_back((double)q.x, (double)q.y);
            g.push_back(gr);
        }
        Json vals = Json::object();
        vals["shift_x_px"] = py_round(dx, 2);
        vals["shift_y_px"] = py_round(dy, 2);
        vals["angle_deg"] = py_round(ang, 3);
        vals["uniqueness"] = last_margin_ ? Json(py_round(*last_margin_, 3)) : Json();
        vals["inliers"] = method == "features" ? Json(last_inliers_) : Json();
        vals["method"] = method;
        if (ok && !M.empty()) {
            Json mat = Json::array();
            for (int i = 0; i < 2; i++) {
                Json row = Json::array();
                for (int j = 0; j < 3; j++) row.push_back(py_round((double)M.at<float>(i, j), 6));
                mat.push_back(row);
            }
            vals["matrix"] = mat;
        } else {
            vals["matrix"] = Json();
        }
        return ToolResult(id, type_name, ok ? "pass" : "fail", score, vals, g, why, now_ms() - t0);
    }

private:
    std::optional<double> last_margin_, margin_;
    int last_inliers_ = 0;

    // ---------------------------------------------------------- template
    std::pair<cv::Mat, double> templ(const cv::Mat& ref_g, const cv::Mat& img_g) {
        int H = ref_g.rows, W = ref_g.cols;
        std::array<double, 4> r = roi ? *roi : std::array<double, 4>{0, 0, (double)W, (double)H};
        int x = std::max(0, (int)r[0]), y = std::max(0, (int)r[1]);
        int w = std::max(8, (int)std::min(r[2], (double)(W - x))), h = std::max(8, (int)std::min(r[3], (double)(H - y)));
        cv::Mat tpl = slice(ref_g, x, y, x + w, y + h);

        double lo = 0, hi = 0;
        if (p.contains("angle_range") && p["angle_range"].is_array() && p["angle_range"].size() >= 2) {
            lo = p["angle_range"][0].get<double>();
            hi = p["angle_range"][1].get<double>();
        }
        double fine_step = pf("angle_step", 1.0);
        if (fine_step == 0.0) fine_step = 1.0;
        auto arange = [](double a, double b, double st) {
            std::vector<double> v;
            long n = (long)std::ceil((b - a) / st);
            for (long i = 0; i < n; i++) v.push_back(a + i * st);
            return v;
        };
        Best best;
        if (std::abs(hi - lo) < 1e-9) {
            best = search(tpl, img_g, {lo == 0 ? 0.0 : lo});
        } else if (pb("pyramid", true)) {
            double sc = 0.5;
            cv::Mat tpl_s = resize_fx(tpl, sc), img_s = resize_fx(img_g, sc);
            double coarse_step = std::max(fine_step, (hi - lo) / 16.0);
            auto coarse = arange(lo, hi + 1e-9, coarse_step);
            double a0 = 0.0;
            if (std::min(tpl_s.rows, tpl_s.cols) >= 8) {
                Best c = search(tpl_s, img_s, coarse);
                a0 = c.loc ? c.angle : 0.0;
            }
            std::vector<double> fine;
            for (double a : arange(a0 - coarse_step, a0 + coarse_step + 1e-9, fine_step))
                if (lo - 1e-9 <= a && a <= hi + 1e-9) fine.push_back(a);
            best = search(tpl, img_g, fine.empty() ? std::vector<double>{a0} : fine);
        } else {
            double coarse_step = std::max(fine_step, (hi - lo) / 16.0);
            best = search(tpl, img_g, arange(lo, hi + 1e-9, coarse_step));
            if (best.loc) {
                double a0 = best.angle;
                std::vector<double> fine;
                for (double a : arange(a0 - coarse_step, a0 + coarse_step + 1e-9, fine_step))
                    if (lo - 1e-9 <= a && a <= hi + 1e-9) fine.push_back(a);
                Best cand = search(tpl, img_g, fine);
                if (cand.score > best.score) best = cand;
            }
        }
        if (!best.loc) return {cv::Mat(), 0.0};
        last_margin_ = margin_;
        double cx_ref = x + w / 2.0, cy_ref = y + h / 2.0;
        double cx_cur = best.loc->first + w / 2.0, cy_cur = best.loc->second + h / 2.0;
        double rad = best.angle * (kPi / 180.0);
        double ca = std::cos(rad), sa = std::sin(rad);
        cv::Mat M = (cv::Mat_<float>(3, 3) << (float)ca, (float)sa, (float)(cx_cur - (ca * cx_ref + sa * cy_ref)),
                     (float)-sa, (float)ca, (float)(cy_cur - (-sa * cx_ref + ca * cy_ref)), 0.f, 0.f, 1.f);
        if (pb("refine", true)) {
            cv::Mat refined = refine_ecc(ref_g, img_g, M);
            if (!refined.empty() && fit(ref_g, img_g, refined) >= fit(ref_g, img_g, M)) M = refined;
        }
        return {M, best.score};
    }

    struct Best {
        std::optional<std::pair<double, double>> loc;
        double score = -1.0, angle = 0.0;
    };

    Best search(const cv::Mat& tpl, const cv::Mat& img_g, const std::vector<double>& angles) {
        int h = tpl.rows, w = tpl.cols;
        Best b;
        margin_.reset();
        for (double a : angles) {
            cv::Mat t;
            if (std::abs(a) < 1e-9) {
                t = tpl;
            } else {
                cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(w / 2.0f, h / 2.0f), a, 1.0);
                cv::warpAffine(tpl, t, R, cv::Size(w, h), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            }
            if (img_g.rows < h || img_g.cols < w) continue;
            cv::Mat res;
            cv::matchTemplate(img_g, t, res, cv::TM_CCOEFF_NORMED);
            double mn, mx;
            cv::Point mnl, loc;
            cv::minMaxLoc(res, &mn, &mx, &mnl, &loc);
            if (mx > b.score) {
                b.score = (double)(float)mx;
                b.angle = a;
                b.loc = subpixel(res, loc);
                cv::Mat masked = res.clone();
                int x0 = std::max(0, loc.x - w / 2), y0 = std::max(0, loc.y - h / 2);
                cv::Mat roi = slice(masked, x0, y0, loc.x + w / 2, loc.y + h / 2);
                roi.setTo(-1.0f);
                double rmx = -1.0;
                if (!masked.empty()) cv::minMaxLoc(masked, nullptr, &rmx);
                margin_ = (double)(float)mx - (double)(float)rmx;
            }
        }
        return b;
    }

    // parabolic peak, in float32 as numpy computes it on the result array
    static std::pair<double, double> subpixel(const cv::Mat& res, cv::Point l) {
        int px = l.x, py = l.y, h = res.rows, w = res.cols;
        double fx = px, fy = py;
        auto at = [&](int yy, int xx) { return res.at<float>(yy, xx); };
        if (0 < px && px < w - 1) {
            float L = at(py, px - 1), C = at(py, px), R = at(py, px + 1);
            float d = L - 2.0f * C + R;
            if (std::abs(d) > 1e-12) fx = px + (double)std::clamp(0.5f * (L - R) / d, -1.0f, 1.0f);
        }
        if (0 < py && py < h - 1) {
            float U = at(py - 1, px), C = at(py, px), D = at(py + 1, px);
            float d = U - 2.0f * C + D;
            if (std::abs(d) > 1e-12) fy = py + (double)std::clamp(0.5f * (U - D) / d, -1.0f, 1.0f);
        }
        return {fx, fy};
    }

    // Align._fit: whole-picture correlation of the reference warped by M
    static double fit(const cv::Mat& ref_g, const cv::Mat& img_g, const cv::Mat& M, int size = 240) {
        try {
            double sc = std::min(1.0, size / (double)std::max(img_g.rows, img_g.cols));
            cv::Mat A = M.rowRange(0, 2).clone();
            A.at<float>(0, 2) *= (float)sc;
            A.at<float>(1, 2) *= (float)sc;
            cv::Mat rs = resize_fx(ref_g, sc), isg = resize_fx(img_g, sc), warped;
            cv::warpAffine(rs, warped, A, isg.size(), cv::INTER_LINEAR);
            std::vector<float> a, b;
            for (int yy = 0; yy < warped.rows; yy++)
                for (int xx = 0; xx < warped.cols; xx++)
                    if (warped.at<uint8_t>(yy, xx) > 0) {
                        a.push_back(warped.at<uint8_t>(yy, xx));
                        b.push_back(isg.at<uint8_t>(yy, xx));
                    }
            if (a.size() < 50) return -1.0;
            float ma = npy::mean_f32(a.data(), a.size()), mb = npy::mean_f32(b.data(), b.size());
            for (auto& v : a) v -= ma;
            for (auto& v : b) v -= mb;
            float d = npy::norm_f32(a.data(), a.size()) * npy::norm_f32(b.data(), b.size());
            if ((double)d <= 1e-9) return -1.0;
            return (double)(npy::dot_f32(a.data(), b.data(), a.size()) / d);
        } catch (const cv::Exception&) {
            return -1.0;
        }
    }

    cv::Mat refine_ecc(const cv::Mat& ref_g, const cv::Mat& img_g, const cv::Mat& M) {
        try {
            double target = pf("refine_size", 420);
            double scale = std::min(1.0, target / std::max(ref_g.rows, ref_g.cols));
            cv::Mat warp = M.rowRange(0, 2).clone();
            cv::Mat r = ref_g, i = img_g;
            if (scale < 1.0) {
                r = resize_fx(ref_g, scale);
                i = resize_fx(img_g, scale);
                warp.at<float>(0, 2) *= (float)scale;
                warp.at<float>(1, 2) *= (float)scale;
            }
            cv::TermCriteria crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, pi("refine_iters", 30), 1e-5);
            cv::findTransformECC(r, i, warp, cv::MOTION_EUCLIDEAN, crit, cv::noArray(), 5);
            if (scale < 1.0) {
                warp.at<float>(0, 2) /= (float)scale;
                warp.at<float>(1, 2) /= (float)scale;
            }
            cv::Mat out = cv::Mat::eye(3, 3, CV_32F);
            warp.copyTo(out.rowRange(0, 2));
            return all_finite(out) ? out : cv::Mat();
        } catch (const cv::Exception&) {
            return cv::Mat();
        }
    }

    // ---------------------------------------------------------- features
    std::pair<Win, Win> feature_windows(int H, int W) const {
        if (!roi) return {Win{0, 0, W, H}, Win{0, 0, W, H}};
        int x = (int)(*roi)[0], y = (int)(*roi)[1], w = (int)(*roi)[2], h = (int)(*roi)[3];
        int pad = (has("search_pad_px") && pf("search_pad_px", 0) != 0) ? (int)pf("search_pad_px", 0) : (int)(0.35 * std::max(w, h));
        Win rw{std::max(0, x), std::max(0, y), std::min(W, x + w), std::min(H, y + h)};
        Win iw{std::max(0, x - pad), std::max(0, y - pad), std::min(W, x + w + pad), std::min(H, y + h + pad)};
        return {rw, iw};
    }

    double fit_roi(const cv::Mat& ref_g, const cv::Mat& img_g, const cv::Mat& M, int size = 640) const {
        int H = ref_g.rows, W = ref_g.cols;
        std::array<double, 4> r = roi ? *roi : std::array<double, 4>{0, 0, (double)W, (double)H};
        int x = std::max(0, (int)r[0]), y = std::max(0, (int)r[1]);
        int w = std::max(8, std::min((int)r[2], W - x)), h = std::max(8, std::min((int)r[3], H - y));
        double sc = std::min(1.0, size / (double)std::max(w, h));
        try {
            cv::Mat M64;
            M.rowRange(0, 2).convertTo(M64, CV_64F);
            cv::Mat Minv;
            cv::invertAffineTransform(M64, Minv);
            cv::Matx33d S(sc, 0, -x * sc, 0, sc, -y * sc, 0, 0, 1);
            cv::Matx33d B(Minv.at<double>(0, 0), Minv.at<double>(0, 1), Minv.at<double>(0, 2),
                          Minv.at<double>(1, 0), Minv.at<double>(1, 1), Minv.at<double>(1, 2), 0, 0, 1);
            cv::Matx33d A = mm(S, B);
            cv::Mat A2 = (cv::Mat_<double>(2, 3) << A(0, 0), A(0, 1), A(0, 2), A(1, 0), A(1, 1), A(1, 2));
            cv::Size dsz((int)(w * sc), (int)(h * sc));
            cv::Mat back, ref;
            cv::warpAffine(img_g, back, A2, dsz, cv::INTER_AREA);
            cv::resize(slice(ref_g, x, y, x + w, y + h), ref, dsz, 0, 0, cv::INTER_AREA);
            size_t n = (size_t)ref.total();
            cv::Mat rc = ref.isContinuous() ? ref : ref.clone(), bc = back.isContinuous() ? back : back.clone();
            float mr = (float)npy::mean_u8(rc.ptr<uint8_t>(), n), mb = (float)npy::mean_u8(bc.ptr<uint8_t>(), n);
            std::vector<float> a(n), b(n), ab(n);
            for (size_t k = 0; k < n; k++) {
                a[k] = (float)rc.ptr<uint8_t>()[k] - mr;
                b[k] = (float)bc.ptr<uint8_t>()[k] - mb;
                ab[k] = a[k] * b[k];
            }
            float d = npy::norm_f32(a.data(), n) * npy::norm_f32(b.data(), n);
            if ((double)d <= 1e-9) return 0.0;
            return std::max(0.0, (double)(npy::sum_f32(ab.data(), n) / d));
        } catch (const cv::Exception&) {
            return 0.0;
        }
    }

    cv::Mat refine_ecc_window(const cv::Mat& ref_g, const cv::Mat& img_g, const cv::Mat& M, Win rw, Win iw) {
        try {
            cv::Mat r = slice(ref_g, rw[0], rw[1], rw[2], rw[3]), i = slice(img_g, iw[0], iw[1], iw[2], iw[3]);
            double target = pf("refine_size", 420);
            double scale = std::min(1.0, target / std::max(std::max(r.rows, r.cols), std::max(i.rows, i.cols)));
            cv::Matx33d T_r(1, 0, rw[0], 0, 1, rw[1], 0, 0, 1), T_i(1, 0, -iw[0], 0, 1, -iw[1], 0, 0, 1);
            cv::Matx33d S(scale, 0, 0, 0, scale, 0, 0, 0, 1), Sinv(1 / scale, 0, 0, 0, 1 / scale, 0, 0, 0, 1);
            cv::Matx33d Mc = mm(mm(mm(mm(S, T_i), m33(M)), T_r), Sinv);
            cv::Mat warp(2, 3, CV_32F);
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 3; b++) warp.at<float>(a, b) = (float)Mc(a, b);
            if (scale < 1.0) {
                r = resize_fx(r, scale);
                i = resize_fx(i, scale);
            }
            cv::TermCriteria crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, pi("refine_iters", 30), 1e-5);
            cv::findTransformECC(r, i, warp, cv::MOTION_EUCLIDEAN, crit, cv::noArray(), 5);
            cv::Matx33d Mc2((double)warp.at<float>(0, 0), (double)warp.at<float>(0, 1), (double)warp.at<float>(0, 2),
                            (double)warp.at<float>(1, 0), (double)warp.at<float>(1, 1), (double)warp.at<float>(1, 2), 0, 0, 1);
            cv::Matx33d Ti_inv(1, 0, iw[0], 0, 1, iw[1], 0, 0, 1), Tr_inv(1, 0, -rw[0], 0, 1, -rw[1], 0, 0, 1);
            cv::Mat out = to_f32_3x3(mm(mm(mm(mm(Ti_inv, Sinv), Mc2), S), Tr_inv));
            return all_finite(out) ? out : cv::Mat();
        } catch (const cv::Exception&) {
            return cv::Mat();
        }
    }

    std::pair<cv::Mat, double> features(const cv::Mat& ref_g, const cv::Mat& img_g) {
        int n = pi("n_features", 2000);
        auto det = cv::SIFT::create(std::max(n, 1000));
        auto [rw, iw] = feature_windows(ref_g.rows, ref_g.cols);
        int long_side = std::max(iw[2] - iw[0], iw[3] - iw[1]);
        double sc = std::min(1.0, pf("feature_px", 1400) / std::max(1, long_side));
        auto detect = [&](const cv::Mat& gray, Win win) {
            cv::Mat crop = slice(gray, win[0], win[1], win[2], win[3]).clone();
            if (sc < 1.0) crop = resize_fx(crop, sc);
            std::vector<cv::KeyPoint> kps;
            RefFeatures f;
            det->detectAndCompute(crop, cv::noArray(), kps, f.desc);
            for (auto& k : kps) f.pts.emplace_back((double)k.pt.x / sc + win[0], (double)k.pt.y / sc + win[1]);
            return f;
        };
        // the reference's features, computed once (Python keys them the same way)
        long long ssum = 0;
        for (int yy = 0; yy < ref_g.rows; yy += 97)
            for (int xx = 0; xx < ref_g.cols; xx += 89) ssum += ref_g.at<uint8_t>(yy, xx);
        char key[256];
        std::snprintf(key, sizeof key, "%dx%d|%d,%d,%d,%d|%d|%.17g|%lld", ref_g.cols, ref_g.rows, rw[0], rw[1], rw[2], rw[3], n, sc, ssum);
        RefFeatures rf;
        {
            std::lock_guard<std::mutex> lk(g_cache_mx);
            auto it = g_ref_cache.find(key);
            if (it != g_ref_cache.end()) rf = it->second;
        }
        if (rf.pts.empty() && rf.desc.empty()) {
            rf = detect(ref_g, rw);
            std::lock_guard<std::mutex> lk(g_cache_mx);
            if (g_ref_cache.size() > 32) g_ref_cache.clear();
            g_ref_cache[key] = rf;
        }
        RefFeatures cur = detect(img_g, iw);
        last_inliers_ = 0;
        if (rf.desc.empty() || cur.desc.empty() || rf.pts.size() < 8 || cur.pts.size() < 8) return {cv::Mat(), 0.0};
        cv::BFMatcher bf(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch>> pairs;
        bf.knnMatch(rf.desc, cur.desc, pairs, 2);
        std::vector<cv::Point2f> src, dst;
        for (auto& pr : pairs)
            if (pr.size() == 2 && (double)pr[0].distance < 0.75 * (double)pr[1].distance) {
                auto& a = rf.pts[pr[0].queryIdx];
                auto& b = cur.pts[pr[0].trainIdx];
                src.emplace_back((float)a.first, (float)a.second);
                dst.emplace_back((float)b.first, (float)b.second);
            }
        if (src.size() < 8) return {cv::Mat(), 0.0};
        cv::Mat mask;
        cv::Mat M = cv::estimateAffinePartial2D(src, dst, mask, cv::RANSAC, 4.0);
        if (M.empty() || mask.empty()) return {cv::Mat(), 0.0};
        int inliers = cv::countNonZero(mask);
        last_inliers_ = inliers;
        if (inliers < pi("min_inliers", 12)) return {cv::Mat(), 0.0};
        cv::Mat M3 = cv::Mat::eye(3, 3, CV_32F);
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 3; b++) M3.at<float>(a, b) = (float)M.at<double>(a, b);
        if (pb("refine", true)) {
            cv::Mat refined = refine_ecc_window(ref_g, img_g, M3, rw, iw);
            if (!refined.empty() && fit_roi(ref_g, img_g, refined) >= fit_roi(ref_g, img_g, M3)) M3 = refined;
        }
        return {M3, fit_roi(ref_g, img_g, M3)};
    }
};

VC_REGISTER("align", Align)

}  // namespace vc
