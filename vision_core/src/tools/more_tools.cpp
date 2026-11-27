// vision_station/tools/more_tools.py: find_ellipse, sharpness, color_segment, classify.
#include "geom.hpp"

#include "../core/npy.hpp"
#include "../core/onnx.hpp"
#include "../core/pyfmt.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>

namespace vc {

namespace geom {

// more_tools._fit_ellipse_robust: cv2.fitEllipse, drop the worst point, refit -- `ignore` times
EllipseFit fit_ellipse_robust(const std::vector<double>& pts, int ignore) {
    int n = (int)(pts.size() / 2);
    std::vector<int> keep(n);
    for (int i = 0; i < n; i++) keep[i] = i;
    for (int it = 0; it <= ignore; it++) {
        std::vector<cv::Point2f> fp;
        for (int i : keep) fp.emplace_back((float)pts[2 * i], (float)pts[2 * i + 1]);
        cv::RotatedRect rr = cv::fitEllipse(fp);
        double cx = rr.center.x, cy = rr.center.y, w = rr.size.width, h = rr.size.height, ang = rr.angle;
        double a = w / 2.0, b = h / 2.0;
        double t = ang * (3.141592653589793 / 180.0);
        double c = std::cos(t), s = std::sin(t);
        double ma = std::max(a, 1e-6), mb = std::max(b, 1e-6), mn = std::min(a, b);
        std::vector<double> resid;
        for (int i : keep) {
            double d0 = pts[2 * i] - cx, d1 = pts[2 * i + 1] - cy;
            double u = d0 * c + d1 * s;
            double v = -d0 * s + d1 * c;
            double uu = u / ma, vv = v / mb;
            double r = std::sqrt(uu * uu + vv * vv);
            resid.push_back((r - 1.0) * mn);
        }
        if ((int)keep.size() <= 6 || it == ignore) return {cx, cy, a, b, ang, resid, keep};
        int worst = 0;
        for (int i = 1; i < (int)resid.size(); i++)
            if (std::abs(resid[i]) > std::abs(resid[worst])) worst = i;
        keep.erase(keep.begin() + worst);
    }
    return {};
}

}  // namespace geom

using namespace geom;

namespace {
constexpr double kPi = 3.141592653589793;

std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string r;
    for (size_t i = 0; i < v.size(); i++) r += (i ? sep : "") + v[i];
    return r;
}
}  // namespace

// ---------------------------------------------------------------- find ellipse
class FindEllipse : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        auto [g, o] = raw_plane(*this, image, ctx);
        if (g.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "empty region");
        int H = g.rows, W = g.cols;
        double cx0 = (W - 1) / 2.0, cy0 = (H - 1) / 2.0, ax = W / 2.0, ay = H / 2.0;
        double f_in = pf("search_inner", 0.45), f_out = pf("search_outer", 1.0);
        int n = std::max(8, pi("num_calipers", 36));
        std::string pol = ps("polarity", "any");
        double minc = pf("min_contrast", 8.0);
        std::vector<Graphic> guides;
        for (double f : {f_in, f_out}) {
            double sub = f == f_out ? 1.0 : 0.0;
            std::vector<Pt> ring;
            for (int i = 0; i < 48; i++) {
                double a = i == 47 ? 2 * kPi : i * (2 * kPi / 47.0);
                ring.emplace_back(cx0 + (ax * f - sub) * std::cos(a), cy0 + (ay * f - sub) * std::sin(a));
            }
            guides.push_back(abs_graphic("poly", to_image(*this, o.x, o.y, ring), "search"));
        }
        const int steps = 48;
        std::vector<double> fs(steps);
        double fstep = (f_out - f_in) / (double)(steps - 1);
        for (int i = 0; i < steps; i++) fs[i] = fstep == 0 ? f_in : (double)i * fstep + f_in;
        fs[steps - 1] = f_out;
        std::vector<Pt> pts;
        for (int k = 0; k < n; k++) {
            double a = 2 * kPi * k / n, ca = std::cos(a), sa = std::sin(a);
            cv::Mat xs(1, steps, CV_32F), ys(1, steps, CV_32F), prof;
            for (int i = 0; i < steps; i++) {
                float x = (float)(cx0 + ax * fs[i] * ca), y = (float)(cy0 + ay * fs[i] * sa);
                xs.at<float>(0, i) = std::min((float)(W - 1), std::max(0.0f, x));
                ys.at<float>(0, i) = std::min((float)(H - 1), std::max(0.0f, y));
            }
            cv::remap(g, prof, xs, ys, cv::INTER_LINEAR);
            std::vector<float> pv((const float*)prof.data, (const float*)prof.data + steps);
            double step = npy::py_hypot(ax * ca, ay * sa) * (f_out - f_in) / (steps - 1);
            auto e = profile_edges(pv, pol, minc, std::max(step, 1e-6), 1);
            if (e.empty()) continue;
            double f = f_in + (f_out - f_in) * e[0].first / (double)(steps - 1);
            pts.emplace_back(cx0 + ax * f * ca, cy0 + ay * f * sa);
        }
        int need = std::max(6, pi("min_found", 12));
        if ((int)pts.size() < need) {
            auto gr = roi_graphic(ctx, "fail");
            gr.insert(gr.end(), guides.begin(), guides.end());
            return ToolResult(id, type_name, "fail", 0.0, Json{{"found", (int)pts.size()}}, gr,
                              "edge found by " + std::to_string(pts.size()) + " calipers, need " + std::to_string(need), now_ms() - t0);
        }
        auto P = to_image(*this, o.x, o.y, pts);
        std::vector<double> Pd;
        for (auto& q : P) {
            Pd.push_back(q.x);
            Pd.push_back(q.y);
        }
        auto e = fit_ellipse_robust(Pd, pi("num_to_ignore", 4));
        std::vector<double> sq;
        for (double r : e.resid) sq.push_back(r * r);
        double rms = std::sqrt(npy::sum_f64(sq.data(), sq.size()) / (double)sq.size());
        double max_rms = pf("max_rms", 1.5);
        bool ok = rms <= max_rms;
        std::string st = ok ? "pass" : "fail";
        ctx.share[id] = Json{{"type", "circle"}, {"center", {e.cx, e.cy}}, {"radius", (e.a + e.b) / 2.0}};
        double t = e.ang * (kPi / 180.0);
        std::vector<cv::Point2f> ring;
        for (int i = 0; i < 64; i++) {
            double u = i == 63 ? 2 * kPi : i * (2 * kPi / 63.0);
            ring.emplace_back((float)(e.cx + e.a * std::cos(u) * std::cos(t) - e.b * std::sin(u) * std::sin(t)),
                              (float)(e.cy + e.a * std::cos(u) * std::sin(t) + e.b * std::sin(u) * std::cos(t)));
        }
        auto gr = roi_graphic(ctx, st);
        gr.insert(gr.end(), guides.begin(), guides.end());
        gr.push_back(abs_graphic("poly", ring, st, id + " " + pyfmt("%.1f", 2 * e.a) + " x " + pyfmt("%.1f", 2 * e.b)));
        gr.push_back(abs_graphic("line", {{(float)(e.cx - 4), (float)e.cy}, {(float)(e.cx + 4), (float)e.cy}}, st));
        gr.push_back(abs_graphic("line", {{(float)e.cx, (float)(e.cy - 4)}, {(float)e.cx, (float)(e.cy + 4)}}, st));
        Json v = {{"cx", py_round(e.cx, 3)}, {"cy", py_round(e.cy, 3)}, {"major_px", py_round(2 * std::max(e.a, e.b), 3)},
                  {"minor_px", py_round(2 * std::min(e.a, e.b), 3)}, {"angle_deg", py_round(e.ang, 2)}, {"rms_px", py_round(rms, 3)},
                  {"found", (int)pts.size()}, {"used", (int)e.keep.size()}};
        if (ctx.mm_per_px > 0) {
            v["major_mm"] = py_round(2 * std::max(e.a, e.b) * ctx.mm_per_px, 4);
            v["minor_mm"] = py_round(2 * std::min(e.a, e.b) * ctx.mm_per_px, 4);
        }
        return ToolResult(id, type_name, st, std::max(0.0, 1.0 - rms / (2 * max_rms)), v, gr,
                          ok ? "" : "fit error " + pyfmt("%.2f", rms) + " px over " + pyfmt("%.2f", max_rms), now_ms() - t0);
    }
};
VC_REGISTER("find_ellipse", FindEllipse)

// ---------------------------------------------------------------- sharpness
namespace {
double lap_var(const cv::Mat& crop) {
    cv::Mat g = to_gray(crop), f, lap;
    g.convertTo(f, CV_32F);
    cv::Laplacian(f, lap, CV_32F, 3);
    if (!lap.isContinuous()) lap = lap.clone();
    return (double)npy::var_f32((const float*)lap.data, lap.total());
}
}  // namespace

class Sharpness : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        auto [crop, o] = this->crop(image, ctx);
        (void)o;
        if (crop.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "empty region");
        double score = lap_var(crop);
        std::optional<double> ref_score;
        if (!ctx.reference.empty() && pb("relative", true)) {
            cv::Mat saved = ctx.fixture;
            ctx.fixture = cv::Mat();
            std::pair<cv::Mat, cv::Point> rc;
            try {
                rc = this->crop(ctx.reference, ctx);
            } catch (...) {
                ctx.fixture = saved;
                throw;
            }
            ctx.fixture = saved;
            if (!rc.first.empty()) ref_score = lap_var(rc.first);
        }
        Json v = {{"sharpness", py_round(score, 2)}};
        bool ok;
        std::string why;
        double limit;
        double sc;
        if (ref_score && *ref_score != 0) {
            double rel = score / *ref_score;
            v["reference"] = py_round(*ref_score, 2);
            v["relative"] = py_round(rel, 3);
            limit = pf("min_relative", 0.6);
            ok = rel >= limit;
            if (!ok) why = "focus " + pyfmt("%.0f", 100 * rel) + "% of the reference (limit " + pyfmt("%.0f", 100 * limit) + "%)";
            sc = std::min(1.0, v["relative"].get<double>());
        } else {
            limit = pf("min_sharpness", 50.0);
            ok = score >= limit;
            if (!ok) why = "sharpness " + pyfmt("%.1f", score) + " under " + pyfmt("%.1f", limit);
            sc = std::min(1.0, score / std::max(limit, 1e-6));
        }
        std::string st = ok ? "pass" : "fail";
        return ToolResult(id, type_name, st, sc, v, roi_graphic(ctx, st, id + " " + pyfmt("%.0f", score)), why, now_ms() - t0);
    }
};
VC_REGISTER("sharpness", Sharpness)

// ---------------------------------------------------------------- colour segmenter
class ColorSegment : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        auto [crop, o] = this->crop(image, ctx);
        if (crop.empty() || crop.channels() != 3) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "needs a colour region");
        cv::Mat hsv, m;
        cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);
        int h0 = pi("hue_min", 0), h1 = pi("hue_max", 179), s0 = pi("sat_min", 60), v0 = pi("val_min", 40);
        if (h0 <= h1) {
            cv::inRange(hsv, cv::Scalar(h0, s0, v0), cv::Scalar(h1, 255, 255), m);
        } else {
            cv::Mat a, b;
            cv::inRange(hsv, cv::Scalar(h0, s0, v0), cv::Scalar(179, 255, 255), a);
            cv::inRange(hsv, cv::Scalar(0, s0, v0), cv::Scalar(h1, 255, 255), b);
            cv::bitwise_or(a, b, m);
        }
        cv::morphologyEx(m, m, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
        long long total;
        if (!crop_mask.empty() && crop_mask.size() == m.size() && crop_mask.type() == m.type()) {
            cv::bitwise_and(m, crop_mask, m);
            total = cv::countNonZero(crop_mask);
        } else {
            total = (long long)m.total();
        }
        long long area = cv::countNonZero(m);
        double pct = 100.0 * (double)area / (double)std::max(1LL, total);
        double lo = pf("min_pct", 0), hi = pf("max_pct", 0);
        std::vector<std::string> why;
        if (lo != 0 && pct < lo) why.push_back(pyfmt("%.1f", pct) + "% of the region is that colour, under " + pyfmt("%.1f", lo) + "%");
        if (hi != 0 && pct > hi) why.push_back(pyfmt("%.1f", pct) + "% of the region is that colour, over " + pyfmt("%.1f", hi) + "%");
        bool ok = why.empty();
        std::string st = ok ? "pass" : "fail";
        if (ctx.want_records) {
            cv::Mat c3;
            cv::cvtColor(m, c3, cv::COLOR_GRAY2BGR);
            ctx.records[id]["LastRun.ColorMask"] = c3;
        }
        auto g = roi_graphic(ctx, st, id + " " + pyfmt("%.1f", pct) + "%");
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(m.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::vector<std::pair<double, size_t>> by_area;
        for (size_t i = 0; i < cnts.size(); i++) by_area.emplace_back(cv::contourArea(cnts[i]), i);
        std::stable_sort(by_area.begin(), by_area.end(), [](auto& a, auto& b) { return a.first > b.first; });
        for (size_t k = 0; k < by_area.size() && k < 12; k++) {
            if (by_area[k].first < 20) continue;
            std::vector<Pt> pp;
            for (auto& q : cnts[by_area[k].second]) pp.emplace_back(q.x, q.y);
            g.push_back(abs_graphic("poly", to_image(*this, o.x, o.y, pp), st));
        }
        return ToolResult(id, type_name, st, ok ? 1.0 : 0.0,
                          Json{{"percent", py_round(pct, 2)}, {"pixels", area}, {"blobs", (int)cnts.size()}}, g, join(why, "; "), now_ms() - t0);
    }
};
VC_REGISTER("color_segment", ColorSegment)

// ---------------------------------------------------------------- classify
class Classify : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        namespace fs = std::filesystem;
        double t0 = now_ms();
        std::string model_p = has("model") ? py_str(p["model"]) : "";
        std::string path = model_p;
        if (!path.empty() && !fs::u8path(path).is_absolute()) path = (fs::u8path(ctx.folder) / fs::u8path(path)).u8string();
        if (path.empty() || !fs::exists(fs::u8path(path)))
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "model file not found: " + (model_p.empty() ? std::string("(none)") : model_p));
        auto [crop, o] = this->crop(image, ctx);
        (void)o;
        if (crop.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "empty region");
        auto model = load_model(path, "");
        auto shape = model->input_shape();
        int hh = shape.size() > 2 && shape[2] > 0 ? (int)shape[2] : 224;
        int ww = shape.size() > 3 && shape[3] > 0 ? (int)shape[3] : 224;
        cv::Mat rgb, sm;
        cv::cvtColor(crop, rgb, crop.channels() == 3 ? cv::COLOR_BGR2RGB : cv::COLOR_GRAY2RGB);
        cv::resize(rgb, sm, cv::Size(ww, hh), 0, 0, cv::INTER_AREA);
        bool imagenet = ps("normalize", "imagenet") == "imagenet";
        const double mean[3] = {0.485, 0.456, 0.406}, sd[3] = {0.229, 0.224, 0.225};
        std::vector<float> x((size_t)3 * hh * ww);
        for (int y = 0; y < hh; y++)
            for (int xx = 0; xx < ww; xx++) {
                auto px = sm.at<cv::Vec3b>(y, xx);
                for (int c = 0; c < 3; c++) {
                    float f = (float)px[c] / 255.0f;
                    x[(size_t)c * hh * ww + (size_t)y * ww + xx] = imagenet ? (float)(((double)f - mean[c]) / sd[c]) : f;
                }
            }
        auto out = model->run(x.data(), {1, 3, hh, ww});
        std::vector<double> logits(out[0].data.begin(), out[0].data.end());
        double mx = *std::max_element(logits.begin(), logits.end());
        std::vector<double> e(logits.size());
        for (size_t i = 0; i < e.size(); i++) e[i] = std::exp(logits[i] - mx);
        double es = npy::sum_f64(e.data(), e.size());
        std::vector<double> probs(e.size());
        for (size_t i = 0; i < e.size(); i++) probs[i] = e[i] / es;
        std::vector<std::string> names;
        if (has("classes")) {
            const Json& c = p["classes"];
            if (c.is_string()) {
                std::string s = c.get<std::string>(), part;
                for (size_t i = 0; i <= s.size(); i++) {
                    if (i == s.size() || s[i] == ',') {
                        auto a = part.find_first_not_of(" \t"), b = part.find_last_not_of(" \t");
                        if (a != std::string::npos) names.push_back(part.substr(a, b - a + 1));
                        part.clear();
                    } else part += s[i];
                }
            } else if (c.is_array()) {
                for (auto& q : c) names.push_back(py_str(q));
            }
        }
        if (names.empty()) {
            auto meta = model->metadata();
            auto it = meta.find("classes");
            if (it != meta.end() && !it->second.empty()) {
                try {
                    for (auto& q : Json::parse(it->second)) names.push_back(py_str(q));
                } catch (...) {
                    names.clear();
                }
            }
        }
        for (size_t i = names.size(); i < probs.size(); i++) names.push_back("class_" + std::to_string(i));
        size_t top = std::max_element(probs.begin(), probs.end()) - probs.begin();
        std::string label = names[top];
        double conf = probs[top];
        std::vector<std::string> accept;
        if (has("accept")) {
            const Json& a = p["accept"];
            std::vector<std::string> parts;
            if (a.is_string()) {
                std::string s = a.get<std::string>(), part;
                for (size_t i = 0; i <= s.size(); i++) {
                    if (i == s.size() || s[i] == ',') parts.push_back(part), part.clear();
                    else part += s[i];
                }
            } else if (a.is_array())
                for (auto& q : a) parts.push_back(py_str(q));
            for (auto& q : parts) {
                auto x0 = q.find_first_not_of(" \t"), x1 = q.find_last_not_of(" \t");
                if (x0 != std::string::npos) accept.push_back(q.substr(x0, x1 - x0 + 1));
            }
        }
        if (accept.empty()) accept.push_back(names[0]);
        double min_conf = pf("min_confidence", 0.6);
        bool in_accept = std::find(accept.begin(), accept.end(), label) != accept.end();
        bool ok = in_accept && conf >= min_conf;
        std::string why;
        if (!ok)
            why = !in_accept ? "classified as " + label + " (" + pyfmt("%.0f", 100 * conf) + "%)"
                             : label + " only " + pyfmt("%.0f", 100 * conf) + "% confident (need " + pyfmt("%.0f", 100 * min_conf) + "%)";
        Json scores = Json::object();
        for (size_t i = 0; i < std::min(names.size(), probs.size()); i++) scores[names[i]] = py_round(probs[i], 4);
        std::string st = ok ? "pass" : "fail";
        return ToolResult(id, type_name, st, conf, Json{{"class", label}, {"confidence", py_round(conf, 4)}, {"scores", scores}},
                          roi_graphic(ctx, st, id + " " + label + " " + pyfmt("%.0f", 100 * conf) + "%"), why, now_ms() - t0);
    }
};
VC_REGISTER("classify", Classify)

}  // namespace vc
