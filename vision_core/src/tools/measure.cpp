// Golden-image differencing (defect.py defect_diff) and edge measurement
// (measure.py measure_length, calibrate_scale).
#include "../core/npy.hpp"
#include "../core/pyfmt.hpp"
#include "../core/tool.hpp"

#include <opencv2/imgproc.hpp>

namespace vc {

namespace {

cv::Mat np_slice(const cv::Mat& m, int x0, int y0, int x1, int y1) {
    x0 = std::clamp(x0, 0, m.cols); y0 = std::clamp(y0, 0, m.rows);
    x1 = std::clamp(x1, x0, m.cols); y1 = std::clamp(y1, y0, m.rows);
    return m(cv::Rect(x0, y0, x1 - x0, y1 - y0));
}

double subpixel_peak(const std::vector<float>& prof, int i) {
    if (i <= 0 || i >= (int)prof.size() - 1) return (double)i;
    double a = prof[i - 1], b = prof[i], c = prof[i + 1];
    double d = a - 2 * b + c;
    return std::abs(d) < 1e-9 ? (double)i : (double)i - 0.5 * (c - a) / d;
}

// measure._edges: [(position, strength)]
std::vector<std::pair<double, double>> edges(const cv::Mat& crop, const std::string& axis, const std::string& polarity, int n, double min_contrast) {
    cv::Mat g = to_gray(crop), gb;
    cv::GaussianBlur(g, gb, cv::Size(3, 3), 0);
    std::vector<float> prof;
    if (axis == "x") {
        for (int x = 0; x < gb.cols; x++) {
            double s = 0;
            for (int y = 0; y < gb.rows; y++) s += gb.at<uint8_t>(y, x);
            prof.push_back((float)(s / gb.rows));
        }
    } else {
        for (int y = 0; y < gb.rows; y++) {
            double s = 0;
            for (int x = 0; x < gb.cols; x++) s += gb.at<uint8_t>(y, x);
            prof.push_back((float)(s / gb.cols));
        }
    }
    auto d = npy::gradient_f32(prof);
    std::vector<float> cand = d;
    if (polarity == "light_to_dark") for (auto& v : cand) v = -v;
    else if (polarity != "dark_to_light") for (auto& v : cand) v = std::abs(v);
    std::vector<int> idx;
    for (int i = 1; i + 1 < (int)cand.size(); i++)
        if (cand[i] >= cand[i - 1] && cand[i] > cand[i + 1] && (double)cand[i] >= min_contrast) idx.push_back(i);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return -cand[a] < -cand[b]; });
    if ((int)idx.size() > n) idx.resize(std::max(0, n));
    std::sort(idx.begin(), idx.end());
    std::vector<std::pair<double, double>> out;
    for (int i : idx) out.emplace_back(subpixel_peak(cand, i), (double)cand[i]);
    return out;
}

}  // namespace

class DefectDiff : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        if (ctx.reference.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "reference image required");
        int h = image.rows, w = image.cols;
        cv::Mat ref = ctx.reference;
        if (!ctx.fixture.empty()) {
            cv::Mat o;
            cv::warpAffine(ref, o, ctx.fixture.rowRange(0, 2), cv::Size(w, h), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            ref = o;
        } else if (ref.rows != h || ref.cols != w) {
            cv::Mat o;
            cv::resize(ref, o, cv::Size(w, h));
            ref = o;
        }
        auto mr = mapped_roi(ctx);
        int x = mr ? (*mr)[0] : 0, y = mr ? (*mr)[1] : 0, cw = mr ? (*mr)[2] : w, ch = mr ? (*mr)[3] : h;
        x = std::max(0, x);
        y = std::max(0, y);
        cw = std::min(cw, w - x);
        ch = std::min(ch, h - y);
        cv::Mat a, b;
        cv::cvtColor(np_slice(ref, x, y, x + cw, y + ch), a, cv::COLOR_BGR2GRAY);
        cv::cvtColor(np_slice(image, x, y, x + cw, y + ch), b, cv::COLOR_BGR2GRAY);
        int blur = pi("illum_blur", 51) | 1;
        cv::Mat ab, bb;
        cv::GaussianBlur(a, ab, cv::Size(blur, blur), 0);
        cv::GaussianBlur(b, bb, cv::Size(blur, blur), 0);
        cv::divide(a, ab, a, 128);
        cv::divide(b, bb, b, 128);
        cv::GaussianBlur(a, a, cv::Size(3, 3), 0);
        cv::GaussianBlur(b, b, cv::Size(3, 3), 0);
        cv::Mat diff, m;
        cv::absdiff(a, b, diff);
        cv::threshold(diff, m, pi("threshold", 22), 255, cv::THRESH_BINARY);
        int k = pi("open_px", 2);
        if (k > 0) cv::morphologyEx(m, m, cv::MORPH_OPEN, cv::Mat::ones(k, k, CV_8U));
        cv::morphologyEx(m, m, cv::MORPH_CLOSE, cv::Mat::ones(3, 3, CV_8U));
        double min_area = pf("min_area", 25), elong = pf("scratch_elongation", 3.5);
        double mm_px = pf("mm_per_px", 0);
        if (mm_px == 0) mm_px = ctx.mm_per_px;
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(m, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        Json defects = Json::array();
        auto g = roi_graphic(ctx);
        int worst = 0, scratches = 0, dents = 0;
        for (auto& c : cnts) {
            double area = cv::contourArea(c);
            if (area < min_area) continue;
            cv::RotatedRect rr = cv::minAreaRect(c);
            double rw = rr.size.width, rh = rr.size.height;
            double long_s = std::max(rw, rh), short_s = std::max(1e-6, std::min(rw, rh));
            std::string kind = long_s / short_s >= elong ? "scratch" : "dent";
            cv::Rect br = cv::boundingRect(c);
            Json d = {{"kind", kind}, {"area_px", (int)area}, {"length_px", py_round(long_s, 1)},
                      {"x", br.x + x}, {"y", br.y + y}, {"w", br.width}, {"h", br.height}};
            if (mm_px != 0) {
                d["length_mm"] = py_round(long_s * mm_px, 3);
                d["area_mm2"] = py_round(area * mm_px * mm_px, 4);
            }
            worst = std::max(worst, (int)area);
            (kind == "scratch" ? scratches : dents)++;
            g.emplace_back("rect", std::vector<std::pair<double, double>>{{(double)(br.x + x), (double)(br.y + y)}, {(double)(br.x + x + br.width), (double)(br.y + y + br.height)}}, "fail", kind);
            defects.push_back(d);
        }
        int max_n = pi("max_defects", 0);
        double max_a = pf("max_area_px", 0);
        bool ok = (int)defects.size() <= max_n && (max_a == 0 || worst <= max_a);
        Json vals = {{"defect_count", defects.size()}, {"scratches", scratches}, {"dents", dents}, {"largest_area_px", worst}, {"defects", defects}};
        return ToolResult(id, type_name, ok ? "pass" : "fail", ok ? 1.0 : 0.0, vals, g,
                          ok ? "" : std::to_string(defects.size()) + " defect(s), largest " + std::to_string(worst) + "px", now_ms() - t0);
    }
};
VC_REGISTER("defect_diff", DefectDiff)

class MeasureLength : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        auto [crp, off] = crop(image, ctx);
        std::string axis = ps("axis", "x");
        auto e = edges(crp, axis, ps("polarity", "any"), pi("max_edges", 2), pf("min_contrast", 2.0));
        if (e.size() < 2)
            return ToolResult(id, type_name, "fail", 0.0, {{"edges_found", e.size()}}, roi_graphic(ctx, "fail"),
                              "fewer than two edges found \xe2\x80\x94 check contrast or polarity", now_ms() - t0);
        double p0 = e.front().first, p1 = e.back().first, px = std::abs(p1 - p0);
        double mm_px = pf("mm_per_px", 0);
        if (mm_px == 0) mm_px = ctx.mm_per_px;
        std::optional<double> mm;
        if (mm_px != 0) mm = px * mm_px;
        Json nom = has("nominal_mm") ? p["nominal_mm"] : Json(), tol = has("tolerance_mm") ? p["tolerance_mm"] : Json();
        bool ok = true;
        std::string why;
        if (!nom.is_null() && mm) {
            double t = tol.is_null() ? 0.05 : tol.get<double>();
            ok = std::abs(*mm - nom.get<double>()) <= t;
            if (!ok) why = pyfmt("%.4f", *mm) + " mm outside " + py_str(nom) + " \xc2\xb1" + py_repr(t);
        }
        auto r = mapped_roi(ctx);
        auto g = roi_graphic(ctx, ok ? "pass" : "fail");
        std::string label = mm ? pyfmt("%.4f", *mm) + " mm" : pyfmt("%.2f", px) + " px";
        double ox = off.x, oy = off.y;
        using P = std::vector<std::pair<double, double>>;
        if (axis == "x") {
            g.emplace_back("line", P{{ox + p0, oy}, {ox + p0, oy + (*r)[3]}}, "pass", "");
            g.emplace_back("line", P{{ox + p1, oy}, {ox + p1, oy + (*r)[3]}}, "pass", "");
            g.emplace_back("text", P{{ox + (p0 + p1) / 2, oy - 6}}, ok ? "pass" : "fail", label);
        } else {
            g.emplace_back("line", P{{ox, oy + p0}, {ox + (*r)[2], oy + p0}}, "pass", "");
            g.emplace_back("line", P{{ox, oy + p1}, {ox + (*r)[2], oy + p1}}, "pass", "");
            g.emplace_back("text", P{{ox + (*r)[2] / 2.0, oy + (p0 + p1) / 2}}, ok ? "pass" : "fail", label);
        }
        Json vals = {{"length_px", py_round(px, 3)}, {"edges_found", e.size()}};
        if (mm) {
            vals["length_mm"] = py_round(*mm, 4);
            vals["nominal_mm"] = nom;
            vals["tolerance_mm"] = tol;
            vals["deviation_mm"] = nom.is_null() ? Json() : Json(py_round(*mm - nom.get<double>(), 4));
        }
        return ToolResult(id, type_name, ok ? "pass" : "fail", ok ? 1.0 : 0.0, vals, g, why, now_ms() - t0);
    }
};
VC_REGISTER("measure_length", MeasureLength)

class CalibrateScale : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        double known = pf("known_mm", 0);
        auto [crp, off] = crop(image, ctx);
        auto e = edges(crp, ps("axis", "x"), ps("polarity", "any"), 2, pf("min_contrast", 2.0));
        if (e.size() < 2 || known <= 0) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "need two edges and a positive known_mm");
        double px = std::abs(e.back().first - e.front().first);
        ctx.mm_per_px = known / px;
        return ToolResult(id, type_name, "pass", 1.0, {{"mm_per_px", py_round(ctx.mm_per_px, 6)}, {"span_px", py_round(px, 3)}}, roi_graphic(ctx), "", now_ms() - t0);
    }
};
VC_REGISTER("calibrate_scale", CalibrateScale)

}  // namespace vc
