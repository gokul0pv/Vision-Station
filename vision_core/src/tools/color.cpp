// Colour (vision_station/tools/color.py): LED colour by CIELAB deltaE, LED lit / not lit.
#include "../core/npy.hpp"
#include "../core/pyfmt.hpp"
#include "../core/tool.hpp"

#include <opencv2/imgproc.hpp>

namespace vc {

namespace {

const Json DEFAULT_SWATCHES = Json::parse(R"({
  "red": [115, 190, 172], "green": [181, 67, 183], "blue": [94, 168, 56], "amber": [200, 145, 200],
  "yellow": [218, 114, 204], "white": [236, 128, 129], "cyan": [210, 92, 118], "magenta": [150, 200, 90],
  "off": [30, 129, 128]})");
constexpr double L_WEIGHT = 0.5;

cv::Mat np_slice(const cv::Mat& m, int x0, int y0, int x1, int y1) {
    x0 = std::clamp(x0, 0, m.cols); y0 = std::clamp(y0, 0, m.rows);
    x1 = std::clamp(x1, x0, m.cols); y1 = std::clamp(y1, y0, m.rows);
    return m(cv::Rect(x0, y0, x1 - x0, y1 - y0));
}

double delta_e(const std::array<double, 3>& a0, const std::array<double, 3>& b0, const Json& lw) {
    float a[3], b[3], d[3];
    for (int i = 0; i < 3; i++) { a[i] = (float)a0[i]; b[i] = (float)b0[i]; }
    for (int i = 1; i < 3; i++) { a[i] -= 128.0f; b[i] -= 128.0f; }
    for (int i = 0; i < 3; i++) d[i] = a[i] - b[i];
    d[0] *= (float)(lw.is_null() ? L_WEIGHT : lw.get<double>());
    return (double)npy::norm_f32(d, 3);
}

// np.median(pixels.reshape(-1, 3), axis=0) of LAB pixels where sel says so
std::array<double, 3> median3(const cv::Mat& lab, const cv::Mat* sel) {
    std::vector<uint8_t> c[3];
    for (int y = 0; y < lab.rows; y++)
        for (int x = 0; x < lab.cols; x++)
            if (!sel || sel->at<uint8_t>(y, x)) {
                auto v = lab.at<cv::Vec3b>(y, x);
                for (int k = 0; k < 3; k++) c[k].push_back(v[k]);
            }
    return {npy::median_u8(c[0]), npy::median_u8(c[1]), npy::median_u8(c[2])};
}

std::array<double, 3> representative_lab(const cv::Mat& image, const cv::Mat& lab, int x, int y, int w, int h, int clip, double halo) {
    int H = lab.rows, W = lab.cols;
    cv::Mat sub_bgr = np_slice(image, x, y, x + w, y + h), sub_lab = np_slice(lab, x, y, x + w, y + h);
    if (sub_lab.empty()) return {0, 128, 128};
    auto maxc = [](const cv::Mat& bgr) {
        cv::Mat m(bgr.rows, bgr.cols, CV_8U);
        for (int yy = 0; yy < bgr.rows; yy++)
            for (int xx = 0; xx < bgr.cols; xx++) {
                auto v = bgr.at<cv::Vec3b>(yy, xx);
                m.at<uint8_t>(yy, xx) = std::max({v[0], v[1], v[2]});
            }
        return m;
    };
    cv::Mat keep = maxc(sub_bgr) < clip;
    if (cv::countNonZero(keep) >= std::max(12.0, 0.05 * keep.total())) return median3(sub_lab, &keep);
    double cx = x + w / 2.0, cy = y + h / 2.0, rad = std::max(w, h) * halo / 2.0;
    int x0 = (int)std::max(0.0, cx - rad), x1 = (int)std::min((double)W, cx + rad);
    int y0 = (int)std::max(0.0, cy - rad), y1 = (int)std::min((double)H, cy + rad);
    cv::Mat rb = np_slice(image, x0, y0, x1, y1), rl = np_slice(lab, x0, y0, x1, y1);
    if (!rl.empty()) {
        cv::Mat m = maxc(rb);
        cv::Mat sel = (m < clip) & (m > 40);
        if (cv::countNonZero(sel) >= 12) return median3(rl, &sel);
    }
    return median3(sub_lab, nullptr);
}

Json blobs_from(const Context& ctx, const std::string& src) {
    auto it = ctx.share.find(src);
    if (it == ctx.share.end()) return Json();
    if (auto j = std::any_cast<Json>(&it->second)) return *j;
    return Json();
}

}  // namespace

class ColorClassify : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        std::string src = ps("source", "");
        Json blobs = blobs_from(ctx, src);
        if (blobs.is_null())
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "no blobs from source '" + (has("source") ? py_str(p["source"]) : "None") + "' \xe2\x80\x94 run a count_blob first");
        Json sw = DEFAULT_SWATCHES;
        if (p.contains("swatches") && p["swatches"].is_object())
            for (auto& [k, v] : p["swatches"].items()) sw[k] = v;
        Json expect = p.value("expect", Json::array());
        double max_de = pf("deltaE_max", 28);
        Json lw = has("lightness_weight") ? p["lightness_weight"] : Json();
        cv::Mat lab;
        cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
        double halo = pf("halo", 1.8);
        int clip = pi("clip_level", 250);
        Json found = Json::array();
        std::vector<Graphic> g;
        std::vector<std::string> fails;
        for (size_t i = 0; i < blobs.size(); i++) {
            auto& b = blobs[i];
            int x = b["x"], y = b["y"], w = b["w"], h = b["h"];
            if (np_slice(lab, x, y, x + w, y + h).empty()) continue;
            auto med = representative_lab(image, lab, x, y, w, h, clip, halo);
            std::string name;
            double de = 1e300;
            for (auto& [k, v] : sw.items()) {
                double e = delta_e(med, {v[0].get<double>(), v[1].get<double>(), v[2].get<double>()}, lw);
                if (e < de) { de = e; name = k; }
            }
            Json want = i < expect.size() ? expect[i] : Json();
            bool ok = de <= max_de && (want.is_null() || name == want.get<std::string>());
            if (!ok) {
                std::string f = "#" + std::to_string(i + 1) + " " + name;
                if (!want.is_null() && py_str(want) != "") f += " (want " + py_str(want) + ")";
                fails.push_back(f + " dE=" + pyfmt("%.0f", de));
            }
            found.push_back({{"index", (int)i + 1}, {"color", name}, {"deltaE", py_round(de, 1)}, {"expected", want}, {"ok", ok}});
            g.emplace_back("rect", std::vector<std::pair<double, double>>{{(double)x, (double)y}, {(double)(x + w), (double)(y + h)}}, ok ? "pass" : "fail", name);
        }
        std::string status = fails.empty() ? "pass" : "fail";
        if (!expect.empty() && blobs.size() != expect.size()) {
            status = "fail";
            fails.push_back(std::to_string(blobs.size()) + " LEDs vs " + std::to_string(expect.size()) + " expected");
        }
        std::string msg;
        for (size_t i = 0; i < fails.size(); i++) msg += (i ? "; " : "") + fails[i];
        double score = 1.0 - std::min(1.0, (double)fails.size() / (double)std::max<size_t>(1, found.size()));
        return ToolResult(id, type_name, status, score, {{"leds", found}}, g, msg, now_ms() - t0);
    }
};
VC_REGISTER("color_classify", ColorClassify)

class LedGlow : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        Json blobs = blobs_from(ctx, ps("source", ""));
        if (blobs.is_null()) {
            auto [crp, off] = crop(image, ctx);
            blobs = Json::array({{{"x", off.x}, {"y", off.y}, {"w", crp.cols}, {"h", crp.rows}}});
        }
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        double v_on = pf("v_on", 140), s_on = pf("s_on", 60);
        Json expect = p.value("expect_state", Json::array());
        Json out = Json::array();
        std::vector<Graphic> g;
        std::vector<std::string> fails;
        int lit_count = 0;
        for (size_t i = 0; i < blobs.size(); i++) {
            auto& b = blobs[i];
            int x = b["x"], y = b["y"], w = b["w"], h = b["h"];
            cv::Mat pch = np_slice(hsv, x, y, x + w, y + h);
            if (pch.empty()) continue;
            std::vector<uint8_t> vv, ss;
            for (int yy = 0; yy < pch.rows; yy++)
                for (int xx = 0; xx < pch.cols; xx++) {
                    auto px = pch.at<cv::Vec3b>(yy, xx);
                    vv.push_back(px[2]);
                    ss.push_back(px[1]);
                }
            double v = npy::percentile_u8(vv, 90), s = npy::percentile_u8(ss, 60);
            bool lit = v >= v_on && s >= s_on;
            bool want = i < expect.size() ? (expect[i].is_boolean() ? expect[i].get<bool>() : !expect[i].is_null() && expect[i] != 0) : true;
            bool ok = lit == want;
            if (!ok) fails.push_back("#" + std::to_string(i + 1) + " " + (lit ? "on" : "off") + ", want " + (want ? "on" : "off"));
            if (lit) lit_count++;
            out.push_back({{"index", (int)i + 1}, {"lit", lit}, {"v", py_round(v, 1)}, {"s", py_round(s, 1)}, {"ok", ok}});
            g.emplace_back("rect", std::vector<std::pair<double, double>>{{(double)x, (double)y}, {(double)(x + w), (double)(y + h)}}, ok ? "pass" : "fail", lit ? "on" : "off");
        }
        std::string msg;
        for (size_t i = 0; i < fails.size(); i++) msg += (i ? "; " : "") + fails[i];
        double score = 1.0 - std::min(1.0, (double)fails.size() / (double)std::max<size_t>(1, out.size()));
        return ToolResult(id, type_name, fails.empty() ? "pass" : "fail", score, {{"leds", out}, {"lit_count", lit_count}}, g, msg, now_ms() - t0);
    }
};
VC_REGISTER("led_glow", LedGlow)

}  // namespace vc
