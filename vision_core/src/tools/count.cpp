// Counting and presence (vision_station/tools/count.py): Part present (with
// its Train-tab OK/NG examples and the seated check), Count by shape, Count
// blobs.
#include "../core/engine.hpp"
#include "../core/npy.hpp"
#include "../core/pyfmt.hpp"
#include "../core/tool.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <mutex>

namespace fs = std::filesystem;

namespace vc {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double CHANGE_LEVEL = 30.0;
constexpr double NG_MARGIN = 0.03;
constexpr double SAME_PICTURE_GREY = 1.0;
constexpr double SAME_PATCH_SHARE = 0.01;
const cv::Size THUMB(64, 48);

cv::Mat np_slice(const cv::Mat& m, int x0, int y0, int x1, int y1) {
    x0 = std::clamp(x0, 0, m.cols); y0 = std::clamp(y0, 0, m.rows);
    x1 = std::clamp(x1, x0, m.cols); y1 = std::clamp(y1, y0, m.rows);
    return m(cv::Rect(x0, y0, x1 - x0, y1 - y0));
}

double std_of(const cv::Mat& g) {
    cv::Mat c = g.isContinuous() ? g : g.clone();
    return npy::std_u8(c.ptr<uint8_t>(), c.total());
}

}  // namespace

// count.upright: the region turned by `angle` about its centre, cut upright, grey
cv::Mat upright(const cv::Mat& img, const std::array<double, 4>& r, double angle, int margin) {
    int x = (int)r[0], y = (int)r[1], w = (int)r[2], h = (int)r[3];
    if (std::abs(angle) < 1e-6) {
        int H = img.rows, W = img.cols;
        int x0 = std::max(0, x - margin), y0 = std::max(0, y - margin);
        return to_gray(np_slice(img, x0, y0, std::min(W, x + w + margin), std::min(H, y + h + margin)));
    }
    double a = angle * kPi / 180.0, ca = std::cos(a), sa = std::sin(a);
    int sw = w + 2 * margin, sh = h + 2 * margin;
    double cx = x + w / 2.0, cy = y + h / 2.0;
    cv::Mat M = (cv::Mat_<float>(2, 3) << (float)ca, (float)-sa, (float)(cx - ca * sw / 2.0 + sa * sh / 2.0),
                 (float)sa, (float)ca, (float)(cy - sa * sw / 2.0 - ca * sh / 2.0));
    cv::Mat out;
    cv::warpAffine(to_gray(img), out, M, cv::Size(sw, sh), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
    return out;
}

// count.change_share: (share, mask of the largest patch that differs)
std::pair<double, cv::Mat> change_share(const cv::Mat& found, const cv::Mat& tpl, double level = CHANGE_LEVEL) {
    if (found.size() != tpl.size() || found.empty()) return {0.0, cv::Mat()};
    cv::Mat a, b;
    found.convertTo(a, CV_32F);
    tpl.convertTo(b, CV_32F);
    auto vec = [](const cv::Mat& m) { std::vector<float> v(m.begin<float>(), m.end<float>()); return v; };
    float a10 = npy::percentile_f32(vec(a), 10), a90 = npy::percentile_f32(vec(a), 90);
    float b10 = npy::percentile_f32(vec(b), 10), b90 = npy::percentile_f32(vec(b), 90);
    double gain = (a90 - a10 > 4.0f) ? (double)(b90 - b10) / (double)(a90 - a10) : 1.0;
    gain = std::min(2.0, std::max(0.5, gain));
    // (a - float(a10)) * gain + float(b10), float32 throughout (NEP 50)
    float ga = (float)gain, fa10 = a10, fb10 = b10;
    a.forEach<float>([&](float& v, const int*) { v = (v - fa10) * ga + fb10; });
    int h = b.rows, w = b.cols;
    cv::Mat bp;
    cv::copyMakeBorder(b, bp, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    cv::Mat d;
    for (int dy = 0; dy < 3; dy++)
        for (int dx = 0; dx < 3; dx++) {
            cv::Mat e = cv::abs(a - bp(cv::Rect(dx, dy, w, h)));
            if (d.empty()) d = e;
            else d = cv::min(d, e);
        }
    cv::GaussianBlur(d, d, cv::Size(3, 3), 0);
    cv::Mat m = d > (float)level;
    m = m / 255;
    if (std::min(h, w) >= 12) cv::morphologyEx(m, m, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
    cv::Mat lab, stats, cent;
    int n = cv::connectedComponentsWithStats(m, lab, stats, cent, 8);
    if (n <= 1) return {0.0, m};
    int k = 1;
    for (int i = 2; i < n; i++)
        if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(k, cv::CC_STAT_AREA)) k = i;
    cv::Mat mask = (lab == k) / 255;
    return {(double)stats.at<int>(k, cv::CC_STAT_AREA) / (double)(w * h), mask};
}

cv::Mat thumb(const cv::Mat& image) {
    cv::Mat t, f;
    cv::resize(to_gray(image), t, THUMB, 0, 0, cv::INTER_AREA);
    t.convertTo(f, CV_32F);
    return f;
}

// count._best_sim: (similarity, region) where the patch fits best in the window
static std::pair<double, cv::Mat> best_sim(const cv::Mat& win, const cv::Mat& patch) {
    int ph = patch.rows, pw = patch.cols;
    if (win.rows < ph || win.cols < pw || patch.empty()) return {0.0, cv::Mat()};
    cv::Mat r;
    double sim;
    cv::Point loc;
    if (std_of(patch) < 4.0) {
        cv::matchTemplate(win, patch, r, cv::TM_SQDIFF);
        double mn;
        cv::minMaxLoc(r, &mn, nullptr, &loc, nullptr);
        mn = (double)(float)mn;
        sim = std::max(0.0, 1.0 - std::sqrt(std::max(0.0, mn) / (double)(pw * ph)) / 64.0);
    } else {
        cv::matchTemplate(win, patch, r, cv::TM_CCOEFF_NORMED);
        double mx;
        cv::minMaxLoc(r, nullptr, &mx, nullptr, &loc);
        sim = (double)(float)mx;
    }
    return {sim, win(cv::Rect(loc.x, loc.y, pw, ph))};
}

struct Example {
    std::string label, path;
    cv::Mat patch, thumb;
};

namespace {
std::mutex g_ex_mx;
std::map<std::string, std::vector<Example>> g_examples;
std::map<std::string, std::pair<cv::Mat, cv::Mat>> g_pictures;   // path+mtime -> (image, thumb)
}  // namespace

void release_pictures() {
    std::lock_guard<std::mutex> lk(g_ex_mx);
    g_pictures.clear();
}

static std::vector<Example> training_examples(const std::string& folder, const std::string& tool_id,
                                              const std::array<double, 4>& roi, const cv::Mat& tpl, double angle) {
    if (folder.empty()) return {};
    fs::path base = fs::u8path(folder) / "training" / fs::u8path(tool_id);
    struct F { std::string label, path; long long mtime; };
    std::vector<F> files;
    for (const char* label : {"ok", "ng"}) {
        fs::path d = base / label;
        std::error_code ec;
        if (!fs::is_directory(d, ec)) continue;
        std::vector<std::string> names;
        for (auto& e : fs::directory_iterator(d, ec)) {
            std::string ext = e.path().extension().u8string();
            for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff")
                names.push_back(e.path().u8string());
        }
        std::sort(names.begin(), names.end());
        for (auto& n : names) {
            std::error_code e2;
            auto t = fs::last_write_time(fs::u8path(n), e2);
            if (!e2) files.push_back({label, n, (long long)t.time_since_epoch().count()});
        }
    }
    if (files.empty()) return {};
    std::string key = base.u8string() + "|" + std::to_string(roi[0]) + "," + std::to_string(roi[1]) + "," + std::to_string(roi[2]) + "," +
                      std::to_string(roi[3]) + "|" + std::to_string(std::round(angle * 1000)) + "|" + std::to_string(tpl.cols) + "x" + std::to_string(tpl.rows);
    for (auto& f : files) key += "|" + f.path + "@" + std::to_string(f.mtime);
    {
        std::lock_guard<std::mutex> lk(g_ex_mx);
        auto it = g_examples.find(key);
        if (it != g_examples.end()) return it->second;
    }
    double x = roi[0], y = roi[1], w = roi[2], h = roi[3];
    int th = tpl.rows, tw = tpl.cols;
    int margin = (int)std::max(30.0, 0.5 * std::max(w, h));
    std::vector<Example> out;
    bool plain = std_of(tpl) < 1e-6;
    for (auto& f : files) {
        cv::Mat img, th_img;
        {
            std::lock_guard<std::mutex> lk(g_ex_mx);
            std::string pk = f.path + "@" + std::to_string(f.mtime);
            auto it = g_pictures.find(pk);
            if (it != g_pictures.end()) {
                img = it->second.first;
                th_img = it->second.second;
            } else {
                img = imread(f.path);
                if (!img.empty()) th_img = thumb(img);
                if (g_pictures.size() >= 40) g_pictures.erase(g_pictures.begin());
                g_pictures[pk] = {img, th_img};
            }
        }
        if (img.empty()) continue;
        double x0, y0;
        if (std::abs(angle) > 1e-6) { x0 = x - margin; y0 = y - margin; }
        else { x0 = std::max(0.0, x - margin); y0 = std::max(0.0, y - margin); }
        cv::Mat win = upright(img, roi, angle, margin);
        if (win.rows < th || win.cols < tw) continue;
        cv::Mat r;
        cv::matchTemplate(win, tpl, r, plain ? cv::TM_SQDIFF : cv::TM_CCOEFF_NORMED);
        double mn, mx;
        cv::Point mnl, mxl;
        cv::minMaxLoc(r, &mn, &mx, &mnl, &mxl);
        cv::Point l = plain ? mnl : mxl;
        int lx = l.x, ly = l.y;
        if (!plain && (double)(float)mx < 0.5) {
            lx = (int)(x - x0);
            ly = (int)(y - y0);
            lx = std::max(0, std::min(win.cols - tw, lx));
            ly = std::max(0, std::min(win.rows - th, ly));
        }
        out.push_back({f.label, f.path, np_slice(win, lx, ly, lx + tw, ly + th).clone(), th_img});
    }
    std::lock_guard<std::mutex> lk(g_ex_mx);
    if (g_examples.size() > 4000) g_examples.clear();
    g_examples[key] = out;
    return out;
}

static std::string stem(const std::string& p) { return fs::u8path(p).stem().u8string(); }

class Presence : public Tool {
public:
    using Tool::Tool;

    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        if (ctx.reference.empty() || !roi) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "roi and reference required");
        auto R = *roi;
        double x = R[0], y = R[1], w = R[2], h = R[3];
        cv::Mat tpl = upright(ctx.reference, R, angle, 0);
        int pad = pi("search_pad", 15);
        roi = std::array<double, 4>{x - pad, y - pad, w + 2 * pad, h + 2 * pad};
        std::pair<cv::Mat, cv::Point> cr;
        try {
            cr = crop(image, ctx);
        } catch (...) {
            roi = R;
            throw;
        }
        roi = R;
        cv::Mat cg = to_gray(cr.first);
        int ox = cr.second.x, oy = cr.second.y;
        double home_x, home_y;
        if (crop_M.empty()) {
            auto m = mapped_roi(ctx);
            double ex0 = m ? (*m)[0] : x, ey0 = m ? (*m)[1] : y;
            home_x = ex0 - ox;
            home_y = ey0 - oy;
        } else {
            home_x = pad;
            home_y = pad;
        }
        if (cg.rows < h || cg.cols < w) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "search window too small");
        double texture = std_of(tpl);
        double mx;
        cv::Point loc;
        std::string mode;
        cv::Mat res;
        if (texture < pf("min_texture", 4.0)) {
            cv::matchTemplate(cg, tpl, res, cv::TM_SQDIFF);
            double mn;
            cv::minMaxLoc(res, &mn, nullptr, &loc, nullptr);
            mn = (double)(float)mn;
            double rms = std::sqrt(std::max(0.0, mn) / (double)(w * h));
            mx = std::max(0.0, 1.0 - rms / 64.0);
            mode = "brightness";
        } else {
            cv::matchTemplate(cg, tpl, res, cv::TM_CCOEFF_NORMED);
            double m2;
            cv::minMaxLoc(res, nullptr, &m2, nullptr, &loc);
            mx = (double)(float)m2;
            mode = "pattern";
        }
        double thr = pf("min_score", 0.65);
        bool ok = mx >= thr;
        double ok_best = mx, ng_best = 0.0;
        std::string ng_path, same_ng;
        auto examples = training_examples(ctx.folder, id, R, tpl, angle);
        bool has_ng = false;
        for (auto& e : examples) has_ng |= e.label == "ng";
        cv::Mat mine;
        if (has_ng) {
            char kb[48];
            std::snprintf(kb, sizeof kb, "_thumb@%p", (const void*)image.data);   // ("_thumb", id(image))
            std::string key = kb;
            auto it = ctx.share.find(key);
            if (it != ctx.share.end()) mine = std::any_cast<cv::Mat>(it->second);
            else {
                mine = thumb(image);
                ctx.share[key] = mine;
            }
        }
        bool need_ok = !(mx >= thr && !has_ng);
        for (auto& e : examples) {
            if (e.label == "ok" && !need_ok) continue;
            auto [sim, region] = best_sim(cg, e.patch);
            if (e.label == "ok") {
                ok_best = std::max(ok_best, sim);
                continue;
            }
            if (sim > ng_best) {
                ng_best = sim;
                ng_path = e.path;
            }
            if (!region.empty() && !e.thumb.empty() && e.thumb.size() == mine.size()) {
                cv::Mat diff = cv::abs(e.thumb - mine);
                cv::Mat dc = diff.isContinuous() ? diff : diff.clone();
                float md = npy::mean_f32(dc.ptr<float>(), dc.total());
                cv::Mat r16, p16, ad;
                region.convertTo(r16, CV_16S);
                e.patch.convertTo(p16, CV_16S);
                ad = cv::abs(r16 - p16) > 30;
                double share = (double)cv::countNonZero(ad) / (double)ad.total();
                if ((double)md < SAME_PICTURE_GREY && share < SAME_PATCH_SHARE) same_ng = e.path;
            }
        }
        std::string why = ok ? "" : "match " + pyfmt("%.2f", mx) + " below " + py_repr(thr);
        if (!examples.empty()) {
            ok = ok_best >= thr;
            why = ok ? "" : "match " + pyfmt("%.2f", ok_best) + " below " + py_repr(thr);
            bool looks = false;
            if (!same_ng.empty()) {
                ok = false;
                why = "this is the picture of NG example " + stem(same_ng);
            } else if (ng_best > ok_best + NG_MARGIN) {
                ok = false;
                why = "looks like NG example " + stem(ng_path) + " (" + pyfmt("%.2f", ng_best) + ") more than a good part (" + pyfmt("%.2f", ok_best) + ")";
                looks = true;
            }
            mx = (looks || !same_ng.empty()) ? std::min(ok_best, std::max(0.0, 1.0 - ng_best)) : ok_best;
        }
        int th = tpl.rows, tw = tpl.cols;
        double shift = std::hypot(loc.x - home_x, loc.y - home_y);
        Json seat = Json::object();
        cv::Mat changed;
        if (ok && pb("check_seated", false)) {
            std::tie(seat, changed) = seated(cg, tpl, loc, shift, examples);
            if (seat.contains("why")) {
                ok = false;
                why = seat["why"].get<std::string>();
            }
        }
        std::string status = ok ? "pass" : "fail";
        auto to_img = [&](std::vector<cv::Point2f> pts) {
            if (crop_M.empty()) {
                for (auto& q : pts) { q.x += (float)ox; q.y += (float)oy; }
                return pts;
            }
            std::vector<cv::Point2f> o;
            cv::perspectiveTransform(pts, o, crop_M);
            return o;
        };
        std::vector<Graphic> g;
        {
            std::string err;
            bool good = guarded([&] {
                auto c = to_img({{(float)(home_x + tw / 2.0), (float)(home_y + th / 2.0)}, {(float)(loc.x + tw / 2.0), (float)(loc.y + th / 2.0)}});
                double ddx = (double)(c[1].x - c[0].x), ddy = (double)(c[1].y - c[0].y);
                g = roi_graphic(ctx, status);
                for (auto& gr : g)
                    for (auto& q : gr.points) { q.first += ddx; q.second += ddy; }
            }, err);
            if (!good) g.clear();
        }
        if (!changed.empty()) {
            int mnx = INT_MAX, mny = INT_MAX, mxx = -1, mxy = -1;
            for (int yy = 0; yy < changed.rows; yy++)
                for (int xx = 0; xx < changed.cols; xx++)
                    if (changed.at<uint8_t>(yy, xx)) {
                        mnx = std::min(mnx, xx); mxx = std::max(mxx, xx);
                        mny = std::min(mny, yy); mxy = std::max(mxy, yy);
                    }
            if (mxx >= 0) {
                auto q = to_img({{(float)(loc.x + mnx), (float)(loc.y + mny)}, {(float)(loc.x + mxx + 1), (float)(loc.y + mny)},
                                 {(float)(loc.x + mxx + 1), (float)(loc.y + mxy + 1)}, {(float)(loc.x + mnx), (float)(loc.y + mxy + 1)}});
                Graphic gr("poly", {}, "fail", "");
                for (auto& v : q) gr.points.emplace_back((double)v.x, (double)v.y);
                g.push_back(gr);
            }
        }
        if (g.empty()) {
            Graphic gr("rect", {{(double)(loc.x + ox), (double)(loc.y + oy)}, {loc.x + ox + w, loc.y + oy + h}}, status, "");
            gr.role = "region";
            g.push_back(gr);
        }
        Json vals = {{"match", py_round(mx, 4)}, {"mode", mode}, {"texture", py_round(texture, 1)}, {"shift_px", py_round(shift, 2)}};
        for (auto& [k, v] : seat.items())
            if (k != "why") vals[k] = v;
        if (!examples.empty()) {
            int nok = 0, nng = 0;
            for (auto& e : examples) (e.label == "ok" ? nok : nng)++;
            vals["good_match"] = py_round(ok_best, 4);
            vals["ng_match"] = py_round(ng_best, 4);
            vals["examples_ok"] = nok;
            vals["examples_ng"] = nng;
        }
        return ToolResult(id, type_name, ok ? "pass" : "fail", mx, vals, g, why, now_ms() - t0);
    }

private:
    std::pair<Json, cv::Mat> seated(const cv::Mat& cg, const cv::Mat& tpl, cv::Point loc, double shift, const std::vector<Example>& examples) {
        Json out = Json::object();
        int th = tpl.rows, tw = tpl.cols;
        double max_shift = pf("max_shift", 3.0);
        if (max_shift > 0) out["shift_limit_px"] = py_round(max_shift, 2);
        if (max_shift > 0 && shift > max_shift) {
            out["why"] = "found " + pyfmt("%.1f", shift) + " px from its place (allowed " + pyfmt("%.1f", max_shift) + ") - lifted, turned or not on its pads?";
            return {out, cv::Mat()};
        }
        double limit = pf("max_change", 4.0) / 100.0;
        if (limit <= 0) return {out, cv::Mat()};
        double level = pf("change_level", CHANGE_LEVEL);
        if (level == 0) level = CHANGE_LEVEL;
        cv::Mat found = np_slice(cg, loc.x, loc.y, loc.x + tw, loc.y + th);
        auto [share, mask] = change_share(found, tpl, level);
        std::optional<double> good;
        double gmax = -1;
        for (auto& e : examples)
            if (e.label == "ok" && e.patch.size() == tpl.size()) gmax = std::max(gmax, change_share(e.patch, tpl, level).first);
        bool any_ok = std::any_of(examples.begin(), examples.end(), [](const Example& e) { return e.label == "ok"; });
        if (any_ok && gmax >= 0) good = gmax;
        if (good) limit = std::max(limit, 1.5 * *good + 0.015);
        out["changed_pct"] = py_round(100.0 * share, 1);
        out["change_limit_pct"] = py_round(100.0 * limit, 1);
        if (good) out["good_changed_pct"] = py_round(100.0 * *good, 1);
        if (share > limit) {
            out["why"] = pyfmt("%.0f", 100 * share) + " % of the part looks different from the reference (allowed " + pyfmt("%.0f", 100 * limit) +
                         " %) - lifted, tilted or solder missing?";
            return {out, mask};
        }
        return {out, cv::Mat()};
    }
};
VC_REGISTER("presence", Presence)

// count.nms: indices kept, best first (scores sorted descending as numpy's
// argsort()[::-1] orders them)
std::vector<int> nms(const std::vector<std::array<float, 4>>& b, const std::vector<float>& s, double iou_thr) {
    std::vector<int> order(b.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(), [&](int i, int j) { return s[i] < s[j]; });
    std::reverse(order.begin(), order.end());
    std::vector<int> keep;
    while (!order.empty()) {
        int i = order[0];
        keep.push_back(i);
        std::vector<int> rest;
        float ax1 = b[i][0], ay1 = b[i][1], ax2 = b[i][0] + b[i][2], ay2 = b[i][1] + b[i][3];
        float ai = (ax2 - ax1) * (ay2 - ay1);
        for (size_t k = 1; k < order.size(); k++) {
            int j = order[k];
            float bx1 = b[j][0], by1 = b[j][1], bx2 = b[j][0] + b[j][2], by2 = b[j][1] + b[j][3];
            float xx1 = std::max(ax1, bx1), yy1 = std::max(ay1, by1), xx2 = std::min(ax2, bx2), yy2 = std::min(ay2, by2);
            float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
            float aj = (bx2 - bx1) * (by2 - by1);
            double iou = (double)inter / ((double)(ai + aj - inter) + 1e-9);
            if (iou <= iou_thr) rest.push_back(j);
        }
        order = rest;
    }
    return keep;
}

class CountTemplate : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        if (!has("template_roi") || ctx.reference.empty())
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "set Sample region: x, y, w, h of one example to count, on the reference image");
        auto tr = p["template_roi"];
        cv::Mat tpl = upright(ctx.reference, {tr[0].get<double>(), tr[1].get<double>(), tr[2].get<double>(), tr[3].get<double>()}, angle, 0);
        int th = tpl.rows, tw = tpl.cols;
        auto [search, off] = crop(image, ctx);
        cv::Mat sg = to_gray(search);
        if (sg.rows < th || sg.cols < tw) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "ROI smaller than template");
        double thr = pf("min_score", 0.7);
        cv::Mat res;
        cv::matchTemplate(sg, tpl, res, cv::TM_CCOEFF_NORMED);
        std::vector<std::array<float, 4>> boxes;
        std::vector<float> scores;
        for (int yy = 0; yy < res.rows; yy++)
            for (int xx = 0; xx < res.cols; xx++)
                if ((double)res.at<float>(yy, xx) >= thr) {
                    boxes.push_back({(float)xx, (float)yy, (float)tw, (float)th});
                    scores.push_back(res.at<float>(yy, xx));
                }
        auto keep = nms(boxes, scores, pf("overlap", 0.3));
        Json expect = has("expect_count") ? p["expect_count"] : Json();
        int n = (int)keep.size();
        bool ok = expect.is_null() || n == (int)expect.get<double>();
        auto g = roi_graphic(ctx, ok ? "pass" : "fail");
        for (int i : keep) {
            float x = boxes[i][0], y = boxes[i][1];
            g.emplace_back("rect", std::vector<std::pair<double, double>>{{x + off.x, y + off.y}, {x + off.x + tw, y + off.y + th}}, "pass", pyfmt("%.2f", scores[i]));
        }
        std::vector<cv::Point2f> c;
        for (int i : keep) c.emplace_back((float)(boxes[i][0] + tw / 2.0), (float)(boxes[i][1] + th / 2.0));
        if (!c.empty()) {
            if (crop_M.empty()) for (auto& q : c) { q.x += (float)off.x; q.y += (float)off.y; }
            else { std::vector<cv::Point2f> o; cv::perspectiveTransform(c, o, crop_M); c = o; }
        }
        Json matches = Json::array();
        double ssum = 0;
        for (size_t k = 0; k < keep.size(); k++) {
            matches.push_back({{"x", (double)c[k].x - tw / 2.0}, {"y", (double)c[k].y - th / 2.0}, {"w", tw}, {"h", th}, {"score", py_round(scores[keep[k]], 4)}});
            ssum += scores[keep[k]];
        }
        std::vector<double> sv;
        for (int i : keep) sv.push_back(scores[i]);
        double mean = sv.empty() ? 0.0 : npy::sum_f64(sv.data(), sv.size()) / sv.size();
        return ToolResult(id, type_name, ok ? "pass" : "fail", mean, {{"count", n}, {"expected", expect}, {"matches", matches}}, g,
                          ok ? "" : "found " + std::to_string(n) + ", expected " + py_str(expect), now_ms() - t0);
    }
};
VC_REGISTER("count_template", CountTemplate)

class CountBlob : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        auto [crp, off] = crop(image, ctx);
        std::string space = ps("space", "gray");
        cv::Mat ch;
        if (space == "hsv_v" || space == "hsv_s") {
            cv::Mat hsv;
            cv::cvtColor(crp, hsv, cv::COLOR_BGR2HSV);
            cv::extractChannel(hsv, ch, space == "hsv_v" ? 2 : 1);
        } else {
            ch = to_gray(crp);
        }
        cv::Mat m;
        if (!has("threshold") || (p["threshold"].is_string() && p["threshold"] == "otsu"))
            cv::threshold(ch, m, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        else
            cv::threshold(ch, m, (int)pf("threshold", 0), 255, cv::THRESH_BINARY);
        if (pb("invert", false)) cv::bitwise_not(m, m);
        int k = pi("open_px", 3);
        if (k > 0) cv::morphologyEx(m, m, cv::MORPH_OPEN, cv::Mat::ones(k, k, CV_8U));
        cv::Mat lab, stats, cent;
        int n_lab = cv::connectedComponentsWithStats(m, lab, stats, cent, 8);
        double lo = pf("min_area", 30), hi = pf("max_area", 1e9);
        struct B { int x, y, w, h, area; double cx, cy; };
        std::vector<B> blobs;
        for (int i = 1; i < n_lab; i++) {
            int a = stats.at<int>(i, cv::CC_STAT_AREA);
            if (lo <= a && a <= hi)
                blobs.push_back({stats.at<int>(i, 0) + off.x, stats.at<int>(i, 1) + off.y, stats.at<int>(i, 2), stats.at<int>(i, 3), a,
                                 cent.at<double>(i, 0) + off.x, cent.at<double>(i, 1) + off.y});
        }
        int tol = std::max(1, pi("row_tol", 20));
        auto fdiv = [](int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; };
        std::stable_sort(blobs.begin(), blobs.end(), [&](const B& a, const B& b) {
            int ka = fdiv(a.y, tol), kb = fdiv(b.y, tol);
            return ka != kb ? ka < kb : a.x < b.x;
        });
        Json expect = has("expect_count") ? p["expect_count"] : Json();
        int n = (int)blobs.size();
        bool ok = expect.is_null() || n == (int)expect.get<double>();
        Json bj = Json::array();
        for (auto& b : blobs) bj.push_back({{"x", b.x}, {"y", b.y}, {"w", b.w}, {"h", b.h}, {"area", b.area}, {"cx", b.cx}, {"cy", b.cy}});
        ctx.share[id] = bj;
        auto g = roi_graphic(ctx, ok ? "pass" : "fail");
        for (size_t i = 0; i < blobs.size(); i++)
            g.emplace_back("rect", std::vector<std::pair<double, double>>{{(double)blobs[i].x, (double)blobs[i].y}, {(double)(blobs[i].x + blobs[i].w), (double)(blobs[i].y + blobs[i].h)}},
                           "pass", std::to_string(i + 1));
        return ToolResult(id, type_name, ok ? "pass" : "fail", ok ? 1.0 : 0.0, {{"count", n}, {"expected", expect}, {"blobs", bj}}, g,
                          ok ? "" : "found " + std::to_string(n) + ", expected " + py_str(expect), now_ms() - t0);
    }
};
VC_REGISTER("count_blob", CountBlob)

}  // namespace vc
