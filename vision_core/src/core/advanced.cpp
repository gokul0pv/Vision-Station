#include "advanced.hpp"

#include "pyfmt.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>

namespace vc::advanced {

namespace fs = std::filesystem;

namespace {

const double AI_MIN = 0.35, SURE_PART = 0.85, SURE_TEXT = 0.80;

std::string kind_of(const std::string& type) {
    if (type == "presence") return "part";
    if (type == "ocv" || type == "ocr") return "text";
    return "";
}

std::string net_device() {
    const char* v = std::getenv("VS_DEVICE");
    std::string s = v ? v : "";
    for (auto& c : s) c = (char)std::toupper((unsigned char)c);
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    while (!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
    return s == "CPU" ? "CPU" : "GPU";
}

// advanced.region_in: (x, y, w, h) round the check's region in the picture
std::optional<std::array<double, 4>> region_in(const Tool& t, const Context& ctx) {
    try {
        if (t.shape == "polygon" && !t.points.empty()) {
            auto r = t.mapped_roi(ctx);
            if (!r) return std::nullopt;
            return std::array<double, 4>{(double)(*r)[0], (double)(*r)[1], (double)(*r)[2], (double)(*r)[3]};
        }
        if (!t.roi) return std::nullopt;
        auto q = t.mapped_quad(ctx);
        if (q.empty()) return std::nullopt;
        float x0 = q[0].x, y0 = q[0].y, x1 = q[0].x, y1 = q[0].y;
        for (auto& p : q) x0 = std::min(x0, p.x), y0 = std::min(y0, p.y), x1 = std::max(x1, p.x), y1 = std::max(y1, p.y);
        return std::array<double, 4>{(double)x0, (double)y0, (double)x1 - (double)x0, (double)y1 - (double)y0};
    } catch (...) {
        return std::nullopt;
    }
}

// numpy / Python float floor division (fmod-based, as npy_divmod)
double py_floordiv(double a, double b) {
    double mod = std::fmod(a, b);
    double div = (a - mod) / b;
    if (mod != 0 && ((b < 0) != (mod < 0))) div -= 1.0;
    double fl;
    if (div != 0) {
        fl = std::floor(div);
        if (div - fl > 0.5) fl += 1.0;
    } else fl = std::copysign(0.0, a / b);
    return fl;
}

double leeway(const Tool& t, const std::array<double, 4>& region) {
    if (t.type_name == "presence") return t.pf("search_pad", 15);
    return std::max(8.0, 0.15 * std::max(region[2], region[3]));
}

}  // namespace

Detector::Detector(const std::string& f, const Json& meta) : folder(f) {
    path = (fs::u8path(folder) / "advanced" / "model.onnx").u8string();
    if (meta.contains("tools") && meta["tools"].is_object())
        for (auto& [k, v] : meta["tools"].items()) tools[k] = (int)v.get<double>();
    zoom = meta.contains("zoom") ? meta["zoom"].get<double>() : 1.0;
    imgsz = meta.contains("imgsz") ? (int)meta["imgsz"].get<double>() : 1280;
    src = (int)py_round_int(imgsz / zoom);
    min_conf = meta.contains("min_conf") ? meta["min_conf"].get<double>() : AI_MIN;
    if (meta.contains("thresholds") && meta["thresholds"].is_object())
        for (auto& [k, v] : meta["thresholds"].items()) thresholds[k] = v.get<double>();
}

double Detector::threshold(const std::string& id) const {
    auto it = thresholds.find(id);
    return it != thresholds.end() ? it->second : min_conf;
}

std::pair<int, int> Detector::tile_for(const std::array<double, 4>& region, int W, int H) const {
    int step = std::max(1, src / 4);
    double cx = region[0] + region[2] / 2.0, cy = region[1] + region[3] / 2.0;
    auto place = [&](double c, double size, int n) {
        if (n <= src) return 0;
        int t = (int)py_round_int((c - src / 2.0) / step) * step;
        t = std::max(0, std::min(n - src, t));
        if (c - size / 2.0 < t || c + size / 2.0 > t + src) {
            long long rr = py_round_int(c - src / 2.0);
            t = (int)std::max(0LL, std::min((long long)(n - src), rr));
        }
        return t;
    };
    return {place(cx, region[2], W), place(cy, region[3], H)};
}

std::vector<std::array<float, 6>> Detector::detect_tile(const cv::Mat& frame, int tx, int ty) const {
    int x1 = std::min(frame.cols, tx + src), y1 = std::min(frame.rows, ty + src);
    cv::Mat tile = frame(cv::Rect(tx, ty, std::max(0, x1 - tx), std::max(0, y1 - ty))).clone();
    if (tile.channels() == 1) cv::cvtColor(tile, tile, cv::COLOR_GRAY2BGR);
    if (tile.rows < src || tile.cols < src)
        cv::copyMakeBorder(tile, tile, 0, src - tile.rows, 0, src - tile.cols, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    if (src != imgsz) {
        cv::Mat r;
        cv::resize(tile, r, cv::Size(imgsz, imgsz), 0, 0, src > imgsz ? cv::INTER_AREA : cv::INTER_LINEAR);
        tile = r;
    }
    std::vector<float> x((size_t)3 * imgsz * imgsz);
    for (int y = 0; y < imgsz; y++)
        for (int xx = 0; xx < imgsz; xx++) {
            auto px = tile.at<cv::Vec3b>(y, xx);
            for (int c = 0; c < 3; c++) x[(size_t)c * imgsz * imgsz + (size_t)y * imgsz + xx] = (float)px[2 - c] / 255.0f;
        }
    auto net = load_model(path, net_device());
    auto outs = net->run(x.data(), {1, 3, imgsz, imgsz});
    const Tensor& raw = outs.at(0);
    float s = (float)(imgsz / (double)src);
    std::vector<std::array<float, 6>> res;
    if (raw.shape.size() == 3 && raw.shape[2] == 6) {           // end to end: x1 y1 x2 y2 score class
        size_t N = (size_t)raw.shape[1];
        for (size_t i = 0; i < N; i++) {
            const float* d = raw.data.data() + i * 6;
            if (!(d[4] >= 0.05f)) continue;
            res.push_back({d[0] / s + (float)tx, d[1] / s + (float)ty, d[2] / s + (float)tx, d[3] / s + (float)ty, d[4], d[5]});
        }
        return res;
    }
    size_t C = (size_t)raw.shape[1], N = (size_t)raw.shape[2];
    auto at = [&](size_t r, size_t i) { return raw.data[r * N + i]; };
    std::vector<size_t> keep;
    std::vector<int64_t> cls;
    std::vector<float> conf;
    for (size_t i = 0; i < N; i++) {
        size_t best = 4;
        for (size_t r = 5; r < C; r++)
            if (at(r, i) > at(best, i)) best = r;
        float cf = at(best, i);
        if (!(cf >= 0.05f)) continue;
        keep.push_back(i);
        cls.push_back((int64_t)(best - 4));
        conf.push_back(cf);
    }
    if (keep.empty()) return res;
    std::vector<std::array<float, 4>> b;
    for (size_t k = 0; k < keep.size(); k++) {
        size_t i = keep[k];
        float cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
        std::array<float, 4> q = {(cx - w / 2) / s, (cy - h / 2) / s, (cx + w / 2) / s, (cy + h / 2) / s};
        q[0] += (float)tx, q[2] += (float)tx, q[1] += (float)ty, q[3] += (float)ty;
        b.push_back(q);
    }
    std::set<int64_t> classes(cls.begin(), cls.end());
    for (int64_t c : classes) {
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        std::vector<size_t> idx_of;
        for (size_t k = 0; k < b.size(); k++)
            if (cls[k] == c) {
                boxes.emplace_back((double)b[k][0], (double)b[k][1], (double)(b[k][2] - b[k][0]), (double)(b[k][3] - b[k][1]));
                scores.push_back(conf[k]);
                idx_of.push_back(k);
            }
        std::vector<int> idx;
        cv::dnn::NMSBoxes(boxes, scores, 0.05f, 0.5f, idx);
        for (int i : idx) {
            size_t k = idx_of[(size_t)i];
            res.push_back({b[k][0], b[k][1], b[k][2], b[k][3], conf[k], (float)c});
        }
    }
    return res;
}

bool Detector::contains(std::pair<int, int> tile, const std::array<double, 4>& r, int W, int H, double margin) const {
    double sw = std::min(src, W), sh = std::min(src, H);
    return r[0] - margin >= tile.first && r[1] - margin >= tile.second && r[0] + r[2] + margin <= tile.first + sw &&
           r[1] + r[3] + margin <= tile.second + sh;
}

std::map<std::string, std::pair<int, int>> Detector::plan(const std::vector<std::pair<std::string, std::array<double, 4>>>& regions, int W, int H) const {
    std::map<std::string, std::pair<int, int>> out;
    if (regions.empty()) return out;
    int step = std::max(1, src / 4);
    auto grid = [&](int n) {
        std::set<int> v;
        for (int x = 0; x < std::max(1, n - src) + step; x += step) v.insert(std::min(std::max(0, n - src), x));
        return std::vector<int>(v.begin(), v.end());
    };
    auto xs = grid(W), ys = grid(H);
    size_t K = regions.size();
    const double m = 16.0;
    double sw = std::min(src, W), sh = std::min(src, H);
    std::vector<std::pair<int, int>> cand;
    for (int ty : ys)
        for (int tx : xs) cand.emplace_back(tx, ty);
    size_t T = cand.size();
    std::vector<std::vector<char>> inside(T, std::vector<char>(K));
    std::vector<double> band(K);
    for (size_t k = 0; k < K; k++) {
        const auto& R = regions[k].second;
        double x0 = R[0] - m, y0 = R[1] - m, x1 = R[0] + R[2] + m, y1 = R[1] + R[3] + m;
        for (size_t t = 0; t < T; t++)
            inside[t][k] = x0 >= cand[t].first && y0 >= cand[t].second && x1 <= cand[t].first + sw && y1 <= cand[t].second + sh;
        band[k] = py_floordiv(R[1], (double)std::max(1, src / 2)) * 1e6 + R[0];
    }
    std::vector<char> left(K, 1), holdable(K, 0);
    for (size_t k = 0; k < K; k++)
        for (size_t t = 0; t < T; t++) holdable[k] |= inside[t][k];
    for (;;) {
        long long anchor = -1;
        double bmin = INFINITY;
        bool any = false;
        for (size_t k = 0; k < K; k++)
            if (left[k] && holdable[k]) {
                any = true;
                if (anchor < 0 || band[k] < bmin) bmin = band[k], anchor = (long long)k;
            }
        if (!any) break;
        long long best = -1, bestg = 0;
        for (size_t t = 0; t < T; t++) {
            long long g = -1;
            if (inside[t][(size_t)anchor]) {
                g = 0;
                for (size_t k = 0; k < K; k++) g += inside[t][k] && left[k];
            }
            if (best < 0 || g > bestg) best = (long long)t, bestg = g;
        }
        if (bestg <= 0) break;
        for (size_t k = 0; k < K; k++)
            if (inside[(size_t)best][k] && left[k]) out[regions[k].first] = cand[(size_t)best];
        for (size_t k = 0; k < K; k++)
            if (inside[(size_t)best][k]) left[k] = 0;
    }
    for (size_t k = 0; k < K; k++)
        if (left[k]) out[regions[k].first] = tile_for(regions[k].second, W, H);
    return out;
}

std::pair<double, std::optional<std::array<double, 4>>> Detector::find(const cv::Mat& frame, const std::array<double, 4>& region, int cls,
                                                                       std::map<std::pair<int, int>, std::vector<std::array<float, 6>>>& cache,
                                                                       std::optional<std::pair<int, int>> tile, double pad) const {
    int H = frame.rows, W = frame.cols;
    if (!tile || !contains(*tile, region, W, H, 0)) tile = tile_for(region, W, H);
    auto it = cache.find(*tile);
    if (it == cache.end()) it = cache.emplace(*tile, detect_tile(frame, tile->first, tile->second)).first;
    double x = region[0], y = region[1], w = region[2], h = region[3];
    double cx = x + w / 2.0, cy = y + h / 2.0, area = std::max(1.0, w * h);
    double tol_x = pad + 0.25 * w, tol_y = pad + 0.25 * h;
    double best = 0.0;
    std::optional<std::array<double, 4>> box;
    for (auto& d : it->second) {
        float x1 = d[0], y1 = d[1], x2 = d[2], y2 = d[3], s = d[4];
        if ((long long)d[5] != cls || (double)s <= best) continue;
        float dcx = (x1 + x2) / 2.0f - (float)cx, dcy = (y1 + y2) / 2.0f - (float)cy;
        if (std::abs(dcx) > (float)tol_x || std::abs(dcy) > (float)tol_y) continue;
        float a = (x2 - x1) * (y2 - y1);
        float ratio = a / (float)area;
        if (!(0.5f <= ratio && ratio <= 2.0f)) continue;
        best = s;
        box = std::array<double, 4>{x1, y1, (double)(x2 - x1), (double)(y2 - y1)};
    }
    return {best, box};
}

std::shared_ptr<Detector> load(const std::string& folder, const Json& spec) {
    Json st = spec.contains("advanced") && spec["advanced"].is_object() ? spec["advanced"] : Json::object();
    bool trained = st.contains("trained") && ((st["trained"].is_boolean() && st["trained"].get<bool>()) ||
                                              (st["trained"].is_number() && st["trained"].get<double>() != 0) ||
                                              (st["trained"].is_string() && !st["trained"].get<std::string>().empty()));
    fs::path dir = fs::u8path(folder) / "advanced";
    if (!trained || !fs::is_regular_file(dir / "model.onnx") || !fs::is_regular_file(dir / "model.json")) return nullptr;
    try {
        std::ifstream f(dir / "model.json");
        Json meta = Json::parse(f);
        return std::make_shared<Detector>(folder, meta);
    } catch (...) {
        return nullptr;
    }
}

void apply(const Detector& det, Tool& tool, ToolResult& r, const cv::Mat& frame, Context& ctx) {
    std::string kind = kind_of(tool.type_name);
    if (!det.tools.count(tool.id) || kind.empty()) return;
    if (!tool.pb("ai", true) || r.status == "error" || frame.empty()) return;
    auto region = region_in(tool, ctx);
    if (!region) return;
    using Cache = std::map<std::pair<int, int>, std::vector<std::array<float, 6>>>;
    using Plan = std::map<std::string, std::pair<int, int>>;
    if (!ctx.share.count("_advanced_tiles")) ctx.share["_advanced_tiles"] = std::make_shared<Cache>();
    auto cache = std::any_cast<std::shared_ptr<Cache>>(ctx.share["_advanced_tiles"]);
    if (!ctx.share.count("_advanced_plan")) {
        std::vector<std::pair<std::string, std::array<double, 4>>> regs;
        for (Tool* t : ctx.tool_order)
            if (det.tools.count(t->id) && !kind_of(t->type_name).empty() && t->enabled && t->pb("ai", true))
                if (auto rg = region_in(*t, ctx)) regs.emplace_back(t->id, *rg);
        ctx.share["_advanced_plan"] = std::make_shared<Plan>(det.plan(regs, frame.cols, frame.rows));
    }
    auto plan = std::any_cast<std::shared_ptr<Plan>>(ctx.share["_advanced_plan"]);
    double pad = leeway(tool, *region);
    std::optional<std::pair<int, int>> tile;
    if (plan->count(tool.id)) tile = plan->at(tool.id);
    double conf;
    try {
        conf = det.find(frame, *region, det.tools.at(tool.id), *cache, tile, pad).first;
    } catch (const std::exception& e) {
        r.values["ai"] = std::string("error: ") + e.what();
        return;
    }
    bool found = conf >= det.threshold(tool.id);
    r.values["ai_confidence"] = py_round(conf, 3);
    r.values["ai_found"] = found;
    double sure = kind == "text" ? SURE_TEXT : SURE_PART;
    if (r.status == "pass" && !found && r.score >= sure) {
        r.values["ai_overruled"] = false;
        r.values["ai_note"] = "AI unsure (" + pyfmt("%.2f", conf) + "); the check itself is sure (" + pyfmt("%.2f", r.score) + ")";
        return;
    }
    if (r.status == "pass" && !found) {
        r.status = "fail";
        r.message = std::string("AI did not find the ") + (kind == "text" ? "text" : "part") + " here (" + pyfmt("%.2f", conf) + ")";
        for (auto& g : r.graphics)
            if (g.status == "pass") g.status = "fail";
    }
}

}  // namespace vc::advanced
