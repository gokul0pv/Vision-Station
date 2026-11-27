// trained_surface (vision_station/tools/learned.py, deep_surface.py): a region's
// normal appearance learned from good parts -- the per-pixel statistical model,
// or the deep model (ResNet-18 features against a memory bank of good patches).
// Models are the .npz files Vision Station trains.
#include "../core/linalg.hpp"
#include "../core/npy.hpp"
#include "../core/npz.hpp"
#include "../core/onnx.hpp"
#include "../core/paths.hpp"
#include "../core/pyfmt.hpp"
#include "../core/tool.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <mutex>

namespace vc {

namespace fs = std::filesystem;

namespace {

struct Model {
    bool deep = false;
    int w = 0, h = 0;                   // "size" (width, height)
    double threshold = 0, k = 0;
    bool flatten = false;
    bool f64 = false;                   // mean/std stored as float64
    std::vector<float> mean, std_;      // statistical, h x w
    std::vector<double> mean64, std64;
    std::vector<float> bank;            // deep: M x D (float16 in the file, float32 here)
    int D = 0, M = 0;
};

std::shared_ptr<Model> load(const std::string& path) {
    static std::mutex m;
    static std::map<std::string, std::shared_ptr<Model>> cache;
    auto ap = fs::absolute(fs::u8path(path));
    std::string key = ap.u8string() + "|" + std::to_string(fs::last_write_time(ap).time_since_epoch().count());
    std::lock_guard<std::mutex> lk(m);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto z = npz::load(path);
    auto mdl = std::make_shared<Model>();
    auto sz = z.at("size").f32();
    mdl->w = (int)sz.at(0);
    mdl->h = (int)sz.at(1);
    mdl->threshold = z.at("threshold").scalar();
    if (z.count("kind") && z.at("kind").str() == "deep") {
        mdl->deep = true;
        const auto& b = z.at("bank");
        mdl->bank = b.f32();
        mdl->M = (int)b.shape.at(0);
        mdl->D = (int)b.shape.at(1);
    } else {
        mdl->k = z.at("k").scalar();
        mdl->flatten = z.count("flatten") ? z.at("flatten").scalar() != 0 : false;
        const auto& mn = z.at("mean");
        const auto& sd = z.at("std");
        mdl->f64 = mn.descr == "<f8" || sd.descr == "<f8";
        if (mdl->f64) {
            auto cvt = [](const npz::Array& a) {
                std::vector<double> v(a.count());
                if (a.descr == "<f8") std::memcpy(v.data(), a.data.data(), v.size() * 8);
                else {
                    auto f = a.f32();
                    for (size_t i = 0; i < v.size(); i++) v[i] = f[i];
                }
                return v;
            };
            mdl->mean64 = cvt(mn);
            mdl->std64 = cvt(sd);
        } else {
            mdl->mean = mn.f32();
            mdl->std_ = sd.f32();
        }
    }
    cache[key] = mdl;
    return mdl;
}

// learned._normalise on a float32 image
cv::Mat normalise(const cv::Mat& g32) {
    cv::Mat g = g32.isContinuous() ? g32 : g32.clone();
    const float* p = (const float*)g.data;
    float m = npy::mean_f32(p, g.total()), s = npy::std_f32(p, g.total());
    float d = (double)s > 1e-3 ? s : 1.0f;
    cv::Mat out(g.size(), CV_32F);
    float* o = (float*)out.data;
    for (size_t i = 0; i < g.total(); i++) o[i] = (p[i] - m) / d;
    return out;
}

// learned._prepare
cv::Mat prepare(const cv::Mat& patch, int w, int h, bool flatten) {
    cv::Mat g = to_gray(patch), r, b, f;
    cv::resize(g, r, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    cv::GaussianBlur(r, b, cv::Size(3, 3), 0);
    b.convertTo(f, CV_32F);
    if (flatten) {
        double sigma = std::max(4.0, std::max(w, h) / 24.0);
        cv::Mat bl;
        cv::GaussianBlur(f, bl, cv::Size(0, 0), sigma);
        f = f - bl;
    }
    return normalise(f);
}

// learned.deviation_score: (area of the largest deviating blob, mask)
int deviation_score(const cv::Mat& prep, const Model& m, double k, cv::Mat& mask) {
    mask = cv::Mat::zeros(prep.size(), CV_8U);
    const float* p = (const float*)prep.data;
    size_t n = prep.total();
    for (size_t i = 0; i < n; i++) {
        bool over;
        if (m.f64) {
            double dev = std::abs((double)p[i] - m.mean64[i]) / m.std64[i];
            over = dev > k;
        } else {
            float dev = std::abs(p[i] - m.mean[i]) / m.std_[i];
            over = dev > (float)k;
        }
        mask.data[i] = over ? 1 : 0;
    }
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::Mat::ones(2, 2, CV_8U));
    cv::Mat labels, stats, cent;
    int nl = cv::connectedComponentsWithStats(mask, labels, stats, cent);
    int area = 0;
    for (int i = 1; i < nl; i++) area = std::max(area, stats.at<int>(i, cv::CC_STAT_AREA));
    return area;
}

// ---------------------------------------------------------------- the deep model
std::string backbone_path() {
    if (const char* e = std::getenv("VS_BACKBONE")) return e;
    auto d = paths::find_dir("models/backbone");
    return d.empty() ? std::string() : (d / "resnet18_features.onnx").u8string();
}

std::string deep_device() {
    const char* v = std::getenv("VS_DEVICE");
    std::string s = v ? v : "";
    for (auto& c : s) c = (char)std::toupper((unsigned char)c);
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    while (!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
    return s == "CPU" ? "CPU" : "GPU";
}

// deep_surface.features: (vectors N x D, rows, cols)
std::vector<float> features(const cv::Mat& patch, int w, int h, int& rows, int& cols, int& D) {
    cv::Mat img = patch.channels() == 3 ? patch : cv::Mat(), r;
    if (img.empty()) cv::cvtColor(patch, img, cv::COLOR_GRAY2BGR);
    cv::resize(img, r, cv::Size(w, h), 0, 0, img.cols >= w ? cv::INTER_AREA : cv::INTER_LINEAR);
    const float MEAN[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f}, STD[3] = {0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f};
    std::vector<float> x((size_t)3 * w * h);
    for (int y = 0; y < h; y++)
        for (int xx = 0; xx < w; xx++) {
            auto px = r.at<cv::Vec3b>(y, xx);
            for (int c = 0; c < 3; c++) x[(size_t)c * w * h + (size_t)y * w + xx] = ((float)px[2 - c] - MEAN[c]) / STD[c];
        }
    auto net = load_model(backbone_path(), deep_device());
    auto out = net->run(x.data(), {1, 3, h, w});
    const Tensor &f2 = out.at(0), &f3 = out.at(1);
    int C2 = (int)f2.shape[1], C3 = (int)f3.shape[1];
    rows = (int)f2.shape[2];
    cols = (int)f2.shape[3];
    int r3 = (int)f3.shape[2], c3 = (int)f3.shape[3];
    D = C2 + C3;
    std::vector<cv::Mat> ch;
    for (int c = 0; c < C2; c++) {
        cv::Mat a(rows, cols, CV_32F, const_cast<float*>(f2.data.data()) + (size_t)c * rows * cols), b;
        cv::blur(a, b, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_REFLECT);
        ch.push_back(b);
    }
    for (int c = 0; c < C3; c++) {
        cv::Mat a(r3, c3, CV_32F, const_cast<float*>(f3.data.data()) + (size_t)c * r3 * c3), b, s;
        cv::blur(a, b, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_REFLECT);
        cv::resize(b, s, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);
        ch.push_back(s);
    }
    size_t N = (size_t)rows * cols;
    std::vector<float> v(N * D);
    for (int c = 0; c < D; c++) {
        const float* p = (const float*)ch[c].data;
        for (size_t i = 0; i < N; i++) v[i * D + c] = p[i];
    }
    return v;
}

// deep_surface.nearest: distance from each vector to its nearest bank vector
std::vector<float> nearest(const std::vector<float>& v, size_t N, const Model& m) {
    int D = m.D, M = m.M;
    std::vector<float> b2(M);
    for (int j = 0; j < M; j++) {
        std::vector<float> sq(D);
        for (int c = 0; c < D; c++) sq[c] = m.bank[(size_t)j * D + c] * m.bank[(size_t)j * D + c];
        b2[j] = npy::sum_f32(sq.data(), D);
    }
    std::vector<float> out(N);
    const size_t chunk = 4096;
    for (size_t s = 0; s < N; s += chunk) {
        int n = (int)std::min(chunk, N - s);
        const float* part = v.data() + s * D;
        auto ab = la::sgemm(false, true, n, M, D, part, D, m.bank.data(), D);
        for (int i = 0; i < n; i++) {
            std::vector<float> sq(D);
            for (int c = 0; c < D; c++) sq[c] = part[(size_t)i * D + c] * part[(size_t)i * D + c];
            float a2 = npy::sum_f32(sq.data(), D);
            float best = INFINITY;
            for (int j = 0; j < M; j++) {
                float d = (a2 + b2[j]) - 2.0f * ab[(size_t)i * M + j];
                if (d < best || j == 0) best = j == 0 ? d : std::min(best, d);
            }
            out[s + i] = std::sqrt(std::max(best, 0.0f));
        }
    }
    return out;
}

}  // namespace

class TrainedSurface : public Tool {
public:
    using Tool::Tool;

    std::string model_path(const std::string& folder) const {
        std::string p0 = has("model") ? py_str(p["model"]) : "";
        if (p0.empty()) p0 = "models/" + id + ".npz";
        if (fs::u8path(p0).is_absolute() || fs::exists(fs::u8path(p0))) return p0;
        return (fs::u8path(folder) / fs::u8path(p0)).u8string();
    }

    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        std::string path = model_path(ctx.folder);
        if (!fs::exists(fs::u8path(path)))
            return ToolResult(id, type_name, "error", 0.0, Json::object(), roi_graphic(ctx, "error"), "not trained yet - add OK images and press Train",
                              now_ms() - t0);
        auto m = load(path);
        auto [patch, o] = this->crop(image, ctx);
        if (!patch.empty()) {
            size_t n = patch.total(), clip = 0;
            for (int y = 0; y < patch.rows; y++)
                for (int x = 0; x < patch.cols; x++) {
                    int v;
                    if (patch.channels() == 3) {
                        auto px = patch.at<cv::Vec3b>(y, x);
                        v = std::max({px[0], px[1], px[2]});
                    } else v = patch.at<uint8_t>(y, x);
                    clip += v >= 254;
                }
            double clipped = (double)clip / (double)n;
            double limit = pf("max_clipped", 0.15);
            if (clipped > limit)
                return ToolResult(id, type_name, "error", 0.0, Json{{"clipped_pct", py_round(100 * clipped, 1)}}, roi_graphic(ctx, "error"),
                                  "overexposed: " + pyfmt("%.0f", 100 * clipped) + "% of the region is clipped at white - reduce the exposure or lighting",
                                  now_ms() - t0);
        }
        if (patch.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), roi_graphic(ctx, "error"), "empty region", now_ms() - t0);
        if (m->deep) return run_deep(*m, patch, o, ctx, t0);
        cv::Mat prep = prepare(patch, m->w, m->h, m->flatten);
        double k = pf("sensitivity_sigma", 0);
        if (k == 0) k = m->k;
        cv::Mat mask;
        int area = deviation_score(prep, *m, k, mask);
        double over = pf("max_defect_area", 0);
        double thr = over > 0 ? over : m->threshold;
        bool ok = area <= thr;
        auto g = roi_graphic(ctx, ok ? "pass" : "fail");
        if (!ok) {
            double sx = patch.cols / (double)m->w, sy = patch.rows / (double)m->h;
            std::vector<std::vector<cv::Point>> cnts;
            cv::findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            std::vector<std::pair<double, size_t>> ranked;
            for (size_t i = 0; i < cnts.size(); i++) ranked.emplace_back(cv::contourArea(cnts[i]), i);
            std::stable_sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.first > b.first; });
            for (size_t i = 0; i < ranked.size() && i < 3; i++) {
                cv::Rect r = cv::boundingRect(cnts[ranked[i].second]);
                g.push_back(Graphic("rect", {{o.x + r.x * sx, o.y + r.y * sy}, {o.x + (r.x + r.width) * sx, o.y + (r.y + r.height) * sy}}, "fail", "defect"));
            }
        }
        double score = std::max(0.0, 1.0 - area / std::max(1.0, 2.0 * thr));
        return ToolResult(id, type_name, ok ? "pass" : "fail", score, Json{{"defect_area", area}, {"threshold", py_round(thr, 1)}}, g,
                          ok ? "" : "defect area " + std::to_string(area) + " over " + pyfmt("%.0f", thr), now_ms() - t0);
    }

    ToolResult run_deep(const Model& m, const cv::Mat& patch, cv::Point o, Context& ctx, double t0) {
        double score, thr;
        std::vector<std::array<double, 4>> boxes;
        try {
            int rows, cols, D;
            auto v = features(patch, m.w, m.h, rows, cols, D);
            if (D != m.D) throw std::runtime_error("ValueError: the model's features (" + std::to_string(m.D) + ") do not match the network's (" + std::to_string(D) + ")");
            auto d = nearest(v, (size_t)rows * cols, m);
            cv::Mat dm(rows, cols, CV_32F, d.data()), amap;
            cv::GaussianBlur(dm, amap, cv::Size(0, 0), 1.0);
            double sens = pf("deep_sensitivity", 0);
            if (sens == 0) sens = 1.0;
            thr = m.threshold * (sens > 0 ? sens : 1.0);
            std::vector<float> flat((const float*)amap.data, (const float*)amap.data + amap.total());
            std::sort(flat.begin(), flat.end());
            int k = (int)std::max<size_t>(4, std::min<size_t>(16, flat.size() / 400));
            score = (double)npy::mean_f32(flat.data() + flat.size() - k, (size_t)k);
            if (score > thr) {
                cv::Mat mask = amap > (float)(thr * 0.85);
                mask = mask / 255;
                double sx = patch.cols / (double)cols, sy = patch.rows / (double)rows;
                std::vector<std::vector<cv::Point>> cnts;
                cv::findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                std::vector<std::pair<double, size_t>> ranked;
                for (size_t i = 0; i < cnts.size(); i++) {
                    cv::Rect r = cv::boundingRect(cnts[i]);
                    double mx;
                    cv::minMaxLoc(amap(r), nullptr, &mx);
                    ranked.emplace_back(-(double)(float)mx, i);
                }
                std::stable_sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < ranked.size() && i < 6; i++) {
                    cv::Rect r = cv::boundingRect(cnts[ranked[i].second]);
                    boxes.push_back({r.x * sx, r.y * sy, r.width * sx, r.height * sy});
                }
            }
        } catch (const std::exception& e) {
            std::string w = e.what();
            return ToolResult(id, type_name, "error", 0.0, Json::object(), roi_graphic(ctx, "error"), w.substr(0, 200), now_ms() - t0);
        }
        bool ok = score <= thr;
        auto g = roi_graphic(ctx, ok ? "pass" : "fail");
        for (auto& b : boxes) g.push_back(Graphic("rect", {{o.x + b[0], o.y + b[1]}, {o.x + b[0] + b[2], o.y + b[1] + b[3]}}, "fail", "defect"));
        return ToolResult(id, type_name, ok ? "pass" : "fail", std::max(0.0, 1.0 - score / std::max(1e-6, 2.0 * thr)),
                          Json{{"anomaly_score", py_round(score, 3)}, {"threshold", py_round(thr, 3)}, {"defect_area", py_round(score, 3)}, {"model", "deep"}}, g,
                          ok ? "" : "surface differs from the good parts (score " + pyfmt("%.2f", score) + " over " + pyfmt("%.2f", thr) + ")",
                          now_ms() - t0);
    }
};
VC_REGISTER("trained_surface", TrainedSurface)

}  // namespace vc
