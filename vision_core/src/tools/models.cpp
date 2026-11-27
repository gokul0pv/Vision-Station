// Checks that run a trained ONNX model: yolo_detect, yolo_defect
// (vision_station/tools/yolo.py) and anomaly (defect.py).
#include "../core/npsort.hpp"
#include "../core/npy.hpp"
#include "../core/onnx.hpp"
#include "../core/pyfmt.hpp"
#include "../core/tool.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <mutex>
#include <set>

namespace vc {

namespace fs = std::filesystem;

std::vector<int> nms(const std::vector<std::array<float, 4>>& b, const std::vector<float>& s, double iou_thr);   // count.cpp

namespace {

// ast.literal_eval of YOLO's names metadata: "{0: 'person', 1: 'car'}"
std::map<std::string, std::string> parse_names(const std::string& s) {
    std::map<std::string, std::string> out;
    size_t i = 0, n = s.size();
    auto ws = [&] { while (i < n && std::isspace((unsigned char)s[i])) i++; };
    auto lit = [&](std::string& v) -> bool {
        ws();
        if (i < n && (s[i] == '\'' || s[i] == '"')) {
            char q = s[i++];
            v.clear();
            while (i < n && s[i] != q) {
                if (s[i] == '\\' && i + 1 < n) i++;
                v += s[i++];
            }
            if (i >= n) return false;
            i++;
            return true;
        }
        size_t st = i;
        while (i < n && (std::isdigit((unsigned char)s[i]) || s[i] == '-')) i++;
        v = s.substr(st, i - st);
        return !v.empty();
    };
    ws();
    if (i >= n || s[i] != '{') return {};
    i++;
    for (;;) {
        ws();
        if (i < n && s[i] == '}') return out;
        std::string k, v;
        if (!lit(k)) return {};
        ws();
        if (i >= n || s[i] != ':') return {};
        i++;
        if (!lit(v)) return {};
        out[k] = v;
        ws();
        if (i < n && s[i] == ',') {
            i++;
            continue;
        }
        if (i < n && s[i] == '}') return out;
        return {};
    }
}

struct Yolo {
    std::shared_ptr<OnnxModel> model;
    int size;
    std::map<std::string, std::string> names;
};

Yolo load_yolo(const std::string& path, bool force_cpu) {
    static std::mutex m;
    static std::map<std::string, Yolo> cache;
    std::string dev = force_cpu ? "" : "GPU";
    std::string key = fs::absolute(fs::u8path(path)).u8string() + "|" + dev;
    std::lock_guard<std::mutex> lk(m);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    Yolo y;
    y.model = load_model(path, dev);
    auto meta = y.model->metadata();
    auto nm = meta.find("names");
    if (nm != meta.end()) y.names = parse_names(nm->second);
    auto shape = y.model->input_shape();
    y.size = shape.size() > 2 && shape[2] > 0 ? (int)shape[2] : 640;
    cache[key] = y;
    return y;
}

struct Det {
    int class_id;
    std::string label;
    double confidence;
    int x, y, w, h;
};

// yolo.infer
std::vector<Det> infer(const std::string& path, const cv::Mat& image, double conf, double iou, bool force_cpu) {
    Yolo y = load_yolo(path, force_cpu);
    int size = y.size, h = image.rows, w = image.cols;
    double r = std::min((double)size / h, (double)size / w);
    int nh = (int)py_round_int(h * r), nw = (int)py_round_int(w * r);
    cv::Mat resized, canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat img3 = image.channels() == 3 ? image : cv::Mat();
    if (img3.empty()) cv::cvtColor(image, img3, cv::COLOR_GRAY2BGR);
    cv::resize(img3, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    int top = (size - nh) / 2, left = (size - nw) / 2;
    resized.copyTo(canvas(cv::Rect(left, top, nw, nh)));
    std::vector<float> blob((size_t)3 * size * size);
    for (int yy = 0; yy < size; yy++)
        for (int xx = 0; xx < size; xx++) {
            auto px = canvas.at<cv::Vec3b>(yy, xx);
            for (int c = 0; c < 3; c++) blob[(size_t)c * size * size + (size_t)yy * size + xx] = (float)px[2 - c] / 255.0f;
        }
    auto outs = y.model->run(blob.data(), {1, 3, size, size});
    const Tensor& out = outs[0];
    std::vector<int64_t> dims;
    for (auto d : out.shape)
        if (d != 1) dims.push_back(d);
    if (dims.size() != 2) return {};
    int64_t R = dims[0], C = dims[1];
    bool tr = R < C;                      // (4+nc, N) -> (N, 4+nc)
    int64_t rows = tr ? C : R, cols = tr ? R : C;
    auto at = [&](int64_t i, int64_t j) { return tr ? out.data[(size_t)(j * C + i)] : out.data[(size_t)(i * C + j)]; };
    float fconf = (float)conf, fr = (float)r;
    std::vector<std::array<float, 4>> boxes;
    std::vector<float> confs;
    std::vector<int64_t> cls;
    auto fdiv = [](float a, float b) { return a / b; };
    if (cols == 6 && y.names.size() != 2) {
        for (int64_t i = 0; i < rows; i++) {
            if (!(at(i, 4) >= fconf)) continue;
            float x1 = at(i, 0), y1 = at(i, 1), x2 = at(i, 2), y2 = at(i, 3);
            confs.push_back(at(i, 4));
            cls.push_back((int64_t)at(i, 5));
            boxes.push_back({fdiv(x1 - (float)left, fr), fdiv(y1 - (float)top, fr), fdiv(x2 - x1, fr), fdiv(y2 - y1, fr)});
        }
    } else {
        if (cols <= 4) return {};
        for (int64_t i = 0; i < rows; i++) {
            int64_t best = 4;
            for (int64_t j = 5; j < cols; j++)
                if (at(i, j) > at(i, best)) best = j;
            float cf = at(i, best);
            if (!(cf >= fconf)) continue;
            float cx = at(i, 0), cy = at(i, 1), bw = at(i, 2), bh = at(i, 3);
            confs.push_back(cf);
            cls.push_back(best - 4);
            boxes.push_back({fdiv(cx - bw / 2.0f - (float)left, fr), fdiv(cy - bh / 2.0f - (float)top, fr), fdiv(bw, fr), fdiv(bh, fr)});
        }
    }
    std::vector<Det> dets;
    for (int i : nms(boxes, confs, iou)) {
        auto& b = boxes[i];
        int cid = (int)cls[i];
        std::string key = std::to_string(cid);
        auto nmit = y.names.find(key);
        std::string label = nmit != y.names.end() ? nmit->second : "class_" + key;
        dets.push_back({cid, label, py_round((double)confs[i], 4), (int)py_round_int(b[0]), (int)py_round_int(b[1]),
                        (int)py_round_int(b[2]), (int)py_round_int(b[3])});
    }
    auto fl = [](int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; };
    std::stable_sort(dets.begin(), dets.end(), [&](const Det& a, const Det& b) {
        int ka = fl(a.y, 24), kb = fl(b.y, 24);
        return ka != kb ? ka < kb : a.x < b.x;
    });
    return dets;
}

Json det_json(const Det& d) {
    return Json{{"class_id", d.class_id}, {"label", d.label}, {"confidence", d.confidence}, {"x", d.x}, {"y", d.y}, {"w", d.w}, {"h", d.h}};
}

// yolo._detections: (detections in picture coordinates) or an error result
std::optional<ToolResult> detections(Tool& t, const cv::Mat& image, Context& ctx, double t0, std::vector<Det>& dets) {
    std::string model_p = t.has("model") ? py_str(t.p["model"]) : "";
    std::string path = model_p;
    if (!path.empty() && !fs::u8path(path).is_absolute()) path = (fs::u8path(ctx.folder) / fs::u8path(path)).u8string();
    if (path.empty() || !fs::exists(fs::u8path(path))) {
        std::string shown = model_p.empty() ? "models/best.onnx" : model_p;
        std::string base = fs::u8path(shown).filename().u8string();
        return ToolResult(t.id, t.type_name, "error", 0.0, Json::object(), t.roi_graphic(ctx, "error"),
                          "no trained model yet - this check needs a YOLO model trained on your parts (see Train on Colab in the manual); save it as models\\" +
                              base + " in this job's folder",
                          now_ms() - t0);
    }
    auto [crop, o] = t.crop(image, ctx);
    try {
        dets = infer(path, crop, t.pf("confidence", 0.35), t.pf("iou", 0.45), !t.pb("use_gpu", false));
    } catch (const std::exception& e) {
        return ToolResult(t.id, t.type_name, "error", 0.0, Json::object(), {}, e.what());
    }
    if (t.has("only_classes")) {
        const Json& only = t.p["only_classes"];
        bool any = (only.is_array() && !only.empty()) || (only.is_string() && !only.get<std::string>().empty());
        if (any) {
            std::vector<Det> keep;
            for (auto& d : dets) {
                bool in = false;
                if (only.is_array()) {
                    for (auto& q : only) in |= q.is_string() && q.get<std::string>() == d.label;
                } else in = only.get<std::string>().find(d.label) != std::string::npos;
                if (in) keep.push_back(d);
            }
            dets = keep;
        }
    }
    int min_area = t.pi("min_area", 0);
    std::vector<Det> keep;
    for (auto& d : dets)
        if (d.w * d.h >= min_area) keep.push_back(d);
    dets = keep;
    for (auto& d : dets) {
        d.x += o.x;
        d.y += o.y;
    }
    return std::nullopt;
}

Graphic rect_g(const Det& d, const std::string& st) {
    return Graphic("rect", {{(double)d.x, (double)d.y}, {(double)(d.x + d.w), (double)(d.y + d.h)}}, st, d.label + " " + pyfmt("%.2f", d.confidence));
}

}  // namespace

// ---------------------------------------------------------------- yolo detect
class YoloDetect : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        std::vector<Det> dets;
        if (auto err = detections(*this, image, ctx, t0, dets)) return *err;
        Json counts = Json::object();
        for (auto& d : dets) counts[d.label] = counts.value(d.label, 0) + 1;
        Json expect = has("expect_counts") && p["expect_counts"].is_object() ? p["expect_counts"] : Json::object();
        Json total_expect = has("expect_total") ? p["expect_total"] : Json();
        auto empty_total = [&] {
            return total_expect.is_null() || (total_expect.is_string() && total_expect.get<std::string>().empty()) ||
                   (total_expect.is_number() && total_expect.get<double>() == 0);
        };
        auto as_int = [](const Json& v) { return v.is_string() ? std::stoi(v.get<std::string>()) : (int)v.get<double>(); };
        std::vector<std::string> problems;
        for (auto& [name, want] : expect.items()) {
            int got = counts.value(name, 0);
            if (got != as_int(want)) problems.push_back(name + ": found " + std::to_string(got) + ", expected " + py_str(want));
        }
        if (!empty_total() && (int)dets.size() != as_int(total_expect))
            problems.push_back("total " + std::to_string(dets.size()) + ", expected " + py_str(total_expect));
        if (expect.empty() && empty_total() && dets.empty()) problems.push_back("nothing detected");
        bool reject = pb("reject_when_found", false);
        if (reject) {
            int allowed = pi("max_defects", 0);
            problems.clear();
            if ((int)dets.size() > allowed) {
                std::string s;
                bool first = true;
                for (auto& [k, v] : counts.items()) {
                    s += (first ? "" : ", ") + k + " x" + std::to_string(v.get<int>());
                    first = false;
                }
                problems.push_back(std::to_string(dets.size()) + " found: " + s);
            }
        }
        bool ok = problems.empty();
        auto g = roi_graphic(ctx, ok ? "pass" : "fail");
        std::set<std::string> bad;
        for (auto& [name, want] : expect.items())
            if (counts.value(name, 0) != as_int(want)) bad.insert(name);
        for (auto& d : dets) g.push_back(rect_g(d, (reject || bad.count(d.label)) ? "fail" : "pass"));
        double score = 0.0;
        if (!dets.empty()) {
            std::vector<double> c;
            for (auto& d : dets) c.push_back(d.confidence);
            score = npy::sum_f64(c.data(), c.size()) / (double)c.size();
        }
        Json dl = Json::array();
        for (auto& d : dets) dl.push_back(det_json(d));
        std::string msg;
        for (size_t i = 0; i < problems.size(); i++) msg += (i ? "; " : "") + problems[i];
        return ToolResult(id, type_name, ok ? "pass" : "fail", score,
                          Json{{"total", (int)dets.size()}, {"counts", counts}, {"mean_confidence", py_round(score, 3)}, {"detections", dl}}, g, msg,
                          now_ms() - t0);
    }
};
VC_REGISTER("yolo_detect", YoloDetect)

// ---------------------------------------------------------------- yolo defect
class YoloDefect : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        std::vector<Det> dets;
        if (auto err = detections(*this, image, ctx, t0, dets)) return *err;
        int allowed = pi("max_defects", 0);
        bool ok = (int)dets.size() <= allowed;
        Json counts = Json::object();
        for (auto& d : dets) counts[d.label] = counts.value(d.label, 0) + 1;
        auto g = roi_graphic(ctx, ok ? "pass" : "fail");
        for (auto& d : dets) g.push_back(rect_g(d, "fail"));
        double worst = 0.0;
        for (auto& d : dets) worst = std::max(worst, d.confidence);
        Json dl = Json::array();
        for (auto& d : dets) dl.push_back(det_json(d));
        std::string msg;
        if (!ok) {
            msg = std::to_string(dets.size()) + " defect(s): ";
            bool first = true;
            for (auto& [k, v] : counts.items()) {
                msg += (first ? "" : ", ") + k + "\xc3\x97" + std::to_string(v.get<int>());
                first = false;
            }
        }
        return ToolResult(id, type_name, ok ? "pass" : "fail", worst,
                          Json{{"defect_count", (int)dets.size()}, {"counts", counts}, {"worst_confidence", py_round(worst, 3)}, {"defects", dl}}, g,
                          msg, now_ms() - t0);
    }
};
VC_REGISTER("yolo_defect", YoloDefect)

// ---------------------------------------------------------------- anomaly (earlier recipes)
class Anomaly : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        std::string path = has("model") ? py_str(p["model"]) : "None";
        if (!has("model") || path.empty() || !fs::exists(fs::u8path(path)))
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "model not found: " + (has("model") ? path : std::string("None")));
        std::shared_ptr<OnnxModel> sess;
        try {
            sess = load_model(path, pb("use_gpu", false) ? "GPU" : "");
        } catch (const std::exception& e) {
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, std::string("inference runtime: ") + e.what());
        }
        auto [crop, o] = this->crop(image, ctx);
        int size = pi("input_size", 256);
        cv::Mat sm;
        cv::resize(crop, sm, cv::Size(size, size));
        const double mean[3] = {0.485, 0.456, 0.406}, sd[3] = {0.229, 0.224, 0.225};
        std::vector<float> inp((size_t)3 * size * size);
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++) {
                auto px = sm.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; c++) {
                    float f = (float)px[2 - c] / 255.0f;
                    inp[(size_t)c * size * size + (size_t)y * size + x] = (float)(((double)f - mean[c]) / sd[c]);
                }
            }
        auto out = sess->run(inp.data(), {1, 3, size, size});
        const Tensor& t = out[0];
        std::vector<int64_t> dims;
        for (auto d : t.shape)
            if (d != 1) dims.push_back(d);
        double thr = pf("threshold", 0.5);
        float mx = *std::max_element(t.data.begin(), t.data.end());
        double score = mx;
        auto g = roi_graphic(ctx, score < thr ? "pass" : "fail");
        if (dims.size() == 2) {
            cv::Mat heat((int)dims[0], (int)dims[1], CV_32F, const_cast<float*>(t.data.data())), hm, mask;
            cv::resize(heat, hm, crop.size());
            mask = hm >= (float)thr;
            std::vector<std::vector<cv::Point>> cnts;
            cv::findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            for (auto& c : cnts) {
                if (cv::contourArea(c) < pf("min_area", 20)) continue;
                cv::Rect b = cv::boundingRect(c);
                g.push_back(Graphic("rect", {{(double)(b.x + o.x), (double)(b.y + o.y)}, {(double)(b.x + o.x + b.width), (double)(b.y + o.y + b.height)}},
                                    "fail", "anomaly"));
            }
        }
        bool ok = score < thr;
        return ToolResult(id, type_name, ok ? "pass" : "fail", score, Json{{"anomaly_score", py_round(score, 4)}, {"threshold", thr}}, g,
                          ok ? "" : "anomaly score " + pyfmt("%.3f", score) + " over " + py_repr(thr), now_ms() - t0);
    }
};
VC_REGISTER("anomaly", Anomaly)

}  // namespace vc
