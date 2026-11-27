#include "../core/npsort.hpp"
#include "ocr_engine.hpp"

#include "../core/npy.hpp"
#include "../core/onnx.hpp"
#include "../core/paths.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <cstring>
#include <functional>

namespace fs = std::filesystem;

namespace vc::ocr {

// ------------------------------------------------------------------ text
std::u32string from_utf8(const std::string& s) {
    std::u32string o;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        char32_t cp;
        int n;
        if (c < 0x80) { cp = c; n = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; n = 3; }
        else { cp = c & 0x07; n = 4; }
        for (int k = 1; k < n && i + k < s.size(); k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        o.push_back(cp);
        i += n;
    }
    return o;
}

std::string to_utf8(const std::u32string& s) {
    std::string o;
    for (char32_t c : s) {
        if (c < 0x80) o += (char)c;
        else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
        else { o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3F)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
    }
    return o;
}

bool is_space(char32_t c) {
    return (c >= 9 && c <= 13) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
bool is_alpha(char32_t c) { return c <= 0xFFFF && std::iswalpha((wint_t)c) != 0; }
bool is_alnum(char32_t c) { return c <= 0xFFFF && std::iswalnum((wint_t)c) != 0; }
bool is_upper(char32_t c) { return c <= 0xFFFF && std::iswupper((wint_t)c) != 0; }
bool is_lower(char32_t c) { return c <= 0xFFFF && std::iswlower((wint_t)c) != 0; }

std::u32string strip(const std::u32string& s) {
    size_t a = 0, b = s.size();
    while (a < b && is_space(s[a])) a++;
    while (b > a && is_space(s[b - 1])) b--;
    return s.substr(a, b - a);
}
std::u32string upper(const std::u32string& s) {
    std::u32string o = s;
    for (auto& c : o)
        if (c <= 0xFFFF) c = (char32_t)std::towupper((wint_t)c);
    return o;
}
std::u32string lower(const std::u32string& s) {
    std::u32string o = s;
    for (auto& c : o)
        if (c <= 0xFFFF) c = (char32_t)std::towlower((wint_t)c);
    return o;
}

// ------------------------------------------------------------------ boxes
namespace {

constexpr int TILE = 1280, OVERLAP = 192, REC_H = 48, REC_MAX_W = 3200, REC_BATCH = 16;

// A box as numpy holds it: four corners, float32 or float64 (which one
// decides how later arithmetic rounds, as in Vision Station).
struct Box {
    double v[4][2];
    bool f32 = true;
};

struct Models {
    std::unique_ptr<OnnxModel> det, rec;
    std::vector<std::u32string> chars;
    std::string device;
};
std::mutex g_mx;
std::unique_ptr<Models> g_models;
bool g_failed = false;

fs::path model_dir() {
    if (const char* e = std::getenv("VS_OCR_MODELS")) return fs::u8path(e);
    return paths::find_dir("models/ocr");
}

fs::path first_of(std::vector<std::string> names, bool cpu) {
    if (cpu) std::reverse(names.begin(), names.end());
    fs::path d = model_dir();
    for (auto& n : names)
        if (!d.empty() && fs::exists(d / n)) return d / n;
    return {};
}

Models* models() {
    std::lock_guard<std::mutex> lk(g_mx);
    if (g_models) return g_models.get();
    if (g_failed) return nullptr;
    bool cpu = ocr_device() == "CPU";
    fs::path det = first_of({"det_medium.onnx", "det_server.onnx", "det_small.onnx", "det_mobile.onnx"}, cpu);
    fs::path rec = first_of({"rec_medium.onnx", "rec_server.onnx", "rec_small.onnx", "rec_mobile.onnx"}, cpu);
    if (det.empty() || rec.empty()) {
        g_failed = true;
        return nullptr;
    }
    auto m = std::make_unique<Models>();
    m->det = std::make_unique<OnnxModel>(det.u8string(), ocr_device());
    m->rec = std::make_unique<OnnxModel>(rec.u8string(), m->det->device);
    m->device = m->rec->device;
    fs::path dict = rec;
    dict.replace_extension(".txt");
    if (!fs::exists(dict)) dict = model_dir() / "rec_dict.txt";
    std::ifstream f(dict, std::ios::binary);
    if (!f) throw std::runtime_error("no character dictionary beside " + rec.filename().u8string());
    std::vector<std::u32string> chars;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        chars.push_back(from_utf8(line));
    }
    while (!chars.empty() && chars.back().empty()) chars.pop_back();
    m->chars.push_back(U"blank");
    m->chars.insert(m->chars.end(), chars.begin(), chars.end());
    m->chars.push_back(U" ");
    g_models = std::move(m);
    return g_models.get();
}

// np.linalg.norm(a - b) for the box's dtype (both add two products once)
double nrm(const Box& b, int i, int j) {
    if (b.f32) {
        float dx = (float)b.v[i][0] - (float)b.v[j][0], dy = (float)b.v[i][1] - (float)b.v[j][1];
        return (double)std::sqrt((float)((double)(dx * dx) + (double)(dy * dy)));
    }
    double dx = b.v[i][0] - b.v[j][0], dy = b.v[i][1] - b.v[j][1];
    return std::sqrt(dx * dx + dy * dy);
}

std::array<double, 4> rect_of(const Box& b) {
    double x0 = b.v[0][0], y0 = b.v[0][1], x1 = x0, y1 = y0;
    for (auto& p : b.v) {
        x0 = std::min(x0, p[0]); y0 = std::min(y0, p[1]);
        x1 = std::max(x1, p[0]); y1 = std::max(y1, p[1]);
    }
    return {x0, y0, x1, y1};
}

// ocr_engine._order: corners as top-left, top-right, bottom-right, bottom-left (float32)
Box order(std::vector<std::array<double, 2>> pts) {
    std::stable_sort(pts.begin(), pts.end(), [](auto& a, auto& b) { return a[0] < b[0]; });
    std::vector<std::array<double, 2>> left(pts.begin(), pts.begin() + 2), right(pts.begin() + 2, pts.end());
    std::stable_sort(left.begin(), left.end(), [](auto& a, auto& b) { return a[1] < b[1]; });
    std::stable_sort(right.begin(), right.end(), [](auto& a, auto& b) { return a[1] < b[1]; });
    Box o;
    const std::array<double, 2>* q[4] = {&left[0], &right[0], &right[1], &left[1]};
    for (int k = 0; k < 4; k++) { o.v[k][0] = (float)(*q[k])[0]; o.v[k][1] = (float)(*q[k])[1]; }
    o.f32 = true;
    return o;
}

Box box_points(cv::RotatedRect r) {
    cv::Point2f p[4];
    r.points(p);
    return order({{p[0].x, p[0].y}, {p[1].x, p[1].y}, {p[2].x, p[2].y}, {p[3].x, p[3].y}});
}

std::vector<std::pair<Box, double>> boxes_from_map(const cv::Mat& prob, double thresh = 0.2, double box_thresh = 0.45, double unclip = 1.5, double min_side = 4) {
    cv::Mat mask = prob > (float)thresh;
    mask = mask / 255;
    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(mask, cnts, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::pair<Box, double>> out;
    for (size_t k = 0; k < cnts.size() && k < 3000; k++) {
        auto& c = cnts[k];
        cv::RotatedRect rr = cv::minAreaRect(c);
        double cx = rr.center.x, cy = rr.center.y, rw = rr.size.width, rh = rr.size.height, ang = rr.angle;
        if (std::min(rw, rh) < 2) continue;
        cv::Rect br = cv::boundingRect(c);
        cv::Mat sub = cv::Mat::zeros(br.height, br.width, CV_8U);
        std::vector<cv::Point> sh;
        for (auto& p : c) sh.emplace_back(p.x - br.x, p.y - br.y);
        std::vector<std::vector<cv::Point>> polys{sh};
        cv::fillPoly(sub, polys, cv::Scalar(1));
        double score = cv::mean(prob(br), sub)[0];
        if (score < box_thresh) continue;
        double grow = (rw * rh) * unclip / std::max(1e-6, 2.0 * (rw + rh));
        double rw2 = rw + 2.0 * grow, rh2 = rh + 2.0 * grow;
        if (std::min(rw2, rh2) < min_side) continue;
        out.emplace_back(box_points(cv::RotatedRect(cv::Point2f((float)cx, (float)cy), cv::Size2f((float)rw2, (float)rh2), (float)ang)), score);
    }
    return out;
}

std::vector<float> det_blob(const cv::Mat& img0, int& H, int& W) {
    int h = img0.rows, w = img0.cols;
    H = (int)std::ceil(h / 32.0) * 32;
    W = (int)std::ceil(w / 32.0) * 32;
    cv::Mat img = img0;
    if (H != h || W != w) cv::copyMakeBorder(img0, img, 0, H - h, 0, W - w, cv::BORDER_REPLICATE);
    const float mean[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    const float stdv[3] = {0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f};
    std::vector<float> x((size_t)3 * H * W);
    for (int y = 0; y < H; y++) {
        const uint8_t* r = img.ptr<uint8_t>(y);
        for (int xx = 0; xx < W; xx++)
            for (int c = 0; c < 3; c++) x[(size_t)c * H * W + (size_t)y * W + xx] = ((float)r[xx * 3 + c] - mean[c]) / stdv[c];
    }
    return x;
}

std::vector<std::pair<Box, double>> detect_one(Models& m, const cv::Mat& img, double scale = 1.0) {
    cv::Mat work = img;
    if (scale != 1.0) cv::resize(img, work, cv::Size(), scale, scale, scale > 1 ? cv::INTER_CUBIC : cv::INTER_AREA);
    int H, W;
    auto blob = det_blob(work, H, W);
    auto out = m.det->run(blob.data(), {1, 3, H, W});
    const Tensor& t = out[0];
    int oh = (int)t.shape[t.shape.size() - 2], ow = (int)t.shape.back();
    cv::Mat full(oh, ow, CV_32F, const_cast<float*>(t.data.data()));
    cv::Mat prob = full(cv::Rect(0, 0, std::min(ow, work.cols), std::min(oh, work.rows)));
    auto boxes = boxes_from_map(prob);
    for (auto& [b, s] : boxes)
        for (auto& p : b.v) {      // b / scale: float32 with a weak Python float
            p[0] = (float)((float)p[0] / (float)scale);
            p[1] = (float)((float)p[1] / (float)scale);
        }
    return boxes;
}

std::vector<std::array<int, 4>> tiles(int h, int w, int tile = TILE, int overlap = OVERLAP) {
    auto starts = [&](int n) {
        std::set<int> s;
        if (n <= tile) return std::vector<int>{0};
        for (int v = 0; v < n - tile; v += tile - overlap) s.insert(v);
        s.insert(n - tile);
        return std::vector<int>(s.begin(), s.end());
    };
    std::vector<std::array<int, 4>> o;
    for (int y : starts(h))
        for (int x : starts(w)) o.push_back({x, y, std::min(tile, w), std::min(tile, h)});
    return o;
}

struct Found {
    Box b;
    double s;
    int tile;
};

std::vector<std::pair<Box, double>> merge_across_tiles(const std::vector<Found>& items) {
    size_t n = items.size();
    std::vector<std::pair<Box, double>> out;
    if (n < 2) {
        for (auto& f : items) out.emplace_back(f.b, f.s);
        return out;
    }
    std::vector<std::array<float, 4>> rects;
    for (auto& f : items) {
        auto r = rect_of(f.b);
        rects.push_back({(float)r[0], (float)r[1], (float)r[2], (float)r[3]});
    }
    std::vector<int> parent(n);
    for (size_t i = 0; i < n; i++) parent[i] = (int)i;
    std::function<int(int)> find = [&](int i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    std::vector<int> ord;
    {   // np.argsort(rects[:, 0])
        std::vector<float> x0;
        for (size_t i = 0; i < n; i++) x0.push_back(rects[i][0]);
        for (auto k : npy::argsort(x0.data(), x0.size())) ord.push_back((int)k);
    }
    for (size_t pos = 0; pos < n; pos++) {
        int i = ord[pos];
        float ax0 = rects[i][0], ay0 = rects[i][1], ax1 = rects[i][2], ay1 = rects[i][3];
        float aw = ax1 - ax0, ah = ay1 - ay0;
        for (size_t q = pos + 1; q < n; q++) {
            int k = ord[q];
            float bx0 = rects[k][0], by0 = rects[k][1], bx1 = rects[k][2], by1 = rects[k][3];
            if (bx0 > ax1) break;
            if (items[i].tile == items[k].tile) continue;
            float iw = std::min(ax1, bx1) - std::max(ax0, bx0), ih = std::min(ay1, by1) - std::max(ay0, by0);
            if (iw <= 0 || ih <= 0) continue;
            float bw = bx1 - bx0, bh = by1 - by0;
            bool same = (double)(iw * ih) >= 0.5 * (double)std::min(aw * ah, bw * bh);
            if (!same) {
                if ((aw >= ah) != (bw >= bh)) continue;
                float ta = aw >= ah ? ah : aw, tb = aw >= ah ? bh : bw, across = aw >= ah ? ih : iw;
                if ((double)std::max(ta, tb) > 1.4 * (double)std::min(ta, tb) || (double)across < 0.7 * (double)std::min(ta, tb)) continue;
            }
            int ri = find(i), rk = find(k);
            if (ri != rk) parent[rk] = ri;
        }
    }
    std::vector<int> roots;
    std::map<int, std::vector<int>> groups;
    for (size_t i = 0; i < n; i++) {
        int r = find((int)i);
        if (!groups.count(r)) roots.push_back(r);
        groups[r].push_back((int)i);
    }
    for (int r : roots) {
        auto& mem = groups[r];
        if (mem.size() == 1) {
            out.emplace_back(items[mem[0]].b, items[mem[0]].s);
            continue;
        }
        std::vector<cv::Point2f> pts;
        double best = -1e300;
        for (int i : mem) {
            for (auto& p : items[i].b.v) pts.emplace_back((float)p[0], (float)p[1]);
            best = std::max(best, items[i].s);
        }
        out.emplace_back(box_points(cv::minAreaRect(pts)), best);
    }
    return out;
}

std::vector<std::pair<Box, double>> detect(Models& m, const cv::Mat& img, const std::string& detail) {
    int h = img.rows, w = img.cols, long_side = std::max(h, w);
    if (detail == "fast" || (detail != "fine" && long_side <= TILE)) {
        double scale = 1.0;
        if (long_side < 960 && detail != "fast") scale = std::min(3.0, 960.0 / long_side);
        else if (long_side > 1600) scale = 1600.0 / long_side;
        return detect_one(m, img, scale);
    }
    double up = detail == "fine" ? 2.0 : 1.0;
    cv::Mat work = img;
    if (up != 1.0) cv::resize(img, work, cv::Size(), up, up, cv::INTER_CUBIC);
    std::vector<Found> found;
    auto ts = tiles(work.rows, work.cols);
    for (size_t ti = 0; ti < ts.size(); ti++) {
        auto [x, y, tw, th] = ts[ti];
        for (auto& [b, s] : detect_one(m, work(cv::Rect(x, y, std::min(tw, work.cols - x), std::min(th, work.rows - y))))) {
            Box o;
            o.f32 = false;   // float32 + a list of ints is float64 in numpy
            for (int k = 0; k < 4; k++) {
                o.v[k][0] = (b.v[k][0] + x) / up;
                o.v[k][1] = (b.v[k][1] + y) / up;
            }
            found.push_back({o, s, (int)ti});
        }
    }
    auto fine = merge_across_tiles(found);
    double coarse_scale = std::min(1.0, (double)TILE / long_side);
    auto shortside = [](const Box& b) { return std::min(nrm(b, 0, 3), nrm(b, 0, 1)); };
    for (auto& [bb, bs] : detect_one(m, img, coarse_scale)) {
        if (!(shortside(bb) > 0.5 * OVERLAP)) continue;
        std::vector<cv::Point2f> poly;
        for (auto& p : bb.v) poly.emplace_back((float)p[0], (float)p[1]);
        std::vector<int> inside;
        for (size_t i = 0; i < fine.size(); i++) {
            const Box& b = fine[i].first;
            double mx, my;
            if (b.f32) {
                float sx = (((float)b.v[0][0] + (float)b.v[1][0]) + (float)b.v[2][0]) + (float)b.v[3][0];
                float sy = (((float)b.v[0][1] + (float)b.v[1][1]) + (float)b.v[2][1]) + (float)b.v[3][1];
                mx = sx / 4.0f;
                my = sy / 4.0f;
            } else {
                mx = (((b.v[0][0] + b.v[1][0]) + b.v[2][0]) + b.v[3][0]) / 4.0;
                my = (((b.v[0][1] + b.v[1][1]) + b.v[2][1]) + b.v[3][1]) / 4.0;
            }
            if (cv::pointPolygonTest(poly, cv::Point2f((float)mx, (float)my), false) >= 0) inside.push_back((int)i);
        }
        bool small = false;
        for (int i : inside) small |= shortside(fine[i].first) < 0.4 * shortside(bb);
        if (small) continue;
        std::vector<std::pair<Box, double>> keep;
        for (size_t i = 0; i < fine.size(); i++)
            if (std::find(inside.begin(), inside.end(), (int)i) == inside.end()) keep.push_back(fine[i]);
        keep.emplace_back(bb, bs);
        fine = keep;
    }
    return fine;
}

// ------------------------------------------------------------ recognition
cv::Mat crop_line(const cv::Mat& img, const Box& box) {
    int w = (int)std::nearbyint(std::max(nrm(box, 0, 1), nrm(box, 3, 2)));
    int h = (int)std::nearbyint(std::max(nrm(box, 0, 3), nrm(box, 1, 2)));
    w = std::max(2, w);
    h = std::max(2, h);
    std::vector<cv::Point2f> src, dst = {{0, 0}, {(float)w, 0}, {(float)w, (float)h}, {0, (float)h}};
    for (auto& p : box.v) src.emplace_back((float)p[0], (float)p[1]);
    cv::Mat M = cv::getPerspectiveTransform(src, dst), out;
    cv::warpPerspective(img, out, M, cv::Size(w, h), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    return out;
}

struct Read {
    std::u32string text;
    double conf;
    std::vector<double> confs;
};

Read decode(const float* probs, int T, int C, const std::vector<std::u32string>& chars) {
    Read r;
    int prev = 0;
    std::vector<double> confs;
    for (int t = 0; t < T; t++) {
        const float* row = probs + (size_t)t * C;
        int i = 0;
        float v = row[0];
        for (int c = 1; c < C; c++)
            if (row[c] > v) { v = row[c]; i = c; }
        if (i != 0 && i != prev && i < (int)chars.size()) {
            r.text += chars[i];
            confs.push_back((double)v);
        }
        prev = i;
    }
    r.confs = confs;
    r.conf = confs.empty() ? 0.0 : npy::sum_f64(confs.data(), confs.size()) / (double)confs.size();
    return r;
}

std::vector<Read> recognise(Models& m, const std::vector<cv::Mat>& crops) {
    std::vector<Read> out(crops.size());
    std::vector<int> ord(crops.size());
    for (size_t i = 0; i < ord.size(); i++) ord[i] = (int)i;
    auto ratio = [&](int i) { return crops[i].cols / (double)crops[i].rows; };
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return ratio(a) < ratio(b); });
    for (size_t s = 0; s < ord.size(); s += REC_BATCH) {
        std::vector<int> idx(ord.begin() + s, ord.begin() + std::min(ord.size(), s + REC_BATCH));
        double rmax = 0;
        for (int i : idx) rmax = std::max(rmax, ratio(i));
        int width = (int)std::min((double)REC_MAX_W, std::max(160.0, std::ceil(REC_H * rmax / 160.0) * 160));
        std::vector<float> blob((size_t)idx.size() * 3 * REC_H * width, 0.0f);
        for (size_t j = 0; j < idx.size(); j++) {
            const cv::Mat& c = crops[idx[j]];
            int nw = std::max(8, std::min(width, (int)std::ceil(REC_H * c.cols / (double)c.rows)));
            cv::Mat r;
            cv::resize(c, r, cv::Size(nw, REC_H), 0, 0, cv::INTER_LINEAR);
            for (int y = 0; y < REC_H; y++) {
                const uint8_t* row = r.ptr<uint8_t>(y);
                for (int x = 0; x < nw; x++)
                    for (int ch = 0; ch < 3; ch++)
                        blob[(((size_t)j * 3 + ch) * REC_H + y) * width + x] = ((float)row[x * 3 + ch] / 255.0f - 0.5f) / 0.5f;
            }
        }
        auto res = m.rec->run(blob.data(), {(int64_t)idx.size(), 3, REC_H, width});
        const Tensor& t = res[0];
        int T = (int)t.shape[1], C = (int)t.shape[2];
        for (size_t j = 0; j < idx.size(); j++) out[idx[j]] = decode(t.data.data() + j * (size_t)T * C, T, C, m.chars);
    }
    return out;
}

double score_of(const std::u32string& t, double conf) {
    if (t.empty()) return 0.0;
    int a = 0;
    for (char32_t c : t) a += is_alnum(c);
    double alnum = a / (double)t.size();
    return conf * (0.5 + 0.5 * alnum);
}

std::vector<int> turns_for(const cv::Mat& c) {
    if (c.cols >= 1.5 * c.rows) return {0, 2};
    if (c.rows >= 1.5 * c.cols) return {1, 3};
    return {0, 2, 1, 3};
}

cv::Mat rot90(const cv::Mat& c, int k) {   // np.rot90: anticlockwise
    cv::Mat o;
    if (k == 1) cv::rotate(c, o, cv::ROTATE_90_COUNTERCLOCKWISE);
    else if (k == 2) cv::rotate(c, o, cv::ROTATE_180);
    else if (k == 3) cv::rotate(c, o, cv::ROTATE_90_CLOCKWISE);
    else o = c;
    return o;
}

bool looks_like_a_pattern(const std::u32string& text) {
    std::u32string t;
    for (char32_t c : text)
        if (!is_space(c)) t += c;
    if (t.size() < 7) return false;
    std::map<char32_t, int> cnt;
    for (char32_t c : t) cnt[c]++;
    char32_t top = 0;
    int best = -1;
    for (auto& [c, n] : cnt)
        if (n > best) { best = n; top = c; }
    return is_alpha(top) && best >= 0.6 * t.size();
}

struct Cand {
    int k;
    std::u32string t;
    double c, sc;
    std::vector<double> cc;
};

std::vector<Line> read_lines(Models& m, const cv::Mat& img, const std::vector<std::pair<Box, double>>& boxes) {
    std::vector<cv::Mat> crops;
    for (auto& [b, s] : boxes) crops.push_back(crop_line(img, b));
    if (crops.empty()) return {};
    std::vector<cv::Mat> jobs;
    std::vector<std::pair<int, int>> owner;
    for (size_t i = 0; i < crops.size(); i++)
        for (int k : turns_for(crops[i])) {
            jobs.push_back(k ? rot90(crops[i], k).clone() : crops[i]);
            owner.emplace_back((int)i, k);
        }
    auto rd = recognise(m, jobs);
    std::vector<std::vector<Cand>> cands(crops.size());
    for (size_t j = 0; j < owner.size(); j++) {
        auto& r = rd[j];
        std::u32string ts = strip(r.text);
        size_t lead = 0;
        while (lead < r.text.size() && is_space(r.text[lead])) lead++;
        std::vector<double> cc;
        for (size_t q = lead; q < lead + ts.size() && q < r.confs.size(); q++) cc.push_back(r.confs[q]);
        cands[owner[j].first].push_back({owner[j].second, ts, r.conf, score_of(ts, r.conf), cc});
    }
    std::vector<std::pair<int, double>> votes;   // insertion ordered
    for (auto& cs : cands) {
        std::vector<Cand> ranked = cs;
        std::stable_sort(ranked.begin(), ranked.end(), [](const Cand& a, const Cand& b) { return -a.sc < -b.sc; });
        if (ranked.size() < 2 || ranked[0].t.size() < 3) continue;
        double margin = ranked[0].sc - ranked[1].sc;
        if (margin >= 0.15) {
            double add = margin * std::min((double)ranked[0].t.size(), 12.0);
            auto it = std::find_if(votes.begin(), votes.end(), [&](auto& v) { return v.first == ranked[0].k; });
            if (it == votes.end()) votes.emplace_back(ranked[0].k, 0.0 + add);
            else it->second += add;
        }
    }
    std::optional<int> common;
    if (!votes.empty()) {
        auto best = votes[0];
        for (auto& v : votes)
            if (v.second > best.second) best = v;
        common = best.first;
    }
    std::vector<Line> out;
    for (size_t i = 0; i < boxes.size(); i++) {
        auto& cs = cands[i];
        const Cand* best = &cs[0];
        for (auto& c : cs)
            if (c.sc > best->sc) best = &c;
        if (common && best->k != *common) {
            const Cand* same = nullptr;
            for (auto& c : cs)
                if (c.k == *common) { same = &c; break; }
            if (same && !same->t.empty() && same->sc >= best->sc - 0.15) best = same;
        }
        if (best->t.empty() || looks_like_a_pattern(best->t)) continue;
        auto r = rect_of(boxes[i].first);
        Line l;
        l.text = best->t;
        l.confidence = py_round(best->c, 4);
        for (double v : best->cc) l.char_conf.push_back(py_round(v, 4));
        for (auto& p : boxes[i].first.v) l.poly.push_back({p[0], p[1]});
        l.box = {(int)r[0], (int)r[1], (int)std::nearbyint(r[2] - r[0]), (int)std::nearbyint(r[3] - r[1])};
        l.turn = best->k * 90;
        l.upside_down = best->k == 2;
        l.detector_score = py_round(boxes[i].second, 3);
        out.push_back(std::move(l));
    }
    return out;
}

std::vector<Line> reading_order(std::vector<Line> lines) {
    if (lines.empty()) return lines;
    std::vector<int> hs;
    for (auto& l : lines) hs.push_back(std::min(l.box[2], l.box[3]));
    std::sort(hs.begin(), hs.end());
    double row = std::max(6.0, 0.7 * hs[hs.size() / 2]);
    auto key = [&](const Line& l) { return std::make_pair((long long)std::floor((l.box[1] + l.box[3] / 2.0) / row), l.box[0]); };
    std::stable_sort(lines.begin(), lines.end(), [&](const Line& a, const Line& b) { return key(a) < key(b); });
    return lines;
}

cv::Mat stretch(const cv::Mat& img) {
    std::vector<float> v(img.total() * img.channels());
    size_t k = 0;
    cv::Mat c = img.isContinuous() ? img : img.clone();
    for (size_t i = 0; i < v.size(); i++) v[k++] = (float)c.ptr<uint8_t>()[i];
    float lo = npy::percentile_f32(v, 1), hi = npy::percentile_f32(v, 99);
    cv::Mat out(img.size(), img.type());
    float den = std::max(1.0f, hi - lo);
    for (size_t i = 0; i < v.size(); i++) {
        float g = std::min(255.0f, std::max(0.0f, ((float)c.ptr<uint8_t>()[i] - lo) * 255.0f / den));
        out.ptr<uint8_t>()[i] = (uint8_t)(int)g;
    }
    return out;
}

cv::Mat clahe(const cv::Mat& img) {
    cv::Mat lab;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch;
    cv::split(lab, ch);
    cv::createCLAHE(3.0, cv::Size(8, 8))->apply(ch[0], ch[0]);
    cv::merge(ch, lab);
    cv::Mat out;
    cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
    return out;
}

// one rescue attempt: the image changed as `which` says; false when OpenCV refused
bool rescue_variant(const cv::Mat& img, int which, cv::Mat& v, double& scale, const char*& name) {
    scale = 1.0;
    try {
        if (which == 0) {
            cv::GaussianBlur(img, v, cv::Size(0, 0), 1.2);
            name = "smoothed";
        } else if (which == 1) {
            cv::Mat b;
            cv::GaussianBlur(img, b, cv::Size(0, 0), 1.5);
            v = clahe(b);
            name = "smoothed, contrast raised";
        } else {
            cv::Mat u, b;
            cv::resize(img, u, cv::Size(), 2, 2, cv::INTER_CUBIC);
            cv::GaussianBlur(u, b, cv::Size(0, 0), 2.5);
            v = stretch(b);
            scale = 2.0;
            name = "enlarged, smoothed, stretched";
        }
        return true;
    } catch (const cv::Exception&) {
        return false;
    }
}

bool rescue_read(Models& m, const cv::Mat& v, std::vector<Line>& lines) {
    try {
        lines = read_lines(m, v, detect(m, v, "auto"));
        return true;
    } catch (const cv::Exception&) {
        return false;
    }
}

std::vector<Line> rescue(Models& m, const cv::Mat& img) {
    std::vector<Line> best;
    double best_score = 0.0;
    for (int which = 0; which < 3; which++) {
        cv::Mat v;
        double scale = 1.0;
        const char* name = "";
        std::vector<Line> lines;
        if (!rescue_variant(img, which, v, scale, name) || !rescue_read(m, v, lines)) continue;
        std::vector<Line> keep;
        double score = 0;
        for (auto& l : lines)
            if (l.confidence >= 0.5) {
                keep.push_back(l);
                score += l.confidence * std::min((double)l.text.size(), 20.0);
            }
        if (score > best_score) {
            for (auto& l : keep) {
                for (auto& p : l.poly) {
                    p[0] /= scale;
                    p[1] /= scale;
                }
                for (auto& bx : l.box) bx = (int)(bx / scale);
                l.read_from = name;
            }
            best = keep;
            best_score = score;
        }
    }
    return best;
}

}  // namespace

bool available() {
    try {
        return models() != nullptr;
    } catch (...) {
        return false;
    }
}

std::string engine_name() {
    Models* m = models();
    if (!m) return "none";
    std::string d = m->device;
    for (auto& c : d) c = (char)std::tolower((unsigned char)c);
    return "pp-ocr (" + d + ")";
}

static cv::Mat as_bgr(const cv::Mat& image) {
    if (image.channels() == 1) { cv::Mat o; cv::cvtColor(image, o, cv::COLOR_GRAY2BGR); return o; }
    if (image.channels() == 4) { cv::Mat o; cv::cvtColor(image, o, cv::COLOR_BGRA2BGR); return o; }
    return image;
}

std::vector<Line> read(const cv::Mat& image, const std::string& detail) {
    if (image.empty()) return {};
    cv::Mat img = as_bgr(image);
    Models* m = models();
    if (!m) throw std::runtime_error("no text reader installed - put the OCR models in models/ocr");
    auto boxes = detect(*m, img, detail);
    if (detail == "auto" && !boxes.empty() && (long long)img.rows * img.cols <= 8000000LL) {
        std::vector<double> sh;
        for (auto& [b, s] : boxes) sh.push_back(std::min(nrm(b, 0, 3), nrm(b, 0, 1)));
        std::sort(sh.begin(), sh.end());
        if (sh[sh.size() / 2] < 26 && std::max(img.rows, img.cols) > 480) {
            auto closer = detect(*m, img, "fine");
            if (closer.size() >= boxes.size()) boxes = closer;
        }
    }
    auto lines = read_lines(*m, img, boxes);
    bool any = false;
    for (auto& l : lines) any |= l.confidence >= 0.5;
    if (!any && (long long)img.rows * img.cols <= 4000000LL) {
        auto r = rescue(*m, img);
        if (!r.empty()) lines = r;
    }
    return reading_order(lines);
}

std::optional<Line> read_single_line(const cv::Mat& image) {
    if (image.empty()) return std::nullopt;
    Models* m = models();
    if (!m) return std::nullopt;
    cv::Mat img = as_bgr(image);
    Box b;
    double w = img.cols, h = img.rows;
    double pts[4][2] = {{0, 0}, {w, 0}, {w, h}, {0, h}};
    std::memcpy(b.v, pts, sizeof pts);
    b.f32 = true;
    auto got = read_lines(*m, img, {{b, 0.0}});
    if (got.empty()) return std::nullopt;
    return got[0];
}

// ------------------------------------------------------- repeatable results
static std::optional<std::u32string> core_text(const Line& l) {
    const auto& text = l.text;
    const auto& cc = l.char_conf;
    if (cc.size() != text.size() || text.size() < 3) return std::nullopt;
    std::vector<std::pair<char32_t, double>> keep;
    for (size_t i = 0; i < text.size(); i++)
        if (is_alnum(text[i]) || text[i] == U' ' || cc[i] >= 0.85) keep.emplace_back(text[i], cc[i]);
    size_t lo = 0, hi = keep.size();
    int n = 0;
    while (n < 2 && hi - lo > 2 && (keep[lo].second < 0.85 || keep[lo].first == U' ')) {
        n += keep[lo].first != U' ';
        lo++;
    }
    n = 0;
    while (n < 2 && hi - lo > 2 && (keep[hi - 1].second < 0.85 || keep[hi - 1].first == U' ')) {
        n += keep[hi - 1].first != U' ';
        hi--;
    }
    std::u32string out;
    std::vector<double> sure;
    for (size_t i = lo; i < hi; i++) {
        out += keep[i].first;
        if (keep[i].first != U' ') sure.push_back(keep[i].second);
    }
    out = strip(out);
    if (sure.empty() || *std::min_element(sure.begin(), sure.end()) < 0.95 || out == strip(text)) return std::nullopt;
    return out;
}

std::vector<Line> clean_lines(std::vector<Line> lines) {
    for (auto& l : lines)
        if (auto c = core_text(l)) l.core = c;
    auto nospace = [](const std::u32string& t) {
        size_t n = 0;
        for (char32_t c : t) n += c != U' ';
        return n;
    };
    const Line* ref = nullptr;
    for (auto& l : lines)
        if (nospace(l.text) >= 3 && (!ref || l.confidence > ref->confidence)) ref = &l;
    if (!ref) return lines;
    double tall = std::min(ref->box[2], ref->box[3]);
    std::vector<Line> out;
    for (auto& l : lines)
        if (&l == ref || nospace(l.text) > 2 || std::min(l.box[2], l.box[3]) >= 0.5 * tall) out.push_back(l);
    return out;
}

}  // namespace vc::ocr
