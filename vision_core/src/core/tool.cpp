#include "tool.hpp"
#include "npy.hpp"

#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>

namespace vc {

namespace {

constexpr double kPi = 3.14159265358979323846;

// numpy slicing a[y0:y1, x0:x1] for non-negative starts: clipped, maybe empty
cv::Mat np_slice(const cv::Mat& m, int x0, int y0, int x1, int y1) {
    x0 = std::clamp(x0, 0, m.cols);
    y0 = std::clamp(y0, 0, m.rows);
    x1 = std::clamp(x1, x0, m.cols);
    y1 = std::clamp(y1, y0, m.rows);
    return m(cv::Rect(x0, y0, x1 - x0, y1 - y0));
}

std::vector<cv::Point2f> transform2(const std::vector<cv::Point2f>& pts, const cv::Mat& fixture) {
    if (fixture.empty() || pts.empty()) return pts;
    std::vector<cv::Point2f> out;
    cv::transform(pts, out, fixture.rowRange(0, 2));
    return out;
}

float norm2(cv::Point2f a, cv::Point2f b) {
    float d[2] = {a.x - b.x, a.y - b.y};
    return npy::norm_f32(d, 2);
}

}  // namespace

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

cv::Mat to_gray(const cv::Mat& m) {
    if (m.channels() == 1) return m;
    cv::Mat g;
    cv::cvtColor(m, g, cv::COLOR_BGR2GRAY);
    return g;
}

Graphic abs_graphic(const std::string& kind, const std::vector<cv::Point2f>& pts, const std::string& status, const std::string& label) {
    Graphic g;
    g.kind = kind;
    for (auto& p : pts) g.points.emplace_back((double)p.x, (double)p.y);
    g.status = status;
    g.label = label;
    g.absolute = true;
    return g;
}

std::map<std::string, ToolFactory>& registry() {
    static std::map<std::string, ToolFactory> r;
    return r;
}

std::unique_ptr<Tool> make_tool(const Json& spec) {
    std::string t = spec.value("type", "");
    auto& reg = registry();
    auto it = reg.find(t);
    if (it == reg.end()) throw std::runtime_error("unknown tool type '" + t + "'");
    return it->second(spec.value("id", ""), spec.contains("params") ? spec["params"] : Json::object());
}

Tool::Tool(std::string id_, Json params) : id(std::move(id_)), p(std::move(params)) {
    if (p.contains("roi") && p["roi"].is_array() && p["roi"].size() >= 4) {
        std::array<double, 4> r{};
        for (int i = 0; i < 4; i++) r[i] = p["roi"][i].get<double>();
        roi = r;
    }
    if (p.contains("angle") && p["angle"].is_number()) angle = p["angle"].get<double>();
    if (p.contains("shape") && p["shape"].is_string()) shape = p["shape"].get<std::string>();
    if (p.contains("points") && p["points"].is_array())
        for (auto& q : p["points"])
            if (q.is_array() && q.size() >= 2) points.push_back({q[0].get<double>(), q[1].get<double>()});
    enabled = p.contains("enabled") ? p["enabled"].get<bool>() : true;
    critical = p.contains("critical") ? p["critical"].get<bool>() : true;
}

bool Tool::has(const char* k) const { return p.contains(k) && !p[k].is_null(); }
double Tool::pf(const char* k, double d) const {
    if (!p.contains(k) || p[k].is_null()) return d;
    if (p[k].is_number()) return p[k].get<double>();
    if (p[k].is_boolean()) return p[k].get<bool>() ? 1.0 : 0.0;
    if (p[k].is_string()) return std::stod(p[k].get<std::string>());
    return d;
}
int Tool::pi(const char* k, int d) const {
    if (!p.contains(k) || p[k].is_null()) return d;
    if (p[k].is_number_integer()) return p[k].get<int>();
    return (int)pf(k, d);   // Python int(float) truncates
}
bool Tool::pb(const char* k, bool d) const {
    if (!p.contains(k) || p[k].is_null()) return d;
    const auto& v = p[k];
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v.get<double>() != 0.0;
    if (v.is_string()) return !v.get<std::string>().empty();
    if (v.is_array() || v.is_object()) return !v.empty();
    return d;
}
std::string Tool::ps(const char* k, const std::string& d) const {
    if (!p.contains(k) || !p[k].is_string()) return d;
    return p[k].get<std::string>();
}

std::vector<cv::Point2f> Tool::polygon_points(const Context& ctx) const {
    if (points.size() < 3) return {};
    std::vector<cv::Point2f> pts;
    for (auto& q : points) pts.emplace_back((float)q[0], (float)q[1]);
    return transform2(pts, ctx.fixture);
}

std::optional<std::array<int, 4>> Tool::mapped_roi(const Context& ctx) const {
    if (shape == "polygon" && !points.empty()) {
        auto pts = polygon_points(ctx);
        if (!pts.empty()) {
            float x0 = pts[0].x, y0 = pts[0].y, x1 = pts[0].x, y1 = pts[0].y;
            for (auto& q : pts) {
                x0 = std::min(x0, q.x); y0 = std::min(y0, q.y);
                x1 = std::max(x1, q.x); y1 = std::max(y1, q.y);
            }
            return std::array<int, 4>{(int)x0, (int)y0, (int)(x1 - x0), (int)(y1 - y0)};
        }
    }
    if (!roi) return std::nullopt;
    auto [x, y, w, h] = *roi;
    std::vector<cv::Point2f> c = {{(float)x, (float)y}, {(float)(x + w), (float)y}, {(float)(x + w), (float)(y + h)}, {(float)x, (float)(y + h)}};
    c = transform2(c, ctx.fixture);
    float x0 = c[0].x, y0 = c[0].y, x1 = c[0].x, y1 = c[0].y;
    for (auto& q : c) {
        x0 = std::min(x0, q.x); y0 = std::min(y0, q.y);
        x1 = std::max(x1, q.x); y1 = std::max(y1, q.y);
    }
    return std::array<int, 4>{(int)py_round_int(x0), (int)py_round_int(y0),
                              (int)py_round_int((double)(x1 - x0)), (int)py_round_int((double)(y1 - y0))};
}

std::vector<cv::Point2f> Tool::mapped_quad(const Context& ctx) const {
    if (!roi) return {};
    auto [x, y, w, h] = *roi;
    double cx = x + w / 2.0, cy = y + h / 2.0;
    double a = angle * kPi / 180.0;
    double ca = std::cos(a), sa = std::sin(a);
    std::vector<cv::Point2f> pts;
    const double d[4][2] = {{-w / 2, -h / 2}, {w / 2, -h / 2}, {w / 2, h / 2}, {-w / 2, h / 2}};
    for (auto& v : d) pts.emplace_back((float)(cx + v[0] * ca - v[1] * sa), (float)(cy + v[0] * sa + v[1] * ca));
    return transform2(pts, ctx.fixture);
}

std::array<int, 4> Tool::points_bbox() const {
    float x0 = (float)points[0][0], y0 = (float)points[0][1], x1 = x0, y1 = y0;
    for (auto& q : points) {
        x0 = std::min(x0, (float)q[0]); y0 = std::min(y0, (float)q[1]);
        x1 = std::max(x1, (float)q[0]); y1 = std::max(y1, (float)q[1]);
    }
    double x = x0, yy = y0;
    return {(int)x, (int)yy, (int)((double)x1 - x), (int)((double)y1 - yy)};
}

cv::Mat Tool::apply_shape(const cv::Mat& patch) {
    crop_mask.release();
    if (patch.empty()) return patch;
    if (shape != "ellipse" && shape != "polygon") return patch;
    int h = patch.rows, w = patch.cols;
    cv::Mat mask = cv::Mat::zeros(h, w, CV_8U);
    if (shape == "polygon") {
        if (points.size() < 3) return patch;
        int bx, by;
        if (patch_origin) { bx = (*patch_origin)[0]; by = (*patch_origin)[1]; }
        else { auto b = points_bbox(); bx = b[0]; by = b[1]; }
        std::vector<cv::Point> pts;
        for (auto& q : points) pts.emplace_back((int)(q[0] - bx), (int)(q[1] - by));
        std::vector<std::vector<cv::Point>> polys{pts};
        cv::fillPoly(mask, polys, cv::Scalar(255));
    } else {
        cv::ellipse(mask, cv::Point(w / 2, h / 2), cv::Size(std::max(1, w / 2), std::max(1, h / 2)), 0, 0, 360, cv::Scalar(255), -1);
    }
    crop_mask = mask;
    int nin = cv::countNonZero(mask);
    if (!nin) return patch;
    cv::Mat out = patch.clone();
    int cn = patch.channels();
    std::vector<double> sum(cn, 0.0);
    for (int yy = 0; yy < h; yy++) {
        const uint8_t* mr = mask.ptr<uint8_t>(yy);
        const uint8_t* pr = patch.ptr<uint8_t>(yy);
        for (int xx = 0; xx < w; xx++)
            if (mr[xx])
                for (int c = 0; c < cn; c++) sum[c] += pr[xx * cn + c];
    }
    // u8 sums are exact in double whatever the order; mean -> astype(uint8) truncates
    std::vector<uint8_t> fill(cn);
    for (int c = 0; c < cn; c++) fill[c] = (uint8_t)(int)(sum[c] / nin);
    for (int yy = 0; yy < h; yy++) {
        const uint8_t* mr = mask.ptr<uint8_t>(yy);
        uint8_t* orow = out.ptr<uint8_t>(yy);
        for (int xx = 0; xx < w; xx++)
            if (!mr[xx])
                for (int c = 0; c < cn; c++) orow[xx * cn + c] = fill[c];
    }
    return out;
}

std::pair<cv::Mat, cv::Point> Tool::crop(const cv::Mat& image, const Context& ctx) {
    crop_M.release();
    poly_mask.release();
    if (shape == "polygon") {
        auto pts = polygon_points(ctx);
        if (pts.empty()) return {image, {0, 0}};
        float mnx = pts[0].x, mny = pts[0].y, mxx = pts[0].x, mxy = pts[0].y;
        for (auto& q : pts) {
            mnx = std::min(mnx, q.x); mny = std::min(mny, q.y);
            mxx = std::max(mxx, q.x); mxy = std::max(mxy, q.y);
        }
        int x0 = (int)std::max(0.0f, std::floor(mnx)), y0 = (int)std::max(0.0f, std::floor(mny));
        int x1 = (int)std::min((float)image.cols, std::ceil(mxx)), y1 = (int)std::min((float)image.rows, std::ceil(mxy));
        if (x1 - x0 < 2 || y1 - y0 < 2) return {image, {0, 0}};
        int w = x1 - x0, h = y1 - y0;
        cv::Mat m = cv::Mat::zeros(h, w, CV_8U);
        std::vector<cv::Point> ip;
        // numpy: float32 points minus a list of ints is computed in float64
        for (auto& q : pts) ip.emplace_back((int)((double)q.x - x0), (int)((double)q.y - y0));
        std::vector<std::vector<cv::Point>> polys{ip};
        cv::fillPoly(m, polys, cv::Scalar(255));
        poly_mask = m;
        return {apply_shape(np_slice(image, x0, y0, x0 + w, y0 + h)), {x0, y0}};
    }
    auto r = mapped_roi(ctx);
    if (!r) return {image, {0, 0}};
    double fix_rot = 0.0;
    bool shifted = false;
    if (!ctx.fixture.empty()) {
        // float32 from align, float64 from fixture_npoint -- as the Python keeps them
        auto q = [&](int i, int j) {
            return ctx.fixture.type() == CV_64F ? ctx.fixture.at<double>(i, j) : (double)ctx.fixture.at<float>(i, j);
        };
        fix_rot = std::atan2(q(1, 0), q(0, 0)) * 180.0 / kPi;
        if (roi) {
            double tx = q(0, 0) * (*roi)[0] + q(0, 1) * (*roi)[1] + q(0, 2);
            double ty = q(1, 0) * (*roi)[0] + q(1, 1) * (*roi)[1] + q(1, 2);
            shifted = std::abs(tx - std::nearbyint(tx)) > 0.01 || std::abs(ty - std::nearbyint(ty)) > 0.01;
        }
    }
    bool rotated = std::abs(angle) > 1e-6 || std::abs(fix_rot) > 0.05 || shifted;
    if (!rotated) {
        auto [x, y, w, h] = *r;
        x = std::max(0, x);
        y = std::max(0, y);
        w = std::max(1, std::min(w, image.cols - x));
        h = std::max(1, std::min(h, image.rows - y));
        patch_origin = std::array<int, 4>{x, y, w, h};
        return {apply_shape(np_slice(image, x, y, x + w, y + h)), {x, y}};
    }
    auto quad = mapped_quad(ctx);
    int w = (int)py_round_int(std::max(norm2(quad[1], quad[0]), norm2(quad[2], quad[3])));
    int h = (int)py_round_int(std::max(norm2(quad[3], quad[0]), norm2(quad[2], quad[1])));
    w = std::max(1, w);
    h = std::max(1, h);
    std::vector<cv::Point2f> dst = {{0, 0}, {(float)(w - 1), 0}, {(float)(w - 1), (float)(h - 1)}, {0, (float)(h - 1)}};
    cv::Mat M = cv::getPerspectiveTransform(quad, dst);
    cv::Mat patch;
    cv::warpPerspective(image, patch, M, cv::Size(w, h), cv::INTER_LINEAR);
    crop_M = cv::getPerspectiveTransform(dst, quad);
    float mnx = quad[0].x, mny = quad[0].y;
    for (auto& q : quad) { mnx = std::min(mnx, q.x); mny = std::min(mny, q.y); }
    return {apply_shape(patch), {(int)py_round_int(mnx), (int)py_round_int(mny)}};
}

ToolResult& Tool::correct_graphics(ToolResult& r, const Context& ctx) const {
    if (crop_M.empty() || r.graphics.empty()) return r;
    if (!mapped_roi(ctx)) return r;
    auto quad = mapped_quad(ctx);
    float mnx = quad[0].x, mny = quad[0].y;
    for (auto& q : quad) { mnx = std::min(mnx, q.x); mny = std::min(mny, q.y); }
    double ox = (double)py_round_int(mnx), oy = (double)py_round_int(mny);
    for (auto& g : r.graphics) {
        if (g.absolute) continue;
        if (g.kind == "rect" && g.points.size() == 2) {
            double x0 = g.points[0].first - ox, y0 = g.points[0].second - oy;
            double x1 = g.points[1].first - ox, y1 = g.points[1].second - oy;
            std::vector<cv::Point2f> box = {{(float)x0, (float)y0}, {(float)x1, (float)y0}, {(float)x1, (float)y1}, {(float)x0, (float)y1}}, out;
            cv::perspectiveTransform(box, out, crop_M);
            g.kind = "poly";
            g.points.clear();
            for (auto& q : out) g.points.emplace_back((double)q.x, (double)q.y);
        } else if (!g.points.empty()) {
            std::vector<cv::Point2f> pts, out;
            for (auto& q : g.points) pts.emplace_back((float)(q.first - ox), (float)(q.second - oy));
            cv::perspectiveTransform(pts, out, crop_M);
            g.points.clear();
            for (auto& q : out) g.points.emplace_back((double)q.x, (double)q.y);
        }
    }
    return r;
}

std::vector<Graphic> Tool::roi_graphic(const Context& ctx, const std::string& status, const std::string& label) const {
    auto out = roi_graphic_raw(ctx, status, label);
    for (auto& g : out) g.role = "region";
    return out;
}

std::vector<Graphic> Tool::roi_graphic_raw(const Context& ctx, const std::string& status, const std::string& label) const {
    const std::string lb = label.empty() ? id : label;
    if (shape == "polygon") {
        auto pts = polygon_points(ctx);
        if (pts.empty()) return {};
        return {abs_graphic("poly", pts, status, lb)};
    }
    if (!roi) return {};
    if (shape == "ellipse") {
        auto [x, y, w, h] = *roi;
        double cx = x + w / 2.0, cy = y + h / 2.0, rx = w / 2.0, ry = h / 2.0;
        double a = angle * kPi / 180.0, ca = std::cos(a), sa = std::sin(a);
        std::vector<cv::Point2f> pts;
        for (int i = 0; i < 48; i++) {
            double th = 2.0 * kPi * i / 48.0;
            double dx = rx * std::cos(th), dy = ry * std::sin(th);
            pts.emplace_back((float)(cx + dx * ca - dy * sa), (float)(cy + dx * sa + dy * ca));
        }
        return {abs_graphic("poly", transform2(pts, ctx.fixture), status, lb)};
    }
    if (std::abs(angle) > 1e-6) return {abs_graphic("poly", mapped_quad(ctx), status, lb)};
    auto r = mapped_roi(ctx);
    auto [x, y, w, h] = *r;
    Graphic g("rect", {{(double)x, (double)y}, {(double)(x + w), (double)(y + h)}}, status, lb);
    g.absolute = true;
    return {g};
}

}  // namespace vc
