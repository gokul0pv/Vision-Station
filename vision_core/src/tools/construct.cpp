// vision_station/tools/construct_tools.py: fixture_npoint, calibrate_npoint,
// construct, bead, data_analysis.
#include "geom.hpp"

#include "../core/linalg.hpp"
#include "../core/npy.hpp"
#include "../core/pyfmt.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <mutex>
#include <tuple>

namespace vc {

namespace {

constexpr double kPi = 3.141592653589793;
constexpr double kDeg = 180.0 / kPi;
using D2 = std::array<double, 2>;

std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string r;
    for (size_t i = 0; i < v.size(); i++) r += (i ? sep : "") + v[i];
    return r;
}

std::string strip(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// construct_tools._ids
std::vector<std::string> ids_of(const Json& v) {
    std::vector<std::string> out;
    if (v.is_array()) {
        for (auto& x : v) {
            std::string s = strip(py_str(x));
            if (!s.empty()) out.push_back(s);
        }
        return out;
    }
    std::string s = v.is_null() ? "" : py_str(v), part;
    for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == ',') {
            std::string t = strip(part);
            if (!t.empty()) out.push_back(t);
            part.clear();
        } else part += s[i];
    }
    return out;
}

D2 j2(const Json& a) { return {a[0].get<double>(), a[1].get<double>()}; }

void need_dict(const Json& f) {
    if (f.is_array() && !f.empty()) throw std::runtime_error("AttributeError: 'list' object has no attribute 'get'");
}

// construct_tools._point
std::optional<D2> point_of(const std::optional<Json>& f) {
    if (!f || f->is_null() || (f->is_object() && f->empty()) || (f->is_array() && f->empty())) return std::nullopt;
    need_dict(*f);
    std::string t = f->value("type", "");
    if (t == "circle") return j2((*f)["center"]);
    if (t == "point") return j2((*f)["point"]);
    if (t == "line") {
        if (f->contains("p0") && f->contains("p1")) {
            D2 a = j2((*f)["p0"]), b = j2((*f)["p1"]);
            return D2{(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0};
        }
        return j2((*f)["point"]);
    }
    return std::nullopt;
}

bool truthy(const Json& v) { return !(v.is_null() || (v.is_array() && v.empty()) || (v.is_string() && v.get<std::string>().empty())); }

// construct_tools._line: (point, unit direction)
std::optional<std::pair<D2, D2>> line_of(const std::optional<Json>& f) {
    if (!f || f->is_null() || (f->is_object() && f->empty()) || (f->is_array() && f->empty())) return std::nullopt;
    need_dict(*f);
    if (f->value("type", "") != "line") return std::nullopt;
    D2 d;
    if (f->contains("direction") && truthy((*f)["direction"])) d = j2((*f)["direction"]);
    else {
        D2 a = j2((*f)["p0"]), b = j2((*f)["p1"]);
        d = {b[0] - a[0], b[1] - a[1]};
    }
    D2 p = f->contains("point") && truthy((*f)["point"]) ? j2((*f)["point"]) : j2((*f)["p0"]);
    double n = la::norm(d.data(), 2);
    if (n == 0) n = 1.0;
    return std::make_pair(p, D2{d[0] / n, d[1] / n});
}

Json line_feature(D2 p, D2 d, double length = 200.0) {
    double n = la::norm(d.data(), 2);
    if (n == 0) n = 1.0;
    d = {d[0] / n, d[1] / n};
    return Json{{"type", "line"}, {"point", {p[0], p[1]}}, {"direction", {d[0], d[1]}},
                {"p0", {p[0] - d[0] * length / 2, p[1] - d[1] * length / 2}}, {"p1", {p[0] + d[0] * length / 2, p[1] + d[1] * length / 2}},
                {"angle", std::atan2(d[1], d[0]) * kDeg}};
}

std::vector<cv::Point2f> f2(std::initializer_list<D2> pts) {
    std::vector<cv::Point2f> v;
    for (auto& p : pts) v.emplace_back((float)p[0], (float)p[1]);
    return v;
}

}  // namespace

// ---------------------------------------------------------------- fixture from points
class FixtureNPoint : public Tool {
public:
    using Tool::Tool;
    bool is_fixture() const override { return true; }

    std::optional<std::vector<D2>> reference_points(Context& ctx, const std::vector<std::string>& ids) {
        static std::mutex m;
        static std::map<std::string, std::vector<D2>> cache;
        char kb[32];
        std::snprintf(kb, sizeof kb, "%p", (const void*)ctx.reference.data);
        std::string key = std::string(kb) + "|" + join(ids, "\x1f") + "|" + ctx.folder;
        {
            std::lock_guard<std::mutex> lk(m);
            auto it = cache.find(key);
            if (it != cache.end()) return it->second;
        }
        Context ref_ctx;
        ref_ctx.reference = ctx.reference;
        ref_ctx.folder = ctx.folder;
        std::vector<D2> pts;
        for (auto& tid : ids) {
            auto it = ctx.tools.find(tid);
            if (it == ctx.tools.end()) return std::nullopt;
            ToolResult r = it->second->run(ctx.reference, ref_ctx);
            if (r.status != "pass") return std::nullopt;
            auto s = ref_ctx.share.find(tid);
            std::optional<Json> f;
            if (s != ref_ctx.share.end())
                if (auto j = std::any_cast<Json>(&s->second)) f = *j;
            auto p = point_of(f);
            if (!p) return std::nullopt;
            pts.push_back(*p);
        }
        std::lock_guard<std::mutex> lk(m);
        if (cache.size() > 64) cache.clear();
        cache[key] = pts;
        return pts;
    }

    ToolResult run(const cv::Mat&, Context& ctx) override {
        double t0 = now_ms();
        auto ids = ids_of(has("points") ? p["points"] : Json());
        if (ids.size() < 2) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "name at least two feature tools in 'points'");
        std::vector<D2> cur;
        std::vector<std::string> missing;
        for (auto& i : ids) {
            auto q = point_of(geom::passed_feature(ctx, i));
            if (!q) missing.push_back(i);
            else cur.push_back(*q);
        }
        if (!missing.empty())
            return ToolResult(id, type_name, "fail", 0.0, Json::object(), {}, "no good feature from " + join(missing, ", ") + " - it failed or did not run",
                              now_ms() - t0);
        if (ctx.reference.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "needs a reference image");
        auto ref = reference_points(ctx, ids);
        if (!ref) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "could not find the features on the reference image");
        const auto& src = *ref;
        const auto& dst = cur;
        int n = (int)src.size();
        double M2[6];
        if (n == 2) {
            D2 vr = {src[1][0] - src[0][0], src[1][1] - src[0][1]}, vc = {dst[1][0] - dst[0][0], dst[1][1] - dst[0][1]};
            double nr = la::norm(vr.data(), 2);
            double s = la::norm(vc.data(), 2) / (nr != 0 ? nr : 1.0);
            double a = std::atan2(vc[1], vc[0]) - std::atan2(vr[1], vr[0]);
            double R[4] = {s * std::cos(a), s * -std::sin(a), s * std::sin(a), s * std::cos(a)};
            auto Rs = la::matvec(R, src[0].data(), 2, 2);
            M2[0] = R[0], M2[1] = R[1], M2[2] = dst[0][0] - Rs[0];
            M2[3] = R[2], M2[4] = R[3], M2[5] = dst[0][1] - Rs[1];
        } else {
            cv::Mat S(n, 2, CV_64F), D(n, 2, CV_64F);
            for (int i = 0; i < n; i++) {
                S.at<double>(i, 0) = src[i][0], S.at<double>(i, 1) = src[i][1];
                D.at<double>(i, 0) = dst[i][0], D.at<double>(i, 1) = dst[i][1];
            }
            cv::Mat E = cv::estimateAffinePartial2D(S, D, cv::noArray(), cv::LMEDS);
            if (E.empty()) return ToolResult(id, type_name, "fail", 0.0, Json::object(), {}, "points do not agree");
            for (int i = 0; i < 6; i++) M2[i] = E.at<double>(i / 3, i % 3);
        }
        // pred = (M2[:, :2] @ src.T).T + M2[:, 2]
        std::vector<double> sflat;
        for (auto& q : src) sflat.push_back(q[0]), sflat.push_back(q[1]);
        auto P = la::gemm(false, true, 2, n, 2, M2, 3, sflat.data(), 2);
        std::vector<double> per(n);
        for (int i = 0; i < n; i++) {
            double ex = P[i] + M2[2] - dst[i][0], ey = P[n + i] + M2[5] - dst[i][1];
            per[i] = (0.0 + ex * ex) + ey * ey;
        }
        double rms = std::sqrt(npy::sum_f64(per.data(), per.size()) / (double)n);
        double scale = npy::py_hypot(M2[0], M2[3]);
        double max_rms = pf("max_rms", 2.0);
        bool ok = rms <= max_rms && std::abs(scale - 1.0) <= pf("max_scale_change", 0.05);
        if (ok) {
            cv::Mat M = (cv::Mat_<double>(3, 3) << M2[0], M2[1], M2[2], M2[3], M2[4], M2[5], 0, 0, 1);
            ctx.fixture = M;
        }
        double ang = std::atan2(M2[3], M2[0]) * kDeg;
        std::string st = ok ? "pass" : "fail";
        std::vector<Graphic> g;
        for (int i = 0; i + 1 < n; i++) g.push_back(abs_graphic("line", f2({dst[i], dst[i + 1]}), st));
        for (auto& q : dst) g.push_back(abs_graphic("rect", f2({{q[0] - 5, q[1] - 5}, {q[0] + 5, q[1] + 5}}), st));
        std::string why = ok ? "" : rms > max_rms ? "points disagree by " + pyfmt("%.2f", rms) + " px" : "scale changed by " + pyfmt("%.1f", 100 * (scale - 1)) + "%";
        Json matrix;
        if (ok) matrix = Json{{py_round(M2[0], 6), py_round(M2[1], 6), py_round(M2[2], 6)}, {py_round(M2[3], 6), py_round(M2[4], 6), py_round(M2[5], 6)}};
        Json v = {{"shift_x_px", py_round(M2[2], 2)}, {"shift_y_px", py_round(M2[5], 2)}, {"angle_deg", py_round(ang, 3)},
                  {"scale", py_round(scale, 5)}, {"rms_px", py_round(rms, 3)}, {"matrix", matrix}};
        return ToolResult(id, type_name, st, std::max(0.0, 1.0 - rms / (2 * max_rms)), v, g, why, now_ms() - t0);
    }
};
VC_REGISTER("fixture_npoint", FixtureNPoint)

// ---------------------------------------------------------------- n-point calibration
namespace {
double py_float(const std::string& s) {
    std::string t = s;
    size_t used = 0;
    double v = 0;
    bool ok = true;
    try {
        v = std::stod(t, &used);
    } catch (...) {
        ok = false;
    }
    if (!ok || used != t.size()) throw std::runtime_error("ValueError: could not convert string to float: '" + s + "'");
    return v;
}

std::pair<std::vector<D2>, std::vector<D2>> parse_pairs(const std::string& text) {
    std::string s = text;
    for (auto& c : s)
        if (c == '\n') c = ';';
    std::vector<D2> px, wd;
    size_t start = 0;
    for (;;) {
        size_t e = s.find(';', start);
        std::string part = s.substr(start, e == std::string::npos ? std::string::npos : e - start);
        size_t eq = part.find('=');
        if (eq != std::string::npos) {
            auto nums = [](std::string t) {
                t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
                std::vector<double> out;
                size_t a = 0;
                for (;;) {
                    size_t c = t.find(',', a);
                    std::string v = t.substr(a, c == std::string::npos ? std::string::npos : c - a);
                    if (!v.empty()) out.push_back(py_float(v));
                    if (c == std::string::npos) break;
                    a = c + 1;
                }
                return out;
            };
            auto pa = nums(part.substr(0, eq)), pb = nums(part.substr(eq + 1));
            if (pa.size() == 2 && pb.size() == 2) {
                px.push_back({pa[0], pa[1]});
                wd.push_back({pb[0], pb[1]});
            }
        }
        if (e == std::string::npos) break;
        start = e + 1;
    }
    return {px, wd};
}
}  // namespace

class CalibrateNPoint : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat&, Context& ctx) override {
        double t0 = now_ms();
        auto [px, wd] = parse_pairs(has("pairs") ? py_str(p["pairs"]) : "");
        int n = (int)px.size();
        if (n < 3) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "needs at least three 'px,py = wx,wy' pairs");
        std::vector<double> A, W;
        for (int i = 0; i < n; i++) {
            A.push_back(px[i][0]), A.push_back(px[i][1]), A.push_back(1.0);
            W.push_back(wd[i][0]), W.push_back(wd[i][1]);
        }
        auto sol = la::lstsq(A.data(), W.data(), n, 3, 2);           // 3 x 2
        double M[6] = {sol[0], sol[2], sol[4], sol[1], sol[3], sol[5]};   // sol.T, 2 x 3
        auto pred = la::gemm(false, false, n, 2, 3, A.data(), 3, sol.data(), 2);
        std::vector<double> per(n);
        for (int i = 0; i < n; i++) {
            double ex = pred[2 * i] - wd[i][0], ey = pred[2 * i + 1] - wd[i][1];
            per[i] = (0.0 + ex * ex) + ey * ey;
        }
        double rms = std::sqrt(npy::sum_f64(per.data(), per.size()) / (double)n);
        double m22[4] = {M[0], M[1], M[3], M[4]};
        double scale = std::sqrt(std::abs(la::det(m22, 2)));
        ctx.mm_per_px = scale;
        ctx.world = (cv::Mat_<double>(2, 3) << M[0], M[1], M[2], M[3], M[4], M[5]);
        double limit = pf("max_rms", 0.05);
        bool ok = rms <= limit;
        Json mat = Json::array();
        for (int r = 0; r < 2; r++) mat.push_back({py_round(M[3 * r], 8), py_round(M[3 * r + 1], 8), py_round(M[3 * r + 2], 8)});
        return ToolResult(id, type_name, ok ? "pass" : "fail", ok ? 1.0 : 0.0,
                          Json{{"units_per_px", py_round(scale, 6)}, {"rms", py_round(rms, 5)}, {"points", n}, {"matrix", mat}}, {},
                          ok ? "" : "the points fit to " + pyfmt("%.4f", rms) + ", over " + pyfmt("%.4f", limit), now_ms() - t0);
    }
};
VC_REGISTER("calibrate_npoint", CalibrateNPoint)

// ---------------------------------------------------------------- create geometry
class Construct : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat&, Context& ctx) override {
        double t0 = now_ms();
        std::string op = ps("operation", "line_through_points");
        std::optional<Json> F[3];
        const char* keys[3] = {"first", "second", "third"};
        for (int i = 0; i < 3; i++) F[i] = geom::passed_feature(ctx, has(keys[i]) ? py_str(p[keys[i]]) : "");
        auto fail = [&](const std::string& why) { return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, why); };
        Json feat, vals;
        std::vector<Graphic> g;
        if (op == "line_through_points" || op == "midpoint") {
            auto a = point_of(F[0]);
            auto b = point_of(F[1]);
            if (!a || !b) return fail("needs two point features as first and second");
            if (op == "midpoint") {
                D2 m = {((*a)[0] + (*b)[0]) / 2.0, ((*a)[1] + (*b)[1]) / 2.0};
                feat = Json{{"type", "point"}, {"point", {m[0], m[1]}}};
                vals = {{"x", py_round(m[0], 3)}, {"y", py_round(m[1], 3)}};
                g = {abs_graphic("line", f2({*a, *b}), "search"), abs_graphic("line", f2({{m[0] - 6, m[1]}, {m[0] + 6, m[1]}}), "pass"),
                     abs_graphic("line", f2({{m[0], m[1] - 6}, {m[0], m[1] + 6}}), "pass", id)};
            } else {
                D2 d = {(*b)[0] - (*a)[0], (*b)[1] - (*a)[1]};
                double len = la::norm(d.data(), 2);
                if (len < 1e-9) return fail("the two points coincide");
                feat = line_feature({((*a)[0] + (*b)[0]) / 2.0, ((*a)[1] + (*b)[1]) / 2.0}, d, len);
                vals = {{"angle_deg", py_round(feat["angle"].get<double>(), 3)}, {"length_px", py_round(len, 3)}};
                g = {abs_graphic("line", f2({*a, *b}), "pass", id)};
            }
        } else if (op == "perpendicular" || op == "parallel") {
            auto ln = line_of(F[0]);
            auto pt = point_of(F[1]);
            if (!ln || !pt) return fail("needs a line as first and a point as second");
            D2 d = op == "parallel" ? ln->second : D2{-ln->second[1], ln->second[0]};
            feat = line_feature(*pt, d);
            vals = {{"angle_deg", py_round(feat["angle"].get<double>(), 3)}};
            g = {abs_graphic("line", f2({j2(feat["p0"]), j2(feat["p1"])}), "pass", id)};
        } else if (op == "circle_through_points") {
            std::optional<D2> pts[3];
            for (int i = 0; i < 3; i++) pts[i] = point_of(F[i]);
            if (!pts[0] || !pts[1] || !pts[2]) return fail("needs three point features");
            double x1 = (*pts[0])[0], y1 = (*pts[0])[1], x2 = (*pts[1])[0], y2 = (*pts[1])[1], x3 = (*pts[2])[0], y3 = (*pts[2])[1];
            double dd = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
            if (std::abs(dd) < 1e-9) return fail("the three points are in a line");
            double s1 = x1 * x1 + y1 * y1, s2 = x2 * x2 + y2 * y2, s3 = x3 * x3 + y3 * y3;
            double ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / dd;
            double uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / dd;
            double r = npy::py_hypot(x1 - ux, y1 - uy);
            feat = Json{{"type", "circle"}, {"center", {ux, uy}}, {"radius", r}};
            vals = {{"cx", py_round(ux, 3)}, {"cy", py_round(uy, 3)}, {"radius_px", py_round(r, 3)}};
            std::vector<cv::Point2f> ring;
            for (int i = 0; i < 64; i++) {
                double t = i == 63 ? 2 * kPi : i * (2 * kPi / 63.0);
                ring.emplace_back((float)(ux + r * std::cos(t)), (float)(uy + r * std::sin(t)));
            }
            g = {abs_graphic("poly", ring, "pass", id)};
        } else {
            return fail("unknown operation " + op);
        }
        ctx.share[id] = feat;
        return ToolResult(id, type_name, "pass", 1.0, vals, g, "", now_ms() - t0);
    }
};
VC_REGISTER("construct", Construct)

// ---------------------------------------------------------------- bead inspect
class BeadInspect : public Tool {
public:
    using Tool::Tool;

    std::vector<double> path(const Context& ctx) const {
        std::vector<double> out;
        if (shape == "polygon" && points.size() >= 2) {
            for (auto& q : polygon_points(ctx)) out.push_back(q.x), out.push_back(q.y);
            return out;
        }
        double x = (*roi)[0], y = (*roi)[1], w = (*roi)[2], h = (*roi)[3];
        double pts[4];
        if (w >= h) pts[0] = x, pts[1] = y + h / 2, pts[2] = x + w, pts[3] = y + h / 2;
        else pts[0] = x + w / 2, pts[1] = y, pts[2] = x + w / 2, pts[3] = y + h;
        if (!ctx.fixture.empty()) {
            cv::Mat M;
            ctx.fixture.convertTo(M, CV_64F);
            double m[4] = {M.at<double>(0, 0), M.at<double>(0, 1), M.at<double>(1, 0), M.at<double>(1, 1)};
            auto r = la::gemm(false, true, 2, 2, 2, m, 2, pts, 2);    // (M[:2, :2] @ pts.T).T
            // r is 2 x 2: row i = coordinate i of both points
            double t0 = M.at<double>(0, 2), t1 = M.at<double>(1, 2);
            return {r[0] + t0, r[2] + t1, r[1] + t0, r[3] + t1};
        }
        return {pts[0], pts[1], pts[2], pts[3]};
    }

    std::vector<Graphic> region_graphic(const Context& ctx, const std::string& status, const std::string& label) const {
        if (shape != "polygon" || points.size() < 2) return roi_graphic(ctx, status, label);
        auto pth = path(ctx);
        std::vector<Graphic> out;
        for (size_t k = 0; k + 1 < pth.size() / 2; k++) {
            size_t i = 2 * k;
            Graphic g = abs_graphic("line", {{(float)pth[i], (float)pth[i + 1]}, {(float)pth[i + 2], (float)pth[i + 3]}}, status, k == 0 ? label : "");
            g.role = "region";
            out.push_back(g);
        }
        return out;
    }

    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        if (!roi && points.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "no path");
        cv::Mat gb, g;
        cv::GaussianBlur(to_gray(image), gb, cv::Size(3, 3), 0);
        gb.convertTo(g, CV_32F);
        auto pth = path(ctx);
        int np_ = (int)(pth.size() / 2), nseg = np_ - 1;
        std::vector<double> sx(nseg), sy(nseg), lens(nseg);
        for (int i = 0; i < nseg; i++) {
            sx[i] = pth[2 * (i + 1)] - pth[2 * i];
            sy[i] = pth[2 * (i + 1) + 1] - pth[2 * i + 1];
            lens[i] = std::hypot(sx[i], sy[i]);
        }
        double total = npy::sum_f64(lens.data(), lens.size());
        int n = std::max(4, pi("stations", 30));
        double half = pf("search_half_width", 25), want = pf("width", 12), tol = pf("width_tolerance", 4);
        double max_off = pf("max_offset", 6), minc = pf("min_contrast", 15);
        bool dark = ps("polarity", "dark") == "dark";
        std::vector<double> cum{0.0};
        double acc = 0;
        for (double l : lens) cum.push_back(acc += l);
        struct St { int k; double width, offset; std::string status, why; };
        std::vector<St> stations;
        std::vector<Graphic> gfx;
        int m = (int)(2 * half) + 1;
        std::vector<double> u(m);
        double ustep = m > 1 ? (half - -half) / (double)(m - 1) : 0;
        for (int i = 0; i < m; i++) u[i] = ustep == 0 ? -half : (double)i * ustep + -half;
        if (m > 1) u[m - 1] = half;
        float mc2 = (float)(minc / 2);
        for (int k = 0; k < n; k++) {
            double s = (k + 0.5) * total / n;
            int ss = (int)(std::upper_bound(cum.begin(), cum.end(), s) - cum.begin());
            int i = std::min(nseg - 1, ss - 1);
            double li = lens[i] != 0 ? lens[i] : 1.0;
            double t = (s - cum[i]) / li;
            D2 c = {pth[2 * i] + sx[i] * t, pth[2 * i + 1] + sy[i] * t};
            D2 d = {sx[i] / li, sy[i] / li};
            D2 nrm = {-d[1], d[0]};
            cv::Mat xs(1, m, CV_32F), ys(1, m, CV_32F), prof;
            for (int q = 0; q < m; q++) {
                xs.at<float>(0, q) = (float)(c[0] + nrm[0] * u[q]);
                ys.at<float>(0, q) = (float)(c[1] + nrm[1] * u[q]);
            }
            cv::remap(g, prof, xs, ys, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            std::vector<float> pv((const float*)prof.data, (const float*)prof.data + m);
            if (!dark)
                for (auto& v : pv) v = -v;
            auto grad = npy::gradient_f32(pv);
            int a = (int)(std::min_element(grad.begin(), grad.end()) - grad.begin());
            int b = (int)(std::max_element(grad.begin(), grad.end()) - grad.begin());
            bool ok_edges = a < b && -grad[a] >= mc2 && grad[b] >= mc2;
            St st;
            if (ok_edges) {
                double width = u[b] - u[a], off = (u[a] + u[b]) / 2.0;
                bool bw = std::abs(width - want) > tol, bad = bw || std::abs(off) > max_off;
                st = {k, py_round(width, 2), py_round(off, 2), bad ? "fail" : "pass",
                      bad ? (bw ? "width " + pyfmt("%.1f", width) : "offset " + pyfmt("%.1f", off)) : ""};
                gfx.push_back(abs_graphic("line", f2({{c[0] + nrm[0] * u[a], c[1] + nrm[1] * u[a]}, {c[0] + nrm[0] * u[b], c[1] + nrm[1] * u[b]}}), st.status));
            } else {
                st = {k, 0.0, 0.0, "fail", "gap"};
                gfx.push_back(abs_graphic("rect", f2({{c[0] - 4, c[1] - 4}, {c[0] + 4, c[1] + 4}}), "fail"));
            }
            stations.push_back(st);
        }
        int gaps = 0;
        std::vector<const St*> bad;
        std::vector<double> widths;
        for (auto& s : stations) {
            gaps += s.why == "gap";
            if (s.status == "fail") bad.push_back(&s);
            if (s.why != "gap") widths.push_back(s.width);
        }
        int allowed = pi("max_bad_stations", 0);
        bool ok = (int)bad.size() <= allowed;
        for (int i = 0; i < nseg; i++)
            gfx.push_back(abs_graphic("line", f2({{pth[2 * i], pth[2 * i + 1]}, {pth[2 * i + 2], pth[2 * i + 3]}}), "search"));
        std::string why;
        if (!ok) {
            std::vector<std::string> parts;
            for (size_t i = 0; i < bad.size() && i < 6; i++) parts.push_back("#" + std::to_string(bad[i]->k) + " " + bad[i]->why);
            why = std::to_string(bad.size()) + " station(s) out: " + join(parts, ", ");
        }
        Json prof = Json::array();
        for (auto& s : stations)
            prof.push_back({{"station", s.k}, {"width", s.width}, {"offset", s.offset}, {"status", s.status}, {"why", s.why}});
        Json v = {{"stations", n}, {"bad_stations", (int)bad.size()}, {"gaps", gaps},
                  {"min_width", widths.empty() ? 0.0 : py_round(*std::min_element(widths.begin(), widths.end()), 2)},
                  {"max_width", widths.empty() ? 0.0 : py_round(*std::max_element(widths.begin(), widths.end()), 2)},
                  {"mean_width", widths.empty() ? 0.0 : py_round(npy::sum_f64(widths.data(), widths.size()) / (double)widths.size(), 2)},
                  {"profile", prof}};
        std::string stt = ok ? "pass" : "fail";
        auto gr = region_graphic(ctx, stt, id);
        gr.insert(gr.end(), gfx.begin(), gfx.end());
        return ToolResult(id, type_name, stt, 1.0 - (double)bad.size() / (double)n, v, gr, why, now_ms() - t0);
    }
};
VC_REGISTER("bead", BeadInspect)

// ---------------------------------------------------------------- data analysis
namespace {
std::mutex series_lock;
std::map<std::pair<std::string, std::string>, std::vector<double>> series_all;
}  // namespace

void reset_series(const std::string* folder) {
    std::lock_guard<std::mutex> lk(series_lock);
    for (auto it = series_all.begin(); it != series_all.end();)
        it = (!folder || it->first.first == *folder) ? series_all.erase(it) : std::next(it);
}

class DataAnalysis : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat&, Context& ctx) override {
        double t0 = now_ms();
        std::string ref = has("source") ? py_str(p["source"]) : "";
        auto dot = ref.find('.');
        if (dot == std::string::npos) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "set source as tool.value");
        std::string tid = ref.substr(0, dot), key = ref.substr(dot + 1);
        std::optional<double> v;
        auto r = ctx.results.find(tid);
        if (r != ctx.results.end()) {
            if (key == "score") v = r->second.score;
            else if (r->second.values.is_object() && r->second.values.contains(key)) {
                const Json& x = r->second.values[key];
                if (x.is_number()) v = x.get<double>();
            }
        }
        if (!v) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, ref + " has no value");
        std::vector<double> a;
        {
            std::lock_guard<std::mutex> lk(series_lock);
            auto& series = series_all[{ctx.folder, id}];
            series.push_back(*v);
            long long window = pi("window", 100);
            // del series[:-window]
            long long L = (long long)series.size(), end = -window;
            if (end < 0) end = std::max(0LL, L + end);
            end = std::min(end, L);
            series.erase(series.begin(), series.begin() + end);
            a = series;
        }
        auto lim = [&](const char* k) -> std::optional<double> {
            if (!has(k)) return std::nullopt;
            const Json& x = p[k];
            if (x.is_string() && x.get<std::string>().empty()) return std::nullopt;
            return pf(k, 0);
        };
        auto lsl = lim("lower_limit"), usl = lim("upper_limit");
        size_t n = a.size();
        if (n == 0) throw std::runtime_error("ValueError: zero-size array to reduction operation minimum which has no identity");
        double mean = npy::sum_f64(a.data(), n) / (double)n, sd = 0.0;
        if (n > 1) {
            std::vector<double> d2(n);
            for (size_t i = 0; i < n; i++) d2[i] = (a[i] - mean) * (a[i] - mean);
            sd = std::sqrt(npy::sum_f64(d2.data(), n) / (double)(n - 1));
        }
        Json vals = {{"value", py_round(*v, 5)}, {"count", (int)n}, {"mean", py_round(mean, 5)}, {"std", py_round(sd, 5)},
                     {"min", py_round(*std::min_element(a.begin(), a.end()), 5)}, {"max", py_round(*std::max_element(a.begin(), a.end()), 5)}};
        std::vector<std::string> why;
        if (lsl && *v < *lsl) why.push_back(pyfmt("%.4g", *v) + " under " + pyfmt("%.4g", *lsl));
        if (usl && *v > *usl) why.push_back(pyfmt("%.4g", *v) + " over " + pyfmt("%.4g", *usl));
        if (sd > 0 && lsl && usl) vals["cp"] = py_round((*usl - *lsl) / (6 * sd), 3);
        if (sd > 0 && (lsl || usl)) {
            std::optional<double> cpk;
            if (usl) cpk = (*usl - mean) / (3 * sd);
            if (lsl) {
                double c2 = (mean - *lsl) / (3 * sd);
                if (!cpk || c2 < *cpk) cpk = c2;
            }
            vals["cpk"] = py_round(*cpk, 3);
            double min_cpk = pf("min_cpk", 0);
            if (min_cpk != 0 && (long long)n >= pi("min_samples", 25) && *cpk < min_cpk)
                why.push_back("Cpk " + pyfmt("%.2f", *cpk) + " under " + pyfmt("%.2f", min_cpk) + " - drifting towards a limit");
        }
        bool ok = why.empty();
        return ToolResult(id, type_name, ok ? "pass" : "fail", ok ? 1.0 : 0.0, vals, {}, join(why, "; "), now_ms() - t0);
    }
};
VC_REGISTER("data_analysis", DataAnalysis)

}  // namespace vc
