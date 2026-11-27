// vision_station/tools/geometry.py: find_line, find_circle, histogram, distance, caliper.
// Fits go through numpy's own LAPACK (core/linalg) and the consensus search
// through numpy's own random generator (core/nprandom), so every number
// matches Vision Station's.
#include "geom.hpp"

#include "../core/linalg.hpp"
#include "../core/nprandom.hpp"
#include "../core/npy.hpp"
#include "../core/pyfmt.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace vc {
namespace geom {

constexpr double kPi = 3.141592653589793;
constexpr double kRadToDeg = 180.0 / kPi;   // CPython math.degrees

double subpixel_peak(const std::vector<float>& prof, int i) {
    if (i <= 0 || i >= (int)prof.size() - 1) return (double)i;
    double a = prof[i - 1], b = prof[i], c = prof[i + 1];
    double d = a - 2 * b + c;
    return std::abs(d) < 1e-9 ? (double)i : (double)i - 0.5 * (c - a) / d;
}

cv::Mat plane(const cv::Mat& img, const std::string& channel) {
    if (img.channels() == 1) return img;
    cv::Mat grey, b, g, r;
    cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
    cv::extractChannel(img, b, 0);
    cv::extractChannel(img, g, 1);
    cv::extractChannel(img, r, 2);
    if (channel == "grey") return grey;
    if (channel == "blue") return b;
    if (channel == "green") return g;
    if (channel == "red") return r;
    cv::Mat best;
    double best_sd = -1;
    for (const cv::Mat* m : {&grey, &b, &g, &r}) {
        double sd = npy::std_u8(m->data, m->total());
        if (sd > best_sd) {
            best_sd = sd;
            best = *m;
        }
    }
    return best;
}

static std::vector<float> blur_gradient(const std::vector<float>& prof) {
    cv::Mat row(1, (int)prof.size(), CV_32F, const_cast<float*>(prof.data())), sm;
    cv::GaussianBlur(row, sm, cv::Size(1, 5), 0);
    std::vector<float> s((const float*)sm.data, (const float*)sm.data + prof.size());
    return npy::gradient_f32(s);
}

std::vector<std::pair<double, double>> profile_edges(const std::vector<float>& prof, const std::string& polarity,
                                                     double min_contrast, double step, int most) {
    if (prof.empty()) return {};
    auto d = blur_gradient(prof);
    float st = (float)step;
    for (auto& v : d) v = v / st;
    std::vector<float> cand = d;
    if (polarity == "light_to_dark") for (auto& v : cand) v = -v;
    else if (polarity != "dark_to_light") for (auto& v : cand) v = std::abs(v);
    if (cand.size() < 3) return {};
    float mc = (float)min_contrast;
    int top = 1;
    for (int i = 2; i + 1 < (int)cand.size(); i++)
        if (cand[i] > cand[top]) top = i;
    if (cand[top] < mc) return {};
    std::vector<int> peaks;
    for (int i = 1; i + 1 < (int)cand.size(); i++)
        if (cand[i] >= cand[i - 1] && cand[i] > cand[i + 1] && cand[i] >= mc) peaks.push_back(i);
    std::stable_sort(peaks.begin(), peaks.end(), [&](int a, int b) { return -cand[a] < -cand[b]; });
    std::vector<int> pick{top};
    for (int i : peaks)
        if (i != top && (int)pick.size() < most) pick.push_back(i);
    std::vector<std::pair<double, double>> out;
    for (int i : pick) out.emplace_back(subpixel_peak(cand, i), (double)cand[i]);
    return out;
}

// ---------------------------------------------------------------- fits
template <class T>
static void mean_rows(const std::vector<T>& pts, T c[2]) {
    size_t n = pts.size() / 2;
    T s0 = 0, s1 = 0;
    for (size_t i = 0; i < n; i++) {
        s0 += pts[2 * i];
        s1 += pts[2 * i + 1];
    }
    c[0] = s0 / (T)n;
    c[1] = s1 / (T)n;
}

template <class T>
LineFit<T> fit_line(const std::vector<T>& pts) {
    LineFit<T> f;
    int n = (int)(pts.size() / 2);
    mean_rows(pts, f.c);
    std::vector<T> cen(pts.size());
    std::vector<double> cd(pts.size());
    for (int i = 0; i < n; i++) {
        cen[2 * i] = pts[2 * i] - f.c[0];
        cen[2 * i + 1] = pts[2 * i + 1] - f.c[1];
        cd[2 * i] = (double)cen[2 * i];
        cd[2 * i + 1] = (double)cen[2 * i + 1];
    }
    auto vh = la::svd_vh(cd.data(), n, 2);
    f.d[0] = (T)vh[0];
    f.d[1] = (T)vh[1];
    T nrm[2] = {-f.d[1], f.d[0]};
    f.resid = la::matvec(cen.data(), nrm, n, 2);
    return f;
}

template <class T>
CircleFit<T> fit_circle(const std::vector<T>& pts) {
    int n = (int)(pts.size() / 2);
    std::vector<T> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = pts[2 * i];
        y[i] = pts[2 * i + 1];
    }
    std::vector<double> A((size_t)n * 3), b(n);
    for (int i = 0; i < n; i++) {
        A[3 * i] = (double)x[i];
        A[3 * i + 1] = (double)y[i];
        A[3 * i + 2] = 1.0;
        b[i] = (double)(T)(x[i] * x[i] + y[i] * y[i]);
    }
    auto sol = la::lstsq(A.data(), b.data(), n, 3);
    T cx = (T)sol[0] / (T)2, cy = (T)sol[1] / (T)2;
    T s = (T)sol[2] + cx * cx;
    s = s + cy * cy;
    T r = (T)std::sqrt(s > (T)1e-9 ? (double)s : 1e-9);
    for (int it = 0; it < 5; it++) {
        std::vector<double> J((size_t)n * 3), nres(n);
        for (int i = 0; i < n; i++) {
            T dx = x[i] - cx, dy = y[i] - cy;
            T di = (T)std::sqrt(dx * dx + dy * dy) + (T)1e-12;
            J[3 * i] = (double)(-dx / di);
            J[3 * i + 1] = (double)(-dy / di);
            J[3 * i + 2] = -1.0;
            nres[i] = (double)(-(di - r));
        }
        auto st = la::lstsq(J.data(), nres.data(), n, 3);
        cx = cx + (T)st[0];
        cy = cy + (T)st[1];
        r = r + (T)st[2];
    }
    CircleFit<T> f{cx, cy, r, {}};
    f.resid.resize(n);
    for (int i = 0; i < n; i++) {
        T dx = x[i] - cx, dy = y[i] - cy;
        f.resid[i] = (T)std::sqrt(dx * dx + dy * dy) - r;
    }
    return f;
}

template LineFit<float> fit_line(const std::vector<float>&);
template LineFit<double> fit_line(const std::vector<double>&);
template CircleFit<float> fit_circle(const std::vector<float>&);
template CircleFit<double> fit_circle(const std::vector<double>&);

// ---------------------------------------------------------------- models for the consensus
using Model = std::vector<double>;   // line: cx cy dx dy; circle: cx cy r

static std::optional<Model> model_from(Kind k, const std::vector<Pt>& p) {
    if (k == Kind::line) {
        double dx = p[1].first - p[0].first, dy = p[1].second - p[0].second;
        double n = std::hypot(dx, dy);
        if (n < 1e-6) return std::nullopt;
        return Model{p[0].first, p[0].second, dx / n, dy / n};
    }
    double x1 = p[0].first, y1 = p[0].second, x2 = p[1].first, y2 = p[1].second, x3 = p[2].first, y3 = p[2].second;
    double a = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
    if (std::abs(a) < 1e-6) return std::nullopt;
    double s1 = x1 * x1 + y1 * y1, s2 = x2 * x2 + y2 * y2, s3 = x3 * x3 + y3 * y3;
    double cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / a;
    double cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / a;
    return Model{cx, cy, std::hypot(x1 - cx, y1 - cy)};
}

static std::vector<double> residual(Kind k, const Model& m, const std::vector<double>& P) {
    size_t n = P.size() / 2;
    if (k == Kind::line) {
        std::vector<double> cen(P.size());
        for (size_t i = 0; i < n; i++) {
            cen[2 * i] = P[2 * i] - m[0];
            cen[2 * i + 1] = P[2 * i + 1] - m[1];
        }
        double nrm[2] = {-m[3], m[2]};
        return la::matvec(cen.data(), nrm, (int)n, 2);
    }
    std::vector<double> r(n);
    for (size_t i = 0; i < n; i++) r[i] = std::hypot(P[2 * i] - m[0], P[2 * i + 1] - m[1]) - m[2];
    return r;
}

static Model fit_model(Kind k, const std::vector<double>& pts) {
    if (k == Kind::line) {
        auto f = fit_line<double>(pts);
        return {f.c[0], f.c[1], f.d[0], f.d[1]};
    }
    auto f = fit_circle<double>(pts);
    return {f.cx, f.cy, f.r};
}

static std::vector<double> fit_resid(Kind k, const std::vector<double>& pts) {
    return k == Kind::line ? fit_line<double>(pts).resid : fit_circle<double>(pts).resid;
}

static std::vector<double> flat(const std::vector<Pt>& p) {
    std::vector<double> v;
    v.reserve(p.size() * 2);
    for (auto& q : p) {
        v.push_back(q.first);
        v.push_back(q.second);
    }
    return v;
}

static double rms_of(const std::vector<double>& r) {
    std::vector<double> sq(r.size());
    for (size_t i = 0; i < r.size(); i++) sq[i] = r[i] * r[i];
    return std::sqrt(npy::sum_f64(sq.data(), sq.size()) / (double)sq.size());
}

// geometry._robust on float64 points: the kept indices
static std::vector<int> robust_keep(Kind k, const std::vector<double>& pts, int ignore, std::vector<double>& last_resid) {
    int n = (int)(pts.size() / 2);
    std::vector<int> keep(n);
    std::iota(keep.begin(), keep.end(), 0);
    auto out = fit_resid(k, pts);
    for (int it = 0; it < ignore; it++) {
        if ((int)keep.size() <= 3) break;
        int worst = 0;
        for (int i = 1; i < (int)out.size(); i++)
            if (std::abs(out[i]) > std::abs(out[worst])) worst = i;
        keep.erase(keep.begin() + worst);
        std::vector<double> sub;
        for (int i : keep) {
            sub.push_back(pts[2 * i]);
            sub.push_back(pts[2 * i + 1]);
        }
        out = fit_resid(k, sub);
    }
    last_resid = out;
    return keep;
}

static std::pair<std::optional<Model>, std::vector<std::pair<int, int>>> consensus(const Rays& cands, Kind k, int n_min, double tol) {
    std::vector<int> live;
    for (int i = 0; i < (int)cands.size(); i++)
        if (!cands[i].empty()) live.push_back(i);
    if ((int)live.size() < n_min) return {std::nullopt, {}};
    npr::Generator rng(0);
    struct F { int i, k; Pt p; double s; };
    std::vector<F> fl;
    for (int i : live)
        for (int q = 0; q < (int)cands[i].size(); q++) fl.push_back({i, q, cands[i][q].p, cands[i][q].strength});
    std::vector<double> P;
    double smax = -INFINITY;
    for (auto& f : fl) {
        P.push_back(f.p.first);
        P.push_back(f.p.second);
        smax = std::max(smax, f.s);
    }
    double div = std::max(1e-9, smax);
    std::vector<double> strength(fl.size());
    for (size_t i = 0; i < fl.size(); i++) strength[i] = fl[i].s / div;
    std::optional<Model> best;
    std::vector<int> best_idx;
    long long best_n = -1;
    double best_s = -1.0;
    for (int it = 0; it < 400; it++) {
        auto pick = rng.choice((int64_t)live.size(), n_min);
        std::vector<Pt> pts;
        for (auto j : pick) {
            const auto& ray = cands[live[(size_t)j]];
            pts.push_back(ray[(size_t)rng.integers((int64_t)ray.size())].p);
        }
        auto m = model_from(k, pts);
        if (!m) continue;
        auto r = residual(k, *m, P);
        for (auto& v : r) v = std::abs(v);
        std::vector<std::pair<int, int>> chosen;   // owner -> flat index, in first-insertion order
        for (int idx = 0; idx < (int)r.size(); idx++) {
            if (!(r[idx] <= tol)) continue;
            int o = fl[idx].i;
            auto it2 = std::find_if(chosen.begin(), chosen.end(), [&](auto& c) { return c.first == o; });
            if (it2 == chosen.end()) chosen.emplace_back(o, idx);
            else if (r[idx] < r[it2->second]) it2->second = idx;
        }
        if (chosen.empty()) continue;
        double s = 0;
        for (auto& c : chosen) s += strength[c.second];
        long long cnt = (long long)chosen.size();
        if (cnt > best_n || (cnt == best_n && s > best_s)) {
            best_n = cnt;
            best_s = s;
            best = m;
            best_idx.clear();
            for (auto& c : chosen) best_idx.push_back(c.second);
            std::sort(best_idx.begin(), best_idx.end());
        }
    }
    if (!best) return {std::nullopt, {}};
    std::vector<std::pair<int, int>> agree;
    for (int i : best_idx) agree.emplace_back(fl[i].i, fl[i].k);
    return {best, agree};
}

std::pair<std::vector<Pt>, bool> edge_points(const Rays& cands, Kind kind, int need, int ignore, double max_rms) {
    int n_min = kind == Kind::line ? 2 : 3;
    std::vector<Pt> strongest;
    for (auto& c : cands)
        if (!c.empty()) strongest.push_back(c[0].p);
    if ((int)strongest.size() >= need) {
        std::vector<double> last;
        auto keep = robust_keep(kind, flat(strongest), ignore, last);
        if (rms_of(last) <= max_rms) {
            std::vector<Pt> out;
            for (int i : keep) out.push_back(strongest[i]);
            return {out, true};
        }
    }
    double tol = std::max(1.5, max_rms * 1.5);
    auto [model, agree] = consensus(cands, kind, n_min, tol);
    std::vector<Pt> pts;
    for (auto& [i, q] : agree) pts.push_back(cands[i][q].p);
    for (int it = 0; it < 2; it++) {
        if ((int)pts.size() < std::max(n_min + 1, 4)) break;
        Model m = fit_model(kind, flat(pts));
        std::vector<Pt> again;
        for (auto& ray : cands) {
            if (ray.empty()) continue;
            std::vector<double> P;
            for (auto& c : ray) {
                P.push_back(c.p.first);
                P.push_back(c.p.second);
            }
            auto r = residual(kind, m, P);
            int j = 0;
            for (int q = 1; q < (int)r.size(); q++)
                if (std::abs(r[q]) < std::abs(r[j])) j = q;
            if (std::abs(r[j]) <= tol) again.push_back(ray[j].p);
        }
        if (again.size() <= pts.size()) break;
        pts = again;
    }
    bool enough = (int)pts.size() >= need;
    return {pts, enough};
}

std::vector<cv::Point2f> to_image(const Tool& t, int ox, int oy, const std::vector<Pt>& pts) {
    std::vector<cv::Point2f> in;
    for (auto& p : pts) in.emplace_back((float)p.first, (float)p.second);
    if (t.crop_M.empty()) {
        for (auto& p : in) {
            p.x = p.x + (float)ox;
            p.y = p.y + (float)oy;
        }
        return in;
    }
    std::vector<cv::Point2f> out;
    if (!in.empty()) cv::perspectiveTransform(in, out, t.crop_M);
    return out;
}

std::pair<cv::Mat, cv::Point> raw_plane(Tool& t, const cv::Mat& image, const Context& ctx) {
    std::string saved = t.shape;
    t.shape = "rect";
    std::pair<cv::Mat, cv::Point> c;
    try {
        c = t.crop(image, ctx);
    } catch (...) {
        t.shape = saved;
        throw;
    }
    t.shape = saved;
    if (c.first.empty()) return {cv::Mat(), cv::Point(0, 0)};
    cv::Mat f, g;
    plane(c.first, t.ps("channel", "auto")).convertTo(f, CV_32F);
    cv::GaussianBlur(f, g, cv::Size(3, 3), 0);
    return {g, c.second};
}

std::optional<Json> passed_feature(const Context& ctx, const std::string& id) {
    auto r = ctx.results.find(id);
    if (r != ctx.results.end() && r->second.status != "pass") return std::nullopt;
    auto s = ctx.share.find(id);
    if (s == ctx.share.end()) return std::nullopt;
    if (auto j = std::any_cast<Json>(&s->second)) return *j;
    return std::nullopt;
}

std::vector<float> mean_axis0(const cv::Mat& g) {
    std::vector<float> s((size_t)g.cols, 0.0f);
    for (int y = 0; y < g.rows; y++) {
        const float* r = g.ptr<float>(y);
        for (int x = 0; x < g.cols; x++) s[x] += r[x];
    }
    for (auto& v : s) v = v / (float)g.rows;
    return s;
}

std::vector<float> mean_axis1(const cv::Mat& g) {
    std::vector<float> s((size_t)g.rows);
    for (int y = 0; y < g.rows; y++) s[y] = npy::mean_f32(g.ptr<float>(y), (size_t)g.cols);
    return s;
}

}  // namespace geom

using namespace geom;

namespace {

std::vector<cv::Point2f> pts2f(const std::vector<std::pair<double, double>>& p) {
    std::vector<cv::Point2f> v;
    for (auto& q : p) v.emplace_back((float)q.first, (float)q.second);
    return v;
}

ToolResult too_few_points(Tool& t, const Context& ctx, int found, int need, double t0, const std::vector<Graphic>& guides) {
    auto g = t.roi_graphic(ctx, "fail");
    g.insert(g.end(), guides.begin(), guides.end());
    return ToolResult(t.id, t.type_name, "fail", 0.0, Json{{"found", found}}, g,
                      "edge found by " + std::to_string(found) + " calipers, need " + std::to_string(need), now_ms() - t0);
}

Json arr(const float* v, int n) {
    Json a = Json::array();
    for (int i = 0; i < n; i++) a.push_back((double)v[i]);
    return a;
}

int pint(const Tool& t, const char* k, int d) { return t.pi(k, d); }

}  // namespace

// ---------------------------------------------------------------- find line
class FindLine : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        auto [crop, o] = this->crop(image, ctx);
        if (crop.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "empty region");
        cv::Mat g;
        plane(crop, ps("channel", "auto")).convertTo(g, CV_32F);
        int H = g.rows, W = g.cols;
        int n = std::max(3, pi("num_calipers", 12));
        int width = std::max(1, pi("caliper_width", 6));
        std::string pol = ps("polarity", "any");
        double minc = pf("min_contrast", 8.0);
        Rays cands;
        for (int i = 0; i < n; i++) {
            double xc = (i + 0.5) * W / (double)n;
            int x0 = (int)std::max(0.0, xc - width / 2.0);
            int x1 = std::min(W, x0 + width);
            std::vector<Cand> ray;
            if (x1 > x0) {
                auto prof = mean_axis1(g.colRange(x0, x1));
                for (auto& [pos, st] : profile_edges(prof, pol, minc)) ray.push_back({{xc, pos}, st});
            }
            cands.push_back(ray);
        }
        int need = std::max(3, pi("min_found", 6));
        double max_rms = pf("max_rms", 1.5);
        auto [pts, enough] = edge_points(cands, Kind::line, need, pi("num_to_ignore", 2), max_rms);
        std::vector<Graphic> guides;
        for (int i = 0; i < n; i++) {
            double x = (i + 0.5) * W / (double)n;
            guides.push_back(abs_graphic("line", to_image(*this, o.x, o.y, {{x, 0.0}, {x, (double)(H - 1)}}), "search"));
        }
        if (!enough) return too_few_points(*this, ctx, (int)pts.size(), need, t0, guides);

        auto P = to_image(*this, o.x, o.y, pts);
        std::vector<float> Pf;
        for (auto& q : P) {
            Pf.push_back(q.x);
            Pf.push_back(q.y);
        }
        auto f = fit_line<float>(Pf);
        int N = (int)P.size();
        std::vector<float> cen(Pf.size());
        for (int i = 0; i < N; i++) {
            cen[2 * i] = Pf[2 * i] - f.c[0];
            cen[2 * i + 1] = Pf[2 * i + 1] - f.c[1];
        }
        auto t = la::matvec(cen.data(), f.d, N, 2);
        float tmin = *std::min_element(t.begin(), t.end()), tmax = *std::max_element(t.begin(), t.end());
        float p0[2] = {f.c[0] + f.d[0] * tmin, f.c[1] + f.d[1] * tmin};
        float p1[2] = {f.c[0] + f.d[0] * tmax, f.c[1] + f.d[1] * tmax};
        if (p1[0] < p0[0]) {
            std::swap(p0[0], p1[0]);
            std::swap(p0[1], p1[1]);
        }
        double angle = std::atan2((double)(p1[1] - p0[1]), (double)(p1[0] - p0[0])) * kRadToDeg;
        std::vector<float> sq(f.resid.size());
        for (size_t i = 0; i < sq.size(); i++) sq[i] = f.resid[i] * f.resid[i];
        double rms = (double)std::sqrt(npy::mean_f32(sq.data(), sq.size()));
        bool ok = rms <= max_rms;
        std::string st = ok ? "pass" : "fail";
        ctx.share[id] = Json{{"type", "line"}, {"p0", arr(p0, 2)}, {"p1", arr(p1, 2)}, {"point", arr(f.c, 2)},
                             {"direction", arr(f.d, 2)}, {"angle", angle}};
        auto gr = roi_graphic(ctx, st);
        gr.insert(gr.end(), guides.begin(), guides.end());
        gr.push_back(abs_graphic("line", {{p0[0], p0[1]}, {p1[0], p1[1]}}, st, id + " " + pyfmt("%.2f", angle) + " deg"));
        for (auto& q : P)
            gr.push_back(abs_graphic("rect", {{(float)(q.x - 1.5), (float)(q.y - 1.5)}, {(float)(q.x + 1.5), (float)(q.y + 1.5)}}, "pass"));
        int found = 0;
        for (auto& c : cands) found += !c.empty();
        Json v = {{"angle_deg", py_round(angle, 3)}, {"rms_px", py_round(rms, 3)}, {"found", found}, {"used", N},
                  {"x0", py_round(p0[0], 2)}, {"y0", py_round(p0[1], 2)}, {"x1", py_round(p1[0], 2)}, {"y1", py_round(p1[1], 2)}};
        return ToolResult(id, type_name, st, std::max(0.0, 1.0 - rms / (2 * max_rms)), v, gr,
                          ok ? "" : "fit error " + pyfmt("%.2f", rms) + " px over " + pyfmt("%.2f", max_rms), now_ms() - t0);
    }
};
VC_REGISTER("find_line", FindLine)

// ---------------------------------------------------------------- find circle
class FindCircle : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        auto [g, o] = raw_plane(*this, image, ctx);
        if (g.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "empty region");
        int H = g.rows, W = g.cols;
        double cx0 = (W - 1) / 2.0, cy0 = (H - 1) / 2.0;
        double R = std::min(W, H) / 2.0;
        double r_in = R * pf("search_inner", 0.45);
        double r_out = R * pf("search_outer", 1.0) - 1.0;
        int n = std::max(6, pi("num_calipers", 24));
        std::string pol = ps("polarity", "any");
        double minc = pf("min_contrast", 8.0);
        int steps = std::max(8, (int)((r_out - r_in) * 2));
        std::vector<double> rs(steps);
        double dstep = (r_out - r_in) / (double)(steps - 1);
        for (int i = 0; i < steps; i++) rs[i] = (double)i * dstep + r_in;
        rs[steps - 1] = r_out;
        Rays cands;
        for (int k = 0; k < n; k++) {
            double a = 2 * kPi * k / n;
            double ca = std::cos(a), sa = std::sin(a);
            cv::Mat xs(1, steps, CV_32F), ys(1, steps, CV_32F), prof;
            for (int i = 0; i < steps; i++) {
                xs.at<float>(0, i) = (float)(cx0 + rs[i] * ca);
                ys.at<float>(0, i) = (float)(cy0 + rs[i] * sa);
            }
            cv::remap(g, prof, xs, ys, cv::INTER_LINEAR);
            std::vector<float> pv((const float*)prof.data, (const float*)prof.data + steps);
            std::vector<Cand> ray;
            for (auto& [pos, st] : profile_edges(pv, pol, minc, (r_out - r_in) / (double)(steps - 1))) {
                double rr = r_in + (r_out - r_in) * pos / (double)(steps - 1);
                ray.push_back({{cx0 + rr * ca, cy0 + rr * sa}, st});
            }
            cands.push_back(ray);
        }
        int need = std::max(4, pi("min_found", 10));
        double max_rms = pf("max_rms", 1.5);
        auto [pts, enough] = edge_points(cands, Kind::circle, need, pi("num_to_ignore", 3), max_rms);
        std::vector<Graphic> guides;
        for (double rr : {r_in, r_out}) {
            std::vector<Pt> ring;
            for (int i = 0; i < 48; i++) {
                double a = i == 47 ? 2 * kPi : (double)i * (2 * kPi / 47.0);
                ring.emplace_back(cx0 + rr * std::cos(a), cy0 + rr * std::sin(a));
            }
            guides.push_back(abs_graphic("poly", to_image(*this, o.x, o.y, ring), "search"));
        }
        if (!enough) return too_few_points(*this, ctx, (int)pts.size(), need, t0, guides);
        auto P = to_image(*this, o.x, o.y, pts);
        std::vector<float> Pf;
        for (auto& q : P) {
            Pf.push_back(q.x);
            Pf.push_back(q.y);
        }
        auto f = fit_circle<float>(Pf);
        std::vector<float> sq(f.resid.size());
        for (size_t i = 0; i < sq.size(); i++) sq[i] = f.resid[i] * f.resid[i];
        double rms = (double)std::sqrt(npy::mean_f32(sq.data(), sq.size()));
        double rmin = pf("min_radius", 0), rmax = pf("max_radius", 0);
        double r = (double)f.r;
        std::vector<std::string> why;
        if (rms > max_rms) why.push_back("fit error " + pyfmt("%.2f", rms) + " px over " + pyfmt("%.2f", max_rms));
        if (rmin != 0 && r < rmin) why.push_back("radius " + pyfmt("%.2f", r) + " under " + pyfmt("%.2f", rmin));
        if (rmax != 0 && r > rmax) why.push_back("radius " + pyfmt("%.2f", r) + " over " + pyfmt("%.2f", rmax));
        bool ok = why.empty();
        std::string st = ok ? "pass" : "fail";
        ctx.share[id] = Json{{"type", "circle"}, {"center", {(double)f.cx, (double)f.cy}}, {"radius", r}};
        auto gr = roi_graphic(ctx, st);
        gr.insert(gr.end(), guides.begin(), guides.end());
        std::vector<cv::Point2f> circ;
        for (int i = 0; i < 49; i++) {
            double a = i == 48 ? 2 * kPi : (double)i * (2 * kPi / 48.0);
            circ.emplace_back((float)(f.cx + r * std::cos(a)), (float)(f.cy + r * std::sin(a)));
        }
        gr.push_back(abs_graphic("poly", circ, st, id + " r=" + pyfmt("%.2f", r)));
        gr.push_back(abs_graphic("line", {{f.cx - 4, f.cy}, {f.cx + 4, f.cy}}, "pass"));
        gr.push_back(abs_graphic("line", {{f.cx, f.cy - 4}, {f.cx, f.cy + 4}}, "pass"));
        int found = 0;
        for (auto& c : cands) found += !c.empty();
        Json v = {{"cx", py_round(f.cx, 3)}, {"cy", py_round(f.cy, 3)}, {"radius_px", py_round(r, 3)},
                  {"rms_px", py_round(rms, 3)}, {"found", found}, {"used", (int)P.size()}};
        if (ctx.mm_per_px > 0) {
            // 2 * r * mm in float32 and numpy's float32 round, as the Python computes it
            float d = (float)2 * f.r * (float)ctx.mm_per_px;
            v["diameter_mm"] = (double)(std::nearbyint(d * 10000.0f) / 10000.0f);
        }
        std::string msg;
        for (size_t i = 0; i < why.size(); i++) msg += (i ? "; " : "") + why[i];
        return ToolResult(id, type_name, st, std::max(0.0, 1.0 - rms / (2 * max_rms)), v, gr, msg, now_ms() - t0);
    }
};
VC_REGISTER("find_circle", FindCircle)

// ---------------------------------------------------------------- histogram
void record_histogram(Tool& t, Context& ctx, const std::vector<float>& px, int lo, int hi, double mean, const std::string& status,
                      const std::string& title);

class Histogram : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        auto [crop, o] = this->crop(image, ctx);
        (void)o;
        if (crop.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "empty region");
        cv::Mat g = to_gray(crop);
        std::vector<float> px;
        bool use_mask = !crop_mask.empty() && crop_mask.size() == g.size();
        for (int y = 0; y < g.rows; y++)
            for (int x = 0; x < g.cols; x++)
                if (!use_mask || crop_mask.at<uint8_t>(y, x) > 0) px.push_back((float)g.at<uint8_t>(y, x));
        if (px.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "no pixels in region");
        int lo_g = pi("band_low", 0), hi_g = pi("band_high", 255);
        float mean = npy::mean_f32(px.data(), px.size()), sd = npy::std_f32(px.data(), px.size());
        float mn = *std::min_element(px.begin(), px.end()), mx = *std::max_element(px.begin(), px.end());
        std::vector<float> s = px;
        size_t n = s.size(), h = n / 2;
        std::nth_element(s.begin(), s.begin() + h, s.end());
        float med = s[h];
        if (n % 2 == 0) {
            float a = *std::max_element(s.begin(), s.begin() + h);
            med = (a + med) / 2.0f;
        }
        size_t inband = 0;
        for (float v : px) inband += (v >= (float)lo_g && v <= (float)hi_g);
        Json v = {{"mean", py_round(mean, 2)}, {"std", py_round(sd, 2)}, {"min", (int)mn}, {"max", (int)mx},
                  {"median", py_round(med, 1)}, {"band_pct", py_round(100.0 * ((double)inband / (double)n), 2)},
                  {"pixels", (int)n}};
        std::vector<std::string> why;
        auto limit = [&](const char* key, const char* lo_key, const char* hi_key) {
            double val = v[key].get<double>();
            if (has(lo_key) && pf(lo_key, 0) > 0 && val < pf(lo_key, 0))
                why.push_back(std::string(key) + " " + pyfmt("%.1f", val) + " under " + pyfmt("%.1f", pf(lo_key, 0)));
            if (has(hi_key) && pf(hi_key, 0) > 0 && val > pf(hi_key, 0))
                why.push_back(std::string(key) + " " + pyfmt("%.1f", val) + " over " + pyfmt("%.1f", pf(hi_key, 0)));
        };
        limit("mean", "mean_min", "mean_max");
        limit("std", "std_min", "std_max");
        limit("band_pct", "band_pct_min", "band_pct_max");
        bool ok = why.empty();
        std::string st = ok ? "pass" : "fail";
        if (ctx.want_records)
            record_histogram(*this, ctx, px, lo_g, hi_g, v["mean"].get<double>(), st,
                             id + " - " + std::to_string(n) + " pixels, mean " + pyfmt("%.1f", v["mean"].get<double>()) +
                                 ", spread " + pyfmt("%.1f", v["std"].get<double>()));
        std::string msg;
        for (size_t i = 0; i < why.size(); i++) msg += (i ? "; " : "") + why[i];
        return ToolResult(id, type_name, st, ok ? 1.0 : 0.0, v, roi_graphic(ctx, st, id + " " + pyfmt("%.0f", v["mean"].get<double>())),
                          msg, now_ms() - t0);
    }
};
VC_REGISTER("histogram", Histogram)

// ---------------------------------------------------------------- distance / angle
namespace {

std::array<double, 2> point_of(const Json& f) {
    std::string t = f.value("type", "");
    if (t == "circle") return {f["center"][0].get<double>(), f["center"][1].get<double>()};
    if (t == "line")
        return {(f["p0"][0].get<double>() + f["p1"][0].get<double>()) / 2.0, (f["p0"][1].get<double>() + f["p1"][1].get<double>()) / 2.0};
    if (f.contains("point")) return {f["point"][0].get<double>(), f["point"][1].get<double>()};
    return {0.0, 0.0};
}

double point_line(const std::array<double, 2>& pt, const Json& line) {
    double c[2] = {line["point"][0].get<double>(), line["point"][1].get<double>()};
    double d[2] = {line["direction"][0].get<double>(), line["direction"][1].get<double>()};
    double nrm[2] = {-d[1], d[0]}, v[2] = {pt[0] - c[0], pt[1] - c[1]};
    return std::abs(la::dot(v, nrm, 2));
}

}  // namespace

class Distance : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        (void)image;
        double t0 = now_ms();
        std::string a_id = has("first") ? py_str(p["first"]) : "", b_id = has("second") ? py_str(p["second"]) : "";
        auto A = passed_feature(ctx, a_id);
        std::optional<Json> B;
        if (!b_id.empty()) B = passed_feature(ctx, b_id);
        std::vector<std::string> missing;
        if (!a_id.empty() && !A) missing.push_back(a_id);
        if (!b_id.empty() && !B) missing.push_back(b_id);
        if (a_id.empty() || !missing.empty()) {
            std::string m;
            for (size_t i = 0; i < missing.size(); i++) m += (i ? ", " : "") + missing[i];
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {},
                              "no result from " + (m.empty() ? std::string("'first'") : m) + " - it must run earlier and succeed", now_ms() - t0);
        }
        Json vals = Json::object();
        std::vector<Graphic> gr;
        std::string unit = "px", what;
        double value;
        std::string ta = (*A).value("type", "");
        if (!B) {
            if (ta != "circle") return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "a single feature must be a circle");
            value = 2 * (*A)["radius"].get<double>();
            what = "diameter";
        } else {
            std::string tb = (*B).value("type", "");
            if (ta == "line" && tb == "line") {
                double da[2] = {(*A)["direction"][0].get<double>(), (*A)["direction"][1].get<double>()};
                double db[2] = {(*B)["direction"][0].get<double>(), (*B)["direction"][1].get<double>()};
                double cosang = std::abs(la::dot(da, db, 2)) / (la::norm(da, 2) * la::norm(db, 2));
                value = std::acos(std::min(1.0, cosang)) * (180.0 / 3.141592653589793);
                what = "angle";
                unit = "deg";
                vals["separation_px"] = py_round(point_line(point_of(*B), *A), 3);
            } else if (ta == "line" || tb == "line") {
                const Json& line = ta == "line" ? *A : *B;
                const Json& other = ta == "line" ? *B : *A;
                auto pt = point_of(other);
                value = point_line(pt, line);
                what = "point_to_line";
                double c[2] = {line["point"][0].get<double>(), line["point"][1].get<double>()};
                double d[2] = {line["direction"][0].get<double>(), line["direction"][1].get<double>()};
                double v[2] = {pt[0] - c[0], pt[1] - c[1]};
                double s = la::dot(v, d, 2);
                gr.push_back(abs_graphic("line", {{(float)pt[0], (float)pt[1]}, {(float)(c[0] + d[0] * s), (float)(c[1] + d[1] * s)}}, "pass",
                                         pyfmt("%.2f", value)));
            } else {
                auto pa = point_of(*A), pb = point_of(*B);
                double dv[2] = {pa[0] - pb[0], pa[1] - pb[1]};
                value = la::norm(dv, 2);
                what = "point_to_point";
                gr.push_back(abs_graphic("line", {{(float)pa[0], (float)pa[1]}, {(float)pb[0], (float)pb[1]}}, "pass", pyfmt("%.2f", value)));
            }
        }
        double shown = value;
        if (unit == "px" && ctx.mm_per_px > 0) {
            shown = value * ctx.mm_per_px;
            unit = "mm";
        }
        vals["measure"] = what;
        vals["value"] = py_round(shown, 4);
        vals["unit"] = unit;
        vals["value_px"] = py_round(value, 3);
        double lo = pf("min_value", 0), hi = pf("max_value", 0);
        std::vector<std::string> why;
        if (lo != 0 && shown < lo) why.push_back(what + " " + pyfmt("%.3f", shown) + " " + unit + " under " + pyfmt("%.3f", lo));
        if (hi != 0 && shown > hi) why.push_back(what + " " + pyfmt("%.3f", shown) + " " + unit + " over " + pyfmt("%.3f", hi));
        bool ok = why.empty();
        for (auto& g : gr) g.status = ok ? "pass" : "fail";
        std::string msg;
        for (size_t i = 0; i < why.size(); i++) msg += (i ? "; " : "") + why[i];
        return ToolResult(id, type_name, ok ? "pass" : "fail", ok ? 1.0 : 0.0, vals, gr, msg, now_ms() - t0);
    }
};
VC_REGISTER("distance", Distance)

// ---------------------------------------------------------------- caliper
void record_profile(Tool& t, Context& ctx, const std::vector<float>& prof, const std::vector<float>& grad,
                    const std::vector<std::tuple<double, std::string, std::string>>& marks,
                    const std::vector<std::tuple<double, double, std::string>>& spans, const std::string& title);

class Caliper : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        auto [crop, o] = this->crop(image, ctx);
        if (crop.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "empty region");
        cv::Mat g;
        plane(crop, ps("channel", "auto")).convertTo(g, CV_32F);
        auto prof = mean_axis0(g);
        // geometry._signed_edges
        auto d = blur_gradient(prof);
        float mc = (float)pf("min_contrast", 8.0);
        struct E { double pos; int sign; double c; };
        std::vector<E> edges;
        std::vector<float> ad(d.size());
        for (size_t i = 0; i < d.size(); i++) ad[i] = std::abs(d[i]);
        for (int i = 1; i + 1 < (int)d.size(); i++) {
            float a = ad[i];
            if (a >= mc && a >= ad[i - 1] && a > ad[i + 1]) edges.push_back({subpixel_peak(ad, i), d[i] > 0 ? 1 : -1, (double)a});
        }
        int first = ps("pair_polarity", "dark") == "dark" ? -1 : 1;
        double exp_w = pf("expected_width", 0), tol_w = pf("width_tolerance", 0);
        struct Pair { double start, end, width, centre, contrast; };
        std::vector<Pair> pairs;
        for (size_t i = 0; i < edges.size(); i++) {
            if (edges[i].sign != first) continue;
            for (size_t j = i + 1; j < edges.size(); j++) {
                if (edges[j].sign == -edges[i].sign) {
                    double w = edges[j].pos - edges[i].pos;
                    if (exp_w != 0 && tol_w != 0 && std::abs(w - exp_w) > tol_w) break;
                    pairs.push_back({edges[i].pos, edges[j].pos, w, (edges[i].pos + edges[j].pos) / 2.0, std::min(edges[i].c, edges[j].c)});
                    break;
                }
                if (edges[j].sign == edges[i].sign) break;
            }
        }
        int n_max = pi("max_results", 20);
        std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return -a.contrast < -b.contrast; });
        if ((int)pairs.size() > n_max) pairs.resize(std::max(0, n_max));
        std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.centre < b.centre; });
        double mm = ctx.mm_per_px;
        if (ctx.want_records) {
            std::vector<std::tuple<double, std::string, std::string>> marks;
            for (auto& e : edges) {
                bool in = false;
                for (auto& q : pairs) in |= std::abs(e.pos - q.start) < 1e-6 || std::abs(e.pos - q.end) < 1e-6;
                marks.emplace_back(e.pos, in ? "pass" : "fail", "");
            }
            std::vector<std::tuple<double, double, std::string>> spans;
            for (auto& q : pairs) spans.emplace_back(q.start, q.end, "pass");
            record_profile(*this, ctx, prof, d, marks, spans,
                           id + " - " + std::to_string(edges.size()) + " edges, " + std::to_string(pairs.size()) + " pairs (green: in a pair)");
        }
        double k = mm > 0 ? mm : 1.0;
        std::vector<double> widths, centres;
        for (auto& q : pairs) {
            widths.push_back(q.width * k);
            centres.push_back(q.centre);
        }
        double pitch = 0.0;
        if (centres.size() > 1) {
            std::vector<double> df;
            for (size_t i = 1; i < centres.size(); i++) df.push_back(centres[i] - centres[i - 1]);
            pitch = npy::sum_f64(df.data(), df.size()) / (double)df.size() * k;
        }
        std::string unit = mm > 0 ? "mm" : "px";
        std::vector<std::string> why;
        int n_exp = pi("expect_count", 0);
        if (n_exp && (int)pairs.size() != n_exp)
            why.push_back("found " + std::to_string(pairs.size()) + " pairs, expected " + std::to_string(n_exp));
        if (pairs.empty()) why.push_back("no edge pair found");
        double nom = pf("nominal_width", 0), tol = pf("tolerance", 0);
        if (nom != 0 && tol != 0) {
            int bad = 0;
            for (double w : widths) bad += std::abs(w - nom) > tol;
            if (bad)
                why.push_back(std::to_string(bad) + " width" + (bad > 1 ? "s" : "") + " out of " + pyfmt("%.3f", nom) + " +- " + pyfmt("%.3f", tol) +
                              " " + unit);
        }
        bool ok = why.empty();
        std::string st = ok ? "pass" : "fail";
        int H = g.rows, Wc = g.cols;
        double mid = H / 2.0;
        auto gr = roi_graphic(ctx, st);
        double hx = Wc - 5 - std::min(12.0, Wc / 6.0), hy = std::min(6.0, H / 3.0);
        gr.push_back(abs_graphic("line", to_image(*this, o.x, o.y, {{4.0, mid}, {(double)(Wc - 5), mid}}), "search"));
        gr.push_back(abs_graphic("poly", to_image(*this, o.x, o.y, {{hx, mid - hy}, {(double)(Wc - 5), mid}, {hx, mid + hy}}), "search"));
        for (auto& q : pairs) {
            gr.push_back(abs_graphic("line", to_image(*this, o.x, o.y, {{q.start, 0.0}, {q.start, (double)(H - 1)}}), st));
            gr.push_back(abs_graphic("line", to_image(*this, o.x, o.y, {{q.end, 0.0}, {q.end, (double)(H - 1)}}), st, pyfmt("%.2f", q.width * k)));
        }
        Json wl = Json::array();
        for (double w : widths) wl.push_back(py_round(w, 4));
        Json v = {{"count", (int)pairs.size()}, {"unit", unit}, {"widths", wl}, {"pitch", py_round(pitch, 4)}};
        if (!widths.empty()) v["mean_width"] = py_round(npy::sum_f64(widths.data(), widths.size()) / (double)widths.size(), 4);
        std::string msg;
        for (size_t i = 0; i < why.size(); i++) msg += (i ? "; " : "") + why[i];
        return ToolResult(id, type_name, st, ok ? 1.0 : 0.0, v, gr, msg, now_ms() - t0);
    }
};
VC_REGISTER("caliper", Caliper)

}  // namespace vc
