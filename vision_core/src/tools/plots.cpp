// vision_station/tools/plots.py: the record images a caliper or histogram keeps
// for the Studio (LastRun.Projection, LastRun.Histogram), drawn with OpenCV.
#include "../core/pyfmt.hpp"
#include "../core/tool.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <tuple>

namespace vc {

namespace {

const cv::Scalar BG(246, 246, 246), GRID(222, 222, 222), AXIS(150, 150, 150), INK(60, 60, 60);
const cv::Scalar PROFILE(180, 110, 40), GRADIENT(40, 140, 230), PASS(74, 158, 30), FAIL(58, 52, 214);
const int FONT = cv::FONT_HERSHEY_SIMPLEX;

struct Box { int left, top, right, bottom; };

cv::Mat frame(int width, int height, const std::string& title, Box& box) {
    cv::Mat img(height, width, CV_8UC3, BG);
    int left = 46, right = 12, top = title.empty() ? 10 : 24, bottom = 24;
    for (int k = 1; k < 4; k++) {
        int y = top + (height - top - bottom) * k / 4;
        cv::line(img, {left, y}, {width - right, y}, GRID, 1);
    }
    cv::rectangle(img, {left, top}, {width - right, height - bottom}, AXIS, 1);
    if (!title.empty()) cv::putText(img, title, {left, 16}, FONT, 0.45, INK, 1, cv::LINE_AA);
    box = {left, top, width - right, height - bottom};
    return img;
}

int X(double i, int n, const Box& b) {
    int span = std::max(1, n - 1);
    return (int)py_round_int(b.left + (b.right - b.left) * i / span);
}

int Y(double v, double lo, double hi, const Box& b) {
    double rng = (hi - lo) != 0 ? (hi - lo) : 1.0;
    return (int)py_round_int(b.bottom - (b.bottom - b.top) * (v - lo) / rng);
}

cv::Mat shade(const cv::Mat& img, int x0, int x1, const Box& b, const cv::Scalar& c) {
    cv::Mat over = img.clone(), out;
    cv::rectangle(over, {std::min(x0, x1), b.top + 1}, {std::max(x0, x1), b.bottom - 1}, c, -1);
    cv::addWeighted(over, 0.12, img, 0.88, 0, out);
    return out;
}

}  // namespace

void record_profile(Tool& t, Context& ctx, const std::vector<float>& prof, const std::vector<float>& grad,
                    const std::vector<std::tuple<double, std::string, std::string>>& marks,
                    const std::vector<std::tuple<double, double, std::string>>& spans, const std::string& title) {
    const int width = 720, height = 260;
    Box b;
    cv::Mat img = frame(width, height, title, b);
    int n = (int)prof.size();
    if (n >= 2) {
        for (auto& [a, e, st] : spans) img = shade(img, X(a, n, b), X(e, n, b), b, st == "pass" ? PASS : FAIL);
        double lo = *std::min_element(prof.begin(), prof.end()), hi = *std::max_element(prof.begin(), prof.end());
        std::vector<cv::Point> pts;
        for (int i = 0; i < n; i++) pts.emplace_back(X(i, n, b), Y(prof[i], lo, hi, b));
        cv::polylines(img, std::vector<std::vector<cv::Point>>{pts}, false, PROFILE, 2, cv::LINE_AA);
        if (!grad.empty()) {
            double m = 0;
            for (float v : grad) m = std::max(m, (double)std::abs(v));
            if (m == 0) m = 1.0;
            int zero = Y(0, -m, m, b);
            cv::line(img, {b.left, zero}, {b.right, zero}, AXIS, 1, cv::LINE_4);
            std::vector<cv::Point> gp;
            for (int i = 0; i < (int)grad.size(); i++) gp.emplace_back(X(i, n, b), Y(grad[i], -m, m, b));
            cv::polylines(img, std::vector<std::vector<cv::Point>>{gp}, false, GRADIENT, 1, cv::LINE_AA);
        }
        for (auto& [pos, st, label] : marks) {
            int x = X(pos, n, b);
            cv::Scalar c = st == "pass" ? PASS : FAIL;
            cv::line(img, {x, b.top}, {x, b.bottom}, c, 1);
            if (!label.empty()) cv::putText(img, label, {x + 3, b.top + 12}, FONT, 0.38, c, 1, cv::LINE_AA);
        }
        cv::putText(img, "brightness", {b.left + 4, b.bottom - 6}, FONT, 0.35, PROFILE, 1, cv::LINE_AA);
        if (!grad.empty()) cv::putText(img, "edge strength", {b.left + 80, b.bottom - 6}, FONT, 0.35, GRADIENT, 1, cv::LINE_AA);
        cv::putText(img, "0", {b.left - 10, height - 8}, FONT, 0.35, INK, 1, cv::LINE_AA);
        cv::putText(img, std::to_string(n - 1), {b.right - 24, height - 8}, FONT, 0.35, INK, 1, cv::LINE_AA);
    }
    ctx.records[t.id]["LastRun.Projection"] = img;
}

void record_histogram(Tool& t, Context& ctx, const std::vector<float>& px, int lo, int hi, double mean, const std::string& status,
                      const std::string& title) {
    const int width = 720, height = 260;
    Box b;
    cv::Mat img = frame(width, height, title, b);
    if (!px.empty()) {
        std::vector<float> hist(256, 0.0f);
        for (float v : px) hist[std::clamp((int)(long long)v, 0, 255)] += 1.0f;
        double hmax = *std::max_element(hist.begin(), hist.end());
        if (hmax == 0) hmax = 1.0;
        cv::Scalar c = status == "pass" ? PASS : FAIL;
        img = shade(img, X(lo, 256, b), X(hi, 256, b), b, c);
        for (int g = 0; g < 256; g++)
            if (hist[g] != 0) cv::line(img, {X(g, 256, b), b.bottom - 1}, {X(g, 256, b), Y(hist[g], 0.0, hmax, b)}, PROFILE, 2);
        int x = X(mean, 256, b);
        cv::line(img, {x, b.top}, {x, b.bottom}, c, 1);
        cv::putText(img, "mean " + pyfmt("%.1f", mean), {x + 3, b.top + 12}, FONT, 0.38, c, 1, cv::LINE_AA);
        for (int g : {0, 128, 255}) cv::putText(img, std::to_string(g), {X(g, 256, b) - 8, height - 8}, FONT, 0.35, INK, 1, cv::LINE_AA);
    }
    ctx.records[t.id]["LastRun.Histogram"] = img;
}

}  // namespace vc
