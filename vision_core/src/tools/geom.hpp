// Shared pieces of vision_station/tools/geometry.py, used by geometry.cpp and
// the tools built on it (find_ellipse, intersect, construct, fixture_npoint...).
#pragma once

#include "../core/tool.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace vc::geom {

using Pt = std::pair<double, double>;
struct Cand {
    Pt p;
    double strength;
};
using Rays = std::vector<std::vector<Cand>>;

// measure._subpixel_peak
double subpixel_peak(const std::vector<float>& prof, int i);
// geometry._plane: grey, blue, green or red -- 'auto' the one that varies most
cv::Mat plane(const cv::Mat& img, const std::string& channel);
// geometry._profile_edges: the strongest few edges, strongest first
std::vector<std::pair<double, double>> profile_edges(const std::vector<float>& prof, const std::string& polarity,
                                                     double min_contrast, double step = 1.0, int most = 4);

// fits, as numpy computes them in T (float for float32 point arrays, double for float64)
template <class T>
struct LineFit {
    T c[2], d[2];
    std::vector<T> resid;
};
template <class T>
LineFit<T> fit_line(const std::vector<T>& pts);  // pts: N x 2 row-major
template <class T>
struct CircleFit {
    T cx, cy, r;
    std::vector<T> resid;
};
template <class T>
CircleFit<T> fit_circle(const std::vector<T>& pts);

enum class Kind { line, circle };
// geometry._edge_points: (points to fit, enough calipers found the edge)
std::pair<std::vector<Pt>, bool> edge_points(const Rays& cands, Kind kind, int need, int ignore, double max_rms);

// geometry._ToImage: crop coordinates -> picture coordinates (float32 as numpy)
std::vector<cv::Point2f> to_image(const Tool& t, int ox, int oy, const std::vector<Pt>& pts);
// geometry._raw_plane: the smoothed plane of the unmasked region, or empty
std::pair<cv::Mat, cv::Point> raw_plane(Tool& t, const cv::Mat& image, const Context& ctx);
// base.passed_feature: another tool's published feature, if that tool passed
std::optional<Json> passed_feature(const Context& ctx, const std::string& id);

// numpy float32 mean / sum over rows (axis 0) and along rows (axis 1) of a CV_32F matrix
std::vector<float> mean_axis0(const cv::Mat& g);
std::vector<float> mean_axis1(const cv::Mat& g);

}  // namespace vc::geom
