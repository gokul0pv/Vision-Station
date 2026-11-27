// A job's own trained network (vision_station/core/advanced.py, inspection
// side): after a part or text check passes, the network must also find that
// part or text where Align puts it -- unless the check itself is sure.
// Training stays in Vision Station; Vision Core runs what it trained.
#pragma once

#include "onnx.hpp"
#include "tool.hpp"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vc::advanced {

class Detector {
public:
    Detector(const std::string& folder, const Json& meta);
    double threshold(const std::string& tool_id) const;
    std::pair<int, int> tile_for(const std::array<double, 4>& region, int W, int H) const;
    // [[x1, y1, x2, y2, score, cls], ...] (float32) in the frame's pixels
    std::vector<std::array<float, 6>> detect_tile(const cv::Mat& frame, int tx, int ty) const;
    bool contains(std::pair<int, int> tile, const std::array<double, 4>& region, int W, int H, double margin = 16) const;
    std::map<std::string, std::pair<int, int>> plan(const std::vector<std::pair<std::string, std::array<double, 4>>>& regions, int W, int H) const;
    std::pair<double, std::optional<std::array<double, 4>>> find(const cv::Mat& frame, const std::array<double, 4>& region, int cls,
                                                                  std::map<std::pair<int, int>, std::vector<std::array<float, 6>>>& cache,
                                                                  std::optional<std::pair<int, int>> tile, double pad) const;

    std::string folder, path;
    std::map<std::string, int> tools;            // check id -> class
    double zoom = 1.0, min_conf = 0.35;
    int imgsz = 1280, src = 1280;
    std::map<std::string, double> thresholds;
};

// advanced.load: the job's Detector, or null (none trained, or its files gone)
std::shared_ptr<Detector> load(const std::string& folder, const Json& spec);
// advanced.apply: r confirmed (or overruled) by the network, changed in place
void apply(const Detector& det, Tool& tool, ToolResult& r, const cv::Mat& frame, Context& ctx);

}  // namespace vc::advanced
