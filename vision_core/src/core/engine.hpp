// A side's recipe and its run (vision_station/core/engine.py Project.inspect).
#pragma once

#include "tool.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vc {

struct Inspection {
    std::string id, project, verdict, source;
    std::vector<std::string> captures_used, captures_missing;
    double timestamp = 0, total_ms = 0;
    int width = 0, height = 0;
    std::string aborted, barcode_ng;           // empty = none
    std::vector<std::string> failed_tools;
    std::vector<ToolResult> tools;
    Json to_json() const;
};

class Project {
public:
    explicit Project(const std::string& folder);
    void load();
    Inspection inspect(const cv::Mat& image, const std::string& source = "api", const std::string& barcode = "");
    Inspection inspect(const std::map<std::string, cv::Mat>& frames, const std::string& source = "api", const std::string& barcode = "");

    std::string folder, id;
    Json spec;
    cv::Mat reference;
    Json calibration;                          // null when none
    std::vector<std::unique_ptr<Tool>> tools;
};

// cv2.imread for any path (np.fromfile + imdecode); empty when unreadable
cv::Mat imread(const std::string& path, int flags = 1);
cv::Mat undistort(const cv::Mat& img, const Json& cal);

// run fn; a C++ exception or a Windows fault (access violation, divide by
// zero, ...) becomes an error message instead of ending the process
bool guarded(const std::function<void()>& fn, std::string& error);

}  // namespace vc
