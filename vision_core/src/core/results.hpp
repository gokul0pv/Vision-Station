// The run log (vision_station/core/results.py ResultStore): every inspection
// into <side>/results/results.db -- the same tables Vision Station writes, so
// its history views and Vision Next read Vision Core's runs too -- written by
// a background thread from a bounded queue. Logging never stops inspection:
// a full queue drops the oldest entry, a failed write is reported and skipped.
#pragma once

#include "types.hpp"

#include <memory>
#include <string>

namespace vc {

class ResultStore {
public:
    // archive: "none" | "fail" | "all" -- which pictures are kept as JPEG (quality 85)
    ResultStore(const std::string& side_folder, const std::string& project_id, const std::string& archive = "none", int max_archived = 2000);
    ~ResultStore();
    void write(const Json& inspection, const cv::Mat& picture, const std::string& operator_name = "");
    std::string last_error() const;
    long long written() const;

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace vc
