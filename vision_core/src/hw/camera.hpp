// A Basler camera through pylon (vision_station/hardware/camera.py): opened
// by IP address (GigE discovery does not reach every network), serial
// number, user name or index; frames converted to BGR by pylon's own
// converter, exactly as Vision Station converts them.
#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>

namespace vc::hw {

struct CameraSettings {
    double exposure_us = 0;      // 0 = leave as the camera has it
    double gain = -1;            // < 0 = leave
    int rotate = 0;              // 0, 90, 180, 270 (clockwise)
};

class Camera {
public:
    Camera();
    ~Camera();
    // throws std::runtime_error with an operator-readable message
    void open(const std::string& address, int index = 0);
    void apply(const CameraSettings& s);
    void start();
    // one BGR frame, or empty on a soft timeout; `error` says why a frame failed
    cv::Mat grab(int timeout_ms, std::string* error = nullptr);
    void close();
    bool is_open() const;
    std::string model() const;

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace vc::hw
