// One loaded ONNX model (vision_station/core/inference.py Model): the same
// session settings -- full graph optimisation; on DirectML no memory pattern
// and sequential execution -- and one run at a time on the GPU.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vc {

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

class OnnxModel {
public:
    // device "GPU" (DirectML, falls back to CPU when it will not start) or "CPU"
    OnnxModel(const std::string& path, const std::string& device);
    ~OnnxModel();
    // input: NCHW float32; returns every output
    std::vector<Tensor> run(const float* data, const std::vector<int64_t>& shape);
    std::vector<int64_t> input_shape() const;
    std::string device;            // where it actually runs
    std::string path;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// "GPU" unless VS_DEVICE=CPU (Vision Station's text reader choice)
std::string ocr_device();

}  // namespace vc
