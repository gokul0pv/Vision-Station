// One loaded ONNX model (vision_station/core/inference.py Model): the same
// session settings -- full graph optimisation; on DirectML no memory pattern
// and sequential execution -- and one run at a time on the GPU.
#pragma once

#include <map>
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
    // custom metadata (YOLO keeps its class names here)
    std::map<std::string, std::string> metadata() const;
    std::string device;            // where it actually runs
    std::string path;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// "GPU" unless VS_DEVICE=CPU (Vision Station's text reader choice)
std::string ocr_device();
// inference.resolve_device: "auto"/"" -> "GPU" only when VS_DEVICE=GPU, else "CPU"
std::string resolve_device(const std::string& device = "");
// inference.load: one cached model per (absolute path, device); models are
// never destroyed while the process runs (DirectML sessions freed mid-run crash)
std::shared_ptr<OnnxModel> load_model(const std::string& path, const std::string& device = "");

}  // namespace vc
