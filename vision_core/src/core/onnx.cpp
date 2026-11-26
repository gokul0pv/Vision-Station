#include "onnx.hpp"

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <cctype>
#include <cstdlib>
#include <filesystem>

namespace vc {

namespace {
Ort::Env& env() {
    static Ort::Env e(ORT_LOGGING_LEVEL_ERROR, "vision-core");
    return e;
}
std::mutex g_dml_lock;   // DirectML takes one run at a time (as Vision Station's _DML_LOCK)
}  // namespace

std::string ocr_device() {
    const char* v = std::getenv("VS_DEVICE");
    std::string s = v ? v : "";
    for (auto& c : s) c = (char)std::toupper((unsigned char)c);
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    return s == "CPU" ? "CPU" : "GPU";
}

struct OnnxModel::Impl {
    std::unique_ptr<Ort::Session> sess;
    std::string in_name;
    std::vector<std::string> out_names;
    std::mutex own_lock;
    std::mutex* lock = &own_lock;
};

OnnxModel::OnnxModel(const std::string& p, const std::string& dev) : path(p), impl_(std::make_unique<Impl>()) {
    std::wstring wpath = std::filesystem::u8path(p).wstring();
    auto make = [&](bool gpu) {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (gpu) {
            so.DisableMemPattern();
            so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, 0));
        }
        return std::make_unique<Ort::Session>(env(), wpath.c_str(), so);
    };
    device = "CPU";
    if (dev == "GPU") {
        try {
            impl_->sess = make(true);
            device = "GPU";
            impl_->lock = &g_dml_lock;
        } catch (const Ort::Exception&) {
            impl_->sess.reset();
        }
    }
    if (!impl_->sess) impl_->sess = make(false);
    Ort::AllocatorWithDefaultOptions al;
    impl_->in_name = impl_->sess->GetInputNameAllocated(0, al).get();
    for (size_t i = 0; i < impl_->sess->GetOutputCount(); i++) impl_->out_names.push_back(impl_->sess->GetOutputNameAllocated(i, al).get());
}

OnnxModel::~OnnxModel() = default;

std::vector<int64_t> OnnxModel::input_shape() const {
    return impl_->sess->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<Tensor> OnnxModel::run(const float* data, const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(data), n, shape.data(), shape.size());
    const char* in_names[] = {impl_->in_name.c_str()};
    std::vector<const char*> outs;
    for (auto& s : impl_->out_names) outs.push_back(s.c_str());
    std::vector<Ort::Value> res;
    {
        std::lock_guard<std::mutex> lk(*impl_->lock);
        res = impl_->sess->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, outs.data(), outs.size());
    }
    std::vector<Tensor> out;
    for (auto& v : res) {
        Tensor t;
        auto info = v.GetTensorTypeAndShapeInfo();
        t.shape = info.GetShape();
        size_t m = info.GetElementCount();
        const float* p = v.GetTensorData<float>();
        t.data.assign(p, p + m);
        out.push_back(std::move(t));
    }
    return out;
}

}  // namespace vc
