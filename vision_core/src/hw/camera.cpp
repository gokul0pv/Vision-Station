#include "camera.hpp"

#include <pylon/PylonIncludes.h>
#include <opencv2/core.hpp>

#include <cctype>
#include <regex>
#include <stdexcept>

using namespace Pylon;

namespace vc::hw {

namespace {
struct PylonRuntime {
    PylonRuntime() { PylonInitialize(); }
    ~PylonRuntime() { PylonTerminate(); }
};
PylonRuntime& runtime() {
    static PylonRuntime r;
    return r;
}
bool is_ip(const std::string& s) {
    static const std::regex rx(R"(^\d{1,3}(\.\d{1,3}){3}$)");
    return std::regex_match(s, rx);
}
std::string first_line(const std::string& s) { return s.substr(0, s.find('\n')); }
}  // namespace

struct Camera::Impl {
    std::unique_ptr<CInstantCamera> cam;
    CImageFormatConverter conv;
    int rotate = 0;
};

Camera::Camera() : d_(std::make_unique<Impl>()) {
    runtime();
    d_->conv.OutputPixelFormat = PixelType_BGR8packed;
}

Camera::~Camera() { close(); }

bool Camera::is_open() const { return d_->cam && d_->cam->IsOpen(); }

std::string Camera::model() const {
    if (!d_->cam) return "";
    return std::string(d_->cam->GetDeviceInfo().GetModelName().c_str());
}

void Camera::open(const std::string& address0, int index) {
    close();
    std::string address = address0;
    while (!address.empty() && std::isspace((unsigned char)address.back())) address.pop_back();
    try {
        CTlFactory& tlf = CTlFactory::GetInstance();
        IPylonDevice* dev = nullptr;
        if (!address.empty()) {
            CDeviceInfo di;
            std::string how;
            if (is_ip(address)) {
                di.SetIpAddress(address.c_str());
                di.SetDeviceClass("BaslerGigE");
                how = "at " + address;
            } else if (std::all_of(address.begin(), address.end(), ::isdigit)) {
                di.SetSerialNumber(address.c_str());
                how = "with serial number " + address;
            } else {
                di.SetUserDefinedName(address.c_str());
                how = "named " + address;
            }
            try {
                dev = tlf.CreateDevice(di);
            } catch (const GenericException& e) {
                throw std::runtime_error("No Basler camera answered " + how + " (" + first_line(e.GetDescription()) +
                                         "). Check the cable, the power and that the PC's network port is on the camera's subnet.");
            }
        } else {
            DeviceInfoList_t devs;
            tlf.EnumerateDevices(devs);
            if (devs.empty())
                throw std::runtime_error("No Basler camera found. Enter the camera's IP address under Settings > Camera - discovery does not reach every camera, but an address always does.");
            if (index >= (int)devs.size())
                throw std::runtime_error("Camera " + std::to_string(index) + " requested but only " + std::to_string(devs.size()) + " camera(s) were found.");
            dev = tlf.CreateDevice(devs[index]);
        }
        d_->cam = std::make_unique<CInstantCamera>(dev);
        d_->cam->Open();
    } catch (const GenericException& e) {
        d_->cam.reset();
        throw std::runtime_error(std::string("camera: ") + first_line(e.GetDescription()));
    }
}

void Camera::apply(const CameraSettings& s) {
    if (!is_open()) return;
    d_->rotate = s.rotate;
    GenApi::INodeMap& nm = d_->cam->GetNodeMap();
    auto set_num = [&](std::initializer_list<const char*> names, double v) {
        for (auto n : names) {
            GenApi::INode* node = nm.GetNode(n);
            if (!node || !GenApi::IsWritable(node)) continue;
            if (auto f = dynamic_cast<GenApi::IFloat*>(node)) { f->SetValue(std::min(f->GetMax(), std::max(f->GetMin(), v))); return true; }
            if (auto i = dynamic_cast<GenApi::IInteger*>(node)) { i->SetValue(std::min(i->GetMax(), std::max(i->GetMin(), (int64_t)std::llround(v)))); return true; }
        }
        return false;
    };
    if (s.exposure_us > 0) set_num({"ExposureTime", "ExposureTimeAbs"}, s.exposure_us);
    if (s.gain >= 0) set_num({"Gain", "GainRaw"}, s.gain);
}

void Camera::start() {
    if (is_open() && !d_->cam->IsGrabbing()) d_->cam->StartGrabbing(GrabStrategy_LatestImageOnly);
}

cv::Mat Camera::grab(int timeout_ms, std::string* error) {
    if (!is_open() || !d_->cam->IsGrabbing()) return {};
    CGrabResultPtr res;
    try {
        if (!d_->cam->RetrieveResult(timeout_ms, res, TimeoutHandling_Return)) return {};
    } catch (const GenericException& e) {
        throw std::runtime_error(std::string("grab failed: ") + first_line(e.GetDescription()));
    }
    if (!res || !res->GrabSucceeded()) {
        if (error) *error = res ? std::string(res->GetErrorDescription().c_str()).substr(0, 160) : "no frame";
        return {};
    }
    CPylonImage img;
    d_->conv.Convert(img, res);
    cv::Mat m((int)img.GetHeight(), (int)img.GetWidth(), CV_8UC3, img.GetBuffer());
    cv::Mat out;
    switch (d_->rotate) {
        case 90: cv::rotate(m, out, cv::ROTATE_90_CLOCKWISE); break;
        case 180: cv::rotate(m, out, cv::ROTATE_180); break;
        case 270: cv::rotate(m, out, cv::ROTATE_90_COUNTERCLOCKWISE); break;
        default: out = m.clone();
    }
    return out;
}

void Camera::close() {
    if (!d_->cam) return;
    try {
        if (d_->cam->IsGrabbing()) d_->cam->StopGrabbing();
        d_->cam->Close();
        d_->cam->DestroyDevice();
    } catch (...) {
    }
    d_->cam.reset();
}

}  // namespace vc::hw
