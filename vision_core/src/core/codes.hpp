// Barcode / QR / DataMatrix decoders as Vision Station uses them: zbar (the
// libzbar-64.dll pyzbar ships) and libdmtx (pylibdmtx's libdmtx-64.dll),
// loaded at run time from beside the program, its codes\ folder, or
// %VC_ZBAR% / %VC_DMTX%. Each is simply absent when its DLL is.
#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace vc::codes {

struct Code {
    std::string data;        // bytes decoded as UTF-8 with replacement, as Python's decode("utf-8", "replace")
    std::string symbology;
};

bool zbar_available();
bool dmtx_available();
// pyzbar.decode(gray): every symbol zbar finds, in zbar's order
std::vector<Code> zbar_decode(const cv::Mat& gray);
// pylibdmtx.decode(gray, timeout=ms)
std::vector<Code> dmtx_decode(const cv::Mat& gray, int timeout_ms);
// bytes -> UTF-8 with U+FFFD for invalid sequences (Python's errors="replace")
std::string utf8_replace(const std::string& bytes);

}  // namespace vc::codes
