// vc-run: Vision Core without a window.
//   vc-run inspect <side folder> <picture or folder> [--json]   one JSON line per picture
//   vc-run types                                              the check types built in
#include "../core/engine.hpp"

#include <opencv2/core/utility.hpp>

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace vc;

namespace vc { void release_pictures(); }

static std::vector<fs::path> pictures(const fs::path& p) {
    if (fs::is_regular_file(p)) return {p};
    std::vector<fs::path> v;
    for (auto& e : fs::directory_iterator(p)) {
        std::string ext = e.path().extension().u8string();
        for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff") v.push_back(e.path());
    }
    std::sort(v.begin(), v.end());
    return v;
}

int wmain(int argc, wchar_t** wargv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) args.push_back(fs::path(wargv[i]).u8string());
    if (args.empty()) {
        std::cerr << "vc-run inspect <side folder> <pictures> [--json]\nvc-run types\n";
        return 2;
    }
    if (args[0] == "types") {
        for (auto& [k, v] : registry()) std::cout << k << "\n";
        return 0;
    }
    if (args[0] == "inspect" && args.size() >= 3) {
        bool json = std::find(args.begin(), args.end(), "--json") != args.end();
        std::unique_ptr<Project> side;
        std::string err;
        if (!guarded([&] { side = std::make_unique<Project>(args[1]); }, err)) {
            std::cerr << "cannot load " << args[1] << ": " << err << "\n";
            return 1;
        }
        int n = 0, notpass = 0;
        double total = 0;
        for (auto& pic : pictures(fs::u8path(args[2]))) {
            cv::Mat img = imread(pic.u8string());
            if (img.empty()) {
                std::cerr << "cannot read " << pic.u8string() << "\n";
                continue;
            }
            Inspection r = side->inspect(img, pic.u8string());
            release_pictures();
            n++;
            total += r.total_ms;
            if (r.verdict != "PASS") notpass++;
            if (json) {
                Json j = r.to_json();
                for (auto& t : j["tools"]) t.erase("graphics");
                std::cout << j.dump(-1, ' ', false, Json::error_handler_t::replace) << "\n";
            } else {
                std::cout << pic.filename().u8string() << "  " << r.verdict << "  " << r.total_ms << " ms\n";
            }
            std::cout.flush();
        }
        std::cerr << n << " inspections, " << notpass << " not PASS; mean " << (n ? total / n : 0) << " ms\n";
        return 0;
    }
    std::cerr << "unknown command\n";
    return 2;
}
