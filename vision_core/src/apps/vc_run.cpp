// vc-run: Vision Core without a window.
//   vc-run inspect <side folder> <picture or folder> [--json]   one JSON line per picture
//   ... --dump <dir>   also write the images tools made, <picture>__<tool>.png
//   vc-run types                                              the check types built in
#include "../core/engine.hpp"
#include "../core/linalg.hpp"
#include "../core/nprandom.hpp"
#include "../core/npsort.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>
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
    if (args[0] == "npy-selftest") {
        // numbers numpy also prints (tests/npy_selftest.py compares them)
        Json j;
        j["blas"] = la::source();
        npr::Generator g(0);
        Json seq = Json::array();
        for (int i = 0; i < 60; i++) {
            int64_t pop = 3 + (i * 7) % 23, size = 2 + i % 2;
            Json c = g.choice(pop, size);
            seq.push_back(c);
            seq.push_back(g.integers(1 + i % 4));
        }
        j["rng"] = seq;
        double P[12] = {1.5, 2.25, 3.1, 4.7, 5.2, 6.9, 7.3, 8.8, 9.1, 10.4, 11.6, 12.2};
        j["svd"] = la::svd_vh(P, 6, 2);
        double A[18], b[6];
        for (int i = 0; i < 6; i++) { A[i * 3] = P[2 * i]; A[i * 3 + 1] = P[2 * i + 1]; A[i * 3 + 2] = 1; b[i] = P[2 * i] * P[2 * i] + P[2 * i + 1] * P[2 * i + 1]; }
        j["lstsq"] = la::lstsq(A, b, 6, 3);
        double x[2] = {-0.6, 0.8};
        j["matvec"] = la::matvec(P, x, 6, 2);
        float Pf[12], xf[2] = {-0.6f, 0.8f};
        for (int i = 0; i < 12; i++) Pf[i] = (float)P[i] * 1.37f;
        auto mf = la::matvec(Pf, xf, 6, 2);
        j["matvec_f32"] = std::vector<double>(mf.begin(), mf.end());
        j["dot"] = la::dot(P, P + 2, 2);
        // argsort with many ties, sizes 1..300 (same arrays as tests/npy_selftest.py)
        j["det"] = la::det(P, 2);
        j["gemm"] = la::gemm(false, true, 2, 6, 2, P, 2, P, 2);
        j["lstsq2"] = la::lstsq(A, P, 6, 3, 2);
        j["argsort_path"] = npy::argsort_path();
        Json as = Json::array();
        npr::Generator ag(7);
        for (int n = 1; n <= 300; n += (n < 40 ? 1 : 13)) {
            std::vector<float> f(n);
            std::vector<double> d(n);
            std::vector<int64_t> q(n);
            for (int i = 0; i < n; i++) {
                q[i] = ag.integers(6);
                f[i] = (float)q[i] * 0.5f;
                d[i] = (double)q[i] * 0.25;
            }
            as.push_back(npy::argsort(f.data(), n));
            as.push_back(npy::argsort(d.data(), n));
            as.push_back(npy::argsort(q.data(), n));
        }
        j["argsort"] = as;
        std::cout << j.dump() << std::endl;
        return 0;
    }
    if (args[0] == "types") {
        for (auto& [k, v] : registry()) std::cout << k << "\n";
        return 0;
    }
    if (args[0] == "inspect" && args.size() >= 3) {
        bool json = std::find(args.begin(), args.end(), "--json") != args.end();
        fs::path dump;
        for (size_t i = 3; i + 1 < args.size(); i++)
            if (args[i] == "--dump") dump = fs::u8path(args[i + 1]);
        if (!dump.empty()) fs::create_directories(dump);
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
            if (!dump.empty()) for (auto& [tid, im] : r.images) {
                std::vector<uchar> buf;
                if (im.empty() || !cv::imencode(".png", im, buf)) continue;
                std::ofstream f(dump / fs::u8path(pic.filename().u8string() + "__" + tid + ".png"), std::ios::binary);
                if (f) f.write((const char*)buf.data(), (std::streamsize)buf.size());
            }
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
