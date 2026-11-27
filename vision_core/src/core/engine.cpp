#include "engine.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <sstream>

#ifdef _WIN32
#include <eh.h>
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace vc {

// ------------------------------------------------------------------ guard
#ifdef _WIN32
namespace {
struct SehError : std::exception {
    unsigned code;
    char msg[64];
    explicit SehError(unsigned c) : code(c) { std::snprintf(msg, sizeof msg, "system fault 0x%08X", c); }
    const char* what() const noexcept override { return msg; }
};
void se_translator(unsigned code, EXCEPTION_POINTERS*) { throw SehError(code); }
}  // namespace
#endif

bool guarded(const std::function<void()>& fn, std::string& error) {
#ifdef _WIN32
    auto prev = _set_se_translator(se_translator);
#endif
    bool ok = false;
    try {
        fn();
        ok = true;
    } catch (const cv::Exception& e) {
        error = std::string("error: ") + e.what();
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown failure";
    }
#ifdef _WIN32
    _set_se_translator(prev);
#endif
    return ok;
}

// ---------------------------------------------------------------- pictures
cv::Mat imread(const std::string& path, int flags) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    if (!f) return {};
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (data.empty()) return {};
    try {
        return cv::imdecode(data, flags);
    } catch (const cv::Exception&) {
        return {};
    }
}

cv::Mat undistort(const cv::Mat& img, const Json& cal) {
    if (cal.is_null() || img.empty()) return img;
    int h = img.rows, w = img.cols;
    cv::Mat K(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) K.at<double>(i, j) = cal["camera_matrix"][i][j].get<double>();
    std::vector<double> dv;
    for (auto& v : cal["dist_coeffs"]) {
        if (v.is_array()) for (auto& u : v) dv.push_back(u.get<double>());
        else dv.push_back(v.get<double>());
    }
    cv::Mat d(dv, true);
    int cw = cal["image_size"][0].get<int>(), ch = cal["image_size"][1].get<int>();
    if (w != cw || h != ch) {
        cv::Mat s = (cv::Mat_<double>(3, 3) << w / (double)cw, 0, 0, 0, h / (double)ch, 0, 0, 0, 1);
        K = s * K;
    }
    static std::map<std::string, std::pair<cv::Mat, cv::Mat>> maps;
    std::ostringstream key;
    key << w << "x" << h;
    for (int i = 0; i < 9; i++) key << "," << K.at<double>(i / 3, i % 3);
    for (double v : dv) key << "," << v;
    auto it = maps.find(key.str());
    if (it == maps.end()) {
        cv::Mat m1, m2;
        cv::initUndistortRectifyMap(K, d, cv::noArray(), K, cv::Size(w, h), CV_16SC2, m1, m2);
        it = maps.emplace(key.str(), std::make_pair(m1, m2)).first;
    }
    cv::Mat out;
    cv::remap(img, out, it->second.first, it->second.second, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

// ----------------------------------------------------------------- project
Project::Project(const std::string& f) {
    folder = fs::absolute(fs::u8path(f)).u8string();
    std::string t = folder;
    while (!t.empty() && (t.back() == '/' || t.back() == '\\')) t.pop_back();
    id = fs::u8path(t).filename().u8string();
    load();
}

void Project::load() {
    std::ifstream in(fs::u8path(folder) / "project.json");
    if (!in) throw std::runtime_error("cannot open " + folder + "/project.json");
    spec = Json::parse(in);
    std::string ref = spec.value("reference_image", "reference.png");
    fs::path rp = fs::u8path(ref).is_absolute() ? fs::u8path(ref) : fs::u8path(folder) / fs::u8path(ref);
    reference = fs::exists(rp) ? imread(rp.u8string()) : cv::Mat();
    calibration = Json();
    fs::path cp = fs::u8path(folder) / "calibration.json";
    if (fs::exists(cp)) {
        std::ifstream c(cp);
        calibration = Json::parse(c);
        if (!reference.empty()) reference = undistort(reference, calibration);
    }
    tools.clear();
    for (auto t : spec.value("tools", Json::array())) {
        Json params = t.contains("params") ? t["params"] : Json::object();
        if (params.contains("model") && params["model"].is_string() && !params["model"].get<std::string>().empty()) {
            fs::path m = fs::u8path(params["model"].get<std::string>());
            if (!m.is_absolute()) params["model"] = (fs::u8path(folder) / m).u8string();
        }
        t["params"] = params;
        if (!registry().count(t.value("type", "")) && std::getenv("VC_SKIP_UNKNOWN")) continue;   // tests only
        tools.push_back(make_tool(t));
    }
}

static std::string new_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    char b[13];
    std::snprintf(b, sizeof b, "%012llx", (unsigned long long)(rng() & 0xFFFFFFFFFFFFull));
    return b;
}

static std::optional<Json> resolve_link(const std::string& ref, const std::map<std::string, ToolResult>& by_id) {
    auto dot = ref.find('.');
    if (ref.empty() || dot == std::string::npos) return std::nullopt;
    auto it = by_id.find(ref.substr(0, dot));
    if (it == by_id.end()) return std::nullopt;
    std::string key = ref.substr(dot + 1);
    if (key == "score") return Json(it->second.score);
    if (key == "status") return Json(it->second.status);
    const auto& v = it->second.values;
    if (!v.contains(key)) return std::nullopt;
    const auto& x = v[key];
    if (x.is_number() || x.is_string() || x.is_boolean()) return x;
    return std::nullopt;
}

std::vector<std::string> caption(const ToolResult& r, const Json& params);   // annotate.cpp

static void outline(Tool& tool, ToolResult& r, const Context& ctx, bool reads_transformed) {
    bool has_region = false;
    for (auto& g : r.graphics) has_region |= g.role == "region";
    if (!reads_transformed && !has_region) {
        std::string err;
        guarded([&] {
            auto g = tool.roi_graphic(ctx, r.status);
            g.insert(g.end(), r.graphics.begin(), r.graphics.end());
            r.graphics = std::move(g);
        }, err);
    }
    for (auto& g : r.graphics)
        if (g.role == "region") {
            g.status = r.status;
            g.label.clear();
        }
    std::string err;
    if (!guarded([&] { r.caption = caption(r, tool.p); }, err)) r.caption.clear();
}

static void filter_graphics(std::vector<Graphic>& gs, const Json& opt) {
    std::vector<Graphic> out;
    for (auto& g : gs) {
        bool show;
        if (g.role == "region") show = opt.value("region", true);
        else if (g.status == "search") show = opt.value("search", true);
        else show = opt.value("results", true);
        if (!show) continue;
        if (!opt.value("labels", true)) g.label.clear();
        out.push_back(g);
    }
    gs = std::move(out);
}

Inspection Project::inspect(const cv::Mat& image, const std::string& source, const std::string& barcode) {
    return inspect(std::map<std::string, cv::Mat>{{"default", image}}, source, barcode);
}

Inspection Project::inspect(const std::map<std::string, cv::Mat>& in_frames, const std::string& source, const std::string& barcode) {
    double t0 = now_ms();
    std::map<std::string, cv::Mat> frames = in_frames;
    if (!calibration.is_null())
        for (auto& [k, v] : frames) v = undistort(v, calibration);
    cv::Mat primary = frames.count("default") ? frames["default"] : frames.begin()->second;

    double mm = 0.0;
    if (spec.contains("mm_per_px") && spec["mm_per_px"].is_number()) mm = spec["mm_per_px"].get<double>();
    if (mm == 0.0 && !calibration.is_null() && calibration.contains("mm_per_px")) mm = calibration["mm_per_px"].get<double>();
    Context ctx;
    ctx.reference = reference;
    ctx.folder = folder;
    ctx.mm_per_px = mm;
    if (!barcode.empty()) ctx.share["barcode"] = barcode;

    std::vector<ToolResult> results;
    std::string aborted, barcode_ng;
    std::set<std::string> missing;
    std::map<std::string, cv::Mat> fixtures;     // empty Mat = did not find the part
    std::set<std::string> fixture_ran;
    std::map<std::string, ToolResult> by_id;
    std::set<std::string> geo_types;
    for (auto& t : tools) {
        if (t->changes_geometry()) geo_types.insert(t->id);
        ctx.tools[t->id] = t.get();
    }

    for (auto& tp : tools) {
        Tool& tool = *tp;
        if (!tool.enabled) continue;
        std::string want = tool.ps("capture", "default");
        if (want.empty()) want = "default";
        bool reads_transformed = geo_types.count(want) > 0;
        cv::Mat frame = frames.count(want) ? frames[want] : cv::Mat();
        if (frame.empty() && reads_transformed) {
            ToolResult r(tool.id, tool.type_name, "error", 0.0, Json::object(), {}, "reads the image from " + want + ", which was not produced");
            by_id[tool.id] = r;
            ctx.results[tool.id] = r;
            results.push_back(r);
            continue;
        }
        if (frame.empty()) {
            if (want != "default") missing.insert(want);
            frame = primary;
        }
        std::string fx_choice = tool.ps("fixture", "auto");
        if (fx_choice.empty()) fx_choice = "auto";
        cv::Mat fx_saved = ctx.fixture;
        cv::Mat ref_saved = ctx.reference;
        std::string problem;
        if (reads_transformed) {
            fx_choice = "none";
            auto it = ctx.ref_images.find(want);
            ctx.reference = it != ctx.ref_images.end() ? it->second : cv::Mat();
        }
        if (fx_choice == "none") {
            ctx.fixture = cv::Mat();
        } else if (fx_choice != "auto") {
            if (!fixture_ran.count(fx_choice)) problem = "fixture " + fx_choice + " has not run before this tool";
            else if (fixtures[fx_choice].empty()) problem = "fixture " + fx_choice + " did not find the part";
            else ctx.fixture = fixtures[fx_choice];
        }
        if (tool.p.contains("links") && tool.p["links"].is_object())
            for (auto& [param, ref] : tool.p["links"].items()) {
                auto val = ref.is_string() ? resolve_link(ref.get<std::string>(), by_id) : std::nullopt;
                if (!val) {
                    problem = "input " + param + " <- " + (ref.is_string() ? ref.get<std::string>() : std::string("?")) + " has no value";
                    break;
                }
                tool.p[param] = *val;
            }
        ToolResult r;
        if (!problem.empty()) {
            r = ToolResult(tool.id, tool.type_name, "error", 0.0, Json::object(), {}, problem);
        } else {
            std::string err;
            double ts = now_ms();
            bool ok = guarded([&] {
                r = tool.run(frame, ctx);
                tool.correct_graphics(r, ctx);
            }, err);
            if (!ok) r = ToolResult(tool.id, tool.type_name, "error", 0.0, Json::object(), {}, err, now_ms() - ts);
            for (auto& [k, v] : ctx.images)
                if (!frames.count(k)) frames[k] = v;
        }
        outline(tool, r, ctx, reads_transformed);
        if (tool.p.contains("graphics") && tool.p["graphics"].is_object() && !tool.p["graphics"].empty())
            filter_graphics(r.graphics, tool.p["graphics"]);
        bool fixture_tool = tool.type_name == "align" || tool.is_fixture();
        if (fixture_tool) {
            fixture_ran.insert(tool.id);
            fixtures[tool.id] = r.status == "pass" ? ctx.fixture : cv::Mat();
        } else if (fx_choice != "auto") {
            ctx.fixture = fx_saved;
        }
        ctx.reference = ref_saved;
        by_id[tool.id] = r;
        ctx.results[tool.id] = r;
        results.push_back(r);
        if (tool.type_name == "align" && r.status != "pass") {
            aborted = "alignment failed \xe2\x80\x94 remaining tools skipped";
            break;
        }
        if (r.values.contains("barcode_match") && r.values["barcode_match"].is_boolean() && !r.values["barcode_match"].get<bool>()) {
            barcode_ng = r.message.empty() ? "Barcode not good" : r.message;
            aborted = "Barcode not good \xe2\x80\x94 remaining checks skipped";
            break;
        }
    }

    auto critical_of = [&](const std::string& tid) {
        for (auto& t : tools)
            if (t->id == tid) return t->critical;
        return true;
    };
    Inspection out;
    bool crit = false, noncrit = false;
    for (auto& r : results)
        if (r.status == "fail" || r.status == "error") {
            if (critical_of(r.tool_id)) {
                crit = true;
                out.failed_tools.push_back(r.tool_id);
            } else {
                noncrit = true;
            }
        }
    out.verdict = crit ? "FAIL" : (noncrit ? "WARN" : "PASS");
    if (!barcode_ng.empty()) out.verdict = "FAIL";
    out.id = new_id();
    out.project = id;
    out.source = source;
    for (auto& [k, v] : frames) out.captures_used.push_back(k);
    out.captures_missing.assign(missing.begin(), missing.end());
    out.timestamp = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    out.total_ms = py_round(now_ms() - t0, 2);
    out.width = primary.cols;
    out.height = primary.rows;
    out.aborted = aborted;
    out.barcode_ng = barcode_ng;
    out.tools = std::move(results);
    out.images = ctx.images;
    return out;
}

Json Inspection::to_json() const {
    Json t = Json::array();
    for (auto& r : tools) t.push_back(r.to_json());
    return Json{{"id", id}, {"project", project}, {"verdict", verdict}, {"source", source},
                {"captures_used", captures_used}, {"captures_missing", captures_missing},
                {"timestamp", timestamp}, {"total_ms", total_ms}, {"image_size", {width, height}},
                {"aborted", aborted.empty() ? Json() : Json(aborted)},
                {"barcode_ng", barcode_ng.empty() ? Json() : Json(barcode_ng)},
                {"failed_tools", failed_tools}, {"tools", t}};
}

}  // namespace vc
