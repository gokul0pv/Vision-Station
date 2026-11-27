// The check contract (vision_station/tools/base.py Tool): a region in
// reference coordinates, mapped into the current picture by the fixture,
// cut out upright, and the overlay mapped back.
#pragma once

#include "types.hpp"

#include <any>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace vc {

struct Context {
    cv::Mat reference;                         // BGR, or empty
    cv::Mat fixture;                           // 3x3 reference -> picture (CV_32F from align, CV_64F from fixture_npoint), or empty
    double mm_per_px = 0.0;
    std::string folder;
    std::map<std::string, std::any> share;
    std::map<std::string, cv::Mat> images;
    std::map<std::string, ToolResult> results;
    std::map<std::string, cv::Mat> ref_images;
    bool want_records = false;
    std::map<std::string, std::map<std::string, cv::Mat>> records;
    std::map<std::string, class Tool*> tools;  // every tool of the job, by id (fixture_npoint runs some on the reference)
    std::vector<class Tool*> tool_order;       // the same, in the job's order
    cv::Mat world;                             // 2x3 pixel -> world from calibrate_npoint, or empty
};

class Tool {
public:
    Tool(std::string id, Json params);
    virtual ~Tool() = default;

    virtual ToolResult run(const cv::Mat& image, Context& ctx) = 0;
    virtual bool changes_geometry() const { return false; }
    virtual bool is_fixture() const { return false; }

    // ---- as base.py ----
    std::optional<std::array<int, 4>> mapped_roi(const Context& ctx) const;
    // four corners in picture coordinates (float32), empty without a region
    std::vector<cv::Point2f> mapped_quad(const Context& ctx) const;
    // (patch, (ox, oy))
    std::pair<cv::Mat, cv::Point> crop(const cv::Mat& image, const Context& ctx);
    ToolResult& correct_graphics(ToolResult& r, const Context& ctx) const;
    std::vector<Graphic> roi_graphic(const Context& ctx, const std::string& status = "pass", const std::string& label = "") const;
    std::vector<cv::Point2f> polygon_points(const Context& ctx) const;

    // parameter helpers (Python: self.p.get(key, default))
    double pf(const char* k, double d) const;
    int pi(const char* k, int d) const;
    bool pb(const char* k, bool d) const;
    std::string ps(const char* k, const std::string& d) const;
    bool has(const char* k) const;

    std::string id;
    std::string type_name;
    Json p;
    std::optional<std::array<double, 4>> roi;
    double angle = 0.0;
    std::string shape = "rect";
    std::vector<std::array<double, 2>> points;
    bool enabled = true;
    bool critical = true;

    cv::Mat crop_M;                            // 3x3 CV_64F crop -> picture, or empty
    std::optional<std::array<int, 4>> patch_origin;
    cv::Mat crop_mask;
    cv::Mat poly_mask;

protected:
    cv::Mat apply_shape(const cv::Mat& patch);
    std::vector<Graphic> roi_graphic_raw(const Context& ctx, const std::string& status, const std::string& label) const;
    std::array<int, 4> points_bbox() const;
};

using ToolFactory = std::function<std::unique_ptr<Tool>(const std::string&, const Json&)>;
std::map<std::string, ToolFactory>& registry();
std::unique_ptr<Tool> make_tool(const Json& spec);

struct Registrar {
    Registrar(const char* name, ToolFactory f) { registry()[name] = std::move(f); }
};

#define VC_REGISTER(NAME, CLASS)                                                              \
    static ::vc::Registrar _vc_reg_##CLASS(NAME, [](const std::string& id, const ::vc::Json& p) { \
        auto t = std::make_unique<CLASS>(id, p);                                             \
        t->type_name = NAME;                                                                  \
        return std::unique_ptr<::vc::Tool>(std::move(t));                                     \
    });

// helpers every check uses
cv::Mat to_gray(const cv::Mat& m);
Graphic abs_graphic(const std::string& kind, const std::vector<cv::Point2f>& pts, const std::string& status, const std::string& label = "");
double now_ms();

}  // namespace vc
