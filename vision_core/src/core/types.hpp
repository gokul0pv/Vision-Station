// Vision Core: the result every check returns, and the overlay it draws.
// Mirrors Vision Station (vision_station/tools/base.py) field for field, so
// results compare one to one and existing front ends can read them.
#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vector>

namespace vc {

using Json = nlohmann::ordered_json;

struct Graphic {
    std::string kind;                                // rect | poly | line | point | text
    std::vector<std::pair<double, double>> points;   // current-picture pixels
    std::string status = "pass";
    std::string label;
    bool absolute = false;                           // already in picture coordinates
    std::string role;                                // "region" for a check's outline

    Graphic() = default;
    Graphic(std::string k, std::vector<std::pair<double, double>> p, std::string st = "pass", std::string lb = "")
        : kind(std::move(k)), points(std::move(p)), status(std::move(st)), label(std::move(lb)) {}
};

struct ToolResult {
    std::string tool_id;
    std::string type_name;
    std::string status = "pass";                     // pass | fail | warn | error
    double score = 0.0;
    Json values = Json::object();
    std::vector<Graphic> graphics;
    std::string message;
    double ms = 0.0;
    std::vector<std::string> caption;

    ToolResult() = default;
    ToolResult(std::string id, std::string type, std::string st = "pass", double sc = 0.0,
               Json vals = Json::object(), std::vector<Graphic> g = {}, std::string msg = "", double t_ms = 0.0)
        : tool_id(std::move(id)), type_name(std::move(type)), status(std::move(st)), score(sc),
          values(std::move(vals)), graphics(std::move(g)), message(std::move(msg)), ms(t_ms) {}

    static ToolResult error(const std::string& id, const std::string& type, const std::string& msg) {
        return ToolResult(id, type, "error", 0.0, Json::object(), {}, msg);
    }

    Json to_json() const;
};

// Python's round(x, n) for a float: decimal rounding as repr would show it,
// ties to even on the binary value (the same algorithm CPython uses).
double py_round(double x, int ndigits);
// Python's round(x) -> int: nearest, ties to even.
long long py_round_int(double x);

}  // namespace vc
