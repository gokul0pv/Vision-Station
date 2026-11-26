#include "types.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>

namespace vc {

double py_round(double x, int ndigits) {
    if (!std::isfinite(x)) return x;
    // the exact decimal value of x, rounded half-even at ndigits (what
    // CPython's round() does through dtoa), read back as a double
    char buf[512];
    auto r = std::to_chars(buf, buf + sizeof(buf) - 1, x, std::chars_format::fixed, ndigits);
    if (r.ec != std::errc()) return x;
    *r.ptr = 0;
    return std::strtod(buf, nullptr);
}

long long py_round_int(double x) { return (long long)std::nearbyint(x); }

Json ToolResult::to_json() const {
    Json g = Json::array();
    for (const auto& gr : graphics) {
        Json pts = Json::array();
        for (const auto& p : gr.points) pts.push_back(Json::array({p.first, p.second}));
        g.push_back(Json{{"kind", gr.kind}, {"points", pts}, {"status", gr.status}, {"label", gr.label}, {"role", gr.role}});
    }
    return Json{{"tool_id", tool_id}, {"type", type_name}, {"status", status}, {"score", py_round(score, 4)},
                {"values", values}, {"graphics", g}, {"message", message}, {"ms", py_round(ms, 2)},
                {"caption", caption}};
}

}  // namespace vc
