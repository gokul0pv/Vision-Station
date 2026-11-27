// vision_station/core/annotate.py caption(): a check's result as one or two
// short lines for the picture.
#include "pyfmt.hpp"
#include "types.hpp"

#include <cstdio>

namespace vc {

namespace {

std::string word_of(const std::string& status) {
    if (status == "pass") return "OK";
    if (status == "fail") return "NG";
    if (status == "warn") return "CHECK";
    if (status == "error") return "ERROR";
    std::string u = status;
    for (auto& c : u) c = (char)std::toupper((unsigned char)c);
    return u;
}

// annotate._num: "%.*f" with trailing zeros (and point) stripped
std::string num(double v, int digits) {
    std::string s = pyfmt(("%." + std::to_string(digits) + "f").c_str(), v);
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    return s;
}

bool has_val(const Json& v, const char* k) { return v.is_object() && v.contains(k) && !v[k].is_null(); }

double fval(const Json& v, const char* k) {
    const Json& x = v[k];
    if (x.is_number()) return x.get<double>();
    if (x.is_boolean()) return x.get<bool>() ? 1.0 : 0.0;
    return std::stod(x.get<std::string>());
}

bool truthy(const Json& x) {
    if (x.is_null()) return false;
    if (x.is_boolean()) return x.get<bool>();
    if (x.is_number()) return x.get<double>() != 0;
    if (x.is_string()) return !x.get<std::string>().empty();
    if (x.is_array() || x.is_object()) return !x.empty();
    return true;
}

std::string size_of(const Json& v, const char* mm_key, const char* px_key, const std::string& prefix = "") {
    if (mm_key && has_val(v, mm_key)) return prefix + num(fval(v, mm_key), 3) + " mm";
    if (px_key && has_val(v, px_key)) return prefix + num(fval(v, px_key), 1) + " px";
    return "";
}

std::string strip(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// annotate._text_of
std::string text_of(const Json& v) {
    const Json* lines = nullptr;
    if (v.contains("lines") && truthy(v["lines"])) lines = &v["lines"];
    else if (v.contains("texts") && truthy(v["texts"])) lines = &v["texts"];
    if (lines && lines->is_array()) {
        std::string out;
        bool first = true;
        for (auto& x : *lines) {
            std::string s = py_str(x);
            if (strip(s).empty()) continue;
            out += (first ? "" : " / ") + s;
            first = false;
        }
        return out;
    }
    if (v.contains("text") && truthy(v["text"])) return py_str(v["text"]);
    if (v.contains("data") && truthy(v["data"])) return py_str(v["data"]);
    return "";
}

// first n code points of a UTF-8 string
std::string head(const std::string& s, size_t n) {
    size_t i = 0, cps = 0;
    while (i < s.size() && cps < n) {
        unsigned char c = (unsigned char)s[i];
        i += c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
        cps++;
    }
    return s.substr(0, std::min(i, s.size()));
}

std::string first_part(const std::string& msg) { return msg.substr(0, msg.find(';')); }

}  // namespace

std::vector<std::string> caption(const ToolResult& r, const Json& params) {
    const std::string& ty = r.type_name;
    const std::string& status = r.status;
    const Json& v = r.values.is_object() ? r.values : Json::object();
    const Json& p = params.is_object() ? params : Json::object();
    std::string word = word_of(status);
    bool ok = status == "pass";
    if (status == "error") {
        std::string msg = first_part(r.message);
        if (msg.empty()) return {"ERROR"};
        return {"ERROR", head(msg, 48)};
    }
    if (ty == "ocv") {
        std::string read = text_of(v);
        if (read.empty()) read = "nothing read";
        if (ok) return {v.contains("text") && truthy(v["text"]) ? py_str(v["text"]) : read};
        std::string want;
        if (p.contains("expect_text") && truthy(p["expect_text"])) want = py_str(p["expect_text"]);
        else if (p.contains("expect_regex") && truthy(p["expect_regex"])) want = "pattern " + py_str(p["expect_regex"]);
        else if (v.contains("expected") && truthy(v["expected"])) want = py_str(v["expected"]);
        std::vector<std::string> out{!text_of(v).empty() ? "NG  read: " + read : "NG  nothing read"};
        if (!want.empty()) out.push_back("expected: " + want);
        return out;
    }
    if (ty == "ocr" || ty == "ocr_auto" || ty == "text_detect") {
        std::string read = text_of(v);
        if (read.empty()) read = "nothing read";
        if (ok) return {read};
        std::string want = p.contains("expect_contains") && truthy(p["expect_contains"]) ? py_str(p["expect_contains"]) : "";
        std::vector<std::string> out{!text_of(v).empty() ? "NG  read: " + read : "NG  nothing read"};
        if (!want.empty()) out.push_back("expected: " + want);
        return out;
    }
    if (ty == "code_read") {
        std::string data = text_of(v);
        if (ok && !data.empty()) return {data};
        std::vector<std::string> out{word};
        if (!data.empty()) out.push_back(data);
        return out;
    }
    if (ty == "measure_length") {
        std::string s = size_of(v, "length_mm", "length_px");
        if (has_val(v, "deviation_mm") && !ok) {
            char b[64];
            std::snprintf(b, sizeof b, "%+.3f", fval(v, "deviation_mm"));
            s += std::string("  (") + b + " mm)";
        }
        return {s.empty() ? word : word + "  " + s};
    }
    if (ty == "distance") {
        if (has_val(v, "value")) return {word + "  " + num(fval(v, "value"), 3) + " " + (v.contains("unit") ? py_str(v["unit"]) : "")};
        return {word};
    }
    if (ty == "find_circle") {
        std::string s;
        if (has_val(v, "diameter_mm")) s = size_of(v, "diameter_mm", nullptr);
        else if (has_val(v, "radius_px")) s = "\xc3\x98 " + num(2 * fval(v, "radius_px"), 1) + " px";
        if (!s.empty() && s.rfind("\xc3\x98", 0) != 0) s = "\xc3\x98 " + s;
        return {s.empty() ? word : word + "  " + s};
    }
    if (ty == "find_ellipse" && has_val(v, "major_px"))
        return {word + "  " + num(fval(v, "major_px"), 1) + " x " + num(fval(v, "minor_px"), 1) + " px"};
    if (ty == "caliper") {
        if (has_val(v, "count")) {
            std::string unit = v.contains("unit") ? py_str(v["unit"]) : "px";
            std::string s = word + "  " + std::to_string((long long)fval(v, "count")) + " found";
            if (v.contains("pitch") && truthy(v["pitch"])) s += ", pitch " + num(fval(v, "pitch"), 3) + " " + unit;
            return {s};
        }
        return {word};
    }
    if (ty == "count_blob" || ty == "count_template" || ty == "yolo_detect" || ty == "yolo_defect") {
        if (!has_val(v, "count")) return {word};
        std::string s = word + "  " + py_str(v["count"]);
        if (v.contains("expected") && !v["expected"].is_null() && !(v["expected"].is_string() && v["expected"].get<std::string>().empty()))
            s += " / " + py_str(v["expected"]);
        return {s};
    }
    if (ty == "led_glow" && has_val(v, "lit_count")) return {word + "  " + py_str(v["lit_count"]) + " lit"};
    if (ty == "color_match" && has_val(v, "delta_e")) return {word + "  \xce\x94" "E " + num(fval(v, "delta_e"), 1)};
    if (ty == "color_segment" && has_val(v, "percent")) return {word + "  " + num(fval(v, "percent"), 1) + "%"};
    if (ty == "classify" && v.contains("class") && truthy(v["class"])) return {word + "  " + py_str(v["class"])};
    if (ty == "defect_diff" && !ok && v.contains("defect_count") && truthy(v["defect_count"])) {
        long long n = (long long)fval(v, "defect_count");
        return {"NG  " + std::to_string(n) + " defect" + (n == 1 ? "" : "s")};
    }
    if (ty == "calibrate_scale" && v.contains("mm_per_px") && truthy(v["mm_per_px"])) return {num(fval(v, "mm_per_px"), 4) + " mm/px"};
    if (ty == "align") return {ok ? "located" : "NG  part not found"};
    if (ty == "color_classify" && !ok) {
        std::string msg = first_part(r.message);
        if (msg.empty()) return {"NG"};
        return {"NG", head(msg, 48)};
    }
    return {word};
}

}  // namespace vc
