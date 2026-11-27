// The text reader (vision_station/tools/ocr_engine.py): PP-OCR detection on
// tiles, lines cut out along their angle, read every way up, best kept.
#pragma once

#include "../core/types.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace vc::ocr {

struct Line {
    std::u32string text;
    double confidence = 0;
    std::vector<double> char_conf;
    std::vector<std::array<double, 2>> poly;
    std::array<int, 4> box{};
    int turn = 0;
    bool upside_down = false;
    double detector_score = 0;
    std::optional<std::u32string> core;
    std::string read_from;
};

std::vector<Line> read(const cv::Mat& image, const std::string& detail = "auto");
std::optional<Line> read_single_line(const cv::Mat& image);
std::vector<Line> clean_lines(std::vector<Line> lines);
bool available();
std::string engine_name();

// text helpers with Python's str semantics (code points)
std::u32string from_utf8(const std::string& s);
std::string to_utf8(const std::u32string& s);
bool is_space(char32_t c);
bool is_alnum(char32_t c);
bool is_alpha(char32_t c);
bool is_upper(char32_t c);
bool is_lower(char32_t c);
std::u32string strip(const std::u32string& s);
std::u32string upper(const std::u32string& s);
std::u32string lower(const std::u32string& s);

}  // namespace vc::ocr
