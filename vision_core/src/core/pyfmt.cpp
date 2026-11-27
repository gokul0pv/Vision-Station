#include "pyfmt.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace vc {

std::string pyfmt(const char* fmt, double v) {
    // only "%.Nf" is used
    int prec = 6;
    const char* dot = std::strchr(fmt, '.');
    if (dot) prec = std::atoi(dot + 1);
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    char buf[512];
    size_t L = std::strlen(fmt);
    if (L && (fmt[L - 1] == 'g')) {   // "%.Ng": N significant digits, as printf / Python
        auto rg = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::general, prec == 0 ? 1 : prec);
        return std::string(buf, rg.ptr);
    }
    auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::fixed, prec);
    return std::string(buf, r.ptr);
}

std::string py_repr(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
    std::string s(buf, r.ptr);                         // d.ddddde[+-]XX
    bool neg = s[0] == '-';
    if (neg) s.erase(0, 1);
    auto e = s.find('e');
    int exp = std::atoi(s.c_str() + e + 1);
    std::string digits;
    for (size_t i = 0; i < e; i++)
        if (s[i] != '.') digits += s[i];
    std::string out;
    if (exp < -4 || exp >= 16) {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char eb[16];
        std::snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += eb;
    } else if (exp < 0) {
        out = "0." + std::string(-exp - 1, '0') + digits;
    } else if ((int)digits.size() <= exp + 1) {
        out = digits + std::string(exp + 1 - digits.size(), '0') + ".0";
    } else {
        out = digits.substr(0, exp + 1) + "." + digits.substr(exp + 1);
    }
    return neg ? "-" + out : out;
}

std::string py_str(const Json& v) {
    if (v.is_null()) return "None";
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    if (v.is_number_integer()) return std::to_string(v.get<long long>());
    if (v.is_number()) return py_repr(v.get<double>());
    if (v.is_string()) return v.get<std::string>();
    return v.dump();
}

}  // namespace vc
