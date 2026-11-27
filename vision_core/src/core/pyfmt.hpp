// Python's number formatting, so messages read exactly as Vision Station's.
#pragma once

#include "types.hpp"

#include <string>

namespace vc {

// "%.Nf" (or "{:.Nf}") of a float: the exact value rounded half to even; "%.Ng" too
std::string pyfmt(const char* fmt, double v);
// str(float) / repr(float): shortest round-trip digits, Python's layout
std::string py_repr(double v);
// str() of a JSON value as Python prints it (1, 1.5, 'abc' -> abc, None, True)
std::string py_str(const Json& v);

}  // namespace vc
