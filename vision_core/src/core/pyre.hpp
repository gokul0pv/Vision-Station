// Python's re for the patterns a recipe carries (expect_regex): the syntax
// Python accepts and ECMAScript does not -- (?P<name>...), (?P=name), inline
// (?i) (?s) (?m) (?x) flags at the start, \A \Z -- translated, and the common
// mistakes reported in Python's words ("missing ), unterminated subpattern at
// position 0"), so a recipe behaves and fails as in Vision Station.
#pragma once

#include <regex>
#include <stdexcept>
#include <string>

namespace vc::pyre {

struct Error : std::runtime_error { using std::runtime_error::runtime_error; };

// re.compile(pattern); throws Error with Python's message
std::regex compile(const std::string& pattern);
// re.fullmatch(compiled, text) is not None
bool fullmatch(const std::regex& rx, const std::string& text);

}  // namespace vc::pyre
