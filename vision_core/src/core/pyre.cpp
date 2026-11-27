#include "pyre.hpp"

#include <cctype>
#include <map>
#include <vector>

namespace vc::pyre {

namespace {

[[noreturn]] void fail(const std::string& what, size_t pos) { throw Error(what + " at position " + std::to_string(pos)); }

// Python's sre_parse checks, in the order it meets them, and the translation
std::string translate(const std::string& p, std::regex::flag_type& flags) {
    std::string out;
    std::vector<size_t> open;              // positions of unclosed '('
    std::map<std::string, int> names;
    int groups = 0;
    // what the last item was: 0 nothing (start, after '(' or '|'), 1 atom, 2 repeat, 3 anchor
    int last = 0;
    size_t i = 0, n = p.size();
    // leading inline flags: (?imsx)
    while (i + 2 < n && p[i] == '(' && p[i + 1] == '?' && std::strchr("imsxaLu", p[i + 2])) {
        size_t j = i + 2;
        while (j < n && std::strchr("imsxaLu", p[j])) {
            if (p[j] == 'i') flags |= std::regex::icase;
            // (?m) has no effect on fullmatch of a one-line read; MSVC's regex has no multiline flag
            j++;
        }
        if (j < n && p[j] == ')') i = j + 1;
        else break;
    }
    while (i < n) {
        char c = p[i];
        if (c == '\\') {
            if (i + 1 >= n) fail("bad escape (end of pattern)", i);
            char e = p[i + 1];
            if (std::isalpha((unsigned char)e) && !std::strchr("abBdDfnrsStvwWZAxuUN", e)) fail(std::string("bad escape \\") + e, i);
            if (e == 'A') {
                out += "^";
                last = 3;
            } else if (e == 'Z') {
                out += "$";
                last = 3;
            } else {
                out += c;
                out += e;
                last = (e == 'b' || e == 'B') ? 3 : 1;
            }
            i += 2;
            continue;
        }
        if (c == '[') {
            size_t j = i + 1;
            if (j < n && p[j] == '^') j++;
            size_t body = j;
            bool lead_bracket = j < n && p[j] == ']';   // a leading ] is literal in Python
            if (lead_bracket) j++;
            while (j < n && p[j] != ']') j += p[j] == '\\' ? 2 : 1;
            if (j >= n) fail("unterminated character set", i);
            out.append(p, i, body - i);
            if (lead_bracket) out += "\\]";
            out.append(p, body + (lead_bracket ? 1 : 0), j - body - (lead_bracket ? 1 : 0) + 1);
            i = j + 1;
            last = 1;
            continue;
        }
        if (c == '(') {
            open.push_back(i);
            if (i + 1 < n && p[i + 1] == '?') {
                if (i + 2 < n && p[i + 2] == 'P') {
                    if (i + 3 < n && p[i + 3] == '<') {
                        size_t e = p.find('>', i + 4);
                        if (e == std::string::npos) fail("missing >, unterminated name", i + 4);
                        names[p.substr(i + 4, e - i - 4)] = ++groups;
                        out += "(";
                        i = e + 1;
                    } else if (i + 3 < n && p[i + 3] == '=') {
                        size_t e = p.find(')', i + 4);
                        if (e == std::string::npos) fail("missing ), unterminated name", i + 4);
                        std::string nm = p.substr(i + 4, e - i - 4);
                        if (!names.count(nm)) fail("unknown group name '" + nm + "'", i + 4);
                        out += "(?:\\" + std::to_string(names[nm]) + ")";
                        open.pop_back();
                        i = e + 1;
                        last = 1;
                        continue;
                    } else {
                        out += "(?";
                        i += 2;
                    }
                } else {
                    out += "(?";
                    out += i + 2 < n ? std::string(1, p[i + 2]) : "";
                    i += 3;
                }
            } else {
                groups++;
                out += "(";
                i++;
            }
            last = 0;
            continue;
        }
        if (c == ')') {
            if (open.empty()) fail("unbalanced parenthesis", i);
            open.pop_back();
            out += c;
            i++;
            last = 1;
            continue;
        }
        if (c == '|') {
            out += c;
            i++;
            last = 0;
            continue;
        }
        bool brace_repeat = false;
        size_t brace_end = 0;
        if (c == '{') {
            // a valid {m}, {m,}, {,n}, {m,n} is a repeat; anything else is a literal
            size_t j = i + 1;
            bool digits = false, comma = false;
            while (j < n && (std::isdigit((unsigned char)p[j]) || (p[j] == ',' && !comma))) {
                if (p[j] == ',') comma = true;
                else digits = true;
                j++;
            }
            if (j < n && p[j] == '}' && (digits || comma)) brace_repeat = true, brace_end = j;
        }
        if (c == '*' || c == '+' || c == '?' || brace_repeat) {
            if (last == 0 || last == 3) fail("nothing to repeat", i);
            if (last == 2) {
                // '?' after a repeat makes it lazy, '+' possessive (3.11+); another repeat is an error
                if (c == '?') {
                    out += c;
                    i++;
                    last = 1;
                    continue;
                }
                if (c != '+') fail("multiple repeat", i);
            }
            if (brace_repeat) {
                std::string q = p.substr(i, brace_end - i + 1);
                if (q.size() > 2 && q[1] == ',') q.insert(1, "0");   // {,n} -> {0,n}
                out += q;
                i = brace_end + 1;
            } else {
                out += c;
                i++;
            }
            last = 2;
            continue;
        }
        out += c;
        i++;
        last = (c == '^' || c == '$') ? 3 : 1;
    }
    if (!open.empty()) fail("missing ), unterminated subpattern", open.back());
    return out;
}

}  // namespace

std::regex compile(const std::string& pattern) {
    std::regex::flag_type flags = std::regex::ECMAScript;
    std::string ecma = translate(pattern, flags);
    try {
        return std::regex(ecma, flags);
    } catch (const std::regex_error& e) {
        throw Error(std::string("cannot read the pattern: ") + e.what());
    }
}

bool fullmatch(const std::regex& rx, const std::string& text) { return std::regex_match(text, rx); }

}  // namespace vc::pyre
