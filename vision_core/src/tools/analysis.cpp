// vision_station/tools/analysis.py: results_analysis (a safe expression over
// earlier results), intersect, fit_shape, color_match.
//
// The expression language is the subset analysis.py's interpreter allows --
// numbers, strings, comparisons, arithmetic, and/or/not, abs/min/max/round and
// tool.value references -- evaluated with Python's rules (int vs float,
// chained comparisons, the same error messages), never with eval.
#include "geom.hpp"

#include "../core/linalg.hpp"
#include "../core/npy.hpp"
#include "../core/pyfmt.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <variant>

namespace vc {

namespace expr {

struct ExprError : std::runtime_error { using std::runtime_error::runtime_error; };   // ExpressionError
struct PyError : std::runtime_error {                                                     // TypeError / ZeroDivisionError
    std::string kind;
    PyError(std::string k, const std::string& m) : std::runtime_error(m), kind(std::move(k)) {}
};

struct V {
    enum T { INT, FLOAT, BOOL, STR } t = INT;
    long long i = 0;
    double f = 0;
    std::string s;
    static V I(long long v) { V x; x.t = INT; x.i = v; return x; }
    static V F(double v) { V x; x.t = FLOAT; x.f = v; return x; }
    static V B(bool v) { V x; x.t = BOOL; x.i = v; return x; }
    static V S(std::string v) { V x; x.t = STR; x.s = std::move(v); return x; }
    bool num() const { return t != STR; }
    bool intlike() const { return t == INT || t == BOOL; }
    double d() const { return t == FLOAT ? f : (double)i; }
    const char* tn() const { return t == INT ? "int" : t == FLOAT ? "float" : t == BOOL ? "bool" : "str"; }
    bool truth() const { return t == STR ? !s.empty() : t == FLOAT ? f != 0 : i != 0; }
};

// ---- the syntax tree
struct Node {
    std::string type;                       // the Python ast class name
    V value;                                // Constant
    std::string name, attr;                 // Name / Attribute(Name)
    std::string op;                         // operator
    std::vector<std::unique_ptr<Node>> kids;
    std::vector<std::string> ops;           // Compare
    bool keywords = false;                  // Call
    bool allowed_const = true;
};
using P = std::unique_ptr<Node>;

struct Tok { enum K { NUM, STR, NAME, OP, END } k; std::string s; V v; bool bad_const = false; };

std::vector<Tok> lex(const std::string& src) {
    std::vector<Tok> out;
    size_t i = 0, n = src.size();
    auto err = [] { throw ExprError("not a valid expression: invalid syntax"); };
    while (i < n) {
        char c = src[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { i++; continue; }
        if (std::isdigit((unsigned char)c) || (c == '.' && i + 1 < n && std::isdigit((unsigned char)src[i + 1]))) {
            size_t j = i;
            Tok t{Tok::NUM, "", {}};
            if (c == '0' && j + 1 < n && std::strchr("xXoObB", src[j + 1])) {
                int base = std::tolower(src[j + 1]) == 'x' ? 16 : std::tolower(src[j + 1]) == 'o' ? 8 : 2;
                j += 2;
                std::string digits;
                while (j < n && (std::isalnum((unsigned char)src[j]) || src[j] == '_')) { if (src[j] != '_') digits += src[j]; j++; }
                if (digits.empty()) err();
                t.v = V::I(std::stoll(digits, nullptr, base));
            } else {
                std::string num;
                bool flt = false;
                while (j < n && (std::isdigit((unsigned char)src[j]) || src[j] == '_')) { if (src[j] != '_') num += src[j]; j++; }
                if (j < n && src[j] == '.') { flt = true; num += '.'; j++; while (j < n && (std::isdigit((unsigned char)src[j]) || src[j] == '_')) { if (src[j] != '_') num += src[j]; j++; } }
                if (j < n && (src[j] == 'e' || src[j] == 'E')) {
                    size_t k = j + 1;
                    if (k < n && (src[k] == '+' || src[k] == '-')) k++;
                    if (k < n && std::isdigit((unsigned char)src[k])) {
                        flt = true;
                        num += 'e';
                        num += src.substr(j + 1, k - j - 1);
                        j = k;
                        while (j < n && std::isdigit((unsigned char)src[j])) num += src[j++];
                    }
                }
                if (j < n && (src[j] == 'j' || src[j] == 'J')) { j++; t.bad_const = true; }
                if (flt) t.v = V::F(std::strtod(num.c_str(), nullptr));
                else {
                    try { t.v = V::I(std::stoll(num)); } catch (...) { t.v = V::F(std::strtod(num.c_str(), nullptr)); }
                }
            }
            if (j < n && (std::isalpha((unsigned char)src[j]) || src[j] == '_')) err();
            out.push_back(t);
            i = j;
            continue;
        }
        if (std::isalpha((unsigned char)c) || c == '_' || (unsigned char)c >= 0x80) {
            size_t j = i;
            while (j < n && (std::isalnum((unsigned char)src[j]) || src[j] == '_' || (unsigned char)src[j] >= 0x80)) j++;
            std::string w = src.substr(i, j - i);
            // string prefixes
            if (j < n && (src[j] == '\'' || src[j] == '"') && w.size() <= 2) {
                std::string lw;
                for (char ch : w) lw += (char)std::tolower(ch);
                if (lw == "r" || lw == "u" || lw == "b" || lw == "f" || lw == "rb" || lw == "br" || lw == "fr" || lw == "rf") {
                    i = j;
                    char q = src[i++];
                    std::string s;
                    bool raw = lw.find('r') != std::string::npos;
                    while (i < n && src[i] != q) {
                        if (src[i] == '\\' && i + 1 < n && !raw) {
                            char e = src[i + 1];
                            s += e == 'n' ? '\n' : e == 't' ? '\t' : e;
                            i += 2;
                        } else s += src[i++];
                    }
                    if (i >= n) err();
                    i++;
                    Tok t{Tok::STR, "", V::S(s)};
                    t.bad_const = lw.find('b') != std::string::npos || lw.find('f') != std::string::npos;
                    out.push_back(t);
                    continue;
                }
            }
            out.push_back({Tok::NAME, w, {}});
            i = j;
            continue;
        }
        if (c == '\'' || c == '"') {
            char q = c;
            i++;
            std::string s;
            while (i < n && src[i] != q) {
                if (src[i] == '\\' && i + 1 < n) {
                    char e = src[i + 1];
                    s += e == 'n' ? '\n' : e == 't' ? '\t' : e;
                    i += 2;
                } else s += src[i++];
            }
            if (i >= n) err();
            i++;
            out.push_back({Tok::STR, "", V::S(s)});
            continue;
        }
        static const char* ops[] = {"**", "//", "<=", ">=", "==", "!=", "<<", ">>", "+", "-", "*", "/", "%", "<", ">", "(", ")", ",", ".", "|", "^", "&", "~", "=", "[", "]", "@"};
        bool found = false;
        for (const char* o : ops) {
            size_t L = std::strlen(o);
            if (src.compare(i, L, o) == 0) {
                out.push_back({Tok::OP, o, {}});
                i += L;
                found = true;
                break;
            }
        }
        if (!found) err();
    }
    out.push_back({Tok::END, "", {}});
    return out;
}

struct Parser {
    std::vector<Tok> t;
    size_t p = 0;
    int depth = 0;
    [[noreturn]] void err(const std::string& m = "invalid syntax") { throw ExprError("not a valid expression: " + m); }
    const Tok& cur() { return t[p]; }
    bool isop(const char* o) { return cur().k == Tok::OP && cur().s == o; }
    bool isname(const char* w) { return cur().k == Tok::NAME && cur().s == w; }
    P mk(const std::string& type) { auto n = std::make_unique<Node>(); n->type = type; return n; }

    P parse() {
        P e = test();
        if (cur().k != Tok::END) err();
        return e;
    }
    P test() {
        P a = or_test();
        if (isname("if")) {
            p++;
            P c = or_test();
            if (!isname("else")) err();
            p++;
            P b = test();
            P n = mk("IfExp");
            n->kids.push_back(std::move(c));
            n->kids.push_back(std::move(a));
            n->kids.push_back(std::move(b));
            return n;
        }
        return a;
    }
    P boolop(const char* word, P (Parser::*sub)()) {
        P a = (this->*sub)();
        if (!isname(word)) return a;
        P n = mk("BoolOp");
        n->op = word;
        n->kids.push_back(std::move(a));
        while (isname(word)) {
            p++;
            n->kids.push_back((this->*sub)());
        }
        return n;
    }
    P or_test() { return boolop("or", &Parser::and_test); }
    P and_test() { return boolop("and", &Parser::not_test); }
    P not_test() {
        if (isname("not")) {
            p++;
            P n = mk("UnaryOp");
            n->op = "not";
            n->kids.push_back(not_test());
            return n;
        }
        return comparison();
    }
    P comparison() {
        P a = bitor_();
        std::vector<std::string> ops;
        std::vector<P> rhs;
        for (;;) {
            std::string o;
            if (cur().k == Tok::OP && (cur().s == "<" || cur().s == ">" || cur().s == "<=" || cur().s == ">=" || cur().s == "==" || cur().s == "!=")) o = cur().s, p++;
            else if (isname("in")) o = "in", p++;
            else if (isname("is")) {
                p++;
                o = "is";
                if (isname("not")) p++, o = "is not";
            } else if (isname("not") && t[p + 1].k == Tok::NAME && t[p + 1].s == "in") p += 2, o = "not in";
            else break;
            ops.push_back(o);
            rhs.push_back(bitor_());
        }
        if (ops.empty()) return a;
        P n = mk("Compare");
        n->kids.push_back(std::move(a));
        for (auto& r : rhs) n->kids.push_back(std::move(r));
        n->ops = ops;
        return n;
    }
    P binary(const std::vector<std::string>& ops, P (Parser::*sub)()) {
        P a = (this->*sub)();
        for (;;) {
            if (cur().k != Tok::OP || std::find(ops.begin(), ops.end(), cur().s) == ops.end()) return a;
            std::string o = cur().s;
            p++;
            P n = mk("BinOp");
            n->op = o;
            n->kids.push_back(std::move(a));
            n->kids.push_back((this->*sub)());
            a = std::move(n);
        }
    }
    P bitor_() { return binary({"|"}, &Parser::bitxor_); }
    P bitxor_() { return binary({"^"}, &Parser::bitand_); }
    P bitand_() { return binary({"&"}, &Parser::shift_); }
    P shift_() { return binary({"<<", ">>"}, &Parser::arith); }
    P arith() { return binary({"+", "-"}, &Parser::term); }
    P term() { return binary({"*", "/", "//", "%", "@"}, &Parser::factor); }
    P factor() {
        if (cur().k == Tok::OP && (cur().s == "-" || cur().s == "+" || cur().s == "~")) {
            std::string o = cur().s;
            p++;
            P n = mk("UnaryOp");
            n->op = o;
            n->kids.push_back(factor());
            return n;
        }
        return power();
    }
    P power() {
        P a = primary();
        if (isop("**")) {
            p++;
            P n = mk("BinOp");
            n->op = "**";
            n->kids.push_back(std::move(a));
            n->kids.push_back(factor());
            return n;
        }
        return a;
    }
    P primary() {
        P a = atom();
        for (;;) {
            if (isop(".")) {
                p++;
                if (cur().k != Tok::NAME) err();
                std::string attr = cur().s;
                p++;
                P n = mk("Attribute");
                n->attr = attr;
                n->kids.push_back(std::move(a));
                a = std::move(n);
            } else if (isop("(")) {
                p++;
                P n = mk("Call");
                n->kids.push_back(std::move(a));
                while (!isop(")")) {
                    if (cur().k == Tok::NAME && t[p + 1].k == Tok::OP && t[p + 1].s == "=") {
                        p += 2;
                        n->keywords = true;
                        test();
                    } else n->kids.push_back(test());
                    if (isop(",")) p++;
                    else if (!isop(")")) err();
                }
                p++;
                a = std::move(n);
            } else if (isop("[")) {
                p++;
                P n = mk("Subscript");
                n->kids.push_back(std::move(a));
                n->kids.push_back(test());
                if (!isop("]")) err();
                p++;
                a = std::move(n);
            } else return a;
        }
    }
    P atom() {
        const Tok& k = cur();
        if (k.k == Tok::NUM || k.k == Tok::STR) {
            P n = mk("Constant");
            n->value = k.v;
            n->allowed_const = !k.bad_const;
            p++;
            // adjacent string literals concatenate
            while (n->value.t == V::STR && cur().k == Tok::STR) {
                n->value.s += cur().v.s;
                p++;
            }
            return n;
        }
        if (k.k == Tok::NAME) {
            static const char* reserved[] = {"and", "or", "not", "in", "is", "if", "else", "lambda", "for", "while", "def", "class",
                                             "return", "import", "from", "as", "with", "pass", "del", "global", "yield", "try",
                                             "except", "finally", "raise", "assert", "break", "continue", "elif", "nonlocal", "await", "async"};
            for (auto r : reserved)
                if (k.s == r) err();
            P n = mk("Constant");
            if (k.s == "True") n->value = V::B(true);
            else if (k.s == "False") n->value = V::B(false);
            else if (k.s == "None") n->allowed_const = false;
            else {
                n->type = "Name";
                n->name = k.s;
            }
            p++;
            return n;
        }
        if (isop("(")) {
            p++;
            if (isop(")")) {
                p++;
                return mk("Tuple");
            }
            P e = test();
            if (isop(",")) {
                P n = mk("Tuple");
                n->kids.push_back(std::move(e));
                while (isop(",")) {
                    p++;
                    if (isop(")")) break;
                    n->kids.push_back(test());
                }
                e = std::move(n);
            }
            if (!isop(")")) err(cur().k == Tok::END ? "'(' was never closed" : "invalid syntax");
            p++;
            return e;
        }
        if (isop("[")) {
            p++;
            P n = mk("List");
            while (!isop("]")) {
                n->kids.push_back(test());
                if (isop(",")) p++;
                else if (!isop("]")) err();
            }
            p++;
            return n;
        }
        err();
    }
};

using Lookup = std::function<std::optional<V>(const std::string&, const std::string&)>;

std::string unsup(const std::string& op, const V& a, const V& b) {
    return std::string("unsupported operand type(s) for ") + op + ": '" + a.tn() + "' and '" + b.tn() + "'";
}

long long floordiv(long long a, long long b) {
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
    return q;
}

V binop(const std::string& op, const V& a, const V& b) {
    if (op == "+") {
        if (a.t == V::STR && b.t == V::STR) return V::S(a.s + b.s);
        if (a.t == V::STR) throw PyError("TypeError", std::string("can only concatenate str (not \"") + b.tn() + "\") to str");
        if (b.t == V::STR) throw PyError("TypeError", unsup("+", a, b));
        if (a.intlike() && b.intlike()) return V::I(a.i + b.i);
        return V::F(a.d() + b.d());
    }
    if (op == "*") {
        if (a.t == V::STR || b.t == V::STR) {
            const V& s = a.t == V::STR ? a : b;
            const V& k = a.t == V::STR ? b : a;
            if (!k.intlike()) throw PyError("TypeError", std::string("can't multiply sequence by non-int of type '") + k.tn() + "'");
            std::string r;
            for (long long i = 0; i < k.i; i++) r += s.s;
            return V::S(r);
        }
        if (a.intlike() && b.intlike()) return V::I(a.i * b.i);
        return V::F(a.d() * b.d());
    }
    if (op == "%" && a.t == V::STR) throw PyError("TypeError", "not all arguments converted during string formatting");
    if (a.t == V::STR || b.t == V::STR) throw PyError("TypeError", unsup(op, a, b));
    if (op == "-") {
        if (a.intlike() && b.intlike()) return V::I(a.i - b.i);
        return V::F(a.d() - b.d());
    }
    if (op == "/") {
        if (b.d() == 0) throw PyError("ZeroDivisionError", a.intlike() && b.intlike() ? "division by zero" : "float division by zero");
        return V::F(a.d() / b.d());
    }
    if (op == "%") {
        if (a.intlike() && b.intlike()) {
            if (b.i == 0) throw PyError("ZeroDivisionError", "integer modulo by zero");
            return V::I(a.i - floordiv(a.i, b.i) * b.i);
        }
        double x = a.d(), w = b.d();
        if (w == 0) throw PyError("ZeroDivisionError", "float modulo");
        double m = std::fmod(x, w);
        if (m != 0) {
            if ((w < 0) != (m < 0)) m += w;
        } else m = std::copysign(0.0, w);
        return V::F(m);
    }
    if (op == "**") {
        if (a.intlike() && b.intlike() && b.i >= 0) {
            long long r = 1, base = a.i;
            for (long long e = b.i; e > 0; e >>= 1) {
                if (e & 1) r *= base;
                base *= base;
            }
            return V::I(r);
        }
        double x = a.d(), y = b.d();
        if (x == 0 && y < 0) throw PyError("ZeroDivisionError", "0.0 cannot be raised to a negative power");
        if (x < 0 && y != std::floor(y)) throw PyError("ValueError", "complex result");
        return V::F(std::pow(x, y));
    }
    throw ExprError("not allowed in an expression: BinOp");
}

bool compare(const std::string& op, const V& a, const V& b) {
    if (op == "==" || op == "!=") {
        bool eq;
        if (a.t == V::STR || b.t == V::STR) eq = a.t == b.t && a.s == b.s;
        else if (a.intlike() && b.intlike()) eq = a.i == b.i;
        else eq = a.d() == b.d();
        return op == "==" ? eq : !eq;
    }
    if ((a.t == V::STR) != (b.t == V::STR))
        throw PyError("TypeError", "'" + op + "' not supported between instances of '" + a.tn() + "' and '" + b.tn() + "'");
    int c;
    if (a.t == V::STR) c = a.s < b.s ? -1 : a.s > b.s ? 1 : 0;
    else if (a.intlike() && b.intlike()) c = a.i < b.i ? -1 : a.i > b.i ? 1 : 0;
    else c = a.d() < b.d() ? -1 : a.d() > b.d() ? 1 : 0;
    if (op == "<") return c < 0;
    if (op == "<=") return c <= 0;
    if (op == ">") return c > 0;
    return c >= 0;
}

V eval(const Node& n, const Lookup& look) {
    if (n.type == "Constant") {
        if (!n.allowed_const) throw ExprError("not allowed in an expression: Constant");
        return n.value;
    }
    if (n.type == "Attribute" && n.kids[0]->type == "Name") {
        auto v = look(n.kids[0]->name, n.attr);
        if (!v) throw ExprError(n.kids[0]->name + "." + n.attr + " has no value");
        return *v;
    }
    if (n.type == "BoolOp") {
        std::vector<bool> vals;
        for (auto& k : n.kids) vals.push_back(eval(*k, look).truth());
        bool r = n.op == "and";
        for (bool v : vals) r = n.op == "and" ? (r && v) : (r || v);
        return V::B(r);
    }
    if (n.type == "UnaryOp") {
        if (n.op == "not") return V::B(!eval(*n.kids[0], look).truth());
        if (n.op == "-" || n.op == "+") {
            V v = eval(*n.kids[0], look);
            if (v.t == V::STR) throw PyError("TypeError", "bad operand type for unary " + n.op + ": 'str'");
            if (n.op == "+") return v.t == V::BOOL ? V::I(v.i) : v;
            return v.intlike() ? V::I(-v.i) : V::F(-v.f);
        }
    }
    if (n.type == "BinOp" && (n.op == "+" || n.op == "-" || n.op == "*" || n.op == "/" || n.op == "%" || n.op == "**")) {
        V a = eval(*n.kids[0], look);
        V b = eval(*n.kids[1], look);
        return binop(n.op, a, b);
    }
    if (n.type == "Compare") {
        V left = eval(*n.kids[0], look);
        for (size_t i = 0; i < n.ops.size(); i++) {
            const std::string& op = n.ops[i];
            if (op == "in" || op == "not in" || op == "is" || op == "is not") throw ExprError("comparison not allowed");
            V right = eval(*n.kids[i + 1], look);
            if (!compare(op, left, right)) return V::B(false);
            left = right;
        }
        return V::B(true);
    }
    if (n.type == "Call" && n.kids[0]->type == "Name" && !n.keywords) {
        const std::string& f = n.kids[0]->name;
        if (f == "abs" || f == "min" || f == "max" || f == "round") {
            std::vector<V> args;
            for (size_t i = 1; i < n.kids.size(); i++) args.push_back(eval(*n.kids[i], look));
            if (f == "abs") {
                if (args.size() != 1) throw PyError("TypeError", "abs() takes exactly one argument (" + std::to_string(args.size()) + " given)");
                const V& v = args[0];
                if (v.t == V::STR) throw PyError("TypeError", "bad operand type for abs(): 'str'");
                return v.intlike() ? V::I(std::llabs(v.i)) : V::F(std::fabs(v.f));
            }
            if (f == "round") {
                if (args.empty() || args.size() > 2) throw PyError("TypeError", "round() takes at most 2 arguments");
                const V& v = args[0];
                if (v.t == V::STR) throw PyError("TypeError", "type str doesn't define __round__ method");
                if (args.size() == 1) return v.intlike() ? V::I(v.i) : V::I(py_round_int(v.f));
                if (!args[1].intlike()) throw PyError("TypeError", std::string("'") + args[1].tn() + "' object cannot be interpreted as an integer");
                if (v.intlike()) return V::I(v.i);
                return V::F(py_round(v.f, (int)args[1].i));
            }
            // min / max
            if (args.empty()) throw PyError("TypeError", f + " expected at least 1 argument, got 0");
            std::vector<V> items;
            if (args.size() == 1) {
                if (args[0].t != V::STR) throw PyError("TypeError", std::string("'") + args[0].tn() + "' object is not iterable");
                for (char ch : args[0].s) items.push_back(V::S(std::string(1, ch)));
                if (items.empty()) throw PyError("ValueError", f + "() iterable argument is empty");
            } else items = args;
            V best = items[0];
            for (size_t i = 1; i < items.size(); i++)
                if (compare(f == "min" ? "<" : ">", items[i], best)) best = items[i];
            return best;
        }
    }
    throw ExprError("not allowed in an expression: " + n.type);
}

// ast.walk order (breadth first) of tool.value references
std::vector<std::string> references(const Node& root) {
    std::vector<std::string> out;
    std::vector<const Node*> q{&root};
    for (size_t i = 0; i < q.size(); i++) {
        const Node* n = q[i];
        if (n->type == "Attribute" && n->kids[0]->type == "Name") out.push_back(n->kids[0]->name + "." + n->attr);
        // ast child order: Compare(left, comparators), BinOp(left, right), Call(func, args), IfExp(test, body, orelse)
        if (n->type == "IfExp") {
            q.push_back(n->kids[0].get());
            q.push_back(n->kids[1].get());
            q.push_back(n->kids[2].get());
        } else {
            for (auto& k : n->kids) q.push_back(k.get());
        }
    }
    return out;
}

P parse(const std::string& s) {
    Parser ps{lex(s)};
    return ps.parse();
}

Json to_json(const V& v) {
    switch (v.t) {
        case V::INT: return Json(v.i);
        case V::FLOAT: return Json(v.f);
        case V::BOOL: return Json(v.i != 0);
        default: return Json(v.s);
    }
}

}  // namespace expr

namespace {

std::optional<expr::V> result_lookup(const Context& ctx, const std::string& tid, const std::string& key) {
    auto it = ctx.results.find(tid);
    if (it == ctx.results.end()) return std::nullopt;
    const ToolResult& r = it->second;
    if (key == "score") return expr::V::F(r.score);
    if (key == "status") return expr::V::S(r.status);
    if (key == "passed") return expr::V::B(r.status == "pass");
    if (!r.values.is_object() || !r.values.contains(key)) return std::nullopt;
    const Json& v = r.values[key];
    if (v.is_boolean()) return expr::V::B(v.get<bool>());
    if (v.is_number_integer()) return expr::V::I(v.get<long long>());
    if (v.is_number()) return expr::V::F(v.get<double>());
    if (v.is_string()) return expr::V::S(v.get<std::string>());
    return std::nullopt;
}

std::string strip(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string r;
    for (size_t i = 0; i < v.size(); i++) r += (i ? sep : "") + v[i];
    return r;
}

}  // namespace

// ---------------------------------------------------------------- results analysis
class ResultsAnalysis : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat&, Context& ctx) override {
        double t0 = now_ms();
        std::string e = has("expression") && p["expression"].is_string() ? strip(p["expression"].get<std::string>()) : "";
        if (e.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "no expression set");
        auto look = [&](const std::string& t, const std::string& k) { return result_lookup(ctx, t, k); };
        expr::P tree;
        expr::V val;
        try {
            tree = expr::parse(e);
            val = expr::eval(*tree, look);
        } catch (const expr::ExprError& x) {
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, x.what(), now_ms() - t0);
        } catch (const expr::PyError& x) {
            if (x.kind != "TypeError" && x.kind != "ZeroDivisionError") throw std::runtime_error(x.kind + ": " + x.what());
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, x.kind + ": " + x.what(), now_ms() - t0);
        }
        bool ok = val.truth();
        Json v = {{"value", expr::to_json(val)}};
        for (auto& ref : expr::references(*tree)) {
            auto dot = ref.find('.');
            auto r = look(ref.substr(0, dot), ref.substr(dot + 1));
            v[ref] = r ? expr::to_json(*r) : Json();
        }
        return ToolResult(id, type_name, ok ? "pass" : "fail", ok ? 1.0 : 0.0, v, {}, ok ? "" : "false: " + e, now_ms() - t0);
    }
};
VC_REGISTER("results_analysis", ResultsAnalysis)

// ---------------------------------------------------------------- intersect
namespace {
using D2 = std::array<double, 2>;
D2 j2(const Json& a) { return {a[0].get<double>(), a[1].get<double>()}; }

Graphic cross(double x, double y, int arm, const std::string& st, bool vertical, const std::string& label = "") {
    if (!vertical) return abs_graphic("line", {{(float)(x - arm), (float)y}, {(float)(x + arm), (float)y}}, st, label);
    return abs_graphic("line", {{(float)x, (float)(y - arm)}, {(float)x, (float)(y + arm)}}, st, label);
}

std::vector<D2> circle_points(Json A, Json B) {
    if (A["type"] == "circle" && B["type"] == "line") std::swap(A, B);
    if (A["type"] == "line") {
        D2 c = j2(B["center"]), p = j2(A["point"]), d = j2(A["direction"]);
        double r = B["radius"].get<double>();
        double nd = la::norm(d.data(), 2);
        if (nd == 0) nd = 1.0;
        d = {d[0] / nd, d[1] / nd};
        D2 f = {p[0] - c[0], p[1] - c[1]};
        double bq = la::dot(f.data(), d.data(), 2);
        double disc = bq * bq - (la::dot(f.data(), f.data(), 2) - r * r);
        if (disc < 0) return {};
        double rt = std::sqrt(disc);
        std::vector<double> ts{-bq - rt};
        if (-bq + rt != -bq - rt) ts.push_back(-bq + rt);
        std::sort(ts.begin(), ts.end());
        std::vector<D2> out;
        for (double t : ts) out.push_back({p[0] + d[0] * t, p[1] + d[1] * t});
        return out;
    }
    D2 c1 = j2(A["center"]), c2 = j2(B["center"]);
    double r1 = A["radius"].get<double>(), r2 = B["radius"].get<double>();
    D2 dv = {c2[0] - c1[0], c2[1] - c1[1]};
    double dd = la::norm(dv.data(), 2);
    if (dd == 0 || dd > r1 + r2 || dd < std::abs(r1 - r2)) return {};
    double a = (r1 * r1 - r2 * r2 + dd * dd) / (2 * dd);
    double h = std::sqrt(std::max(0.0, r1 * r1 - a * a));
    D2 m = {c1[0] + a * dv[0] / dd, c1[1] + a * dv[1] / dd};
    D2 off = {h * -dv[1] / dd, h * dv[0] / dd};
    std::vector<D2> pts;
    if (h > 1e-9) pts = {{m[0] + off[0], m[1] + off[1]}, {m[0] - off[0], m[1] - off[1]}};
    else pts = {m};
    std::sort(pts.begin(), pts.end());
    return pts;
}
}  // namespace

class Intersect : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat&, Context& ctx) override {
        double t0 = now_ms();
        auto A = geom::passed_feature(ctx, has("first") ? py_str(p["first"]) : "");
        auto B = geom::passed_feature(ctx, has("second") ? py_str(p["second"]) : "");
        for (auto* f : {&A, &B})   // Python: (A or {}).get("type") on a published blob list
            if (*f && (*f)->is_array() && !(*f)->empty()) throw std::runtime_error("AttributeError: 'list' object has no attribute 'get'");
        auto kind = [](const std::optional<Json>& f) { return f && f->is_object() && f->contains("type") ? (*f)["type"].get<std::string>() : std::string("None"); };
        std::string ka = kind(A), kb = kind(B);
        auto okk = [](const std::string& k) { return k == "line" || k == "circle"; };
        bool emptyA = !A || (A->is_object() && A->empty()), emptyB = !B || (B->is_object() && B->empty());
        if (emptyA || emptyB || !okk(ka) || !okk(kb))
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "needs two Find line / Find circle results as first and second");
        if (ka == "circle" || kb == "circle") {
            auto pts = circle_points(*A, *B);
            int need = pi("expect_points", 0);
            bool ok = !pts.empty() && (!need || (int)pts.size() == need);
            std::string st = ok ? "pass" : "fail";
            Json v = {{"points", (int)pts.size()}};
            std::vector<Graphic> g;
            for (size_t i = 0; i < pts.size(); i++) {
                v["x" + std::to_string(i + 1)] = py_round(pts[i][0], 3);
                v["y" + std::to_string(i + 1)] = py_round(pts[i][1], 3);
                g.push_back(cross(pts[i][0], pts[i][1], 8, st, false));
                g.push_back(cross(pts[i][0], pts[i][1], 8, st, true, id + " " + std::to_string(i + 1)));
            }
            if (!pts.empty()) {
                Json all = Json::array();
                for (auto& q : pts) all.push_back({q[0], q[1]});
                ctx.share[id] = Json{{"type", "point"}, {"point", {pts[0][0], pts[0][1]}}, {"points", all}};
            }
            std::string why = ok ? "" : pts.empty() ? "they do not meet"
                                                    : std::to_string(pts.size()) + " intersection(s), expected " + std::to_string(need);
            return ToolResult(id, type_name, st, ok ? 1.0 : 0.0, v, g, why, now_ms() - t0);
        }
        D2 p1 = j2((*A)["point"]), d1 = j2((*A)["direction"]), p2 = j2((*B)["point"]), d2 = j2((*B)["direction"]);
        double den = d1[0] * d2[1] - d1[1] * d2[0];
        double angle = std::acos(std::min(1.0, std::abs(la::dot(d1.data(), d2.data(), 2)))) * (180.0 / 3.141592653589793);
        if (std::abs(den) < 1e-9)
            return ToolResult(id, type_name, "fail", 0.0, Json{{"angle_deg", angle}}, {}, "the lines are parallel: no corner", now_ms() - t0);
        double t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den;
        double x = p1[0] + d1[0] * t, y = p1[1] + d1[1] * t;
        ctx.share[id] = Json{{"type", "point"}, {"point", {x, y}}};
        double lo = pf("min_angle", 0), hi = pf("max_angle", 0);
        std::vector<std::string> why;
        if (lo != 0 && angle < lo) why.push_back("angle " + pyfmt("%.2f", angle) + " under " + pyfmt("%.2f", lo));
        if (hi != 0 && angle > hi) why.push_back("angle " + pyfmt("%.2f", angle) + " over " + pyfmt("%.2f", hi));
        bool ok = why.empty();
        std::string st = ok ? "pass" : "fail";
        std::vector<Graphic> g{cross(x, y, 8, st, false), cross(x, y, 8, st, true, id + " " + pyfmt("%.2f", angle) + " deg")};
        return ToolResult(id, type_name, st, ok ? 1.0 : 0.0, Json{{"x", py_round(x, 3)}, {"y", py_round(y, 3)}, {"angle_deg", py_round(angle, 3)}},
                          g, join(why, "; "), now_ms() - t0);
    }
};
VC_REGISTER("intersect", Intersect)

// ---------------------------------------------------------------- fit shape

namespace {
std::vector<double> centres(const Context& ctx, const std::string& source) {
    std::vector<double> out;
    auto it = ctx.share.find(source);
    if (it != ctx.share.end())
        if (auto j = std::any_cast<Json>(&it->second); j && j->is_array()) {
            for (auto& b : *j) {
                if (b.contains("cx")) {
                    out.push_back(b["cx"].get<double>());
                    out.push_back(b["cy"].get<double>());
                } else {
                    out.push_back(b["x"].get<double>() + b.value("w", 0.0) / 2.0);
                    out.push_back(b["y"].get<double>() + b.value("h", 0.0) / 2.0);
                }
            }
            return out;
        }
    auto r = ctx.results.find(source);
    if (r == ctx.results.end() || !r->second.values.is_object() || !r->second.values.contains("matches")) return out;
    const Json& m = r->second.values["matches"];
    if (!m.is_array()) return out;
    for (auto& q : m)
        if (q.is_object() && q.contains("x")) {
            out.push_back(q["x"].get<double>() + q.value("w", 0.0) / 2.0);
            out.push_back(q["y"].get<double>() + q.value("h", 0.0) / 2.0);
        }
    return out;
}
}  // namespace

class FitShape : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat&, Context& ctx) override {
        double t0 = now_ms();
        std::string source = has("source") ? py_str(p["source"]) : "";
        auto pts = centres(ctx, source);
        int n = (int)(pts.size() / 2);
        std::string fit = ps("fit", "line");
        int need = fit == "circle" ? 3 : fit == "ellipse" ? 6 : 2;
        if (n < need)
            return ToolResult(id, type_name, "fail", 0.0, Json{{"points", n}}, {},
                              std::to_string(n) + " centres from " + (source.empty() ? std::string("(no source)") : source) + ", need " + std::to_string(need),
                              now_ms() - t0);
        double mm = ctx.mm_per_px, k = mm > 0 ? mm : 1.0;
        Json v;
        std::vector<double> resid;
        std::string shape_kind;
        std::vector<cv::Point2f> shape_pts;
        constexpr double PI = 3.141592653589793;
        if (fit == "ellipse") {
            auto e = geom::fit_ellipse_robust(pts, 0);
            ctx.share[id] = Json{{"type", "circle"}, {"center", {e.cx, e.cy}}, {"radius", (e.a + e.b) / 2.0}};
            v = {{"cx", py_round(e.cx, 3)}, {"cy", py_round(e.cy, 3)}, {"major", py_round(2 * std::max(e.a, e.b) * k, 4)},
                 {"minor", py_round(2 * std::min(e.a, e.b) * k, 4)}, {"angle_deg", py_round(e.ang, 2)}};
            double tt = e.ang * (PI / 180.0);
            for (int i = 0; i < 64; i++) {
                double u = i == 63 ? 2 * PI : i * (2 * PI / 63.0);
                shape_pts.emplace_back((float)(e.cx + e.a * std::cos(u) * std::cos(tt) - e.b * std::sin(u) * std::sin(tt)),
                                       (float)(e.cy + e.a * std::cos(u) * std::sin(tt) + e.b * std::sin(u) * std::cos(tt)));
            }
            shape_kind = "poly";
            resid = e.resid;
        } else if (fit == "circle") {
            auto f = geom::fit_circle<double>(pts);
            ctx.share[id] = Json{{"type", "circle"}, {"center", {f.cx, f.cy}}, {"radius", f.r}};
            v = {{"cx", py_round(f.cx, 3)}, {"cy", py_round(f.cy, 3)}, {"radius", py_round(f.r * k, 4)}};
            for (int i = 0; i < 64; i++) {
                double a = i == 63 ? 2 * PI : i * (2 * PI / 63.0);
                shape_pts.emplace_back((float)(f.cx + f.r * std::cos(a)), (float)(f.cy + f.r * std::sin(a)));
            }
            shape_kind = "poly";
            resid = f.resid;
        } else {
            auto f = geom::fit_line<double>(pts);
            std::vector<double> cen(pts.size());
            for (int i = 0; i < n; i++) {
                cen[2 * i] = pts[2 * i] - f.c[0];
                cen[2 * i + 1] = pts[2 * i + 1] - f.c[1];
            }
            auto t = la::matvec(cen.data(), f.d, n, 2);
            double tmin = *std::min_element(t.begin(), t.end()), tmax = *std::max_element(t.begin(), t.end());
            double p0[2] = {f.c[0] + f.d[0] * tmin, f.c[1] + f.d[1] * tmin}, p1[2] = {f.c[0] + f.d[0] * tmax, f.c[1] + f.d[1] * tmax};
            double ang = std::atan2(f.d[1], f.d[0]) * (180.0 / PI);
            ctx.share[id] = Json{{"type", "line"}, {"p0", {p0[0], p0[1]}}, {"p1", {p1[0], p1[1]}}, {"point", {f.c[0], f.c[1]}},
                                 {"direction", {f.d[0], f.d[1]}}, {"angle", ang}};
            v = {{"angle_deg", py_round(ang, 3)}};
            shape_kind = "line";
            shape_pts = {{(float)p0[0], (float)p0[1]}, {(float)p1[0], (float)p1[1]}};
            resid = f.resid;
        }
        double worst = 0;
        for (double r : resid) worst = std::max(worst, std::abs(r));
        worst *= k;
        v["points"] = n;
        v["max_deviation"] = py_round(worst, 4);
        v["unit"] = mm > 0 ? "mm" : "px";
        double limit = pf("max_deviation", 0);
        bool ok = !(limit != 0 && worst > limit);
        std::string st = ok ? "pass" : "fail";
        std::vector<Graphic> g{abs_graphic(shape_kind, shape_pts, st, id + " dev " + pyfmt("%.2f", worst))};
        for (int i = 0; i < n && i < (int)resid.size(); i++) {
            double x = pts[2 * i], y = pts[2 * i + 1];
            g.push_back(abs_graphic("rect", {{(float)(x - 3), (float)(y - 3)}, {(float)(x + 3), (float)(y + 3)}},
                                    limit != 0 && std::abs(resid[i]) * k > limit ? "fail" : "pass"));
        }
        return ToolResult(id, type_name, st, ok ? 1.0 : 0.0, v, g,
                          ok ? "" : "a centre is " + pyfmt("%.3f", worst) + " " + v["unit"].get<std::string>() + " off the " + fit, now_ms() - t0);
    }
};
VC_REGISTER("fit_shape", FitShape)

// ---------------------------------------------------------------- colour match
namespace {
// np.median(lab, axis=0) of float32 Lab pixels -> [L*100/255, a-128, b-128] as float32
std::array<float, 3> mean_lab(const cv::Mat& patch, const cv::Mat& mask) {
    cv::Mat lab;
    cv::cvtColor(patch, lab, cv::COLOR_BGR2Lab);
    bool use = !mask.empty() && mask.rows == patch.rows && mask.cols == patch.cols;
    std::vector<float> c[3];
    for (int y = 0; y < lab.rows; y++)
        for (int x = 0; x < lab.cols; x++)
            if (!use || mask.at<uint8_t>(y, x) > 0) {
                auto px = lab.at<cv::Vec3b>(y, x);
                for (int k = 0; k < 3; k++) c[k].push_back((float)px[k]);
            }
    float med[3];
    for (int k = 0; k < 3; k++) {
        auto& s = c[k];
        if (s.empty()) {
            med[k] = NAN;
            continue;
        }
        size_t n = s.size(), h = n / 2;
        std::nth_element(s.begin(), s.begin() + h, s.end());
        float m = s[h];
        if (n % 2 == 0) m = (*std::max_element(s.begin(), s.begin() + h) + m) / 2.0f;
        med[k] = m;
    }
    return {med[0] * 100.0f / 255.0f, med[1] - 128.0f, med[2] - 128.0f};
}
}  // namespace

class ColorMatch : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        if (ctx.reference.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "needs a reference image");
        auto [cur, o1] = this->crop(image, ctx);
        cv::Mat mask_cur = crop_mask.empty() ? cv::Mat() : crop_mask.clone();
        cv::Mat saved = ctx.fixture;
        ctx.fixture = cv::Mat();
        std::pair<cv::Mat, cv::Point> rc;
        try {
            rc = this->crop(ctx.reference, ctx);
        } catch (...) {
            ctx.fixture = saved;
            throw;
        }
        ctx.fixture = saved;
        cv::Mat mask_ref = crop_mask;
        if (cur.empty() || rc.first.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "empty region");
        auto a = mean_lab(cur, mask_cur), b = mean_lab(rc.first, mask_ref);
        float d[3] = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
        double de = (double)npy::norm_f32(d, 3);
        double limit = pf("max_delta_e", 10.0);
        bool ok = de <= limit;
        Json v = {{"delta_e", py_round(de, 2)}, {"L", py_round(a[0], 1)}, {"a", py_round(a[1], 1)}, {"b", py_round(a[2], 1)},
                  {"ref_L", py_round(b[0], 1)}, {"ref_a", py_round(b[1], 1)}, {"ref_b", py_round(b[2], 1)}};
        std::string st = ok ? "pass" : "fail";
        return ToolResult(id, type_name, st, std::max(0.0, 1.0 - de / (2 * limit)), v, roi_graphic(ctx, st, id + " dE " + pyfmt("%.1f", de)),
                          ok ? "" : "colour differs by dE " + pyfmt("%.1f", de) + " (limit " + pyfmt("%.1f", limit) + ")", now_ms() - t0);
    }
};
VC_REGISTER("color_match", ColorMatch)

}  // namespace vc
