// Read text and Verify text (vision_station/tools/text.py ocr, ocv, and the
// legacy ocr_auto / text_detect, which run the same reader).
#include "../core/npy.hpp"
#include "../core/pyfmt.hpp"
#include "../core/tool.hpp"
#include "ocr_engine.hpp"

#include <regex>

namespace vc {

using ocr::Line;
using std::u32string;

namespace {

u32string U(const std::string& s) { return ocr::from_utf8(s); }
std::string S(const u32string& s) { return ocr::to_utf8(s); }

u32string squeeze_ws(const u32string& t) {
    u32string o;
    for (char32_t c : t)
        if (!ocr::is_space(c)) o += c;
    return o;
}

u32string look_alike(u32string t) {
    for (auto& c : t) switch (c) {
        case U'O': case U'o': case U'Q': c = U'0'; break;
        case U'I': case U'l': case U'|': case U'i': c = U'1'; break;
        case U'S': case U's': c = U'5'; break;
        case U'B': c = U'8'; break;
        case U'Z': case U'z': c = U'2'; break;
        default: break;
    }
    return t;
}

u32string latin_only(const u32string& t) {
    u32string o;
    for (char32_t c : t)
        if (c < 0x250 || c == 0x2103 || c == 0x2126 || c == 0x00B5 || c == 0x2122) o += c;
    return ocr::strip(o);
}

u32string fit_charset(u32string text, const u32string& charset) {
    if (charset.empty()) return text;
    bool any_lower = false, any_upper = false;
    for (char32_t c : charset) { any_lower |= ocr::is_lower(c); any_upper |= ocr::is_upper(c); }
    if (!any_lower) text = ocr::upper(text);
    else if (!any_upper) text = ocr::lower(text);
    u32string o;
    for (char32_t c : text)
        if (c == U' ' || charset.find(c) != u32string::npos) o += c;
    return ocr::strip(o);
}

// Python's repr() of a str
std::string py_repr_str(const u32string& s) {
    bool has_sq = s.find(U'\'') != u32string::npos, has_dq = s.find(U'"') != u32string::npos;
    char32_t q = (has_sq && !has_dq) ? U'"' : U'\'';
    u32string o(1, q);
    for (char32_t c : s) {
        if (c == U'\\') o += U"\\\\";
        else if (c == q) { o += U'\\'; o += c; }
        else if (c == U'\n') o += U"\\n";
        else if (c == U'\r') o += U"\\r";
        else if (c == U'\t') o += U"\\t";
        else if (c < 0x20 || c == 0x7F) {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02x", (unsigned)c);
            o += U(b);
        } else o += c;
    }
    o += q;
    return S(o);
}

u32string head(const u32string& s, size_t n) { return s.substr(0, std::min(n, s.size())); }

double mean_conf(const std::vector<Line>& a) {
    std::vector<double> v;
    for (auto& l : a) v.push_back(l.confidence);
    return v.empty() ? 0.0 : npy::sum_f64(v.data(), v.size()) / (double)v.size();
}

int common_turn(const std::vector<Line>& lines) {
    if (lines.empty()) return 0;
    std::map<int, int> c;   // ascending, as a small-int set iterates
    for (auto& l : lines) c[l.turn]++;
    int best = 0, n = -1;
    for (auto& [t, k] : c)
        if (k > n) { n = k; best = t; }
    return best;
}

std::vector<Graphic> line_graphics(const std::vector<Line>& lines, int ox, int oy, const std::string& status) {
    std::vector<Graphic> g;
    for (auto& l : lines)
        if (!l.poly.empty()) {
            Graphic gr("poly", {}, status, S(head(l.text, 28)));
            for (auto& p : l.poly) gr.points.emplace_back(ox + p[0], oy + p[1]);
            g.push_back(gr);
        }
    return g;
}

std::string unsure_note(const std::vector<Line>& unsure, double min_conf) {
    const Line* b = &unsure[0];
    for (auto& l : unsure)
        if (l.confidence > b->confidence) b = &l;
    return "text read with confidence " + pyfmt("%.2f", b->confidence) + ", below the minimum " + pyfmt("%.2f", min_conf) + ": '" + S(head(b->text, 40)) + "'";
}

struct Region {
    std::vector<Line> lines, unsure;
    int ox = 0, oy = 0;
    std::string engine;
};

}  // namespace

class TextTool : public Tool {
public:
    using Tool::Tool;

protected:
    double min_conf() const {
        double v = pf("min_confidence", 0.5);
        return v == 0 ? 0.0 : v;
    }

    Region read_region(const cv::Mat& image, Context& ctx) {
        auto [crp, off] = crop(image, ctx);
        if (crp.empty()) throw std::invalid_argument("empty region");
        std::string detail = ps("detail", "auto");
        if (detail.empty()) detail = "auto";
        auto lines = ocr::read(crp, detail);
        double mc = min_conf();
        if (lines.empty()) {
            auto one = ocr::read_single_line(crp);
            if (one && one->confidence >= std::max(mc, 0.6)) lines = {*one};
        }
        if (pb("ignore_stray_marks", true)) lines = ocr::clean_lines(lines);
        u32string charset = U(ps("charset", ""));
        int min_chars = std::max(1, pi("min_chars", 1) == 0 ? 1 : pi("min_chars", 1));
        std::string script = ps("script", "latin");
        bool latin = (script.empty() ? std::string("latin") : script) != "any";
        auto fit = [&](const u32string& t) { return fit_charset(latin ? latin_only(t) : t, charset); };
        Region r;
        r.ox = off.x;
        r.oy = off.y;
        for (auto ln : lines) {
            u32string t = fit(ln.text);
            if ((int)t.size() < min_chars) continue;
            ln.text = t;
            if (ln.core) {
                u32string c = fit(*ln.core);
                if (c.empty()) ln.core.reset();
                else ln.core = c;
            }
            (ln.confidence >= mc ? r.lines : r.unsure).push_back(ln);
        }
        r.engine = ocr::engine_name();
        return r;
    }

    ToolResult reader_error(Context& ctx, const std::string& msg, double t0) {
        return ToolResult(id, type_name, "error", 0.0, Json::object(), roi_graphic(ctx, "error"), msg, now_ms() - t0);
    }

    Json compare_scanned(Context& ctx, const std::vector<u32string>& read, bool exact = false) {
        if (!pb("compare_barcode", false)) return Json::object();
        auto it = ctx.share.find("barcode");
        if (it == ctx.share.end()) return Json::object();
        std::string code = std::any_cast<std::string>(it->second);
        u32string sc = ocr::strip(U(code));
        if (sc.empty()) return Json::object();
        bool la = !exact && pb("look_alike", true);
        auto sq = [&](const u32string& t) { u32string x = ocr::upper(squeeze_ws(t)); return la ? look_alike(x) : x; };
        u32string want = sq(sc);
        bool match = false;
        if (!want.empty())
            for (auto& t : read) {
                u32string got = sq(t);
                if (exact ? got == want : got.find(want) != u32string::npos) { match = true; break; }
            }
        return {{"barcode_scanned", S(sc)}, {"barcode_match", match}};
    }
};

class Ocr : public TextTool {
public:
    using TextTool::TextTool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        Region R;
        try {
            R = read_region(image, ctx);
        } catch (const std::invalid_argument& e) {
            return reader_error(ctx, e.what(), t0);
        } catch (const std::runtime_error& e) {
            return reader_error(ctx, e.what(), t0);
        }
        auto& lines = R.lines;
        std::vector<u32string> texts;
        for (auto& l : lines) texts.push_back(l.text);
        u32string join = U(has("join") ? ps("join", " ") : " "), text;
        for (size_t i = 0; i < texts.size(); i++) text += (i ? join : U"") + texts[i];
        double conf = mean_conf(!lines.empty() ? lines : R.unsure);
        u32string want = ocr::strip(U(ps("expect_contains", "")));
        bool ok = !lines.empty();
        std::string why = ok ? "" : "no text found in this region";
        if (lines.empty() && !R.unsure.empty()) why = unsure_note(R.unsure, min_conf());
        u32string joined_bar;
        for (size_t i = 0; i < texts.size(); i++) joined_bar += (i ? U" | " : U"") + texts[i];
        if (!lines.empty() && !want.empty()) {
            bool la = pb("look_alike", true);
            auto squeeze = [&](const u32string& t) { u32string x = ocr::upper(squeeze_ws(t)); return la ? look_alike(x) : x; };
            u32string need = squeeze(want);
            std::vector<u32string> cands{text};
            for (auto& l : lines) if (l.core) cands.push_back(*l.core);
            ok = false;
            for (auto& c : cands) if (squeeze(c).find(need) != u32string::npos) { ok = true; break; }
            why = ok ? "" : py_repr_str(want) + " not found; read: " + S(head(joined_bar, 200));
        }
        std::vector<u32string> rd{text};
        rd.insert(rd.end(), texts.begin(), texts.end());
        for (auto& l : lines) if (l.core) rd.push_back(*l.core);
        Json cmp = compare_scanned(ctx, rd);
        if (!cmp.empty() && !cmp["barcode_match"].get<bool>()) {
            ok = false;
            std::string r = S(head(joined_bar, 120));
            why = "Barcode not good: scanned " + cmp["barcode_scanned"].get<std::string>() + ", read " + (r.empty() ? "nothing" : r);
        }
        std::string st = ok ? "pass" : "fail";
        auto g = roi_graphic(ctx, st, lines.empty() ? id : "");
        if (lines.empty()) { /* label is the id */ }
        for (auto& x : line_graphics(lines, R.ox, R.oy, st)) g.push_back(x);
        for (auto& x : line_graphics(R.unsure, R.ox, R.oy, "warn")) g.push_back(x);
        Json tj = Json::array(), blocks = Json::array();
        for (auto& t : texts) tj.push_back(S(t));
        for (auto& l : lines)
            blocks.push_back({{"text", S(l.text)}, {"confidence", py_round(l.confidence, 3)}, {"box", l.box},
                              {"image_box", {R.ox + l.box[0], R.oy + l.box[1], l.box[2], l.box[3]}}});
        Json vals = {{"text", S(text)}, {"lines", tj}, {"texts", tj}, {"blocks_found", lines.size()}, {"blocks", blocks},
                     {"confidence", py_round(conf, 3)}, {"engine", R.engine}, {"skew_deg", 0.0},
                     {"quarter_turn", common_turn(!lines.empty() ? lines : R.unsure)}};
        for (auto& [k, v] : cmp.items()) vals[k] = v;
        return ToolResult(id, type_name, st, conf, vals, g, why, now_ms() - t0);
    }
};
VC_REGISTER("ocr", Ocr)

class LegacyFinder : public Ocr {
public:
    LegacyFinder(std::string id, Json params) : Ocr(std::move(id), fixup(std::move(params))) {}

private:
    static Json fixup(Json p) {
        auto num = [&](const char* k, double d) {
            if (!p.contains(k) || p[k].is_null()) return d;
            double v = p[k].is_number() ? p[k].get<double>() : d;
            return v == 0 ? d : v;
        };
        p["min_chars"] = std::min(2, (int)num("min_chars", 1));
        double mc = p.contains("min_confidence") && p["min_confidence"].is_number() ? p["min_confidence"].get<double>() : 0.0;
        p["min_confidence"] = std::max(0.5, mc);
        return p;
    }
};
using OcrAuto = LegacyFinder;
class TextDetect : public LegacyFinder { public: using LegacyFinder::LegacyFinder; };
VC_REGISTER("ocr_auto", OcrAuto)
VC_REGISTER("text_detect", TextDetect)

class Ocv : public TextTool {
public:
    using TextTool::TextTool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        Region R;
        try {
            R = read_region(image, ctx);
        } catch (const std::invalid_argument& e) {
            return reader_error(ctx, e.what(), t0);
        } catch (const std::runtime_error& e) {
            return reader_error(ctx, e.what(), t0);
        }
        auto& lines = R.lines;
        bool strip_sp = pb("strip_spaces", true);
        auto norm = [&](const u32string& t) { return strip_sp ? squeeze_ws(t) : t; };
        std::vector<u32string> each, cores;
        for (auto& l : lines) each.push_back(norm(l.text));
        for (auto& l : lines) if (l.core) cores.push_back(norm(*l.core));
        u32string text;
        for (size_t i = 0; i < each.size(); i++) text += (i && !strip_sp ? U" " : U"") + each[i];
        double conf = mean_conf(!lines.empty() ? lines : R.unsure);
        u32string pattern = ocr::strip(U(ps("expect_regex", "")));
        u32string exact = norm(ocr::strip(U(ps("expect_text", ""))));
        std::optional<u32string> matched;
        bool ok;
        std::string why;
        std::vector<u32string> cands{text};
        cands.insert(cands.end(), each.begin(), each.end());
        cands.insert(cands.end(), cores.begin(), cores.end());
        if (lines.empty()) {
            ok = false;
            why = "no text read";
            if (!R.unsure.empty()) why = unsure_note(R.unsure, min_conf());
        } else if (!exact.empty()) {
            bool la = pb("look_alike", true);
            for (auto& c : cands)
                if (la ? look_alike(c) == look_alike(exact) : c == exact) { matched = c; break; }
            ok = matched.has_value();
            why = "read '" + S(head(text, 80)) + "', expected '" + S(exact) + "'";
        } else if (!pattern.empty()) {
            std::regex rx;
            try {
                rx = std::regex(S(pattern), std::regex::ECMAScript);
            } catch (const std::regex_error& e) {
                return reader_error(ctx, std::string("the expected pattern is not valid: ") + e.what(), t0);
            }
            for (auto& c : cands)
                if (std::regex_match(S(c), rx)) { matched = c; break; }
            ok = matched.has_value();
            why = "read '" + S(head(text, 80)) + "' does not match /" + S(pattern) + "/";
        } else {
            ok = true;
        }
        std::string st = ok ? "pass" : "fail";
        u32string shown = matched ? *matched : text;
        std::string label = S(head(shown, 28));
        auto g = roi_graphic(ctx, st, label.empty() ? "no read" : label);
        for (auto& x : line_graphics(lines, R.ox, R.oy, st)) g.push_back(x);
        Json lj = Json::array();
        for (auto& l : lines) lj.push_back(S(l.text));
        Json vals = {{"text", S(shown)}, {"lines", lj}, {"confidence", py_round(conf, 3)}, {"engine", R.engine}, {"skew_deg", 0.0},
                     {"quarter_turn", common_turn(!lines.empty() ? lines : R.unsure)}, {"expected", S(!exact.empty() ? exact : pattern)}};
        return ToolResult(id, type_name, st, conf, vals, g, ok ? "" : why, now_ms() - t0);
    }
};
VC_REGISTER("ocv", Ocv)

}  // namespace vc
