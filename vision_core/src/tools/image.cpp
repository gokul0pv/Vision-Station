// Image-making tools: vision_station/tools/imageproc.py (image_process) and
// vision_station/tools/transform.py (affine_transform, polar_unwrap, image_combine).
// Each output goes into ctx.images[id] for later tools that read this id.
#include "../core/npy.hpp"
#include "../core/pyfmt.hpp"
#include "../core/tool.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>

namespace vc {

namespace {

int odd(int k) {
    k = std::max(1, k);
    return k % 2 ? k : k + 1;
}

cv::Mat gray_of(const cv::Mat& img) {
    if (img.channels() == 1) return img;
    cv::Mat g;
    cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    return g;
}

void record(Tool& t, Context& ctx, const std::string& name, const cv::Mat& img) {
    if (ctx.want_records && !img.empty()) ctx.records[t.id][name] = img;
}

// imageproc.apply: image in, image out, same size; grey results come back as 3 channels
cv::Mat apply_op(const cv::Mat& img, const std::string& op, const Tool& t) {
    int k = odd(t.pi("kernel", 5));
    cv::Mat out;
    if (op == "smooth") {
        cv::GaussianBlur(img, out, cv::Size(k, k), 0);
    } else if (op == "median") {
        cv::medianBlur(img, out, k);
    } else if (op == "sharpen") {
        cv::Mat blur;
        cv::GaussianBlur(img, blur, cv::Size(k, k), 0);
        double amt = t.pf("amount", 1.0);
        cv::addWeighted(img, 1.0 + amt, blur, -amt, 0, out);
    } else if (op == "equalize") {
        auto clahe = cv::createCLAHE(t.pf("amount", 2.0), cv::Size(8, 8));
        if (img.channels() == 3) {
            cv::Mat lab;
            cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
            std::vector<cv::Mat> ch;
            cv::split(lab, ch);
            clahe->apply(ch[0], ch[0]);
            cv::merge(ch, lab);
            cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
        } else {
            clahe->apply(img, out);
        }
    } else if (op == "threshold") {
        cv::Mat g = gray_of(img);
        std::string mode = t.ps("threshold_mode", "otsu");
        if (mode == "fixed")
            cv::threshold(g, out, t.pi("level", 128), 255, cv::THRESH_BINARY);
        else if (mode == "adaptive")
            cv::adaptiveThreshold(g, out, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                                  odd(std::max(3, k * 6)), -t.pf("level", 0) / 10.0);
        else
            cv::threshold(g, out, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    } else if (op == "erode" || op == "dilate" || op == "open" || op == "close") {
        cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        int it = std::max(1, t.pi("iterations", 1));
        if (op == "erode") cv::erode(img, out, se, cv::Point(-1, -1), it);
        else if (op == "dilate") cv::dilate(img, out, se, cv::Point(-1, -1), it);
        else cv::morphologyEx(img, out, op == "open" ? cv::MORPH_OPEN : cv::MORPH_CLOSE, se, cv::Point(-1, -1), it);
    } else if (op == "edges") {
        cv::Mat g, gx, gy, mag;
        gray_of(img).convertTo(g, CV_32F);
        cv::Sobel(g, gx, CV_32F, 1, 0, 3);
        cv::Sobel(g, gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, mag);
        cv::Mat flat = mag.isContinuous() ? mag : mag.clone();
        std::vector<float> v((const float*)flat.data, (const float*)flat.data + flat.total());
        double p = (double)npy::percentile_f32(std::move(v), 99.5);
        float scale = (float)(255.0 / std::max(1.0, p));     // Python float -> float32 multiply
        out.create(mag.size(), CV_8U);
        for (int y = 0; y < mag.rows; y++) {
            const float* s = mag.ptr<float>(y);
            uint8_t* d = out.ptr<uint8_t>(y);
            for (int x = 0; x < mag.cols; x++) {
                float f = std::min(255.0f, std::max(0.0f, s[x] * scale));
                d[x] = (uint8_t)f;                              // astype(uint8) truncates
            }
        }
    } else if (op == "invert") {
        cv::bitwise_not(img, out);
    } else if (op == "channel") {
        if (img.channels() == 1) {
            out = img;
        } else {
            std::string c = t.ps("channel", "green");
            int idx = c == "blue" ? 0 : c == "red" ? 2 : 1;
            cv::extractChannel(img, out, idx);
        }
    } else if (op == "gain") {
        float a = (float)t.pf("amount", 1.0), b = (float)t.pf("level", 0);
        out.create(img.size(), img.type());
        int n = img.cols * img.channels();
        for (int y = 0; y < img.rows; y++) {
            const uint8_t* s = img.ptr<uint8_t>(y);
            uint8_t* d = out.ptr<uint8_t>(y);
            for (int x = 0; x < n; x++) {
                float f = (float)s[x] * a + b;
                f = std::min(255.0f, std::max(0.0f, f));
                d[x] = std::isnan(f) ? 0 : (uint8_t)f;
            }
        }
    } else {
        throw std::invalid_argument("ValueError: unknown operator '" + op + "'");
    }
    if (out.channels() == 1) {
        cv::Mat c3;
        cv::cvtColor(out, c3, cv::COLOR_GRAY2BGR);
        out = c3;
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------- image_process
class ImageProcess : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        std::string op = ps("operator", "smooth");
        cv::Mat out;
        try {
            out = apply_op(image, op, *this);
        } catch (const std::invalid_argument& e) {
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, e.what(), now_ms() - t0);
        } catch (const cv::Exception& e) {
            return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, std::string("error: ") + e.err, now_ms() - t0);
        }
        ctx.images[id] = out;
        Json v = {{"operator", op}, {"size", std::to_string(out.cols) + "x" + std::to_string(out.rows)}};
        return ToolResult(id, type_name, "pass", 1.0, v, {}, "", now_ms() - t0);
    }
};
VC_REGISTER("image_process", ImageProcess)

// ---------------------------------------------------------------- transform.py helpers
namespace {

ToolResult out_result(Tool& t, Context& ctx, const cv::Mat& cur, const cv::Mat& ref, double t0, Json vals) {
    ctx.images[t.id] = cur;
    if (!ref.empty()) ctx.ref_images[t.id] = ref;
    record(t, ctx, "LastRun.OutputImage", cur);
    vals["width"] = cur.cols;
    vals["height"] = cur.rows;
    return ToolResult(t.id, t.type_name, "pass", 1.0, vals, t.roi_graphic(ctx, "pass", t.id), "", now_ms() - t0);
}

// fn evaluated on the reference image, unmoved: no fixture
template <class F>
cv::Mat on_reference(Context& ctx, F fn) {
    if (ctx.reference.empty()) return cv::Mat();
    cv::Mat saved = ctx.fixture;
    ctx.fixture = cv::Mat();
    cv::Mat r;
    try {
        r = fn(ctx.reference);
    } catch (...) {
        ctx.fixture = saved;
        throw;
    }
    ctx.fixture = saved;
    return r;
}

}  // namespace

// ---------------------------------------------------------------- affine_transform
class AffineTransform : public Tool {
public:
    using Tool::Tool;
    bool changes_geometry() const override { return true; }
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        double k = pf("scale", 1.0);
        if (k == 0.0) k = 1.0;
        auto make = [&](const cv::Mat& img) -> cv::Mat {
            auto [patch, o] = crop(img, ctx);
            (void)o;
            if (patch.empty()) return cv::Mat();
            if (k != 1.0) {
                cv::Mat r;
                cv::resize(patch, r, cv::Size(), k, k, k > 1 ? cv::INTER_CUBIC : cv::INTER_AREA);
                return r;
            }
            return patch;
        };
        cv::Mat cur = make(image);
        if (cur.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "empty region");
        cv::Mat ref = on_reference(ctx, make);
        return out_result(*this, ctx, cur, ref, t0, Json{{"scale", k}});
    }
};
VC_REGISTER("affine_transform", AffineTransform)

// ---------------------------------------------------------------- polar_unwrap
class PolarUnwrap : public Tool {
public:
    using Tool::Tool;
    bool changes_geometry() const override { return true; }

    cv::Mat unwrap(const cv::Mat& img, const Context& ctx) const {
        if (!roi) return cv::Mat();
        double x = (*roi)[0], y = (*roi)[1], w = (*roi)[2], h = (*roi)[3];
        double cx = x + w / 2.0, cy = y + h / 2.0;
        double start = pf("start_angle", 0.0);
        if (!ctx.fixture.empty()) {
            cv::Mat M;
            ctx.fixture.convertTo(M, CV_64F);           // float32 values, float64 arithmetic as numpy promotes
            auto m = [&](int r, int c) { return M.at<double>(r, c); };
            double nx = m(0, 0) * cx + m(0, 1) * cy + m(0, 2);
            double ny = m(1, 0) * cx + m(1, 1) * cy + m(1, 2);
            cx = nx;
            cy = ny;
            start += std::atan2(m(1, 0), m(0, 0)) * (180.0 / CV_PI);
        }
        double r_out = std::min(w, h) / 2.0;
        double r_in = r_out * pf("inner_radius", 0.5);
        int rows = std::max(4, (int)std::nearbyint(r_out - r_in));   // Python round: half to even
        int width = pi("width", 0);
        if (width == 0) width = std::max(16, (int)std::nearbyint(2 * CV_PI * (r_in + r_out) / 2));
        double sweep = pf("sweep", 360.0);
        double sign = ps("direction", "clockwise") == "counterclockwise" ? -1.0 : 1.0;
        std::vector<double> ang(width), rad(rows);
        for (int i = 0; i < width; i++) ang[i] = (start + sign * sweep * (double)i / (double)width) * (CV_PI / 180.0);
        double den = (double)std::max(1, rows - 1);
        for (int j = 0; j < rows; j++) rad[j] = r_out - (r_out - r_in) * (double)j / den;
        cv::Mat mx(rows, width, CV_32F), my(rows, width, CV_32F);
        for (int j = 0; j < rows; j++) {
            float* px = mx.ptr<float>(j);
            float* py = my.ptr<float>(j);
            for (int i = 0; i < width; i++) {
                px[i] = (float)(cx + rad[j] * std::cos(ang[i]));
                py[i] = (float)(cy + rad[j] * std::sin(ang[i]));
            }
        }
        cv::Mat out;
        cv::remap(img, out, mx, my, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        return out;
    }

    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        cv::Mat cur = unwrap(image, ctx);
        if (cur.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "no region");
        cv::Mat ref = on_reference(ctx, [&](const cv::Mat& img) { return unwrap(img, ctx); });
        return out_result(*this, ctx, cur, ref, t0, Json{{"inner_radius", pf("inner_radius", 0.5)}});
    }
};
VC_REGISTER("polar_unwrap", PolarUnwrap)

// ---------------------------------------------------------------- image_combine
class ImageCombine : public Tool {
public:
    using Tool::Tool;
    ToolResult run(const cv::Mat& image, Context& ctx) override {
        double t0 = now_ms();
        if (ctx.reference.empty()) return ToolResult(id, type_name, "error", 0.0, Json::object(), {}, "needs a reference image");
        cv::Mat ref = ctx.reference;
        if (!ctx.fixture.empty()) {                    // line the reference up with the part
            cv::Mat M;
            ctx.fixture.convertTo(M, CV_64F);
            cv::Mat w;
            cv::warpAffine(ref, w, M.rowRange(0, 2), image.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            ref = w;
        }
        if (ref.size() != image.size() || ref.channels() != image.channels()) {
            cv::Mat r;
            cv::resize(ref, r, image.size());
            ref = r;
        }
        std::string op = ps("operation", "difference");
        cv::Mat out;
        if (op == "subtract") cv::subtract(image, ref, out);
        else if (op == "subtract_reference_from") cv::subtract(ref, image, out);
        else if (op == "minimum") cv::min(image, ref, out);
        else if (op == "maximum") cv::max(image, ref, out);
        else if (op == "average") cv::addWeighted(image, 0.5, ref, 0.5, 0, out);
        else cv::absdiff(image, ref, out);
        ctx.images[id] = out;
        record(*this, ctx, "LastRun.OutputImage", out);
        cv::Mat d;
        cv::absdiff(image, ref, d);
        if (!d.isContinuous()) d = d.clone();
        double diff = npy::mean_u8(d.data, d.total() * d.channels());
        Json v = {{"operation", op}, {"mean_difference", py_round(diff, 2)}};
        return ToolResult(id, type_name, "pass", 1.0, v, {}, "", now_ms() - t0);
    }
};
VC_REGISTER("image_combine", ImageCombine)

}  // namespace vc
