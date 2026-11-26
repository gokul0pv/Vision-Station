// Vision Core Studio: the window. It never inspects anything itself -- it
// asks vc-engine (a separate process) and draws what comes back, so a
// fault in the window cannot stop inspection, and a fault in the engine
// cannot take the window down.
#include "../core/client.hpp"
#include "../core/paths.hpp"
#include "studio.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <windows.h>
#include <shobjidl.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;
using namespace vc;

namespace {

// a path as UTF-8 text, whatever the C++ standard (Slint builds this file as C++20)
template <class P>
std::string u8s(const P& p) {
    auto s = p.u8string();
    return std::string(s.begin(), s.end());
}

std::mutex g_eng_mx;
EngineClient g_eng;
std::atomic<bool> g_alive{true};

Json engine_call(const Json& req, int timeout_ms) {
    std::lock_guard<std::mutex> lk(g_eng_mx);
    Json r = g_eng.call(req, timeout_ms);
    if (!r.value("ok", false) && !g_eng.connected()) {
        // the engine went away: one more try after it is restarted
        if (g_eng.connect(1500)) r = g_eng.call(req, timeout_ms);
    }
    return r;
}

std::string pick_folder(HWND owner, const wchar_t* title) {
    std::string out;
    IFileOpenDialog* dlg = nullptr;
    if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dlg)))) {
        DWORD opt = 0;
        dlg->GetOptions(&opt);
        dlg->SetOptions(opt | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
        dlg->SetTitle(title);
        if (SUCCEEDED(dlg->Show(owner))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dlg->GetResult(&item))) {
                PWSTR p = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &p))) {
                    out = u8s(fs::path(p));
                    CoTaskMemFree(p);
                }
                item->Release();
            }
        }
        dlg->Release();
    }
    return out;
}

cv::Mat read_any(const std::string& path) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (d.empty()) return {};
    return cv::imdecode(d, cv::IMREAD_COLOR);
}

// a screen-sized copy of the picture for display; the overlays stay in the
// picture's own pixels (the viewbox maps them)
slint::Image preview(const cv::Mat& bgr) {
    cv::Mat s = bgr;
    int long_side = std::max(bgr.cols, bgr.rows);
    if (long_side > 2200) {
        double f = 2200.0 / long_side;
        cv::resize(bgr, s, cv::Size(), f, f, cv::INTER_AREA);
    }
    cv::Mat rgba;
    cv::cvtColor(s, rgba, cv::COLOR_BGR2RGBA);
    slint::SharedPixelBuffer<slint::Rgba8Pixel> buf(rgba.cols, rgba.rows);
    std::memcpy(buf.begin(), rgba.data, (size_t)rgba.total() * 4);
    return slint::Image(buf);
}

std::string fmt(double v, int d) {
    char b[64];
    std::snprintf(b, sizeof b, "%.*f", d, v);
    return b;
}

std::string short_json(const Json& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_null()) return "\xe2\x80\x94";
    if (v.is_boolean()) return v.get<bool>() ? "yes" : "no";
    if (v.is_number_float()) return fmt(v.get<double>(), 4);
    if (v.is_number()) return std::to_string(v.get<long long>());
    std::string s = v.dump();
    if (s.size() > 160) s = s.substr(0, 157) + "...";
    return s;
}

std::string path_commands(const Json& g) {
    std::string kind = g.value("kind", "");
    const Json& pts = g["points"];
    std::string c;
    auto P = [&](double x, double y, char op) { c += op; c += ' ' + fmt(x, 1) + ' ' + fmt(y, 1) + ' '; };
    if (kind == "rect" && pts.size() == 2) {
        double x0 = pts[0][0], y0 = pts[0][1], x1 = pts[1][0], y1 = pts[1][1];
        P(x0, y0, 'M'); P(x1, y0, 'L'); P(x1, y1, 'L'); P(x0, y1, 'L');
        c += "Z";
    } else if ((kind == "poly" || kind == "line") && !pts.empty()) {
        for (size_t i = 0; i < pts.size(); i++) P(pts[i][0], pts[i][1], i ? 'L' : 'M');
        if (kind == "poly") c += "Z";
    } else if (kind == "point" && !pts.empty()) {
        double x = pts[0][0], y = pts[0][1], r = 6;
        P(x - r, y, 'M'); P(x + r, y, 'L'); P(x, y - r, 'M'); P(x, y + r, 'L');
    }
    return c;
}

CheckItem mk_check(const std::string& id, const std::string& kind, const std::string& status, const std::string& score, const std::string& msg) {
    CheckItem c;
    c.id = id.c_str(); c.kind = kind.c_str(); c.status = status.c_str(); c.score = score.c_str(); c.message = msg.c_str();
    return c;
}
Overlay mk_overlay(const std::string& cmds, const std::string& status, bool region, const std::string& label) {
    Overlay o;
    o.commands = cmds.c_str(); o.status = status.c_str(); o.region = region; o.label = label.c_str(); o.lx = 0; o.ly = 0;
    return o;
}
Pic mk_pic(const std::string& name, const std::string& path, const std::string& verdict) {
    Pic p;
    p.name = name.c_str(); p.path = path.c_str(); p.verdict = verdict.c_str();
    return p;
}
KV mk_kv(const std::string& k, const std::string& v) {
    KV x;
    x.key = k.c_str(); x.value = v.c_str();
    return x;
}

struct State {
    std::string side;
    std::vector<std::string> pics;
    Json result;                   // the last inspection
    int pic = -1;
    std::vector<std::string> verdicts;
};

}  // namespace

int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    auto ui = StudioWindow::create();
    auto st = std::make_shared<State>();
    slint::ComponentWeakHandle<StudioWindow> weak(ui);

    auto on_ui = [weak](std::function<void(StudioWindow&)> f) {
        slint::invoke_from_event_loop([weak, f] {
            if (auto h = weak.lock()) f(**h);
        });
    };

    // the engine: connect, or start it; then keep an eye on it
    std::thread([on_ui] {
        bool started = false;
        while (g_alive) {
            Json r = engine_call({{"cmd", "ping"}}, 3000);
            bool ok = r.value("ok", false);
            if (!ok && !started) {
                start_engine_process();
                started = true;
            }
            std::string status = ok ? "ready" : (started ? "starting" : "offline");
            on_ui([status](StudioWindow& w) {
                if (w.get_engine_status() != slint::SharedString(status)) w.set_engine_status(status.c_str());
            });
            for (int i = 0; i < 20 && g_alive; i++) Sleep(100);
        }
    }).detach();

    auto set_pictures = [st](StudioWindow& w) {
        std::vector<Pic> v;
        for (size_t i = 0; i < st->pics.size(); i++)
            v.push_back(mk_pic(u8s(fs::u8path(st->pics[i]).filename()), st->pics[i], st->verdicts[i]));
        w.set_pictures(std::make_shared<slint::VectorModel<Pic>>(v));
    };

    auto show_result = [st, set_pictures](StudioWindow& w, const Json& res) {
        st->result = res;
        std::vector<CheckItem> checks;
        std::vector<Overlay> ovs;
        int npass = 0, nfail = 0;
        for (auto& t : res["tools"]) {
            std::string s = t.value("status", "");
            (s == "pass" ? npass : nfail) += (s == "pass" || s == "fail" || s == "error") ? 1 : 0;
            checks.push_back(mk_check(t.value("tool_id", ""), t.value("type", ""), s, fmt(t.value("score", 0.0), 2), t.value("message", "")));
            for (auto& g : t["graphics"]) {
                std::string c = path_commands(g);
                if (c.empty()) continue;
                ovs.push_back(mk_overlay(c, g.value("status", s), g.value("role", "") == "region", g.value("label", "")));
            }
        }
        w.set_checks(std::make_shared<slint::VectorModel<CheckItem>>(checks));
        w.set_overlays(std::make_shared<slint::VectorModel<Overlay>>(ovs));
        std::string verdict = res.value("verdict", "");
        w.set_verdict(verdict.c_str());
        w.set_total_ms((fmt(res.value("total_ms", 0.0) / 1000.0, 2) + " s").c_str());
        w.set_counts((std::to_string(npass) + " pass \xc2\xb7 " + std::to_string(nfail) + " fail").c_str());
        if (st->pic >= 0 && st->pic < (int)st->verdicts.size()) st->verdicts[st->pic] = verdict;
        set_pictures(w);
        w.set_selected_check(-1);
        w.set_detail_title("");
        std::string ab = res.value("aborted", Json()).is_string() ? res["aborted"].get<std::string>() : "";
        w.set_notice(ab.c_str());
    };

    auto run_now = [st, on_ui, show_result](StudioWindow& w) {
        if (st->side.empty() || st->pic < 0) return;
        w.set_busy(true);
        std::string pic = st->pics[st->pic];
        std::thread([pic, on_ui, show_result] {
            Json r = engine_call({{"cmd", "inspect"}, {"picture", pic}}, 600000);
            on_ui([r, show_result](StudioWindow& w) {
                w.set_busy(false);
                if (r.value("ok", false)) show_result(w, r["result"]);
                else w.set_notice(("Inspection failed: " + r.value("error", std::string("unknown"))).c_str());
            });
        }).detach();
    };

    auto open_job = [&, st, on_ui, run_now](std::string side) {
        if (side.empty()) return;
        ui->set_notice("Opening the job...");
        std::thread([side, st, on_ui, run_now] {
            Json r = engine_call({{"cmd", "open"}, {"side", side}}, 120000);
            on_ui([r, side, st, run_now](StudioWindow& w) {
                if (!r.value("ok", false)) {
                    w.set_notice(r.value("error", std::string("cannot open the job")).c_str());
                    return;
                }
                st->side = side;
                w.set_job_name(r.value("name", u8s(fs::u8path(side).filename())).c_str());
                std::vector<CheckItem> checks;
                for (auto& t : r["tools"]) checks.push_back(mk_check(t.value("id", ""), t.value("type", ""), "", "", ""));
                w.set_checks(std::make_shared<slint::VectorModel<CheckItem>>(checks));
                w.set_counts((std::to_string(checks.size()) + " checks").c_str());
                w.set_verdict("");
                w.set_overlays(std::make_shared<slint::VectorModel<Overlay>>());
                w.set_notice("");
                if (st->pic >= 0) run_now(w);    // a picture is already up: inspect it
            });
        }).detach();
    };
    ui->on_open_job([&, open_job] { open_job(pick_folder(GetActiveWindow(), L"Open a job side (the folder holding project.json)")); });

    auto load_pic = [st, on_ui, run_now](int i) {
        if (i < 0 || i >= (int)st->pics.size()) return;
        st->pic = i;
        std::string path = st->pics[i];
        std::thread([path, i, st, on_ui, run_now] {
            cv::Mat m = read_any(path);
            on_ui([m, i, st, run_now](StudioWindow& w) {
                if (m.empty()) {
                    w.set_notice("Cannot read that picture");
                    return;
                }
                w.set_picture(preview(m));
                w.set_img_w((float)m.cols);
                w.set_img_h((float)m.rows);
                w.set_overlays(std::make_shared<slint::VectorModel<Overlay>>());
                w.set_verdict("");
                w.set_current_pic(i);
                if (!st->side.empty()) run_now(w);
            });
        }).detach();
    };

    auto open_pictures = [&, st, load_pic, set_pictures](std::string dir) {
        if (dir.empty()) return;
        st->pics.clear();
        for (auto& e : fs::directory_iterator(fs::u8path(dir))) {
            std::string ext = u8s(e.path().extension());
            for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff") st->pics.push_back(u8s(e.path()));
        }
        std::sort(st->pics.begin(), st->pics.end());
        st->verdicts.assign(st->pics.size(), "");
        ui->set_folder_name((u8s(fs::u8path(dir).filename()) + "  \xc2\xb7  " + std::to_string(st->pics.size())).c_str());
        set_pictures(*ui);
        load_pic(st->pics.empty() ? -1 : 0);
    };
    ui->on_open_pictures([&, open_pictures] { open_pictures(pick_folder(GetActiveWindow(), L"Open a folder of pictures")); });

    // vc-studio [--job <side folder>] [--pictures <folder>]: open them at start
    std::string arg_job, arg_pics;
    for (int i = 1; i + 1 < __argc; i++) {
        std::string a = __argv[i];
        if (a == "--job") arg_job = __argv[++i];
        else if (a == "--pictures") arg_pics = __argv[++i];
    }
    if (!arg_job.empty()) {
        std::thread([arg_job, arg_pics, on_ui, open_job, open_pictures] {
            for (int i = 0; i < 100 && !engine_call({{"cmd", "ping"}}, 1000).value("ok", false); i++) Sleep(300);
            on_ui([arg_job, arg_pics, open_job, open_pictures](StudioWindow&) {
                open_job(arg_job);
                if (!arg_pics.empty()) open_pictures(arg_pics);
            });
        }).detach();
    }

    ui->on_pick_picture([load_pic](int i) { load_pic(i); });
    ui->on_run([&, run_now] { run_now(*ui); });

    ui->on_pick_check([&, st](int i) {
        const Json& tools = st->result.contains("tools") ? st->result["tools"] : Json::array();
        if (i < 0 || i >= (int)tools.size()) return;
        const Json& t = tools[i];
        ui->set_detail_title(t.value("tool_id", "").c_str());
        ui->set_detail_status(t.value("status", "").c_str());
        ui->set_detail_message(t.value("message", "").c_str());
        std::vector<KV> kv{mk_kv("type", t.value("type", "")), mk_kv("score", fmt(t.value("score", 0.0), 4)), mk_kv("time", fmt(t.value("ms", 0.0), 1) + " ms")};
        for (auto& [k, v] : t["values"].items()) kv.push_back(mk_kv(k, short_json(v)));
        ui->set_details(std::make_shared<slint::VectorModel<KV>>(kv));
    });

    ui->run();
    g_alive = false;
    return 0;
}
