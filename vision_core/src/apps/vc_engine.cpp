// vc-engine: Vision Core's inspection engine as a headless process.
// Clients (Studio, the supervisor, a line PC's tools) talk to it over the
// named pipe \\.\pipe\vision-core-engine, one JSON object per line each way:
//
//   {"cmd":"ping"}                         -> {"ok":true,"busy":false,"pid":...}
//   {"cmd":"types"}                        -> {"ok":true,"types":[...]}
//   {"cmd":"open","side":"<folder>"}       -> {"ok":true,"side":...,"tools":[...],"reference":"<path>","size":[w,h]}
//   {"cmd":"inspect","picture":"<path>"}   -> {"ok":true,"result":{...inspection...}}
//   {"cmd":"quit"}
//
// Every request is answered, failures included ({"ok":false,"error":...}).
// Pings are answered while an inspection runs, so a watcher can tell a
// working engine from a hung one.
#include "../core/engine.hpp"
#include "../core/results.hpp"
#include "../hw/camera.hpp"
#include "../hw/light.hpp"
#include "../hw/line.hpp"

#include <opencv2/imgcodecs.hpp>

#include <windows.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;
using namespace vc;

namespace vc { void release_pictures(); }

namespace {

const wchar_t* PIPE = L"\\\\.\\pipe\\vision-core-engine";
std::mutex g_run;                       // one inspection at a time
std::atomic<bool> g_busy{false};
std::atomic<long long> g_busy_since{0};   // GetTickCount64 when the current job started
std::atomic<bool> g_quit{false};
std::unique_ptr<Project> g_side;
std::unique_ptr<hw::Camera> g_cam;
std::unique_ptr<ResultStore> g_store;   // the open job's run log
std::mutex g_last_mx;
Json g_last;                             // the last inspection, for GET on the line
std::string g_code;                      // the barcode scanned for the next part
Json handle(const Json& req);
std::string g_side_path;

Json tool_list(const Project& p) {
    Json a = Json::array();
    for (auto& t : p.tools) {
        Json j = {{"id", t->id}, {"type", t->type_name}, {"enabled", t->enabled}, {"shape", t->shape}, {"angle", t->angle}};
        if (t->roi) j["roi"] = {(*t->roi)[0], (*t->roi)[1], (*t->roi)[2], (*t->roi)[3]};
        if (!t->points.empty()) {
            Json pts = Json::array();
            for (auto& q : t->points) pts.push_back({q[0], q[1]});
            j["points"] = pts;
        }
        a.push_back(j);
    }
    return a;
}

Json handle(const Json& req) {
    std::string cmd = req.value("cmd", "");
    if (cmd == "ping") {
        long long busy_ms = g_busy ? (long long)GetTickCount64() - g_busy_since.load() : 0;
        return {{"ok", true}, {"busy", g_busy.load()}, {"busy_ms", busy_ms}, {"pid", (int)GetCurrentProcessId()}, {"side", g_side_path},
                {"logged", g_store ? g_store->written() : 0}, {"log_error", g_store ? g_store->last_error() : std::string()}};
    }
    if (cmd == "types") {
        Json t = Json::array();
        for (auto& [k, v] : registry()) t.push_back(k);
        return {{"ok", true}, {"types", t}};
    }
    if (cmd == "quit") {
        g_quit = true;
        return {{"ok", true}};
    }
    std::lock_guard<std::mutex> lk(g_run);
    g_busy_since = (long long)GetTickCount64();
    g_busy = true;
    struct Done { ~Done() { g_busy = false; } } done;
    if (cmd == "open") {
        std::string side = req.value("side", "");
        std::unique_ptr<Project> p;
        std::string err;
        if (!guarded([&] { p = std::make_unique<Project>(side); }, err)) return {{"ok", false}, {"error", "cannot open " + side + ": " + err}};
        g_side = std::move(p);
        g_side_path = side;
        g_store.reset();
        if (req.value("log", true))
            g_store = std::make_unique<ResultStore>(g_side->folder, g_side->id, req.value("archive", std::string("none")));
        std::string ref = g_side->spec.value("reference_image", "reference.png");
        fs::path rp = fs::u8path(ref).is_absolute() ? fs::u8path(ref) : fs::u8path(g_side->folder) / fs::u8path(ref);
        return {{"ok", true}, {"side", g_side->folder}, {"name", g_side->spec.value("name", g_side->id)}, {"tools", tool_list(*g_side)},
                {"reference", rp.u8string()}, {"size", {g_side->reference.cols, g_side->reference.rows}}};
    }
    if (cmd == "inspect") {
        if (!g_side) return {{"ok", false}, {"error", "no job open"}};
        std::string pic = req.value("picture", "");
        cv::Mat img = imread(pic);
        if (img.empty()) return {{"ok", false}, {"error", "cannot read the picture " + pic}};
        Json out;
        std::string err;
        bool ok = guarded([&] {
            Inspection r = g_side->inspect(img, pic, req.value("barcode", ""));
            release_pictures();
            out = r.to_json();
        }, err);
        if (!ok) return {{"ok", false}, {"error", "inspection failed: " + err}};
        { std::lock_guard<std::mutex> l2(g_last_mx); g_last = out; }
        if (g_store) g_store->write(out, img);
        return {{"ok", true}, {"result", out}};
    }
    if (cmd == "camera_open") {
        // {"address": "192.168.3.6", "exposure_us": 0, "gain": -1, "rotate": 0}
        try {
            if (!g_cam) g_cam = std::make_unique<hw::Camera>();
            g_cam->open(req.value("address", ""), req.value("index", 0));
            hw::CameraSettings cs;
            cs.exposure_us = req.value("exposure_us", 0.0);
            cs.gain = req.value("gain", -1.0);
            cs.rotate = req.value("rotate", 0);
            g_cam->apply(cs);
            g_cam->start();
            return {{"ok", true}, {"model", g_cam->model()}};
        } catch (const std::exception& e) {
            return {{"ok", false}, {"error", e.what()}};
        }
    }
    if (cmd == "light") {
        // {"action": "on|off|set|read|frame|ports", "port": "COM7", "channel": 1, "value": 128}
        std::string a = req.value("action", "read");
        int ch = req.value("channel", 1);
        try {
            if (a == "ports") return {{"ok", true}, {"ports", hw::serial_ports()}};
            if (a == "frame") return {{"ok", true}, {"frame", hw::light_frame(req.value("light_cmd", std::string("3"))[0], ch, req.value("value", 0))}};
            hw::Light l(req.value("port", ""));
            if (a == "on") l.on(ch);
            else if (a == "off") l.off(ch);
            else if (a == "set") l.set_brightness(ch, req.value("value", 0));
            else if (a == "read") return {{"ok", true}, {"level", l.brightness(ch)}};
            else return {{"ok", false}, {"error", "unknown light action " + a}};
            return {{"ok", true}};
        } catch (const std::exception& e) {
            return {{"ok", false}, {"error", e.what()}};
        }
    }
    if (cmd == "line_start") {
        // {"port": 3000, "barcode": {"prefix": "", "length": 0, "match_case": false}}  (barcode: optional)
        hw::LineHooks h;
        h.busy = [] { return g_busy.load(); };
        h.job_name = [] { return g_side ? g_side->id : std::string(); };
        h.last = [] { std::lock_guard<std::mutex> l(g_last_mx); return g_last; };
        h.trigger = [](const std::string&, std::string& err) -> Json {
            Json r = handle({{"cmd", "snap"}, {"barcode", g_code}});
            g_code.clear();
            if (!r.value("ok", false)) { err = r.value("error", std::string("no frame")); return Json(); }
            if (!r.contains("result")) { err = r.value("inspect_error", std::string("no job open")); return Json(); }
            return r["result"];
        };
        if (req.contains("barcode") && req["barcode"].is_object()) {
            Json b = req["barcode"];
            std::string prefix = b.value("prefix", "");
            int length = b.value("length", 0);
            bool mc = b.value("match_case", false);
            h.code = [prefix, length, mc](const std::string& raw) {
                auto r = hw::barcode_check(raw, prefix, length, mc);
                if (r.first) g_code = r.second;
                return r;
            };
        }
        std::string err;
        if (!hw::start_line_server(req.value("port", 3000), h, err)) return {{"ok", false}, {"error", err}};
        return {{"ok", true}};
    }
    if (cmd == "camera_close") {
        if (g_cam) g_cam->close();
        return {{"ok", true}};
    }
    if (cmd == "snap") {
        // one fresh frame: saved (PNG) where asked or beside the program, and inspected when a job is open
        if (!g_cam || !g_cam->is_open()) return {{"ok", false}, {"error", "no camera open"}};
        cv::Mat img;
        std::string why;
        try {
            for (int i = 0; i < 3 && img.empty(); i++) img = g_cam->grab(1500, &why);   // the first may be a frame in flight
            img = g_cam->grab(3000, &why);
        } catch (const std::exception& e) {
            return {{"ok", false}, {"error", e.what()}};
        }
        if (img.empty()) return {{"ok", false}, {"error", "no frame from the camera" + (why.empty() ? std::string() : ": " + why)}};
        std::string out = req.value("save", "");
        if (out.empty()) out = (fs::temp_directory_path() / "vision-core-snap.png").u8string();
        std::vector<uint8_t> buf;
        cv::imencode(".png", img, buf);
        { std::ofstream f(fs::u8path(out), std::ios::binary); f.write((const char*)buf.data(), buf.size()); }
        Json resp = {{"ok", true}, {"picture", out}};
        if (g_side) {
            std::string err;
            Json res;
            if (guarded([&] { res = g_side->inspect(img, out, req.value("barcode", "")).to_json(); release_pictures(); }, err)) {
                resp["result"] = res;
                if (g_store) g_store->write(res, img);
                std::lock_guard<std::mutex> l2(g_last_mx);
                g_last = res;
            }
            else resp["inspect_error"] = err;
        }
        return resp;
    }
    return {{"ok", false}, {"error", "unknown command '" + cmd + "'"}};
}

void serve(HANDLE pipe) {
    std::string buf;
    char chunk[65536];
    for (;;) {
        DWORD n = 0;
        if (!ReadFile(pipe, chunk, sizeof chunk, &n, nullptr) || n == 0) break;
        buf.append(chunk, n);
        size_t nl;
        while ((nl = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, nl);
            buf.erase(0, nl + 1);
            if (line.empty()) continue;
            Json resp;
            try {
                resp = handle(Json::parse(line));
            } catch (const std::exception& e) {
                resp = {{"ok", false}, {"error", std::string("bad request: ") + e.what()}};
            }
            std::string s = resp.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n";
            DWORD w = 0;
            if (!WriteFile(pipe, s.data(), (DWORD)s.size(), &w, nullptr)) goto done;
        }
        if (g_quit) {
            FlushFileBuffers(pipe);
            ExitProcess(0);
        }
    }
done:
    FlushFileBuffers(pipe);
    DisconnectNamedPipe(pipe);
    CloseHandle(pipe);
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    // a fault in the engine is reported by the supervisor; no error boxes on a line PC
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    if (argc > 1) {
        std::string side = fs::path(argv[1]).u8string();
        handle({{"cmd", "open"}, {"side", side}});
    }
    std::cerr << "vc-engine listening on \\\\.\\pipe\\vision-core-engine\n";
    while (!g_quit) {
        HANDLE pipe = CreateNamedPipeW(PIPE, PIPE_ACCESS_DUPLEX, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                                       PIPE_UNLIMITED_INSTANCES, 1 << 20, 1 << 20, 0, nullptr);
        if (pipe == INVALID_HANDLE_VALUE) {
            std::cerr << "cannot create the pipe (" << GetLastError() << ")\n";
            return 1;
        }
        BOOL ok = ConnectNamedPipe(pipe, nullptr) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
        if (!ok) {
            CloseHandle(pipe);
            continue;
        }
        std::thread(serve, pipe).detach();
    }
    return 0;
}
