// vc-supervisor: keeps vc-engine running. Starts it, pings it every two
// seconds, and restarts it when it has exited, stops answering (10 s), or has
// been busy on one job longer than the limit (a hung check). Each event is
// written to logs/supervisor.log beside the program.
//   vc-supervisor [--studio] [--busy-limit-s 120]
#include "../core/client.hpp"
#include "../core/paths.hpp"

#include <windows.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <thread>

using namespace vc;
namespace fs = std::filesystem;

static void log(const std::string& m) {
    fs::path d = paths::exe_dir() / "logs";
    std::error_code ec;
    fs::create_directories(d, ec);
    std::ofstream f(d / "supervisor.log", std::ios::app);
    std::time_t t = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    f << ts << "  " << m << "\n";
}

int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    long long busy_limit_ms = 120000;
    bool studio = false;
    for (int i = 1; i < argc; i++) {
        std::wstring a = argv[i];
        if (a == L"--studio") studio = true;
        if (a == L"--busy-limit-s" && i + 1 < argc) busy_limit_ms = _wtoll(argv[++i]) * 1000;
    }
    // one supervisor per PC
    HANDLE mutex = CreateMutexW(nullptr, TRUE, L"Local\\vision-core-supervisor");
    if (GetLastError() == ERROR_ALREADY_EXISTS) return 0;
    if (studio) {
        std::wstring exe = (paths::exe_dir() / "vc-studio.exe").wstring();
        STARTUPINFOW si{};
        si.cb = sizeof si;
        PROCESS_INFORMATION pi{};
        std::wstring cmd = L"\"" + exe + L"\"";
        if (CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
        }
    }
    log("supervisor started");
    HANDLE engine = nullptr;
    int restarts = 0;
    EngineClient probe;
    int missed = 0;
    for (;;) {
        if (!engine || WaitForSingleObject(engine, 0) == WAIT_OBJECT_0) {
            if (engine) {
                DWORD code = 0;
                GetExitCodeProcess(engine, &code);
                char b[96];
                std::snprintf(b, sizeof b, "engine stopped (exit code 0x%08lX) - restarting", code);
                log(b);
                CloseHandle(engine);
                engine = nullptr;
                restarts++;
                std::this_thread::sleep_for(std::chrono::milliseconds(std::min(10000, 500 * restarts)));
            }
            void* h = nullptr;
            if (!start_engine_process(&h)) {
                log("cannot start vc-engine.exe");
                std::this_thread::sleep_for(std::chrono::seconds(5));
                continue;
            }
            engine = (HANDLE)h;
            probe.close();
            missed = 0;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        Json r = probe.call({{"cmd", "ping"}}, 3000);
        if (r.value("ok", false)) {
            missed = 0;
            if (r.value("busy", false) && r.value("busy_ms", 0LL) > busy_limit_ms) {
                log("engine busy on one job for " + std::to_string(r.value("busy_ms", 0LL) / 1000) + " s - restarting it");
                TerminateProcess(engine, 0xDEAD);
            }
        } else if (++missed >= 4) {
            log("engine not answering (" + r.value("error", std::string()) + ") - restarting it");
            TerminateProcess(engine, 0xDEAD);
            missed = 0;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    (void)mutex;
}
