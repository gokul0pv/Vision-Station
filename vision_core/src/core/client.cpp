#include "client.hpp"
#include "paths.hpp"

#include <windows.h>

namespace vc {

static const wchar_t* PIPE = L"\\\\.\\pipe\\vision-core-engine";

EngineClient::~EngineClient() { close(); }

void EngineClient::close() {
    if (pipe_) CloseHandle((HANDLE)pipe_);
    pipe_ = nullptr;
    buf_.clear();
}

bool EngineClient::connect(int timeout_ms) {
    close();
    ULONGLONG end = GetTickCount64() + (ULONGLONG)timeout_ms;
    for (;;) {
        HANDLE h = CreateFileW(PIPE, GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            pipe_ = h;
            return true;
        }
        DWORD e = GetLastError();
        if (GetTickCount64() >= end) return false;
        if (e == ERROR_PIPE_BUSY) WaitNamedPipeW(PIPE, 200);
        else Sleep(100);
    }
}

// one overlapped read or write, abandoned after the deadline
static bool io(HANDLE h, bool write, void* data, DWORD len, DWORD& done, ULONGLONG deadline) {
    OVERLAPPED ov{};
    ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    BOOL ok = write ? WriteFile(h, data, len, nullptr, &ov) : ReadFile(h, data, len, nullptr, &ov);
    if (!ok && GetLastError() != ERROR_IO_PENDING) {
        CloseHandle(ov.hEvent);
        return false;
    }
    ULONGLONG now = GetTickCount64();
    DWORD wait = deadline > now ? (DWORD)(deadline - now) : 0;
    if (WaitForSingleObject(ov.hEvent, wait) != WAIT_OBJECT_0) {
        CancelIoEx(h, &ov);
        GetOverlappedResult(h, &ov, &done, TRUE);
        CloseHandle(ov.hEvent);
        return false;
    }
    bool r = GetOverlappedResult(h, &ov, &done, FALSE) != 0;
    CloseHandle(ov.hEvent);
    return r && (done > 0 || write);
}

Json EngineClient::call(const Json& req, int timeout_ms) {
    if (!pipe_ && !connect(2000)) return {{"ok", false}, {"error", "the engine is not running"}};
    ULONGLONG deadline = GetTickCount64() + (ULONGLONG)timeout_ms;
    std::string s = req.dump() + "\n";
    DWORD done = 0;
    if (!io((HANDLE)pipe_, true, s.data(), (DWORD)s.size(), done, deadline)) {
        close();
        return {{"ok", false}, {"error", "the engine did not take the request"}};
    }
    char chunk[65536];
    for (;;) {
        size_t nl = buf_.find('\n');
        if (nl != std::string::npos) {
            std::string line = buf_.substr(0, nl);
            buf_.erase(0, nl + 1);
            try {
                return Json::parse(line);
            } catch (const std::exception& e) {
                return {{"ok", false}, {"error", std::string("bad reply: ") + e.what()}};
            }
        }
        if (!io((HANDLE)pipe_, false, chunk, sizeof chunk, done, deadline)) {
            close();
            return {{"ok", false}, {"error", "the engine did not answer in time"}};
        }
        buf_.append(chunk, done);
    }
}

bool start_engine_process(void** process_handle) {
    std::wstring exe = (paths::exe_dir() / "vc-engine.exe").wstring();
    std::wstring cmd = L"\"" + exe + L"\"";
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return false;
    CloseHandle(pi.hThread);
    if (process_handle) *process_handle = pi.hProcess;
    else CloseHandle(pi.hProcess);
    return true;
}

}  // namespace vc
