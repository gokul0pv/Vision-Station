// "Vision Core.exe": the program to double-click. No console window.
//   Not running yet  -> starts the supervisor, which starts the engine and the Studio.
//   Already running  -> brings the Studio window to the front, or opens the
//                       Studio again if it was closed (the engine keeps running).
#include <windows.h>

#include <string>

static std::wstring exe_dir() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring p(buf, n);
    return p.substr(0, p.find_last_of(L"\\/"));
}

static bool start(const std::wstring& dir, const std::wstring& exe, const std::wstring& args, DWORD flags) {
    std::wstring path = dir + L"\\" + exe;
    if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) {
        MessageBoxW(nullptr, (exe + L" is missing from\n" + dir + L"\n\nReinstall Vision Core (unzip the whole folder).").c_str(),
                    L"Vision Core", MB_ICONERROR | MB_OK);
        return false;
    }
    std::wstring cmd = L"\"" + path + L"\"" + (args.empty() ? L"" : L" " + args);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(path.c_str(), cmd.data(), nullptr, nullptr, FALSE, flags, nullptr, dir.c_str(), &si, &pi)) {
        MessageBoxW(nullptr, (L"Could not start " + exe + L" (error " + std::to_wstring(GetLastError()) + L").").c_str(), L"Vision Core",
                    MB_ICONERROR | MB_OK);
        return false;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return true;
}

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    std::wstring dir = exe_dir();
    // the supervisor holds this while it runs (src/apps/vc_supervisor.cpp)
    HANDLE m = OpenMutexW(SYNCHRONIZE, FALSE, L"Local\\vision-core-supervisor");
    if (!m) return start(dir, L"vc-supervisor.exe", L"--studio", CREATE_NO_WINDOW) ? 0 : 1;
    CloseHandle(m);
    if (HWND w = FindWindowW(nullptr, L"Vision Core Studio")) {
        if (IsIconic(w)) ShowWindow(w, SW_RESTORE);
        SetForegroundWindow(w);
        return 0;
    }
    return start(dir, L"vc-studio.exe", L"", 0) ? 0 : 1;
}
