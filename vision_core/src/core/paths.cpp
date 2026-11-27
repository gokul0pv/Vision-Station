#include "paths.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace vc::paths {

std::filesystem::path exe_dir() {
#ifdef _WIN32
    wchar_t buf[32768];
    DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD)(sizeof buf / sizeof buf[0]));
    return std::filesystem::path(std::wstring(buf, n)).parent_path();
#else
    return std::filesystem::current_path();
#endif
}

std::filesystem::path find_dir(const std::string& rel) {
    auto d = exe_dir();
    for (int i = 0; i < 6; i++) {
        auto p = d / std::filesystem::u8path(rel);
        std::error_code ec;
        if (std::filesystem::is_directory(p, ec)) return p;
        if (!d.has_parent_path() || d.parent_path() == d) break;
        d = d.parent_path();
    }
    return {};
}

}  // namespace vc::paths
