#include "codes.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <tuple>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace vc::codes {

namespace {

namespace fs = std::filesystem;

fs::path exe_dir() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return fs::path(std::wstring(buf, n)).parent_path();
}

HMODULE load_dll(const wchar_t* env, const wchar_t* name) {
    std::vector<fs::path> tries;
    if (const wchar_t* e = _wgetenv(env)) tries.emplace_back(e);
    tries.push_back(exe_dir() / name);
    tries.push_back(exe_dir() / "codes" / name);
    for (auto& p : tries) {
        std::error_code ec;
        if (!fs::exists(p, ec)) continue;
        // its own folder first, so libzbar finds the libiconv.dll beside it
        if (HMODULE h = LoadLibraryExW(p.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)) return h;
    }
    return nullptr;
}

// ---- zbar
struct Zbar {
    void* (*scanner_create)() = nullptr;
    void (*scanner_destroy)(void*) = nullptr;
    void* (*image_create)() = nullptr;
    void (*image_destroy)(void*) = nullptr;
    void (*set_format)(void*, unsigned long) = nullptr;
    void (*set_size)(void*, unsigned, unsigned) = nullptr;
    void (*set_data)(void*, const void*, unsigned long, void*) = nullptr;
    int (*scan)(void*, void*) = nullptr;
    const void* (*first)(const void*) = nullptr;
    const void* (*next)(const void*) = nullptr;
    const char* (*data)(const void*) = nullptr;
    unsigned (*length)(const void*) = nullptr;
    bool ok = false;
};

const Zbar& zbar() {
    static Zbar z;
    static std::once_flag once;
    std::call_once(once, [] {
        HMODULE h = load_dll(L"VC_ZBAR", L"libzbar-64.dll");
        if (!h) return;
        auto f = [&](const char* n) { return (void*)GetProcAddress(h, n); };
        z.scanner_create = (decltype(z.scanner_create))f("zbar_image_scanner_create");
        z.scanner_destroy = (decltype(z.scanner_destroy))f("zbar_image_scanner_destroy");
        z.image_create = (decltype(z.image_create))f("zbar_image_create");
        z.image_destroy = (decltype(z.image_destroy))f("zbar_image_destroy");
        z.set_format = (decltype(z.set_format))f("zbar_image_set_format");
        z.set_size = (decltype(z.set_size))f("zbar_image_set_size");
        z.set_data = (decltype(z.set_data))f("zbar_image_set_data");
        z.scan = (decltype(z.scan))f("zbar_scan_image");
        z.first = (decltype(z.first))f("zbar_image_first_symbol");
        z.next = (decltype(z.next))f("zbar_symbol_next");
        z.data = (decltype(z.data))f("zbar_symbol_get_data");
        z.length = (decltype(z.length))f("zbar_symbol_get_data_length");
        z.ok = z.scanner_create && z.scanner_destroy && z.image_create && z.image_destroy && z.set_format && z.set_size && z.set_data &&
               z.scan && z.first && z.next && z.data && z.length;
    });
    return z;
}

std::string zbar_name(int t) {
    static const std::map<int, const char*> names = {
        {0, "NONE"}, {1, "PARTIAL"}, {2, "EAN2"}, {5, "EAN5"}, {8, "EAN8"}, {9, "UPCE"}, {10, "ISBN10"}, {12, "UPCA"}, {13, "EAN13"},
        {14, "ISBN13"}, {15, "COMPOSITE"}, {25, "I25"}, {34, "DATABAR"}, {35, "DATABAR_EXP"}, {38, "CODABAR"}, {39, "CODE39"},
        {57, "PDF417"}, {64, "QRCODE"}, {80, "SQCODE"}, {93, "CODE93"}, {128, "CODE128"}};
    auto it = names.find(t);
    return it != names.end() ? it->second : "Unrecognised type [" + std::to_string(t) + "]";
}

// ---- libdmtx
struct DmtxTime {
    unsigned long long sec;
    unsigned long usec;
};
struct Dmtx {
    void* (*image_create)(unsigned char*, int, int, int) = nullptr;
    int (*image_destroy)(void**) = nullptr;
    void* (*decode_create)(void*, int) = nullptr;
    int (*decode_destroy)(void**) = nullptr;
    void* (*region_find_next)(void*, DmtxTime*) = nullptr;
    int (*region_destroy)(void**) = nullptr;
    void* (*decode_matrix_region)(void*, void*, int) = nullptr;
    int (*message_destroy)(void**) = nullptr;
    DmtxTime (*time_now)() = nullptr;
    DmtxTime (*time_add)(DmtxTime, long) = nullptr;
    const char* (*version)() = nullptr;
    bool fnc1 = true;       // libdmtx >= 0.7.5 has the fnc1 field in DmtxMessage
    bool ok = false;
};

const Dmtx& dmtx() {
    static Dmtx d;
    static std::once_flag once;
    std::call_once(once, [] {
        HMODULE h = load_dll(L"VC_DMTX", L"libdmtx-64.dll");
        if (!h) return;
        auto f = [&](const char* n) { return (void*)GetProcAddress(h, n); };
        d.image_create = (decltype(d.image_create))f("dmtxImageCreate");
        d.image_destroy = (decltype(d.image_destroy))f("dmtxImageDestroy");
        d.decode_create = (decltype(d.decode_create))f("dmtxDecodeCreate");
        d.decode_destroy = (decltype(d.decode_destroy))f("dmtxDecodeDestroy");
        d.region_find_next = (decltype(d.region_find_next))f("dmtxRegionFindNext");
        d.region_destroy = (decltype(d.region_destroy))f("dmtxRegionDestroy");
        d.decode_matrix_region = (decltype(d.decode_matrix_region))f("dmtxDecodeMatrixRegion");
        d.message_destroy = (decltype(d.message_destroy))f("dmtxMessageDestroy");
        d.time_now = (decltype(d.time_now))f("dmtxTimeNow");
        d.time_add = (decltype(d.time_add))f("dmtxTimeAdd");
        d.version = (decltype(d.version))f("dmtxVersion");
        if (d.version) {
            // pylibdmtx: the fnc1 field from 0.7.5 on
            int a = 0, b = 0, c = 0;
            std::sscanf(d.version(), "%d.%d.%d", &a, &b, &c);
            d.fnc1 = std::make_tuple(a, b, c) >= std::make_tuple(0, 7, 5);
        }
        d.ok = d.image_create && d.image_destroy && d.decode_create && d.decode_destroy && d.region_find_next && d.region_destroy &&
               d.decode_matrix_region && d.message_destroy && d.time_now && d.time_add;
    });
    return d;
}

struct DmtxMessageOld { size_t arraySize, codeSize, outputSize; int outputIdx, padCount; unsigned char *array, *code, *output; };
struct DmtxMessageNew { size_t arraySize, codeSize, outputSize; int outputIdx, padCount, fnc1; unsigned char *array, *code, *output; };

}  // namespace

std::string utf8_replace(const std::string& s) {
    std::string out;
    const unsigned char* p = (const unsigned char*)s.data();
    size_t n = s.size(), i = 0;
    const char* REP = "\xef\xbf\xbd";
    while (i < n) {
        unsigned char c = p[i];
        if (c < 0x80) {
            out += (char)c;
            i++;
            continue;
        }
        int need;
        unsigned lo = 0x80, hi = 0xbf;
        if (c >= 0xc2 && c <= 0xdf) need = 1;
        else if (c >= 0xe0 && c <= 0xef) {
            need = 2;
            if (c == 0xe0) lo = 0xa0;
            if (c == 0xed) hi = 0x9f;
        } else if (c >= 0xf0 && c <= 0xf4) {
            need = 3;
            if (c == 0xf0) lo = 0x90;
            if (c == 0xf4) hi = 0x8f;
        } else {
            out += REP;
            i++;
            continue;
        }
        size_t j = i + 1;
        int got = 0;
        while (got < need && j < n) {
            unsigned char d = p[j];
            unsigned l = got == 0 ? lo : 0x80, h = got == 0 ? hi : 0xbf;
            if (d < l || d > h) break;
            j++;
            got++;
        }
        if (got == need) out.append(s, i, j - i);
        else out += REP;           // the maximal invalid subpart is one replacement
        i = got == need ? j : j;
    }
    return out;
}

bool zbar_available() { return zbar().ok; }
bool dmtx_available() { return dmtx().ok; }

std::vector<Code> zbar_decode(const cv::Mat& gray) {
    const Zbar& z = zbar();
    if (!z.ok) throw std::runtime_error("zbar not installed");
    cv::Mat g = gray.isContinuous() ? gray : gray.clone();
    std::vector<Code> out;
    void* sc = z.scanner_create();
    if (!sc) throw std::runtime_error("Could not create image scanner");
    void* img = z.image_create();
    if (!img) {
        z.scanner_destroy(sc);
        throw std::runtime_error("Could not create zbar image");
    }
    const unsigned long Y800 = 'Y' | ('8' << 8) | ('0' << 16) | ((unsigned long)'0' << 24);
    z.set_format(img, Y800);
    z.set_size(img, (unsigned)g.cols, (unsigned)g.rows);
    z.set_data(img, g.data, (unsigned long)g.total(), nullptr);
    int r = z.scan(sc, img);
    if (r >= 0)
        for (const void* s = z.first(img); s; s = z.next(s)) {
            int type = *(const int*)s;
            out.push_back({utf8_replace(std::string(z.data(s), z.length(s))), zbar_name(type)});
        }
    z.image_destroy(img);
    z.scanner_destroy(sc);
    if (r < 0) throw std::runtime_error("Unsupported image format");
    return out;
}

std::vector<Code> dmtx_decode(const cv::Mat& gray, int timeout_ms) {
    const Dmtx& d = dmtx();
    if (!d.ok) throw std::runtime_error("libdmtx not installed");
    cv::Mat g = gray.isContinuous() ? gray.clone() : gray.clone();
    DmtxTime limit{};
    bool has_limit = timeout_ms != 0;
    if (has_limit) limit = d.time_add(d.time_now(), timeout_ms);
    std::vector<Code> out;
    void* img = d.image_create(g.data, g.cols, g.rows, 300);   // DmtxPack8bppK
    if (!img) throw std::runtime_error("Could not create image");
    void* dec = d.decode_create(img, 1);
    if (!dec) {
        d.image_destroy(&img);
        throw std::runtime_error("Could not create decoder");
    }
    for (;;) {
        void* reg = d.region_find_next(dec, has_limit ? &limit : nullptr);
        if (!reg) break;
        void* msg = d.decode_matrix_region(dec, reg, -1);   // corrections: DmtxUndefined
        if (msg) {
            const unsigned char* o = d.fnc1 ? ((DmtxMessageNew*)msg)->output : ((DmtxMessageOld*)msg)->output;
            out.push_back({utf8_replace(std::string((const char*)o)), "DATAMATRIX"});
            d.message_destroy(&msg);
        }
        d.region_destroy(&reg);
    }
    d.decode_destroy(&dec);
    d.image_destroy(&img);
    return out;
}

}  // namespace vc::codes
