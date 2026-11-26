#include "light.hpp"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace vc::hw {

std::string light_frame(char cmd, int channel, int data) {
    char body[16];
    std::snprintf(body, sizeof body, "$%c%d%03X", cmd, channel, std::max(0, std::min(0xFFF, data)));
    unsigned char x = 0;
    for (const char* p = body; *p; p++) x ^= (unsigned char)*p;
    char out[24];
    std::snprintf(out, sizeof out, "%s%02X", body, x);
    return out;
}

int light_parse_read(const std::string& text, char cmd, int channel) {
    char head[4] = {'$', cmd, (char)('0' + channel), 0};
    size_t i = text.find(head);
    if (i == std::string::npos || text.size() < i + 8) return -1;
    std::string f = text.substr(i, 8);
    int v = (int)std::strtol(f.substr(3, 3).c_str(), nullptr, 16);
    std::string want = light_frame(cmd, channel, v), got = f;
    for (auto& c : want) c = (char)toupper((unsigned char)c);
    for (auto& c : got) c = (char)toupper((unsigned char)c);
    return want == got ? v : -1;
}

std::vector<std::string> serial_ports() {
    std::vector<std::pair<int, std::string>> v;
    char target[1024];
    for (int n = 1; n <= 256; n++) {
        std::string name = "COM" + std::to_string(n);
        if (QueryDosDeviceA(name.c_str(), target, sizeof target)) v.emplace_back(n, name);
    }
    std::vector<std::string> out;
    for (auto& p : v) out.push_back(p.second);
    return out;
}

std::string Light::talk(const std::string& data, size_t answer_len) {
    std::string path = "\\\\.\\" + port_;
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        DWORD e = GetLastError();
        throw std::runtime_error(port_ + (e == ERROR_ACCESS_DENIED ? " is in use by another program" : " cannot be opened"));
    }
    struct Closer { HANDLE h; ~Closer() { CloseHandle(h); } } closer{h};
    DCB dcb{};
    dcb.DCBlength = sizeof dcb;
    GetCommState(h, &dcb);
    dcb.BaudRate = 9600;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;      // as pyserial: DTR and RTS on
    dcb.fRtsControl = RTS_CONTROL_ENABLE;
    dcb.fOutxCtsFlow = dcb.fOutxDsrFlow = FALSE;
    dcb.fOutX = dcb.fInX = FALSE;
    if (!SetCommState(h, &dcb)) throw std::runtime_error(port_ + " cannot be set to 9600 8N1");
    COMMTIMEOUTS to{};
    to.ReadTotalTimeoutConstant = 300;          // answers come in about 10 ms
    to.WriteTotalTimeoutConstant = 1000;
    SetCommTimeouts(h, &to);
    PurgeComm(h, PURGE_RXCLEAR | PURGE_TXCLEAR);
    DWORD n = 0;
    if (!WriteFile(h, data.data(), (DWORD)data.size(), &n, nullptr) || n != data.size())
        throw std::runtime_error("the light controller on " + port_ + " stopped answering");
    std::string got(answer_len, '\0');
    DWORD r = 0;
    ReadFile(h, got.data(), (DWORD)answer_len, &r, nullptr);
    got.resize(r);
    if (!got.empty() && got.size() < answer_len && got[0] != '&') {
        std::string more(answer_len - got.size(), '\0');
        DWORD r2 = 0;
        ReadFile(h, more.data(), (DWORD)more.size(), &r2, nullptr);
        got += more.substr(0, r2);
    }
    if (got.empty()) throw std::runtime_error("no answer from a light controller on " + port_ + " - is it switched on, and is this its port?");
    if (got[0] == '&') throw std::runtime_error("the light controller refused " + data);
    return got;
}

void Light::set(char cmd, int channel, int data) {
    std::string f = light_frame(cmd, channel, data);
    std::string got = talk(f, 1);
    if (got[0] != '$') throw std::runtime_error("unexpected answer to " + f);
}

void Light::on(int channel) { set('1', channel, 0); }
void Light::off(int channel) { set('2', channel, 0); }
void Light::set_brightness(int channel, int value) { set('3', channel, std::max(0, std::min(255, value))); }

int Light::brightness(int channel) {
    std::string got = talk(light_frame('4', channel), 8);
    int v = light_parse_read(got, '4', channel);
    if (v < 0) throw std::runtime_error("unexpected answer to a brightness read");
    return v;
}

}  // namespace vc::hw
