#include "line.hpp"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <thread>

namespace vc::hw {

std::pair<bool, std::string> barcode_check(const std::string& raw, const std::string& prefix, int length, bool match_case) {
    std::string code;
    for (char c : raw)
        if ((unsigned char)c >= 0x20 && c != 0x7f) code += c;
    while (!code.empty() && std::isspace((unsigned char)code.back())) code.pop_back();
    size_t a = 0;
    while (a < code.size() && std::isspace((unsigned char)code[a])) a++;
    code = code.substr(a);
    if (code.empty()) return {false, "nothing was read"};
    auto up = [](std::string s) { for (auto& c : s) c = (char)std::toupper((unsigned char)c); return s; };
    std::string head = code.substr(0, prefix.size());
    bool same = match_case ? head == prefix : up(head) == up(prefix);
    if (!prefix.empty() && !same)
        return {false, "it must start with " + prefix + " - this one starts with " + code.substr(0, std::max<size_t>(1, prefix.size()))};
    if (length > 0 && (int)code.size() != length)
        return {false, "it must be " + std::to_string(length) + " characters - this one has " + std::to_string(code.size())};
    return {true, code};
}

namespace {

std::mutex g_trigger;   // one trigger at a time across clients

void send_line(SOCKET s, const std::string& line) {
    std::string out = line + "\r\n";
    send(s, out.data(), (int)out.size(), 0);
}

std::string result_line(const std::string& job, const Json& res) {
    std::string failed;
    for (auto& t : res["tools"])
        if (t.value("status", "") != "pass") failed += (failed.empty() ? "" : ",") + t.value("tool_id", "");
    char b[64];
    std::snprintf(b, sizeof b, "%.0f", res.value("total_ms", 0.0));
    return "RESULT " + job + " " + res.value("verdict", "") + " " + b + " " + (failed.empty() ? "-" : failed);
}

void handle(SOCKET s, const std::string& line, LineHooks& h) {
    std::string cmd = line.substr(0, line.find(' ')), arg = line.find(' ') == std::string::npos ? "" : line.substr(line.find(' ') + 1);
    for (auto& c : cmd) c = (char)std::toupper((unsigned char)c);
    while (!arg.empty() && std::isspace((unsigned char)arg.back())) arg.pop_back();
    while (!arg.empty() && std::isspace((unsigned char)arg.front())) arg.erase(0, 1);
    if (cmd == "PING") return send_line(s, "PONG");
    if (cmd == "VERSION") return send_line(s, "VISIONSTATION 1");
    if (cmd == "STATUS") return send_line(s, h.busy() ? "BUSY" : "READY");
    if (cmd == "TRIGGER" || cmd == "T") {
        std::unique_lock<std::mutex> lk(g_trigger, std::try_to_lock);
        if (!lk.owns_lock() || h.busy()) return send_line(s, "BUSY");
        std::string job = h.job_name();
        if (job.empty()) return send_line(s, "ERROR no application open");
        if (!arg.empty() && arg != job) return send_line(s, "ERROR no job " + arg);
        std::string err;
        Json res = h.trigger(job, err);
        if (res.is_null()) {
            std::string why = err.empty() ? "not inspected" : err;
            std::replace(why.begin(), why.end(), ' ', '-');
            send_line(s, "RESULT " + job + " ERROR 0 " + why.substr(0, 60));
            return send_line(s, "DONE INCOMPLETE 1");
        }
        send_line(s, result_line(job, res));
        return send_line(s, std::string("DONE ") + (res.value("verdict", "") == "PASS" ? "PASS" : "FAIL") + " 1");
    }
    if (cmd == "GET") {
        auto d1 = arg.find('.'), d2 = arg.find('.', d1 == std::string::npos ? 0 : d1 + 1);
        if (d1 == std::string::npos || d2 == std::string::npos) return send_line(s, "ERROR use GET <job>.<tool>.<value>");
        std::string tool = arg.substr(d1 + 1, d2 - d1 - 1), key = arg.substr(d2 + 1);
        Json last = h.last();
        if (last.is_null() || arg.substr(0, d1) != h.job_name()) return send_line(s, "ERROR no result for " + arg.substr(0, d2));
        for (auto& t : last["tools"])
            if (t.value("tool_id", "") == tool) {
                Json v = (key == "status" || key == "score" || key == "message") ? t[key] : (t["values"].contains(key) ? t["values"][key] : Json());
                if (v.is_null() || v.is_array() || v.is_object()) return send_line(s, "ERROR no value " + arg);
                if (v.is_number_float()) {
                    char b[64];
                    std::snprintf(b, sizeof b, "%.6g", v.get<double>());
                    return send_line(s, std::string("VALUE ") + b);
                }
                if (v.is_boolean()) return send_line(s, std::string("VALUE ") + (v.get<bool>() ? "True" : "False"));
                return send_line(s, "VALUE " + (v.is_string() ? v.get<std::string>() : v.dump()));
            }
        return send_line(s, "ERROR no result for " + arg.substr(0, d2));
    }
    if (cmd == "CODE") {
        if (!h.code) return send_line(s, "ERROR barcode checking is off");
        auto [ok, v] = h.code(arg);
        return send_line(s, ok ? "CODE OK " + v : "CODE NG " + v);
    }
    send_line(s, "ERROR unknown command: " + cmd);
}

void client(SOCKET s, LineHooks h) {
    std::string buf;
    char chunk[4096];
    for (;;) {
        int n = recv(s, chunk, sizeof chunk, 0);
        if (n <= 0) break;
        buf.append(chunk, n);
        size_t nl;
        while ((nl = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, nl);
            buf.erase(0, nl + 1);
            while (!line.empty() && (line.back() == '\r' || std::isspace((unsigned char)line.back()))) line.pop_back();
            if (!line.empty()) {
                try {
                    handle(s, line, h);
                } catch (const std::exception& e) {
                    send_line(s, std::string("ERROR ") + e.what());
                }
            }
        }
    }
    closesocket(s);
}

}  // namespace

bool start_line_server(int port, LineHooks hooks, std::string& error) {
    WSADATA w;
    WSAStartup(MAKEWORD(2, 2), &w);
    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((u_short)port);
    if (bind(ls, (sockaddr*)&a, sizeof a) != 0 || listen(ls, 8) != 0) {
        error = "port " + std::to_string(port) + " is in use or not allowed";
        closesocket(ls);
        return false;
    }
    std::thread([ls, hooks] {
        for (;;) {
            SOCKET c = accept(ls, nullptr, nullptr);
            if (c == INVALID_SOCKET) continue;
            std::thread(client, c, hooks).detach();
        }
    }).detach();
    return true;
}

}  // namespace vc::hw
