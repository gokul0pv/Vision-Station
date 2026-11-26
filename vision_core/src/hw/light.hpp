// The light controller on a serial port (vision_station/hardware/light.py
// LightController): "$" + command + channel + 3 hex digits + XOR checksum,
// 9600 8N1, one answer per command. The port is opened for each command and
// closed after it, so another program can still use the controller.
#pragma once

#include <string>
#include <vector>

namespace vc::hw {

// light.frame: frame('3', 1, 100) -> "$3106414"
std::string light_frame(char cmd, int channel, int data = 0);
// the value in a read's answer, or -1 when it is not one (wrong checksum included)
int light_parse_read(const std::string& reply, char cmd, int channel);
// COM ports on this PC, in number order
std::vector<std::string> serial_ports();

class Light {
public:
    explicit Light(std::string port) : port_(std::move(port)) {}
    // each throws std::runtime_error with an operator-readable message
    void on(int channel = 1);
    void off(int channel = 1);
    void set_brightness(int channel, int value);
    int brightness(int channel = 1);
    const std::string& port() const { return port_; }

private:
    std::string talk(const std::string& data, size_t answer_len);
    void set(char cmd, int channel, int data);
    std::string port_;
};

}  // namespace vc::hw
