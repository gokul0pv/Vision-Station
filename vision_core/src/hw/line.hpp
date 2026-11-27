// The line interface (vision_station/core/comms.py): a TCP server a PLC or
// robot controller talks to in text lines. Same protocol as Vision Station:
//   PING -> PONG          VERSION -> VISIONSTATION 1      STATUS -> READY | BUSY
//   TRIGGER [job] -> RESULT <job> <PASS|FAIL|ERROR> <ms> <failed|-> ... DONE <PASS|FAIL|INCOMPLETE> <n>
//   GET <job>.<tool>.<value> -> VALUE <v>
//   CODE <barcode> -> CODE OK <code> | CODE NG <reason>
// A trigger during a run is answered BUSY, never queued.
#pragma once

#include "../core/types.hpp"

#include <functional>
#include <string>

namespace vc::hw {

struct LineHooks {
    std::function<bool()> busy;
    // inspect once: the inspection JSON, or an error message in `error`
    std::function<Json(const std::string& job, std::string& error)> trigger;
    std::function<std::string()> job_name;                // the open job's short name ("" = none)
    std::function<Json()> last;                          // the last inspection (or null)
    std::function<std::pair<bool, std::string>(const std::string& code)> code;   // barcode rule
};

// listens on `port` in a background thread; false (and `error`) if it cannot
bool start_line_server(int port, LineHooks hooks, std::string& error);

// barcode.check: (ok, reason) for a scanned code against prefix / length rules
std::pair<bool, std::string> barcode_check(const std::string& raw, const std::string& prefix, int length, bool match_case);

}  // namespace vc::hw
