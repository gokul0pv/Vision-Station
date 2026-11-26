// Talking to vc-engine over its named pipe. Every call has a time limit, so a
// client never freezes on an engine that has stopped answering.
#pragma once

#include "types.hpp"

#include <string>

namespace vc {

class EngineClient {
public:
    ~EngineClient();
    bool connect(int timeout_ms = 2000);
    void close();
    bool connected() const { return pipe_ != nullptr; }
    // the reply, or {"ok":false,"error":...} on a timeout or broken pipe (the connection is then closed)
    Json call(const Json& req, int timeout_ms);

private:
    void* pipe_ = nullptr;
    std::string buf_;
};

// start vc-engine.exe from the program's folder (hidden); true when it was started
bool start_engine_process(void** process_handle = nullptr);

}  // namespace vc
