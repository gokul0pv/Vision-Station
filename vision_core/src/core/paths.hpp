// Where things are: beside the program when installed; while developing, in
// the Vision Station folder above (models/ is shared).
#pragma once

#include <filesystem>
#include <string>

namespace vc::paths {

std::filesystem::path exe_dir();
// the first folder `rel` found beside the program or up to five folders above; empty if none
std::filesystem::path find_dir(const std::string& rel);

}  // namespace vc::paths
