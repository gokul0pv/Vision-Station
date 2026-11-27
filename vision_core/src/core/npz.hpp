// numpy .npz files (np.savez / np.savez_compressed): a zip of .npy arrays.
// Vision Station's trained surface models are stored this way.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vc::npz {

struct Array {
    std::string descr;              // '<f4', '<f2', '<f8', '<i8', '<U4', '|b1', ...
    std::vector<int64_t> shape;
    bool fortran = false;
    std::vector<uint8_t> data;      // raw little-endian elements
    size_t count() const;
    std::vector<float> f32() const;      // any numeric dtype -> float32 (as .astype(np.float32))
    double scalar() const;               // float(arr) of a one-element numeric array
    std::string str() const;             // str(arr) of a '<U' / '|S' scalar
};

// every array in the file, by name (without ".npy"); throws std::runtime_error
std::map<std::string, Array> load(const std::string& path);

}  // namespace vc::npz
