#include "npz.hpp"

#include "../../third_party/miniz/miniz.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace vc::npz {

namespace {

// IEEE half -> float, exactly (numpy's float16 -> float32 cast)
float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
    if (exp == 0) {
        float f = std::ldexp((float)man, -24);   // subnormal (or zero): man * 2^-24, exact
        return sign ? -f : f;
    } else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

Array parse_npy(const std::vector<uint8_t>& b, const std::string& name) {
    if (b.size() < 10 || std::memcmp(b.data(), "\x93NUMPY", 6) != 0) throw std::runtime_error(name + ": not a .npy array");
    int major = b[6];
    size_t hlen, off;
    if (major == 1) hlen = b[8] | (b[9] << 8), off = 10;
    else hlen = b[8] | (b[9] << 8) | (b[10] << 16) | ((size_t)b[11] << 24), off = 12;
    std::string h(b.begin() + off, b.begin() + off + hlen);
    Array a;
    auto field = [&](const char* key) {
        size_t k = h.find(std::string("'") + key + "'");
        if (k == std::string::npos) throw std::runtime_error(name + ": no " + key);
        return h.find(':', k) + 1;
    };
    size_t d = h.find('\'', field("descr"));
    a.descr = h.substr(d + 1, h.find('\'', d + 1) - d - 1);
    a.fortran = h.compare(h.find_first_not_of(' ', field("fortran_order")), 4, "True") == 0;
    size_t s = h.find('(', field("shape")), e = h.find(')', s);
    std::string sh = h.substr(s + 1, e - s - 1), num;
    for (char c : sh + ",") {
        if (std::isdigit((unsigned char)c)) num += c;
        else if (c == ',' && !num.empty()) {
            a.shape.push_back(std::stoll(num));
            num.clear();
        }
    }
    a.data.assign(b.begin() + off + hlen, b.end());
    return a;
}

}  // namespace

size_t Array::count() const {
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    return n;
}

std::vector<float> Array::f32() const {
    size_t n = count();
    std::vector<float> out(n);
    char kind = descr.size() > 1 ? descr[1] : '?';
    int sz = descr.size() > 2 ? std::stoi(descr.substr(2)) : 0;
    const uint8_t* p = data.data();
    if (data.size() < n * (size_t)sz) throw std::runtime_error("array data is short");
    for (size_t i = 0; i < n; i++) {
        const uint8_t* q = p + i * sz;
        if (kind == 'f' && sz == 4) std::memcpy(&out[i], q, 4);
        else if (kind == 'f' && sz == 8) {
            double v;
            std::memcpy(&v, q, 8);
            out[i] = (float)v;
        } else if (kind == 'f' && sz == 2) {
            uint16_t v;
            std::memcpy(&v, q, 2);
            out[i] = half_to_float(v);
        } else if ((kind == 'i' || kind == 'u') && sz == 8) {
            int64_t v;
            std::memcpy(&v, q, 8);
            out[i] = (float)v;
        } else if ((kind == 'i' || kind == 'u') && sz == 4) {
            int32_t v;
            std::memcpy(&v, q, 4);
            out[i] = (float)v;
        } else if (kind == 'b' || ((kind == 'i' || kind == 'u') && sz == 1)) out[i] = (float)q[0];
        else throw std::runtime_error("unsupported dtype " + descr);
    }
    // a Fortran-ordered 2-D array: back to C order
    if (fortran && shape.size() == 2) {
        std::vector<float> c(n);
        size_t R = (size_t)shape[0], C = (size_t)shape[1];
        for (size_t r = 0; r < R; r++)
            for (size_t k = 0; k < C; k++) c[r * C + k] = out[k * R + r];
        return c;
    }
    return out;
}

double Array::scalar() const {
    char kind = descr.size() > 1 ? descr[1] : '?';
    int sz = descr.size() > 2 ? std::stoi(descr.substr(2)) : 0;
    if (kind == 'f' && sz == 8) {
        double v;
        std::memcpy(&v, data.data(), 8);
        return v;
    }
    if ((kind == 'i' || kind == 'u') && sz == 8) {
        int64_t v;
        std::memcpy(&v, data.data(), 8);
        return (double)v;
    }
    return (double)f32().at(0);
}

std::string Array::str() const {
    char kind = descr.size() > 1 ? descr[1] : '?';
    int sz = descr.size() > 2 ? std::stoi(descr.substr(2)) : 0;
    std::string out;
    if (kind == 'U') {
        for (int i = 0; i < sz; i++) {
            uint32_t c;
            std::memcpy(&c, data.data() + 4 * i, 4);
            if (!c) break;
            if (c < 0x80) out += (char)c;
            else if (c < 0x800) out += (char)(0xc0 | (c >> 6)), out += (char)(0x80 | (c & 0x3f));
            else out += (char)(0xe0 | (c >> 12)), out += (char)(0x80 | ((c >> 6) & 0x3f)), out += (char)(0x80 | (c & 0x3f));
        }
    } else if (kind == 'S') {
        for (int i = 0; i < sz && data[i]; i++) out += (char)data[i];
    }
    return out;
}

std::map<std::string, Array> load(const std::string& path) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof zip);
    if (!mz_zip_reader_init_mem(&zip, file.data(), file.size(), 0)) throw std::runtime_error(path + " is not a zip (.npz) file");
    std::map<std::string, Array> out;
    try {
        mz_uint n = mz_zip_reader_get_num_files(&zip);
        for (mz_uint i = 0; i < n; i++) {
            mz_zip_archive_file_stat st;
            if (!mz_zip_reader_file_stat(&zip, i, &st)) continue;
            std::string name = st.m_filename;
            size_t sz = 0;
            void* p = mz_zip_reader_extract_to_heap(&zip, i, &sz, 0);
            if (!p) throw std::runtime_error("cannot read " + name + " in " + path);
            std::vector<uint8_t> b((uint8_t*)p, (uint8_t*)p + sz);
            mz_free(p);
            if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0) name.resize(name.size() - 4);
            out[name] = parse_npy(b, name);
        }
    } catch (...) {
        mz_zip_reader_end(&zip);
        throw;
    }
    mz_zip_reader_end(&zip);
    return out;
}

}  // namespace vc::npz
