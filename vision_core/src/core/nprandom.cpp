#include "nprandom.hpp"

#include <intrin.h>

#include <algorithm>
#include <set>
#include <stdexcept>

namespace vc::npr {

namespace {

// numpy/random/bit_generator.pyx SeedSequence
constexpr uint32_t INIT_A = 0x43b0d7e5, MULT_A = 0x931e8875, INIT_B = 0x8b51f9dd, MULT_B = 0x58f38ded;
constexpr uint32_t MIX_MULT_L = 0xca01f9dd, MIX_MULT_R = 0x4973f715;
constexpr int XSHIFT = 16;

uint32_t hashmix(uint32_t value, uint32_t& hc) {
    value ^= hc;
    hc *= MULT_A;
    value *= hc;
    value ^= value >> XSHIFT;
    return value;
}

uint32_t mix(uint32_t x, uint32_t y) {
    uint32_t r = MIX_MULT_L * x - MIX_MULT_R * y;
    r ^= r >> XSHIFT;
    return r;
}

// SeedSequence(seed).generate_state(n_words, uint32)
std::vector<uint32_t> seed_state(uint64_t seed, int n_words) {
    std::vector<uint32_t> entropy;
    do {
        entropy.push_back((uint32_t)(seed & 0xffffffffu));
        seed >>= 32;
    } while (seed);
    uint32_t pool[4];
    uint32_t hc = INIT_A;
    for (int i = 0; i < 4; i++) pool[i] = hashmix(i < (int)entropy.size() ? entropy[i] : 0u, hc);
    for (int s = 0; s < 4; s++)
        for (int d = 0; d < 4; d++)
            if (s != d) pool[d] = mix(pool[d], hashmix(pool[s], hc));
    for (size_t s = 4; s < entropy.size(); s++)
        for (int d = 0; d < 4; d++) pool[d] = mix(pool[d], hashmix(entropy[s], hc));
    std::vector<uint32_t> out(n_words);
    uint32_t h = INIT_B;
    for (int i = 0; i < n_words; i++) {
        uint32_t v = pool[i % 4];
        v ^= h;
        h *= MULT_B;
        v *= h;
        v ^= v >> XSHIFT;
        out[i] = v;
    }
    return out;
}

constexpr uint64_t MUL_HI = 0x2360ED051FC65DA4ULL, MUL_LO = 0x4385DF649FCCF645ULL;

}  // namespace

void Generator::step() {
    // state = state * MUL + inc (128-bit)
    uint64_t carry_hi;
    uint64_t lo = _umul128(lo_, MUL_LO, &carry_hi);
    uint64_t hi = carry_hi + lo_ * MUL_HI + hi_ * MUL_LO;
    uint64_t nlo = lo + inc_lo_;
    hi += inc_hi_ + (nlo < lo ? 1 : 0);
    lo_ = nlo;
    hi_ = hi;
}

Generator::Generator(uint64_t seed) {
    auto w = seed_state(seed, 8);
    // generate_state(4, uint64): little-endian pairs of words
    uint64_t v[4];
    for (int i = 0; i < 4; i++) v[i] = (uint64_t)w[2 * i] | ((uint64_t)w[2 * i + 1] << 32);
    uint64_t s_hi = v[0], s_lo = v[1], q_hi = v[2], q_lo = v[3];
    // pcg_setseq_128_srandom_r
    hi_ = lo_ = 0;
    inc_hi_ = (q_hi << 1) | (q_lo >> 63);
    inc_lo_ = (q_lo << 1) | 1u;
    step();
    uint64_t nlo = lo_ + s_lo;
    hi_ += s_hi + (nlo < lo_ ? 1 : 0);
    lo_ = nlo;
    step();
}

uint64_t Generator::next64() {
    step();
    uint64_t x = hi_ ^ lo_;
    unsigned rot = (unsigned)(hi_ >> 58);
    return (x >> rot) | (x << ((64 - rot) & 63));
}

uint32_t Generator::next32() {
    if (has32_) {
        has32_ = false;
        return u32_;
    }
    uint64_t n = next64();
    has32_ = true;
    u32_ = (uint32_t)(n >> 32);
    return (uint32_t)(n & 0xffffffffu);
}

uint64_t Generator::bounded(uint64_t rng) {
    if (rng == 0) return 0;
    if (rng <= 0xFFFFFFFFULL) {
        if (rng == 0xFFFFFFFFULL) return next32();
        // buffered_bounded_lemire_uint32
        uint32_t excl = (uint32_t)rng + 1;
        uint64_t m = (uint64_t)next32() * excl;
        uint32_t left = (uint32_t)m;
        if (left < excl) {
            uint32_t threshold = (UINT32_MAX - (uint32_t)rng) % excl;
            while (left < threshold) {
                m = (uint64_t)next32() * excl;
                left = (uint32_t)m;
            }
        }
        return m >> 32;
    }
    if (rng == 0xFFFFFFFFFFFFFFFFULL) return next64();
    // bounded_lemire_uint64
    uint64_t excl = rng + 1, hi;
    uint64_t lo = _umul128(next64(), excl, &hi);
    if (lo < excl) {
        uint64_t threshold = (0 - excl) % excl;
        while (lo < threshold) lo = _umul128(next64(), excl, &hi);
    }
    return hi;
}

int64_t Generator::integers(int64_t high) {
    if (high <= 0) throw std::invalid_argument("high <= 0");
    return (int64_t)bounded((uint64_t)(high - 1));
}

std::vector<int64_t> Generator::choice(int64_t pop, int64_t size) {
    if (size > pop) throw std::invalid_argument("Cannot take a larger sample than population when replace is False");
    std::vector<int64_t> idx((size_t)size);
    if (pop > 10000 && size > pop / 50) {
        // tail shuffle
        std::vector<int64_t> all((size_t)pop);
        for (int64_t i = 0; i < pop; i++) all[i] = i;
        int64_t first = std::max<int64_t>(pop - size, 1);
        for (int64_t i = pop - 1; i >= first; i--) std::swap(all[i], all[bounded((uint64_t)i)]);
        std::copy(all.end() - size, all.end(), idx.begin());
        return idx;
    }
    // Floyd's algorithm, then a shuffle
    std::set<int64_t> seen;
    for (int64_t j = pop - size; j < pop; j++) {
        int64_t v = (int64_t)bounded((uint64_t)j);
        if (seen.insert(v).second) {
            idx[j - pop + size] = v;
        } else {
            seen.insert(j);
            idx[j - pop + size] = j;
        }
    }
    for (int64_t i = size - 1; i >= 1; i--) std::swap(idx[i], idx[bounded((uint64_t)i)]);
    return idx;
}

}  // namespace vc::npr
