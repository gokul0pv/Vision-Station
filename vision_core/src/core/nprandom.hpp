// numpy.random.default_rng(seed): SeedSequence -> PCG64, and the two Generator
// calls Vision Station uses, drawing exactly the numbers numpy draws, so a
// seeded consensus search picks the same samples.
#pragma once

#include <cstdint>
#include <vector>

namespace vc::npr {

class Generator {
public:
    explicit Generator(uint64_t seed);
    uint64_t next64();
    uint32_t next32();
    // rng.integers(high): 0 <= v < high
    int64_t integers(int64_t high);
    // rng.choice(pop, size=size, replace=False)
    std::vector<int64_t> choice(int64_t pop, int64_t size);

private:
    uint64_t bounded(uint64_t rng);  // random_bounded_uint64(0, rng)
    uint64_t hi_ = 0, lo_ = 0, inc_hi_ = 0, inc_lo_ = 0;
    bool has32_ = false;
    uint32_t u32_ = 0;
    void step();
};

}  // namespace vc::npr
