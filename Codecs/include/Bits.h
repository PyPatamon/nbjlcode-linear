/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Apr 09, 2016
 */
/**
 * Based on code by
 *     Daniel Lemire, https://github.com/lemire/FastPFor
 * which was available under the Apache License, Version 2.0.
 */

#ifndef BITS_H_
#define BITS_H_

#include "common.h"

template <uint32_t N, typename T>
class padToNBits;

template <typename T>
using padTo32Bits = padToNBits<32, T>;

template <typename T>
using padTo64Bits = padToNBits<64, T>;

template <typename T>
using padTo128Bits = padToNBits<128, T>;

template <typename T>
using padTo64Bytes = padToNBits<256, T>;


template <uint32_t N, typename T>
class padToNBits {
public:
    const T *operator()(T *inbyte) {
        return reinterpret_cast<const T *>((reinterpret_cast<uintptr_t>(inbyte)
                    + k) & ~k);
    }

    static bool needPaddingToNBits(const T *inbyte) {
        return (reinterpret_cast<uintptr_t>(inbyte) & k) != 0;
    }

private:
    static const uint32_t k = N/8 - 1;
};

__attribute__ ((const))
inline uint32_t gccbits(const uint32_t v) {
#ifdef _MSC_VER
    if (v == 0) {
        return 0;
    }

    unsigned long answer;
    _BitScanReverse(&answer, v);
    return answer + 1;
#else
    return v == 0 ? 0 : 32 - __builtin_clz(v);
#endif
}

__attribute__ ((const))
inline uint32_t asmbits(const uint32_t v) {
#ifdef _MSC_VER
    return gccbits(v);
#else
    if (v == 0)
        return 0;
    uint32_t answer;
    __asm__("bsr %1, %0;" :"=r"(answer) :"r"(v));
    return answer + 1;
#endif
}

__attribute__ ((const))
inline uint32_t slowbits(uint32_t v) {
    uint32_t r = 0;
    while (v) {
        ++r;
        v = v >> 1;
    }
    return r;
}

__attribute__ ((const))
inline uint32_t bits(uint32_t v) {
    uint32_t r(0);
    if (v >= (1U << 15)) {
        v >>= 16;
        r += 16;
    }
    if (v >= (1U << 7)) {
        v >>= 8;
        r += 8;
    }
    if (v >= (1U << 3)) {
        v >>= 4;
        r += 4;
    }
    if (v >= (1U << 1)) {
        v >>= 2;
        r += 2;
    }
    if (v >= (1U << 0)) {
        v >>= 1;
        r += 1;
    }
    return r;
}

#ifndef _MSC_VER
__attribute__ ((const))
constexpr uint32_t constexprbits(uint32_t v) {
    return v >= (1U << 15) ? 16 + constexprbits(v>>16) :
            (v >= (1U << 7)) ? 8 + constexprbits(v>>8) :
                    (v >= (1U << 3)) ? 4 + constexprbits(v>>4) :
                            (v >= (1U << 1)) ? 2 + constexprbits(v>>2) :
                                    (v >= (1U << 0)) ? 1 + constexprbits(v>>1) :
                                            0;
}
#else
template <int N>
struct exprbits {
    enum { value = 1 + exprbits<(N>>1)>::value };
};

template <>
struct exprbits<0> {
    enum { value = 0 };
};
#define constexprbits(n) exprbits<n>::value
#endif

template <typename iterator>
__attribute__ ((pure))
uint32_t maxbits(const iterator &begin, const iterator &end) {
    uint32_t accumulator = 0;
    for (iterator k = begin; k != end; ++k) {
        accumulator |= *k;
    }
    return gccbits(accumulator);
}

template <typename iterator>
uint32_t slowmaxbits(const iterator &begin, const iterator &end) {
    uint32_t accumulator = 0;
    for (iterator k = begin; k != end; ++k) {
        const uint32_t tb = gccbits(*k);
        if (tb > accumulator)
            accumulator = tb;
    }
    return accumulator;
}

__attribute__ ((const))
inline uint32_t bytes(const uint32_t value) {
    if (value > 0x000000FF) {
        if (value > 0x0000FFFF) {
            if (value > 0x00FFFFFF) {
                return 4;
            } else {
                return 3;
            }
        } else {
            return 2;
        }
    } else {
        return 1;
    }
}

#endif /* BITS_H_ */
