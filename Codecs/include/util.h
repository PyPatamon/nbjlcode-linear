/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Jan 1, 2015
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "common.h"

#ifdef __linux__
#define USE_O_DIRECT
#endif

//#define STATS
// taken from stackoverflow
#ifndef NDEBUG
# define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (false)
#else
# define ASSERT(condition, message) do { } while (false)
#endif

#define memset32(dest)       \
	__asm__ __volatile__(    \
			"pxor   %%xmm0, %%xmm0\n\t"    \
			"movdqu %%xmm0, %0\n\t"        \
			"movdqu %%xmm0, %1\n\t"        \
			"movdqu %%xmm0, %2\n\t"        \
			"movdqu %%xmm0, %3\n\t"        \
			"movdqu %%xmm0, %4\n\t"        \
			"movdqu %%xmm0, %5\n\t"        \
			"movdqu %%xmm0, %6\n\t"        \
			"movdqu %%xmm0, %7\n\t"        \
			:"=m" (dest[0]), "=m" (dest[4]), "=m" (dest[8]), "=m" (dest[12]) ,     \
			 "=m" (dest[16]), "=m" (dest[20]), "=m" (dest[24]), "=m" (dest[28])    \
			 ::"memory", "%xmm0")

#define memset64(dest)       \
	__asm__ __volatile__(    \
			"pxor   %%xmm0, %%xmm0\n\t"    \
			"movdqu %%xmm0, %0\n\t"        \
			"movdqu %%xmm0, %1\n\t"        \
			"movdqu %%xmm0, %2\n\t"        \
			"movdqu %%xmm0, %3\n\t"        \
			"movdqu %%xmm0, %4\n\t"        \
			"movdqu %%xmm0, %5\n\t"        \
			"movdqu %%xmm0, %6\n\t"        \
			"movdqu %%xmm0, %7\n\t"        \
			"movdqu %%xmm0, %8\n\t"        \
			"movdqu %%xmm0, %9\n\t"        \
			"movdqu %%xmm0, %10\n\t"        \
			"movdqu %%xmm0, %11\n\t"        \
			"movdqu %%xmm0, %12\n\t"        \
			"movdqu %%xmm0, %13\n\t"        \
			"movdqu %%xmm0, %14\n\t"        \
			"movdqu %%xmm0, %15\n\t"        \
			:"=m" (dest[0]), "=m" (dest[4]), "=m" (dest[8]), "=m" (dest[12]) ,     \
			 "=m" (dest[16]), "=m" (dest[20]), "=m" (dest[24]), "=m" (dest[28]),   \
			 "=m" (dest[32]), "=m" (dest[36]), "=m" (dest[40]), "=m" (dest[44]),   \
			 "=m" (dest[48]), "=m" (dest[52]), "=m" (dest[56]), "=m" (dest[60])    \
			 ::"memory", "%xmm0")

#define memcpy32(dest, src)    \
	__asm__ __volatile__(      \
			"movdqu %8,  %%xmm0\n\t"    \
			"movdqu %9,  %%xmm1\n\t"    \
			"movdqu %10, %%xmm2\n\t"    \
			"movdqu %11, %%xmm3\n\t"    \
			"movdqu %12, %%xmm4\n\t"    \
			"movdqu %13, %%xmm5\n\t"    \
			"movdqu %14, %%xmm6\n\t"    \
			"movdqu %15, %%xmm7\n\t"    \
			"movdqu %%xmm0, %0\n\t"     \
			"movdqu %%xmm1, %1\n\t"     \
			"movdqu %%xmm2, %2\n\t"     \
			"movdqu %%xmm3, %3\n\t"     \
			"movdqu %%xmm4, %4\n\t"     \
			"movdqu %%xmm5, %5\n\t"     \
			"movdqu %%xmm6, %6\n\t"     \
			"movdqu %%xmm7, %7\n\t"     \
			:"=m" (dest[0]), "=m" (dest[4]), "=m" (dest[8]), "=m" (dest[12]),      \
			 "=m" (dest[16]), "=m" (dest[20]), "=m" (dest[24]), "=m" (dest[28])    \
			 :"m" (src[0]), "m" (src[4]), "m" (src[8]), "m" (src[12]),             \
              "m" (src[16]), "m" (src[20]), "m" (src[24]), "m" (src[28])           \
			  :"memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7")

/**
 * Computes the greatest common divisor
 */
constexpr __attribute__ ((const))
uint32_t gcd(uint32_t x, uint32_t y) {
    return (x%y) == 0 ? y : gcd(y, x%y);
}

constexpr uint64_t div_roundup(uint64_t v, uint32_t divisor) {
    return (v + (divisor - 1)) / divisor;
}

__attribute__ ((const))
inline bool divisibleby(uint64_t a, uint32_t x) {
    return (a % x == 0);
}

inline void checkifdivisibleby(uint64_t a, uint32_t x) {
    if (!divisibleby(a, x)) {
        std::ostringstream convert;
        convert << a << " not divisible by " << x;
        throw std::logic_error(convert.str());
    }
}

#endif /* UTIL_H_ */
