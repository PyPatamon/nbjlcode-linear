/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Mar 15, 2016
 */
/**
 * Based on code by
 *     Facebook, https://github.com/facebook/folly
 * which was available under the Apache License, Version 2.0.
 */

#ifndef CODECS_PORTABILITY_H_
#define CODECS_PORTABILITY_H_

#include <cstdint>

// always inline
#ifdef _MSC_VER
# define CODECS_ALWAYS_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
# define CODECS_ALWAYS_INLINE inline __attribute__((__always_inline__))
#else
# define CODECS_ALWAYS_INLINE inline
#endif

// detection for 64 bit
#if defined(__x86_64__) || defined(_M_X64)
# define CODECS_X64 1
#else
# define CODECS_X64 0
#endif

#if defined (__powerpc64)
# define CODECS_PPC64 1
#else
# define CODECS_PPC64 0
#endif

// detection for AVX instruction sets
#ifndef CODECS_AVX
# if defined(__AVX2__)
#  define CODECS_AVX 2
#  define CODECS_AVX_MINOR 0
# elif defined(__AVX__)
#  define CODECS_AVX 1
#  define CODECS_AVX_MINOR 0
# else
#  define CODECS_AVX 0
#  define CODECS_AVX_MINOR 0
# endif
#endif

#define CODECS_AVX_PREREQ(major, minor) \
    (CODECS_AVX > major || CODECS_AVX == major && CODECS_AVX_MINOR >= minor)

// detection for SSE instruction sets
#ifndef CODECS_SSE
# if defined(__SSE4_2__)
#  define CODECS_SSE 4
#  define CODECS_SSE_MINOR 2
# elif defined(__SSE4_1__)
#  define CODECS_SSE 4
#  define CODECS_SSE_MINOR 1
# elif defined(__SSSE3__)
#  define CODECS_SSE 3
#  define CODECS_SSE_MINOR 1
# elif defined(__SSE3__)
#  define CODECS_SSE 3
#  define CODECS_SSE_MINOR 0
# elif defined(__SSE2__)
#  define CODECS_SSE 2
#  define CODECS_SSE_MINOR 0
# elif defined(__SSE__)
#  define CODECS_SSE 1
#  define CODECS_SSE_MINOR 0
# else
#  define CODECS_SSE 0
#  define CODECS_SSE_MINOR 0
# endif
#endif

#define CODECS_SSE_PREREQ(major, minor) \
    (CODECS_SSE > major || CODECS_SSE == major && CODECS_SSE_MINOR >= minor)


#if CODECS_AVX_PREREQ(1, 0) || CODECS_SSE_PREREQ(2, 0)
# if defined(__GNUC__) && (CODECS_X64 || defined(__i386__))
#  include<x86intrin.h>
# elif defined(_MSC_VER)
#  include<intrin.h>
# endif
#endif /* __AVX__ || __SSE2__ */

struct Scalar {
	using datatype = uint32_t;
};

#if CODECS_SSE_PREREQ(2, 0)
struct SSE {
	using datatype = __m128i;
};
#endif /* __SSE2__ */

#if CODECS_AVX_PREREQ(1, 0)
struct AVX {
	using datatype = __m256i;
};
#endif /* __AVX__ */

#endif // CODECS_PORTABILITY_H_ 
