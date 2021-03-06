#include <cstdint>
#include "Portability.h"

namespace Codecs {
namespace SIMDMasks {

#if CODECS_SSE_PREREQ(2, 0)
extern const __m128i SSE2_and_msk_m128i[33] = {
	_mm_set1_epi32(0x0),	// 0-bit
	_mm_set1_epi32(0x1),	// 1-bit
	_mm_set1_epi32(0x3),	// 2-bit
	_mm_set1_epi32(0x7),	// 3-bit
	_mm_set1_epi32(0xF),	// 4-bit
	_mm_set1_epi32(0x1F),	// 5-bit
	_mm_set1_epi32(0x3F),	// 6-bit
	_mm_set1_epi32(0x7F),	// 7-bit
	_mm_set1_epi32(0xFF),	// 8-bit
	_mm_set1_epi32(0x1FF),	// 9-bit
	_mm_set1_epi32(0x3FF),	// 10-bit
	_mm_set1_epi32(0x7FF),	// 11-bit
	_mm_set1_epi32(0xFFF),	// 12-bit
	_mm_set1_epi32(0x1FFF),	// 13-bit
	_mm_set1_epi32(0x3FFF),	// 14-bit
	_mm_set1_epi32(0x7FFF),	// 15-bit
	_mm_set1_epi32(0xFFFF),	// 16-bit
	_mm_set1_epi32(0x1FFFF),	// 17-bit
	_mm_set1_epi32(0x3FFFF),	// 18-bit
	_mm_set1_epi32(0x7FFFF),	// 19-bit
	_mm_set1_epi32(0xFFFFF),	// 20-bit
	_mm_set1_epi32(0x1FFFFF),	// 21-bit
	_mm_set1_epi32(0x3FFFFF),	// 22-bit
	_mm_set1_epi32(0x7FFFFF),	// 23-bit
	_mm_set1_epi32(0xFFFFFF),	// 24-bit
	_mm_set1_epi32(0x1FFFFFF),	// 25-bit
	_mm_set1_epi32(0x3FFFFFF),	// 26-bit
	_mm_set1_epi32(0x7FFFFFF),	// 27-bit
	_mm_set1_epi32(0xFFFFFFF),	// 28-bit
	_mm_set1_epi32(0x1FFFFFFF),	// 29-bit
	_mm_set1_epi32(0x3FFFFFFF),	// 30-bit
	_mm_set1_epi32(0x7FFFFFFF),	// 31-bit
	_mm_set1_epi32(0xFFFFFFFF),	// 32-bit
};
#endif /* __SSE2__ */

#if CODECS_AVX_PREREQ(2, 0)
extern const __m256i AVX2_and_msk_m256i[33] = {
	_mm256_set1_epi32(0x0),	// 0-bit
	_mm256_set1_epi32(0x1),	// 1-bit
	_mm256_set1_epi32(0x3),	// 2-bit
	_mm256_set1_epi32(0x7),	// 3-bit
	_mm256_set1_epi32(0xF),	// 4-bit
	_mm256_set1_epi32(0x1F),	// 5-bit
	_mm256_set1_epi32(0x3F),	// 6-bit
	_mm256_set1_epi32(0x7F),	// 7-bit
	_mm256_set1_epi32(0xFF),	// 8-bit
	_mm256_set1_epi32(0x1FF),	// 9-bit
	_mm256_set1_epi32(0x3FF),	// 10-bit
	_mm256_set1_epi32(0x7FF),	// 11-bit
	_mm256_set1_epi32(0xFFF),	// 12-bit
	_mm256_set1_epi32(0x1FFF),	// 13-bit
	_mm256_set1_epi32(0x3FFF),	// 14-bit
	_mm256_set1_epi32(0x7FFF),	// 15-bit
	_mm256_set1_epi32(0xFFFF),	// 16-bit
	_mm256_set1_epi32(0x1FFFF),	// 17-bit
	_mm256_set1_epi32(0x3FFFF),	// 18-bit
	_mm256_set1_epi32(0x7FFFF),	// 19-bit
	_mm256_set1_epi32(0xFFFFF),	// 20-bit
	_mm256_set1_epi32(0x1FFFFF),	// 21-bit
	_mm256_set1_epi32(0x3FFFFF),	// 22-bit
	_mm256_set1_epi32(0x7FFFFF),	// 23-bit
	_mm256_set1_epi32(0xFFFFFF),	// 24-bit
	_mm256_set1_epi32(0x1FFFFFF),	// 25-bit
	_mm256_set1_epi32(0x3FFFFFF),	// 26-bit
	_mm256_set1_epi32(0x7FFFFFF),	// 27-bit
	_mm256_set1_epi32(0xFFFFFFF),	// 28-bit
	_mm256_set1_epi32(0x1FFFFFFF),	// 29-bit
	_mm256_set1_epi32(0x3FFFFFFF),	// 30-bit
	_mm256_set1_epi32(0x7FFFFFFF),	// 31-bit
	_mm256_set1_epi32(0xFFFFFFFF),	// 32-bit
};
#endif /* __AVX2__ */

#if CODECS_SSE_PREREQ(4, 1)
extern const char Hor_shfl_msk_char[33][2][16] = {
	{ // 0-bit
		{
		  -1, -1, -1, -1,
		  -1, -1, -1, -1,
		  -1, -1, -1, -1,
		  -1, -1, -1, -1,
		},
		{
		  -1, -1, -1, -1,
		  -1, -1, -1, -1,
		  -1, -1, -1, -1,
		  -1, -1, -1, -1,
		},
	},
	{ // 1-bit
		{
		  -1, -1, -1,  0,
		  -1, -1, -1,  0,
		  -1, -1, -1,  0,
		  -1, -1, -1,  0,
		},
		{
		  -1, -1, -1,  0,
		  -1, -1, -1,  0,
		  -1, -1, -1,  0,
		  -1, -1, -1,  0,
		},
	},
	{ // 2-bit
		{
		  -1, -1, -1,  0,
		  -1, -1, -1,  0,
		  -1, -1, -1,  0,
		  -1, -1, -1,  0,
		},
		{
		  -1, -1, -1,  1,
		  -1, -1, -1,  1,
		  -1, -1, -1,  1,
		  -1, -1, -1,  1,
		},
	},
	{ // 3-bit
		{
		  -1, -1, -1,  1,
		  -1, -1,  1,  0,
		  -1, -1, -1,  0,
		  -1, -1, -1,  0,
		},
		{
		  -1, -1, -1,  2,
		  -1, -1, -1,  2,
		  -1, -1,  2,  1,
		  -1, -1, -1,  1,
		},
	},
	{ // 4-bit
		{
		  -1, -1, -1,  1,
		  -1, -1, -1,  1,
		  -1, -1, -1,  0,
		  -1, -1, -1,  0,
		},
		{
		  -1, -1, -1,  3,
		  -1, -1, -1,  3,
		  -1, -1, -1,  2,
		  -1, -1, -1,  2,
		},
	},
	{ // 5-bit
		{
		  -1, -1,  2,  1,
		  -1, -1, -1,  1,
		  -1, -1,  1,  0,
		  -1, -1, -1,  0,
		},
		{
		  -1, -1, -1,  4,
		  -1, -1,  4,  3,
		  -1, -1, -1,  3,
		  -1, -1,  3,  2,
		},
	},
	{ // 6-bit
		{
		  -1, -1, -1,  2,
		  -1, -1,  2,  1,
		  -1, -1,  1,  0,
		  -1, -1, -1,  0,
		},
		{
		  -1, -1, -1,  5,
		  -1, -1,  5,  4,
		  -1, -1,  4,  3,
		  -1, -1, -1,  3,
		},
	},
	{ // 7-bit
		{
		  -1, -1,  3,  2,
		  -1, -1,  2,  1,
		  -1, -1,  1,  0,
		  -1, -1, -1,  0,
		},
		{
		  -1, -1, -1,  6,
		  -1, -1,  6,  5,
		  -1, -1,  5,  4,
		  -1, -1,  4,  3,
		},
	},
	{ // 8-bit
		{
		  -1, -1, -1,  3,
		  -1, -1, -1,  2,
		  -1, -1, -1,  1,
		  -1, -1, -1,  0,
		},
		{
		  -1, -1, -1,  7,
		  -1, -1, -1,  6,
		  -1, -1, -1,  5,
		  -1, -1, -1,  4,
		},
	},
	{ // 9-bit
		{
		  -1, -1,  4,  3,
		  -1, -1,  3,  2,
		  -1, -1,  2,  1,
		  -1, -1,  1,  0,
		},
		{
		  -1, -1,  8,  7,
		  -1, -1,  7,  6,
		  -1, -1,  6,  5,
		  -1, -1,  5,  4,
		},
	},
	{ // 10-bit
		{
		  -1, -1,  4,  3,
		  -1, -1,  3,  2,
		  -1, -1,  2,  1,
		  -1, -1,  1,  0,
		},
		{
		  -1, -1,  9,  8,
		  -1, -1,  8,  7,
		  -1, -1,  7,  6,
		  -1, -1,  6,  5,
		},
	},
	{ // 11-bit
		{
		  -1, -1,  5,  4,
		  -1,  4,  3,  2,
		  -1, -1,  2,  1,
		  -1, -1,  1,  0,
		},
		{
		  -1, -1, 10,  9,
		  -1, -1,  9,  8,
		  -1,  8,  7,  6,
		  -1, -1,  6,  5,
		},
	},
	{ // 12-bit
		{
		  -1, -1,  5,  4,
		  -1, -1,  4,  3,
		  -1, -1,  2,  1,
		  -1, -1,  1,  0,
		},
		{
		  -1, -1, 11, 10,
		  -1, -1, 10,  9,
		  -1, -1,  8,  7,
		  -1, -1,  7,  6,
		},
	},
	{ // 13-bit
		{
		  -1,  6,  5,  4,
		  -1, -1,  4,  3,
		  -1,  3,  2,  1,
		  -1, -1,  1,  0,
		},
		{
		  -1, -1, 12, 11,
		  -1, 11, 10,  9,
		  -1, -1,  9,  8,
		  -1,  8,  7,  6,
		},
	},
	{ // 14-bit
		{
		  -1, -1,  6,  5,
		  -1,  5,  4,  3,
		  -1,  3,  2,  1,
		  -1, -1,  1,  0,
		},
		{
		  -1, -1, 13, 12,
		  -1, 12, 11, 10,
		  -1, 10,  9,  8,
		  -1, -1,  8,  7,
		},
	},
	{ // 15-bit
		{
		  -1,  7,  6,  5,
		  -1,  5,  4,  3,
		  -1,  3,  2,  1,
		  -1, -1,  1,  0,
		},
		{
		  -1, -1, 14, 13,
		  -1, 13, 12, 11,
		  -1, 11, 10,  9,
		  -1,  9,  8,  7,
		},
	},
	{ // 16-bit
		{
		  -1, -1,  7,  6,
		  -1, -1,  5,  4,
		  -1, -1,  3,  2,
		  -1, -1,  1,  0,
		},
		{
		  -1, -1, 15, 14,
		  -1, -1, 13, 12,
		  -1, -1, 11, 10,
		  -1, -1,  9,  8,
		},
	},
	{ // 17-bit
		{
		  -1,  8,  7,  6,
		  -1,  6,  5,  4,
		  -1,  4,  3,  2,
		  -1,  2,  1,  0,
		},
		{
		  -1,  8,  7,  6,
		  -1,  6,  5,  4,
		  -1,  4,  3,  2,
		  -1,  2,  1,  0,
		},
	},
	{ // 18-bit
		{
		  -1,  8,  7,  6,
		  -1,  6,  5,  4,
		  -1,  4,  3,  2,
		  -1,  2,  1,  0,
		},
		{
		  -1,  8,  7,  6,
		  -1,  6,  5,  4,
		  -1,  4,  3,  2,
		  -1,  2,  1,  0,
		},
	},
	{ // 19-bit
		{
		  -1,  9,  8,  7,
		   7,  6,  5,  4,
		  -1,  4,  3,  2,
		  -1,  2,  1,  0,
		},
		{
		  -1,  9,  8,  7,
		  -1,  7,  6,  5,
		   5,  4,  3,  2,
		  -1,  2,  1,  0,
		},
	},
	{ // 20-bit
		{
		  -1,  9,  8,  7,
		  -1,  7,  6,  5,
		  -1,  4,  3,  2,
		  -1,  2,  1,  0,
		},
		{
		  -1,  9,  8,  7,
		  -1,  7,  6,  5,
		  -1,  4,  3,  2,
		  -1,  2,  1,  0,
		},
	},
	{ // 21-bit
		{
		  10,  9,  8,  7,
		  -1,  7,  6,  5,
		   5,  4,  3,  2,
		  -1,  2,  1,  0,
		},
		{
		  -1, 10,  9,  8,
		   8,  7,  6,  5,
		  -1,  5,  4,  3,
		   3,  2,  1,  0,
		},
	},
	{ // 22-bit
		{
		  -1, 10,  9,  8,
		   8,  7,  6,  5,
		   5,  4,  3,  2,
		  -1,  2,  1,  0,
		},
		{
		  -1, 10,  9,  8,
		   8,  7,  6,  5,
		   5,  4,  3,  2,
		  -1,  2,  1,  0,
		},
	},
	{ // 23-bit
		{
		  11, 10,  9,  8,
		   8,  7,  6,  5,
		   5,  4,  3,  2,
		  -1,  2,  1,  0,
		},
		{
		  -1, 11, 10,  9,
		   9,  8,  7,  6,
		   6,  5,  4,  3,
		   3,  2,  1,  0,
		},
	},
	{ // 24-bit
		{
		  -1, 11, 10,  9,
		  -1,  8,  7,  6,
		  -1,  5,  4,  3,
		  -1,  2,  1,  0,
		},
		{
		  -1, 11, 10,  9,
		  -1,  8,  7,  6,
		  -1,  5,  4,  3,
		  -1,  2,  1,  0,
		},
	},
	{ // 25-bit
		{
		  12, 11, 10,  9,
		   9,  8,  7,  6,
		   6,  5,  4,  3,
		   3,  2,  1,  0,
		},
		{
		  12, 11, 10,  9,
		   9,  8,  7,  6,
		   6,  5,  4,  3,
		   3,  2,  1,  0,
		},
	},
	{ // 26-bit
		{
		  12, 11, 10,  9,
		   9,  8,  7,  6,
		   6,  5,  4,  3,
		   3,  2,  1,  0,
		},
		{
		  12, 11, 10,  9,
		   9,  8,  7,  6,
		   6,  5,  4,  3,
		   3,  2,  1,  0,
		},
	},
	{ // 27-bit
		{
		  13, 12, 11, 10,
		   9,  8,  7,  6,
		   6,  5,  4,  3,
		   3,  2,  1,  0,
		},
		{
		  13, 12, 11, 10,
		  10,  9,  8,  7,
		   7,  6,  5,  4,
		   3,  2,  1,  0,
		},
	},
	{ // 28-bit
		{
		  13, 12, 11, 10,
		  10,  9,  8,  7,
		   6,  5,  4,  3,
		   3,  2,  1,  0,
		},
		{
		  13, 12, 11, 10,
		  10,  9,  8,  7,
		   6,  5,  4,  3,
		   3,  2,  1,  0,
		},
	},
	{ // 29-bit
		{
		  14, 13, 12, 11,
		  10,  9,  8,  7,
		   7,  6,  5,  4,
		   3,  2,  1,  0,
		},
		{
		  14, 13, 12, 11,
		  10,  9,  8,  7,
		   7,  6,  5,  4,
		   3,  2,  1,  0,
		},
	},
	{ // 30-bit
		{
		  14, 13, 12, 11,
		  10,  9,  8,  7,
		   7,  6,  5,  4,
		   3,  2,  1,  0,
		},
		{
		  14, 13, 12, 11,
		  10,  9,  8,  7,
		   7,  6,  5,  4,
		   3,  2,  1,  0,
		},
	},
	{ // 31-bit
		{
		  14, 13, 12, 11,
		  10,  9,  8,  7,
		  -1, -1, -1, -1,
		  -1, -1, -1, -1,
		},
		{
		  -1, -1, -1, -1,
		  -1, -1, -1, -1,
		   8,  7,  6,  5,
		   4,  3,  2,  1,
		},
	},
	{ // 32-bit
		{
		  15, 14, 13, 12,
		  11, 10,  9,  8,
		   7,  6,  5,  4,
		   3,  2,  1,  0,
		},
		{
		  15, 14, 13, 12,
		  11, 10,  9,  8,
		   7,  6,  5,  4,
		   3,  2,  1,  0,
		},
	},
};

extern const __m128i Hor_SSE4_mul_msk_m128i[33][2] = {
	{ _mm_set_epi32(0x01, 0x01, 0x01, 0x01), _mm_set_epi32(0x01, 0x01, 0x01, 0x01) },	// 0-bit
	{ _mm_set_epi32(0x01, 0x02, 0x04, 0x08), _mm_set_epi32(0x01, 0x02, 0x04, 0x08) },	// 1-bit
	{ _mm_set_epi32(0x01, 0x04, 0x10, 0x40), _mm_set_epi32(0x01, 0x04, 0x10, 0x40) },	// 2-bit
	{ _mm_set_epi32(0x20, 0x01, 0x08, 0x40), _mm_set_epi32(0x04, 0x20, 0x01, 0x08) },	// 3-bit
	{ _mm_set_epi32(0x01, 0x10, 0x01, 0x10), _mm_set_epi32(0x01, 0x10, 0x01, 0x10) },	// 4-bit
	{ _mm_set_epi32(0x01, 0x20, 0x04, 0x80), _mm_set_epi32(0x08, 0x01, 0x20, 0x04) },	// 5-bit
	{ _mm_set_epi32(0x10, 0x04, 0x01, 0x40), _mm_set_epi32(0x10, 0x04, 0x01, 0x40) },	// 6-bit
	{ _mm_set_epi32(0x04, 0x02, 0x01, 0x80), _mm_set_epi32(0x08, 0x04, 0x02, 0x01) },	// 7-bit
	{ _mm_set_epi32(0x01, 0x01, 0x01, 0x01), _mm_set_epi32(0x01, 0x01, 0x01, 0x01) },	// 8-bit
	{ _mm_set_epi32(0x01, 0x02, 0x04, 0x08), _mm_set_epi32(0x01, 0x02, 0x04, 0x08) },	// 9-bit
	{ _mm_set_epi32(0x01, 0x04, 0x10, 0x40), _mm_set_epi32(0x01, 0x04, 0x10, 0x40) },	// 10-bit
	{ _mm_set_epi32(0x20, 0x01, 0x08, 0x40), _mm_set_epi32(0x04, 0x20, 0x01, 0x08) },	// 11-bit
	{ _mm_set_epi32(0x01, 0x10, 0x01, 0x10), _mm_set_epi32(0x01, 0x10, 0x01, 0x10) },	// 12-bit
	{ _mm_set_epi32(0x01, 0x20, 0x04, 0x80), _mm_set_epi32(0x08, 0x01, 0x20, 0x04) },	// 13-bit
	{ _mm_set_epi32(0x10, 0x04, 0x01, 0x40), _mm_set_epi32(0x10, 0x04, 0x01, 0x40) },	// 14-bit
	{ _mm_set_epi32(0x04, 0x02, 0x01, 0x80), _mm_set_epi32(0x08, 0x04, 0x02, 0x01) },	// 15-bit
	{ _mm_set_epi32(0x01, 0x01, 0x01, 0x01), _mm_set_epi32(0x01, 0x01, 0x01, 0x01) },	// 16-bit
	{ _mm_set_epi32(0x01, 0x02, 0x04, 0x08), _mm_set_epi32(0x01, 0x02, 0x04, 0x08) },	// 17-bit
	{ _mm_set_epi32(0x01, 0x04, 0x10, 0x40), _mm_set_epi32(0x01, 0x04, 0x10, 0x40) },	// 18-bit
	{ _mm_set_epi32(0x20, 0x01, 0x08, 0x40), _mm_set_epi32(0x04, 0x20, 0x01, 0x08) },	// 19-bit
	{ _mm_set_epi32(0x01, 0x10, 0x01, 0x10), _mm_set_epi32(0x01, 0x10, 0x01, 0x10) },	// 20-bit
	{ _mm_set_epi32(0x01, 0x20, 0x04, 0x80), _mm_set_epi32(0x08, 0x01, 0x20, 0x04) },	// 21-bit
	{ _mm_set_epi32(0x10, 0x04, 0x01, 0x40), _mm_set_epi32(0x10, 0x04, 0x01, 0x40) },	// 22-bit
	{ _mm_set_epi32(0x04, 0x02, 0x01, 0x80), _mm_set_epi32(0x08, 0x04, 0x02, 0x01) },	// 23-bit
	{ _mm_set_epi32(0x01, 0x01, 0x01, 0x01), _mm_set_epi32(0x01, 0x01, 0x01, 0x01) },	// 24-bit
	{ _mm_set_epi32(0x01, 0x02, 0x04, 0x08), _mm_set_epi32(0x01, 0x02, 0x04, 0x08) },	// 25-bit
	{ _mm_set_epi32(0x01, 0x04, 0x10, 0x40), _mm_set_epi32(0x01, 0x04, 0x10, 0x40) },	// 26-bit
	{ _mm_set_epi32(0x04, 0x08, 0x01, 0x08), _mm_set_epi32(0x01, 0x08, 0x20, 0x02) },	// 27-bit
	{ _mm_set_epi32(0x01, 0x10, 0x01, 0x10), _mm_set_epi32(0x01, 0x10, 0x01, 0x10) },	// 28-bit
	{ _mm_set_epi32(0x01, 0x01, 0x04, 0x04), _mm_set_epi32(0x01, 0x01, 0x04, 0x04) },	// 29-bit
	{ _mm_set_epi32(0x01, 0x01, 0x04, 0x04), _mm_set_epi32(0x01, 0x01, 0x04, 0x04) },	// 30-bit
	{ _mm_set_epi32(0x01, 0x01, 0x01, 0x01), _mm_set_epi32(0x01, 0x02, 0x02, 0x02) },	// 31-bit
	{ _mm_set_epi32(0x01, 0x01, 0x01, 0x01), _mm_set_epi32(0x01, 0x01, 0x01, 0x01) },	// 32-bit
};

extern const int Hor_SSE4_srli_imm_int[33][2] = {
	{ 0, 0 },	// 0-bit
	{ 3, 7 },	// 1-bit
	{ 6, 6 },	// 2-bit
	{ 6, 7 },	// 3-bit
	{ 4, 4 },	// 4-bit
	{ 7, 6 },	// 5-bit
	{ 6, 6 },	// 6-bit
	{ 7, 4 },	// 7-bit
	{ 0, 0 },	// 8-bit
	{ 3, 7 },	// 9-bit
	{ 6, 6 },	// 10-bit
	{ 6, 7 },	// 11-bit
	{ 4, 4 },	// 12-bit
	{ 7, 6 },	// 13-bit
	{ 6, 6 },	// 14-bit
	{ 7, 4 },	// 15-bit
	{ 0, 0 },	// 16-bit
	{ 3, 7 },	// 17-bit
	{ 6, 6 },	// 18-bit
	{ 6, 7 },	// 19-bit
	{ 4, 4 },	// 20-bit
	{ 7, 6 },	// 21-bit
	{ 6, 6 },	// 22-bit
	{ 7, 4 },	// 23-bit
	{ 0, 0 },	// 24-bit
	{ 3, 7 },	// 25-bit
	{ 6, 6 },	// 26-bit
	{ 3, 5 },	// 27-bit
	{ 4, 4 },	// 28-bit
	{ 2, 3 },	// 29-bit
	{ 2, 2 },	// 30-bit
	{ 0, 1 },	// 31-bit
	{ 0, 0 },	// 32-bit
};
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
extern const __m256i Hor_AVX2_shfl_msk_m256i[33] = {
	 // 0-bit
	_mm256_set_epi8(
		-1, -1, -1, -1,
		-1, -1, -1, -1,
		-1, -1, -1, -1,
		-1, -1, -1, -1,
		-1, -1, -1, -1,
		-1, -1, -1, -1,
		-1, -1, -1, -1,
		-1, -1, -1, -1),
	 // 1-bit
	_mm256_set_epi8(
		-1, -1, -1,  0,
		-1, -1, -1,  0,
		-1, -1, -1,  0,
		-1, -1, -1,  0,
		-1, -1, -1,  0,
		-1, -1, -1,  0,
		-1, -1, -1,  0,
		-1, -1, -1,  0),
	 // 2-bit
	_mm256_set_epi8(
		-1, -1, -1,  1,
		-1, -1, -1,  1,
		-1, -1, -1,  1,
		-1, -1, -1,  1,
		-1, -1, -1,  0,
		-1, -1, -1,  0,
		-1, -1, -1,  0,
		-1, -1, -1,  0),
	 // 3-bit
	_mm256_set_epi8(
		-1, -1, -1,  2,
		-1, -1, -1,  2,
		-1, -1,  2,  1,
		-1, -1, -1,  1,
		-1, -1, -1,  1,
		-1, -1,  1,  0,
		-1, -1, -1,  0,
		-1, -1, -1,  0),
	 // 4-bit
	_mm256_set_epi8(
		-1, -1, -1,  3,
		-1, -1, -1,  3,
		-1, -1, -1,  2,
		-1, -1, -1,  2,
		-1, -1, -1,  1,
		-1, -1, -1,  1,
		-1, -1, -1,  0,
		-1, -1, -1,  0),
	 // 5-bit
	_mm256_set_epi8(
		-1, -1, -1,  4,
		-1, -1,  4,  3,
		-1, -1, -1,  3,
		-1, -1,  3,  2,
		-1, -1,  2,  1,
		-1, -1, -1,  1,
		-1, -1,  1,  0,
		-1, -1, -1,  0),
	 // 6-bit
	_mm256_set_epi8(
		-1, -1, -1,  5,
		-1, -1,  5,  4,
		-1, -1,  4,  3,
		-1, -1, -1,  3,
		-1, -1, -1,  2,
		-1, -1,  2,  1,
		-1, -1,  1,  0,
		-1, -1, -1,  0),
	 // 7-bit
	_mm256_set_epi8(
		-1, -1, -1,  6,
		-1, -1,  6,  5,
		-1, -1,  5,  4,
		-1, -1,  4,  3,
		-1, -1,  3,  2,
		-1, -1,  2,  1,
		-1, -1,  1,  0,
		-1, -1, -1,  0),
	 // 8-bit
	_mm256_set_epi8(
		-1, -1, -1,  7,
		-1, -1, -1,  6,
		-1, -1, -1,  5,
		-1, -1, -1,  4,
		-1, -1, -1,  3,
		-1, -1, -1,  2,
		-1, -1, -1,  1,
		-1, -1, -1,  0),
	 // 9-bit
	_mm256_set_epi8(
		-1, -1,  8,  7,
		-1, -1,  7,  6,
		-1, -1,  6,  5,
		-1, -1,  5,  4,
		-1, -1,  4,  3,
		-1, -1,  3,  2,
		-1, -1,  2,  1,
		-1, -1,  1,  0),
	 // 10-bit
	_mm256_set_epi8(
		-1, -1,  9,  8,
		-1, -1,  8,  7,
		-1, -1,  7,  6,
		-1, -1,  6,  5,
		-1, -1,  4,  3,
		-1, -1,  3,  2,
		-1, -1,  2,  1,
		-1, -1,  1,  0),
	 // 11-bit
	_mm256_set_epi8(
		-1, -1, 10,  9,
		-1, -1,  9,  8,
		-1,  8,  7,  6,
		-1, -1,  6,  5,
		-1, -1,  5,  4,
		-1,  4,  3,  2,
		-1, -1,  2,  1,
		-1, -1,  1,  0),
	 // 12-bit
	_mm256_set_epi8(
		-1, -1, 11, 10,
		-1, -1, 10,  9,
		-1, -1,  8,  7,
		-1, -1,  7,  6,
		-1, -1,  5,  4,
		-1, -1,  4,  3,
		-1, -1,  2,  1,
		-1, -1,  1,  0),
	 // 13-bit
	_mm256_set_epi8(
		-1, -1, 12, 11,
		-1, 11, 10,  9,
		-1, -1,  9,  8,
		-1,  8,  7,  6,
		-1,  6,  5,  4,
		-1, -1,  4,  3,
		-1,  3,  2,  1,
		-1, -1,  1,  0),
	 // 14-bit
	_mm256_set_epi8(
		-1, -1, 13, 12,
		-1, 12, 11, 10,
		-1, 10,  9,  8,
		-1, -1,  8,  7,
		-1, -1,  6,  5,
		-1,  5,  4,  3,
		-1,  3,  2,  1,
		-1, -1,  1,  0),
	 // 15-bit
	_mm256_set_epi8(
		-1, -1, 14, 13,
		-1, 13, 12, 11,
		-1, 11, 10,  9,
		-1,  9,  8,  7,
		-1,  7,  6,  5,
		-1,  5,  4,  3,
		-1,  3,  2,  1,
		-1, -1,  1,  0),
	 // 16-bit
	_mm256_set_epi8(
		-1, -1, 15, 14,
		-1, -1, 13, 12,
		-1, -1, 11, 10,
		-1, -1,  9,  8,
		-1, -1,  7,  6,
		-1, -1,  5,  4,
		-1, -1,  3,  2,
		-1, -1,  1,  0),
	 // 17-bit
	_mm256_set_epi8(
		-1,  8,  7,  6,
		-1,  6,  5,  4,
		-1,  4,  3,  2,
		-1,  2,  1,  0,
		-1,  8,  7,  6,
		-1,  6,  5,  4,
		-1,  4,  3,  2,
		-1,  2,  1,  0),
	 // 18-bit
	_mm256_set_epi8(
		-1,  8,  7,  6,
		-1,  6,  5,  4,
		-1,  4,  3,  2,
		-1,  2,  1,  0,
		-1,  8,  7,  6,
		-1,  6,  5,  4,
		-1,  4,  3,  2,
		-1,  2,  1,  0),
	 // 19-bit
	_mm256_set_epi8(
		-1,  9,  8,  7,
		-1,  7,  6,  5,
		 5,  4,  3,  2,
		-1,  2,  1,  0,
		-1,  9,  8,  7,
		 7,  6,  5,  4,
		-1,  4,  3,  2,
		-1,  2,  1,  0),
	 // 20-bit
	_mm256_set_epi8(
		-1,  9,  8,  7,
		-1,  7,  6,  5,
		-1,  4,  3,  2,
		-1,  2,  1,  0,
		-1,  9,  8,  7,
		-1,  7,  6,  5,
		-1,  4,  3,  2,
		-1,  2,  1,  0),
	 // 21-bit
	_mm256_set_epi8(
		-1, 10,  9,  8,
		 8,  7,  6,  5,
		-1,  5,  4,  3,
		 3,  2,  1,  0,
		10,  9,  8,  7,
		-1,  7,  6,  5,
		 5,  4,  3,  2,
		-1,  2,  1,  0),
	 // 22-bit
	_mm256_set_epi8(
		-1, 10,  9,  8,
		 8,  7,  6,  5,
		 5,  4,  3,  2,
		-1,  2,  1,  0,
		-1, 10,  9,  8,
		 8,  7,  6,  5,
		 5,  4,  3,  2,
		-1,  2,  1,  0),
	 // 23-bit
	_mm256_set_epi8(
		-1, 11, 10,  9,
		 9,  8,  7,  6,
		 6,  5,  4,  3,
		 3,  2,  1,  0,
		11, 10,  9,  8,
		 8,  7,  6,  5,
		 5,  4,  3,  2,
		-1,  2,  1,  0),
	 // 24-bit
	_mm256_set_epi8(
		-1, 11, 10,  9,
		-1,  8,  7,  6,
		-1,  5,  4,  3,
		-1,  2,  1,  0,
		-1, 11, 10,  9,
		-1,  8,  7,  6,
		-1,  5,  4,  3,
		-1,  2,  1,  0),
	 // 25-bit
	_mm256_set_epi8(
		12, 11, 10,  9,
		 9,  8,  7,  6,
		 6,  5,  4,  3,
		 3,  2,  1,  0,
		12, 11, 10,  9,
		 9,  8,  7,  6,
		 6,  5,  4,  3,
		 3,  2,  1,  0),
	 // 26-bit
	_mm256_set_epi8(
		12, 11, 10,  9,
		 9,  8,  7,  6,
		 6,  5,  4,  3,
		 3,  2,  1,  0,
		12, 11, 10,  9,
		 9,  8,  7,  6,
		 6,  5,  4,  3,
		 3,  2,  1,  0),
	 // 27-bit
	_mm256_set_epi8(
		13, 12, 11, 10,
		10,  9,  8,  7,
		 7,  6,  5,  4,
		 3,  2,  1,  0,
		13, 12, 11, 10,
		 9,  8,  7,  6,
		 6,  5,  4,  3,
		 3,  2,  1,  0),
	 // 28-bit
	_mm256_set_epi8(
		13, 12, 11, 10,
		10,  9,  8,  7,
		 6,  5,  4,  3,
		 3,  2,  1,  0,
		13, 12, 11, 10,
		10,  9,  8,  7,
		 6,  5,  4,  3,
		 3,  2,  1,  0),
	 // 29-bit
	_mm256_set_epi8(
		14, 13, 12, 11,
		10,  9,  8,  7,
		 7,  6,  5,  4,
		 3,  2,  1,  0,
		14, 13, 12, 11,
		10,  9,  8,  7,
		 7,  6,  5,  4,
		 3,  2,  1,  0),
	 // 30-bit
	_mm256_set_epi8(
		14, 13, 12, 11,
		10,  9,  8,  7,
		 7,  6,  5,  4,
		 3,  2,  1,  0,
		14, 13, 12, 11,
		10,  9,  8,  7,
		 7,  6,  5,  4,
		 3,  2,  1,  0),
	 // 31-bit
	_mm256_set_epi8(
		-1, -1, -1, -1,
		-1, -1, -1, -1,
		 8,  7,  6,  5,
		 4,  3,  2,  1,
		14, 13, 12, 11,
		10,  9,  8,  7,
		-1, -1, -1, -1,
		-1, -1, -1, -1),
	 // 32-bit
	_mm256_set_epi8(
		15, 14, 13, 12,
		11, 10,  9,  8,
		 7,  6,  5,  4,
		 3,  2,  1,  0,
		15, 14, 13, 12,
		11, 10,  9,  8,
		 7,  6,  5,  4,
		 3,  2,  1,  0),
};

extern const __m256i Hor_AVX2_srlv_msk_m256i[33] = {
	_mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0), // 0-bit
	_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0), // 1-bit
	_mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0), // 2-bit
	_mm256_set_epi32(5, 2, 7, 4, 1, 6, 3, 0), // 3-bit
	_mm256_set_epi32(4, 0, 4, 0, 4, 0, 4, 0), // 4-bit
	_mm256_set_epi32(3, 6, 1, 4, 7, 2, 5, 0), // 5-bit
	_mm256_set_epi32(2, 4, 6, 0, 2, 4, 6, 0), // 6-bit
	_mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 0), // 7-bit
	_mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0), // 8-bit
	_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0), // 9-bit
	_mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0), // 10-bit
	_mm256_set_epi32(5, 2, 7, 4, 1, 6, 3, 0), // 11-bit
	_mm256_set_epi32(4, 0, 4, 0, 4, 0, 4, 0), // 12-bit
	_mm256_set_epi32(3, 6, 1, 4, 7, 2, 5, 0), // 13-bit
	_mm256_set_epi32(2, 4, 6, 0, 2, 4, 6, 0), // 14-bit
	_mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 0), // 15-bit
	_mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0), // 16-bit
	_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0), // 17-bit
	_mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0), // 18-bit
	_mm256_set_epi32(5, 2, 7, 4, 1, 6, 3, 0), // 19-bit
	_mm256_set_epi32(4, 0, 4, 0, 4, 0, 4, 0), // 20-bit
	_mm256_set_epi32(3, 6, 1, 4, 7, 2, 5, 0), // 21-bit
	_mm256_set_epi32(2, 4, 6, 0, 2, 4, 6, 0), // 22-bit
	_mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 0), // 23-bit
	_mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0), // 24-bit
	_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0), // 25-bit
	_mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0), // 26-bit
	_mm256_set_epi32(5, 2, 0, 5, 0, 5, 3, 0), // 27-bit
	_mm256_set_epi32(4, 0, 4, 0, 4, 0, 4, 0), // 28-bit
	_mm256_set_epi32(0, 3, 0, 3, 0, 3, 0, 3), // 29-bit
	_mm256_set_epi32(0, 2, 0, 2, 0, 2, 0, 2), // 30-bit
	_mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1), // 31-bit
	_mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0), // 32-bit
};
#endif /* __AVX2__ */

#if CODECS_SSE_PREREQ(4, 1)
extern const __m128i Simple_SSE4_mul_msk_m128i[9] = {
	_mm_set_epi32(0x000008, 0x000004, 0x000002, 0x000001),	// 28 * 1-bit
	_mm_set_epi32(0x000040, 0x000010, 0x000004, 0x000001),	// 14 * 2-bit
	_mm_set_epi32(0x000200, 0x000040, 0x000008, 0x000001),	// 9 * 3-bit
	_mm_set_epi32(0x001000, 0x000100, 0x000010, 0x000001),	// 7 * 4-bit
	_mm_set_epi32(0x008000, 0x000400, 0x000020, 0x000001),	// 5 * 5-bit
	_mm_set_epi32(0x200000, 0x004000, 0x000080, 0x000001),	// 4 * 7-bit
	_mm_set_epi32(0x000001, 0x040000, 0x000200, 0x000001),	// 3 * 9-bit
	_mm_set_epi32(0x000001, 0x000001, 0x004000, 0x000001),	// 2 * 14-bit
	_mm_set_epi32(0x000001, 0x000001, 0x000001, 0x000001),	// 1 * 28-bit
};
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
extern const __m256i Simple_AVX2_srlv_msk_m256i[9][4] = {
	{ // 28 * 1
	  _mm256_set_epi32(20, 21, 22, 23, 24, 25, 26, 27),
	  _mm256_set_epi32(12, 13, 14, 15, 16, 17, 18, 19),
	  _mm256_set_epi32( 4,  5,  6,  7,  8,  9, 10, 11),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  1,  2,  3),
	},
	{ // 14 * 2
	  _mm256_set_epi32(12, 14, 16, 18, 20, 22, 24, 26),
	  _mm256_set_epi32( 0,  0,  0,  2,  4,  6,  8, 10),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	},
	{ // 9 * 3
	  _mm256_set_epi32( 4,  7, 10, 13, 16, 19, 22, 25),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  1),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	},
	{ // 7 * 4
	  _mm256_set_epi32( 0,  0,  4,  8, 12, 16, 20, 24),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	},
	{ // 5 * 5
	  _mm256_set_epi32( 0,  0,  0,  3,  8, 13, 18, 23),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	},
	{ // 4 * 7
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  7, 14, 21),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	},
	{ // 3 * 9
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  1, 10, 19),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	},
	{ // 2 * 14
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0, 14),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	},
	{ // 1 * 28
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	  _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
	},
};
#endif /* __AVX2__ */

} // namespace SIMDMasks
} // namespace Codecs
