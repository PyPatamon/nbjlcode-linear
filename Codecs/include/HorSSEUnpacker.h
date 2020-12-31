/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Mar 23, 2016
 */

#ifndef CODECS_HORSSEUNPACKER_H_ 
#define CODECS_HORSSEUNPACKER_H_ 


/**
 * SSE4-based unpacking 128 0-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c0(__m128i *  __restrict__  out,
		const __m128i * __restrict__ in) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		uint32_t *outPtr = reinterpret_cast<uint32_t *>(out);
		for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
			memset64(outPtr);
			outPtr += 64;
		}
	}
	else { // For Rice and OptRice.
		uint32_t *outPtr = reinterpret_cast<uint32_t *>(out);
		const uint32_t *quoPtr = reinterpret_cast<const uint32_t *>(quotient);
		for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
			memcpy32(outPtr, quoPtr);
            outPtr += 32;
            quoPtr += 32;
        }
	}
}


/**
 * SSE4-based unpacking 128 1-bit values.
 * Load 1 SSE vectors, each containing 128 1-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c1(__m128i *  __restrict__  out,
		const __m128i * __restrict__ in) {
	__m128i c1_load_rslt_m128i = _mm_loadu_si128(in); // 16 bytes; contains 128 values.
	hor_sse4_unpack16_c1<0>(out, c1_load_rslt_m128i);  // Unpack 1st 16 values.
	hor_sse4_unpack16_c1<2>(out, c1_load_rslt_m128i);  // Unpack 2nd 16 values.
	hor_sse4_unpack16_c1<4>(out, c1_load_rslt_m128i);  // Unpack 3rd 16 values.
	hor_sse4_unpack16_c1<6>(out, c1_load_rslt_m128i);  // Unpack 4th 16 values.
	hor_sse4_unpack16_c1<8>(out, c1_load_rslt_m128i);  // Unpack 5th 16 values.
	hor_sse4_unpack16_c1<10>(out, c1_load_rslt_m128i); // Unpack 6th 16 values.
	hor_sse4_unpack16_c1<12>(out, c1_load_rslt_m128i); // Unpack 7th 16 values.
	hor_sse4_unpack16_c1<14>(out, c1_load_rslt_m128i); // Unpack 8th 16 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack16_c1(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	const __m128i Hor_SSE4_c1_shfl_msk_m128i = _mm_set_epi8(
			byte + 1, byte + 1, byte + 1, byte + 1,
			byte + 1, byte + 1, byte + 1, byte + 1,
			byte, byte, byte, byte,
			byte, byte, byte, byte);
	__m128i c1_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c1_shfl_msk_m128i);

	const __m128i Hor_SSE4_c1_and_msk_m128i = _mm_set_epi32(0x80402010, 0x08040201, 0x80402010, 0x08040201);
	__m128i c1_and_rslt_m128i = _mm_and_si128(c1_shfl_rslt_m128i, Hor_SSE4_c1_and_msk_m128i);
	__m128i c1_cmpeq_rslt_m128i = _mm_cmpeq_epi8(c1_and_rslt_m128i, Hor_SSE4_c1_and_msk_m128i);
	c1_and_rslt_m128i = _mm_and_si128(c1_cmpeq_rslt_m128i, _mm_set1_epi8(0x01));

	hor_sse4_unpack4_c1<0>(out, c1_and_rslt_m128i);  // Unpack 1st 4 values.
	hor_sse4_unpack4_c1<4>(out, c1_and_rslt_m128i);  // Unpack 2nd 4 values.
	hor_sse4_unpack4_c1<8>(out, c1_and_rslt_m128i);  // Unpack 3rd 4 values.
	hor_sse4_unpack4_c1<12>(out, c1_and_rslt_m128i); // Unpack 4th 4 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c1(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m128i Hor_SSE4_c1_shf_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 3,
				0xFF, 0xFF, 0xFF, byte + 2,
				0xFF, 0xFF, 0xFF, byte + 1,
				0xFF, 0xFF, 0xFF, byte);
		__m128i c1_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c1_shf_msk_m128i);
		_mm_storeu_si128(out++, c1_rslt_m128i);
	}
	else { // For Rice and OptRice.
		const __m128i Hor_SSE4_c1_shf_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 3,
				0xFF, 0xFF, 0xFF, byte + 2,
				0xFF, 0xFF, 0xFF, byte + 1,
				0xFF, 0xFF, 0xFF, byte);
		__m128i c1_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c1_shf_msk_m128i);
		__m128i c1_rslt_m128i = _mm_or_si128(c1_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 1));
		_mm_storeu_si128(out++, c1_rslt_m128i);
	}
}


///**
// * SSE4-based unpacking 128 1-bit values.
// * Load 1 SSE vectors, each containing 128 1-bit values.
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c1(__m128i *  __restrict__  out,
//		const __m128i * __restrict__ in) {
//	__m128i c1_load_rslt_m128i = _mm_loadu_si128(in);
//	hor_sse4_unpack32_c1<0x00>(out, c1_load_rslt_m128i); // Unpack 1st 32 values.
//	hor_sse4_unpack32_c1<0x55>(out, c1_load_rslt_m128i); // Unpack 2nd 32 values.
//	hor_sse4_unpack32_c1<0xAA>(out, c1_load_rslt_m128i); // Unpack 3rd 32 values.
//	hor_sse4_unpack32_c1<0xFF>(out, c1_load_rslt_m128i); // Unpack 4th 32 values.
//}
//
//template <bool IsRiceCoding>
//template <int imm8>
//void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack32_c1(__m128i *  __restrict__  &out,
//        const __m128i &InReg) {
//    __m128i c1_shfl_rslt_m128i = _mm_shuffle_epi32(InReg, imm8);
//    __m128i c1_mul_rslt_m128i = _mm_mullo_epi32(c1_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[1][0]);
//
//    hor_sse4_unpack4_c1<3>(out, c1_mul_rslt_m128i);             // Unpack 1st 4 values.
//    hor_sse4_unpack4_c1<7>(out, c1_mul_rslt_m128i);             // Unpack 2nd 4 values.
//    hor_sse4_unpack4_c1<11>(out, c1_mul_rslt_m128i);            // Unpack 3rd 4 values.
//    hor_sse4_unpack4_c1<15>(out, c1_mul_rslt_m128i);            // Unpack 4th 4 values.
//    hor_sse4_unpack4_c1<19>(out, c1_mul_rslt_m128i);            // Unpack 5th 4 values.
//    hor_sse4_unpack4_c1<23>(out, c1_mul_rslt_m128i);            // Unpack 6th 4 values.
//    hor_sse4_unpack4_c1<27>(out, c1_mul_rslt_m128i);            // Unpack 7th 4 values.
//    hor_sse4_unpackwithoutmask4_c1<31>(out, c1_mul_rslt_m128i); // Unpack 8th 4 values.
//}
//
//template <bool IsRiceCoding>
//template <int bit>
//void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c1(__m128i *  __restrict__  &out,
//        const __m128i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m128i c1_srli_rslt_m128i = _mm_srli_epi32(InReg, bit);
//		__m128i c1_rslt_m128i = _mm_and_si128(c1_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[1]);
//		_mm_storeu_si128(out++, c1_rslt_m128i);
//	}
//	else { // For Rice and OptRice.
//		__m128i c1_srli_rslt_m128i = _mm_srli_epi32(InReg, bit);
//		__m128i c1_and_rslt_m128i = _mm_and_si128(c1_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[1]);
//		__m128i c1_rslt_m128i = _mm_or_si128(c1_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 1));
//		_mm_storeu_si128(out++, c1_rslt_m128i);
//	}
//}
//
//template <bool IsRiceCoding>
//template <int bit>
//void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpackwithoutmask4_c1(__m128i *  __restrict__  &out,
//        const __m128i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m128i c1_rslt_m128i = _mm_srli_epi32(InReg, bit);
//		_mm_storeu_si128(out++, c1_rslt_m128i);
//	}
//	else { // For Rice and OptRice.
//		__m128i c1_srli_rslt_m128i = _mm_srli_epi32(InReg, bit);
//		__m128i c1_rslt_m128i = _mm_or_si128(c1_srli_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 1));
//		_mm_storeu_si128(out++, c1_rslt_m128i);
//	}
//}


/**
 * SSE4-based unpacking 128 2-bit values.
 * Load 2 SSE vectors, each containing 64 2-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c2(__m128i *  __restrict__  out,
		const __m128i * __restrict__ in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
		__m128i c2_load_rslt_m128i = _mm_loadu_si128(in++);
		hor_sse4_unpack16_c2<0x00>(out, c2_load_rslt_m128i); // Unpack 1st 16 values.
		hor_sse4_unpack16_c2<0x55>(out, c2_load_rslt_m128i); // Unpack 2nd 16 values.
		hor_sse4_unpack16_c2<0xAA>(out, c2_load_rslt_m128i); // Unpack 3rd 16 values.
		hor_sse4_unpack16_c2<0xFF>(out, c2_load_rslt_m128i); // Unpack 4th 16 values.
	}
}

template <bool IsRiceCoding>
template <int imm8>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack16_c2(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	__m128i c2_shfl_rslt_m128i = _mm_shuffle_epi32(InReg, imm8);
	__m128i c2_mul_rslt_m128i = _mm_mullo_epi32(c2_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[2][0]);

	hor_sse4_unpack4_c2<6>(out, c2_mul_rslt_m128i);             // Unpack 1st 4 values.
	hor_sse4_unpack4_c2<14>(out, c2_mul_rslt_m128i);            // Unpack 2nd 4 values.
	hor_sse4_unpack4_c2<22>(out, c2_mul_rslt_m128i);            // Unpack 3rd 4 values.
	hor_sse4_unpackwithoutmask4_c2<30>(out, c2_mul_rslt_m128i); // Unpack 4th 4 values.
}

template <bool IsRiceCoding>
template <int bit>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c2(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		__m128i c2_srli_rslt_m128i = _mm_srli_epi32(InReg, bit);
		__m128i c2_rslt_m128i = _mm_and_si128(c2_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[2]);
		_mm_storeu_si128(out++, c2_rslt_m128i);
	}
	else { // For Rice and OptRice.
		__m128i c2_srli_rslt_m128i = _mm_srli_epi32(InReg, bit);
		__m128i c2_and_rslt_m128i = _mm_and_si128(c2_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[2]);
		__m128i c2_rslt_m128i = _mm_or_si128(c2_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 2));
		_mm_storeu_si128(out++, c2_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <int bit>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpackwithoutmask4_c2(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		__m128i c2_rslt_m128i = _mm_srli_epi32(InReg, bit);
		_mm_storeu_si128(out++, c2_rslt_m128i);
	}
	else { // For Rice and OptRice.
		__m128i c2_srli_rslt_m128i = _mm_srli_epi32(InReg, bit);
		__m128i c2_rslt_m128i = _mm_or_si128(c2_srli_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 2));
		_mm_storeu_si128(out++, c2_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 3-bit values.
 * Load 3 SSE vectors, each containing 42 3-bit values. (43th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c3(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c3_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	hor_sse4_unpack8_c3<0>(out, c3_load_rslt1_m128i);   // Unpack 1st 8 values.
	hor_sse4_unpack8_c3<3>(out, c3_load_rslt1_m128i);   // Unpack 2nd 8 values.
	hor_sse4_unpack8_c3<6>(out, c3_load_rslt1_m128i);   // Unpack 3rd 8 values.
	hor_sse4_unpack8_c3<9>(out, c3_load_rslt1_m128i);   // Unpack 4th 8 values.
	hor_sse4_unpack8_c3<12>(out, c3_load_rslt1_m128i);  // Unpack 5th 8 values.

	__m128i c3_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c3_alignr_rslt1_m128i = _mm_alignr_epi8(c3_load_rslt2_m128i, c3_load_rslt1_m128i, 15);
	hor_sse4_unpack8_c3<0>(out, c3_alignr_rslt1_m128i);  // Unpack 6th 8 values.
	hor_sse4_unpack8_c3<3>(out, c3_alignr_rslt1_m128i);  // Unpack 7th 8 values.
	hor_sse4_unpack8_c3<6>(out, c3_alignr_rslt1_m128i);  // Unpack 8th 8 values.
	hor_sse4_unpack8_c3<9>(out, c3_alignr_rslt1_m128i);  // Unpack 9th 8 values.
	hor_sse4_unpack8_c3<12>(out, c3_alignr_rslt1_m128i); // Unpack 10th 8 values.

	__m128i c3_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c3_alignr_rslt2_m128i = _mm_alignr_epi8(c3_load_rslt3_m128i, c3_load_rslt2_m128i, 14);
	hor_sse4_unpack8_c3<0>(out, c3_alignr_rslt2_m128i);  // Unpack 11th 8 values.
	hor_sse4_unpack8_c3<3>(out, c3_alignr_rslt2_m128i);  // Unpack 12th 8 values.
	hor_sse4_unpack8_c3<6>(out, c3_alignr_rslt2_m128i);  // Unpack 13th 8 values.
	hor_sse4_unpack8_c3<9>(out, c3_alignr_rslt2_m128i);  // Unpack 14th 8 values.
	hor_sse4_unpack8_c3<12>(out, c3_alignr_rslt2_m128i); // Unpack 15th 8 values.
	hor_sse4_unpack8_c3<13>(out, c3_load_rslt3_m128i);   // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack8_c3(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
    const __m128i Hor_SSE4_c3_shfl_msk_m128i = _mm_set_epi8(
            0xFF, byte + 2, byte + 1, byte,
            0xFF, byte + 2, byte + 1, byte,
            0xFF, byte + 2, byte + 1, byte,
            0xFF, byte + 2, byte + 1, byte);
    __m128i c3_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c3_shfl_msk_m128i);
    __m128i c3_mul_rslt_m128i = _mm_mullo_epi32(c3_shfl_rslt_m128i, _mm_set_epi32(0x01, 0x08, 0x40, 0x0200));

    hor_sse4_unpack4_c3<9>(out, c3_mul_rslt_m128i);
    hor_sse4_unpack4_c3<21>(out, c3_mul_rslt_m128i);
}

template <bool IsRiceCoding>
template <int bit>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c3(__m128i *  __restrict__  &out, const __m128i &InReg) {
    if (!IsRiceCoding) { // For NewPFor and OptPFor.
        __m128i c3_srli_rslt_m128i = _mm_srli_epi32(InReg, bit);
        __m128i c3_rslt_m128i = _mm_and_si128(c3_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[3]);
        _mm_storeu_si128(out++, c3_rslt_m128i);
    }
    else { // For Rice and OptRice.
        __m128i c3_srli_rslt_m128i = _mm_srli_epi32(InReg, bit);
        __m128i c3_and_rslt_m128i = _mm_and_si128(c3_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[3]);
        __m128i c3_rslt_m128i = _mm_or_si128(c3_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 3));
        _mm_storeu_si128(out++, c3_rslt_m128i);
    }
}


/**
 * SSE4-based unpacking 128 4-bit values.
 * Load 4 SSE vectors, each containing 32 4-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c4(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
		__m128i c4_load_rslt_m128i = _mm_loadu_si128(in++);
		hor_sse4_unpack32_c4(out, c4_load_rslt_m128i); // Unpack 32 values.
	}
}

template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack32_c4(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	__m128i c4_srli_rslt_m128i = _mm_srli_epi16(InReg, 4);
	const __m128i Hor_SSE4_c4_shfl_msk_m128i = _mm_set_epi8(
			13, 12, 15, 14,
			9, 8, 11, 10,
			5, 4, 7, 6,
			1, 0, 3, 2);
	__m128i c4_shfl_rslt_m128i = _mm_shuffle_epi8(c4_srli_rslt_m128i, Hor_SSE4_c4_shfl_msk_m128i);
	__m128i c4_blend_rslt1_m128i = _mm_blend_epi16(InReg, c4_shfl_rslt_m128i, 0xAA);
	__m128i c4_and_rslt1_m128i = _mm_and_si128(c4_blend_rslt1_m128i, _mm_set1_epi8(0x0F));
	__m128i c4_blend_rslt2_m128i = _mm_blend_epi16(InReg, c4_shfl_rslt_m128i, 0x55);
	__m128i c4_and_rslt2_m128i = _mm_and_si128(c4_blend_rslt2_m128i, _mm_set1_epi8(0x0F));

	hor_sse4_unpack4_c4_f1<0>(out, c4_and_rslt1_m128i);  // Unpack 1st 4 values.
	hor_sse4_unpack4_c4_f2<0>(out, c4_and_rslt2_m128i);  // Unpack 2nd 4 values.
	hor_sse4_unpack4_c4_f1<4>(out, c4_and_rslt1_m128i);  // Unpack 3rd 4 values.
	hor_sse4_unpack4_c4_f2<4>(out, c4_and_rslt2_m128i);  // Unpack 4th 4 values.
	hor_sse4_unpack4_c4_f1<8>(out, c4_and_rslt1_m128i);  // Unpack 5th 4 values.
	hor_sse4_unpack4_c4_f2<8>(out, c4_and_rslt2_m128i);  // Unpack 6th 4 values.
	hor_sse4_unpack4_c4_f1<12>(out, c4_and_rslt1_m128i); // Unpack 7th 4 values.
	hor_sse4_unpack4_c4_f2<12>(out, c4_and_rslt2_m128i); // Unpack 8th 4 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c4_f1(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m128i Hor_SSE4_c4_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 3,
				0xFF, 0xFF, 0xFF, byte + 1,
				0xFF, 0xFF, 0xFF, byte + 2,
				0xFF, 0xFF, 0xFF, byte + 0);
		__m128i c4_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c4_shfl_msk_m128i);
		_mm_storeu_si128(out++, c4_rslt_m128i);
	}
	else { // For Rice and OptRice.
		const __m128i Hor_SSE4_c4_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 3,
				0xFF, 0xFF, 0xFF, byte + 1,
				0xFF, 0xFF, 0xFF, byte + 2,
				0xFF, 0xFF, 0xFF, byte + 0);
		__m128i c4_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c4_shfl_msk_m128i);
		__m128i c4_rslt_m128i = _mm_or_si128(c4_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 4));
		_mm_storeu_si128(out++, c4_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c4_f2(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m128i Hor_SSE4_c4_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 1,
				0xFF, 0xFF, 0xFF, byte + 3,
				0xFF, 0xFF, 0xFF, byte + 0,
				0xFF, 0xFF, 0xFF, byte + 2);
		__m128i c4_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c4_shfl_msk_m128i);
		_mm_storeu_si128(out++, c4_rslt_m128i);
	}
	else { // For Rice and OptRice.
		const __m128i Hor_SSE4_c4_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 1,
				0xFF, 0xFF, 0xFF, byte + 3,
				0xFF, 0xFF, 0xFF, byte + 0,
				0xFF, 0xFF, 0xFF, byte + 2);
		__m128i c4_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c4_shfl_msk_m128i);
		__m128i c4_rslt_m128i = _mm_or_si128(c4_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 4));
		_mm_storeu_si128(out++, c4_rslt_m128i);
	}
}

///**
// * SSE4-based unpacking 128 4-bit values.
// * Load 4 SSE vectors, each containing 32 4-bit values.
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c4(__m128i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
//		__m128i c4_load_rslt_m128i = _mm_loadu_si128(in++);
//		hor_sse4_unpack8_c4<0x00>(out, c4_load_rslt_m128i); // Unpack 1st 8 values.
//		hor_sse4_unpack8_c4<0x55>(out, c4_load_rslt_m128i); // Unpack 2nd 8 values.
//		hor_sse4_unpack8_c4<0xAA>(out, c4_load_rslt_m128i); // Unpack 3rd 8 values.
//		hor_sse4_unpack8_c4<0xFF>(out, c4_load_rslt_m128i); // Unpack 4th 8 values.
//	}
//}
//
//template <bool IsRiceCoding>
//template <int imm8>
//void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack8_c4(__m128i *  __restrict__  &out,
//		const __m128i &InReg) {
//	__m128i c4_shfl_rslt_m128i = _mm_shuffle_epi32(InReg, imm8);
//	__m128i c4_mul_rslt_m128i = _mm_mullo_epi32(c4_shfl_rslt_m128i, _mm_set_epi32(0x01, 0x10, 0x0100, 0x1000));
//
//	hor_sse4_unpack4_c4<12>(out, c4_mul_rslt_m128i);            // Unpack 1st 4 values.
//	hor_sse4_unpackwithoutmask4_c4<28>(out, c4_mul_rslt_m128i); // Unpack 2nd 4 values.
//}
//
//template <bool IsRiceCoding>
//template <int bit>
//void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c4(__m128i *  __restrict__  &out,
//		const __m128i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m128i c4_srli_rslt_m128i = _mm_srli_epi32(InReg, bit);
//		__m128i c4_rslt_m128i = _mm_and_si128(c4_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[4]);
//		_mm_storeu_si128(out++, c4_rslt_m128i);
//	}
//	else { // For Rice and OptRice.
//		__m128i c4_srli_rslt_m128i = _mm_srli_epi32(InReg, bit);
//		__m128i c4_and_rslt_m128i = _mm_and_si128(c4_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[4]);
//		__m128i c4_rslt_m128i = _mm_or_si128(c4_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 4));
//		_mm_storeu_si128(out++, c4_rslt_m128i);
//	}
//}
//
//template <bool IsRiceCoding>
//template <int bit>
//void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpackwithoutmask4_c4(__m128i *  __restrict__  &out,
//		const __m128i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m128i c4_rslt_m128i = _mm_srli_epi32(InReg, bit);
//		_mm_storeu_si128(out++, c4_rslt_m128i);
//	}
//	else { // For Rice and OptRice.
//		__m128i c4_srli_rslt_m128i = _mm_srli_epi32(InReg, bit);
//		__m128i c4_rslt_m128i = _mm_or_si128(c4_srli_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 4));
//		_mm_storeu_si128(out++, c4_rslt_m128i);
//	}
//}


/**
 * SSE4-based unpacking 128 5-bit values.
 * Load 5 SSE vectors, each containing 21 6-bit values. (22nd is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c5(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c5_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	hor_sse4_unpack8_c5<0>(out, c5_load_rslt1_m128i);    // Unpack 1st 8 values.
	hor_sse4_unpack8_c5<5>(out, c5_load_rslt1_m128i);    // Unpack 2nd 8 values.
	hor_sse4_unpack8_c5<10>(out, c5_load_rslt1_m128i);   // Unpack 3rd 8 values.

	__m128i c5_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c5_alignr_rslt1_m128i = _mm_alignr_epi8(c5_load_rslt2_m128i, c5_load_rslt1_m128i, 15);
	hor_sse4_unpack8_c5<0>(out, c5_alignr_rslt1_m128i);  // Unpack 4th 8 values.
	hor_sse4_unpack8_c5<5>(out, c5_alignr_rslt1_m128i);  // Unpack 5th 8 values.
	hor_sse4_unpack8_c5<10>(out, c5_alignr_rslt1_m128i); // Unpack 6th 8 values.

	__m128i c5_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c5_alignr_rslt2_m128i = _mm_alignr_epi8(c5_load_rslt3_m128i, c5_load_rslt2_m128i, 14);
	hor_sse4_unpack8_c5<0>(out, c5_alignr_rslt2_m128i);  // Unpack 7th 8 values.
	hor_sse4_unpack8_c5<5>(out, c5_alignr_rslt2_m128i);  // Unpack 8th 8 values.
	hor_sse4_unpack8_c5<10>(out, c5_alignr_rslt2_m128i); // Unpack 9th 8 values.

	__m128i c5_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c5_alignr_rslt3_m128i = _mm_alignr_epi8(c5_load_rslt4_m128i, c5_load_rslt3_m128i, 13);
	hor_sse4_unpack8_c5<0>(out, c5_alignr_rslt3_m128i);  // Unpack 10th 8 values.
	hor_sse4_unpack8_c5<5>(out, c5_alignr_rslt3_m128i);  // Unpack 11th 8 values.
	hor_sse4_unpack8_c5<10>(out, c5_alignr_rslt3_m128i); // Unpack 12th 8 values.

	__m128i c5_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c5_alignr_rslt4_m128i = _mm_alignr_epi8(c5_load_rslt5_m128i, c5_load_rslt4_m128i, 12);
	hor_sse4_unpack8_c5<0>(out, c5_alignr_rslt4_m128i);  // Unpack 13th 8 values.
	hor_sse4_unpack8_c5<5>(out, c5_alignr_rslt4_m128i);  // Unpack 14th 8 values.
	hor_sse4_unpack8_c5<10>(out, c5_alignr_rslt4_m128i); // Unpack 15th 8 values.
	hor_sse4_unpack8_c5<11>(out, c5_load_rslt5_m128i);   // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack8_c5(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 1st 4 values.
		const __m128i Hor_SSE4_c5_shfl_msk0_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, 0xFF, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0,
				0xFF, 0xFF, 0xFF, byte + 0);
		__m128i c5_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c5_shfl_msk0_m128i);
		__m128i c5_mul_rslt_m128i = _mm_mullo_epi32(c5_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[5][0]);
		__m128i c5_srli_rslt_m128i = _mm_srli_epi32(c5_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[5][0]);
		__m128i c5_rslt_m128i = _mm_and_si128(c5_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[5]);
		_mm_storeu_si128(out++, c5_rslt_m128i);

		// Unpack 2nd 4 values.
		const __m128i Hor_SSE4_c5_shfl_msk1_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 4,
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, 0xFF, 0xFF, byte + 3,
				0xFF, 0xFF, byte + 3, byte + 2);
		c5_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c5_shfl_msk1_m128i);
		c5_mul_rslt_m128i = _mm_mullo_epi32(c5_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[5][1]);
		c5_srli_rslt_m128i = _mm_srli_epi32(c5_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[5][1]);
		c5_rslt_m128i = _mm_and_si128(c5_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[5]);
		_mm_storeu_si128(out++, c5_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 1st 4 values.
		const __m128i Hor_SSE4_c5_shfl_msk1_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, 0xFF, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0,
				0xFF, 0xFF, 0xFF, byte + 0);
		__m128i c5_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c5_shfl_msk1_m128i);
		__m128i c5_mul_rslt_m128i = _mm_mullo_epi32(c5_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[5][0]);
		__m128i c5_srli_rslt_m128i = _mm_srli_epi32(c5_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[5][0]);
		__m128i c5_and_rslt_m128i = _mm_and_si128(c5_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[5]);
		__m128i c5_rslt_m128i = _mm_or_si128(c5_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 5));
		_mm_storeu_si128(out++, c5_rslt_m128i);

		// Unpack 2nd 4 values.
		const __m128i Hor_SSE4_c5_shfl_msk2_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 4,
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, 0xFF, 0xFF, byte + 3,
				0xFF, 0xFF, byte + 3, byte + 2);
		c5_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c5_shfl_msk2_m128i);
		c5_mul_rslt_m128i = _mm_mullo_epi32(c5_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[5][1]);
		c5_srli_rslt_m128i = _mm_srli_epi32(c5_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[5][1]);
		c5_and_rslt_m128i = _mm_and_si128(c5_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[5]);
		c5_rslt_m128i = _mm_or_si128(c5_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 5));
		_mm_storeu_si128(out++, c5_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 6-bit values.
 * Load 6 SSE vectors, each containing 21 6-bit values. (22nd is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c6(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
		__m128i c6_load_rslt1_m128i = _mm_loadu_si128(in++);
		hor_sse4_unpack4_c6<0>(out, c6_load_rslt1_m128i);    // Unpack 1st 4 values
		hor_sse4_unpack4_c6<3>(out, c6_load_rslt1_m128i);    // Unpack 2nd 4 values
		hor_sse4_unpack4_c6<6>(out, c6_load_rslt1_m128i);    // Unpack 3rd 4 values
		hor_sse4_unpack4_c6<9>(out, c6_load_rslt1_m128i);    // Unpack 4th 4 values
		hor_sse4_unpack4_c6<12>(out, c6_load_rslt1_m128i);   // Unpack 5th 4 values

		__m128i c6_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c6_alignr_rslt1_m128i = _mm_alignr_epi8(c6_load_rslt2_m128i, c6_load_rslt1_m128i, 15);
		hor_sse4_unpack4_c6<0>(out, c6_alignr_rslt1_m128i);  // Unpack 6th 4 values
		hor_sse4_unpack4_c6<3>(out, c6_alignr_rslt1_m128i);  // Unpack 7th 4 values
		hor_sse4_unpack4_c6<6>(out, c6_alignr_rslt1_m128i);  // Unpack 8th 4 values
		hor_sse4_unpack4_c6<9>(out, c6_alignr_rslt1_m128i);  // Unpack 9th 4 values
		hor_sse4_unpack4_c6<12>(out, c6_alignr_rslt1_m128i); // Unpack 10th 4 values

		__m128i c6_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c6_alignr_rslt2_m128i = _mm_alignr_epi8(c6_load_rslt3_m128i, c6_load_rslt2_m128i, 14);
		hor_sse4_unpack4_c6<0>(out, c6_alignr_rslt2_m128i);  // Unpack 11th 4 values.
		hor_sse4_unpack4_c6<3>(out, c6_alignr_rslt2_m128i);  // Unpack 12th 4 values.
		hor_sse4_unpack4_c6<6>(out, c6_alignr_rslt2_m128i);  // Unpack 13th 4 values.
		hor_sse4_unpack4_c6<9>(out, c6_alignr_rslt2_m128i);  // Unpack 14th 4 values.
		hor_sse4_unpack4_c6<12>(out, c6_alignr_rslt2_m128i); // Unpack 15th 4 values.
		hor_sse4_unpack4_c6<13>(out, c6_load_rslt3_m128i);   // Unpack 16th 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c6(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c6_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 2,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0,
				0xFF, 0xFF, 0xFF, byte + 0);
		__m128i c6_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c6_shfl_msk_m128i);
		__m128i c6_mul_rslt_m128i = _mm_mullo_epi32(c6_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[6][0]);
		__m128i c6_srli_rslt_m128i = _mm_srli_epi32(c6_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[6][0]);
		__m128i c6_rslt_m128i = _mm_and_si128(c6_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[6]);
		_mm_storeu_si128(out++, c6_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c6_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 2,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0,
				0xFF, 0xFF, 0xFF, byte + 0);
		__m128i c6_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c6_shfl_msk_m128i);
		__m128i c6_mul_rslt_m128i = _mm_mullo_epi32(c6_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[6][0]);
		__m128i c6_srli_rslt_m128i = _mm_srli_epi32(c6_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[6][0]);
		__m128i c6_and_rslt_m128i = _mm_and_si128(c6_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[6]);
		__m128i c6_rslt_m128i = _mm_or_si128(c6_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 6));
		_mm_storeu_si128(out++, c6_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 7-bit values.
 * Load 7 SSE vectors, each containing 18 7-bit values. (19th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c7(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c7_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	hor_sse4_unpack8_c7<0>(out, c7_load_rslt1_m128i);   // Unpack 1st 8 values.
	hor_sse4_unpack8_c7<7>(out, c7_load_rslt1_m128i);   // Unpack 2nd 8 values.

	__m128i c7_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c7_alignr_rslt1_m128i = _mm_alignr_epi8(c7_load_rslt2_m128i, c7_load_rslt1_m128i, 14);
	hor_sse4_unpack8_c7<0>(out, c7_alignr_rslt1_m128i); // Unpack 3rd 8 values.
	hor_sse4_unpack8_c7<7>(out, c7_alignr_rslt1_m128i); // Unpack 4th 8 values.

	__m128i c7_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c7_alignr_rslt2_m128i = _mm_alignr_epi8(c7_load_rslt3_m128i, c7_load_rslt2_m128i, 12);
	hor_sse4_unpack8_c7<0>(out, c7_alignr_rslt2_m128i); // Unpack 5th 8 values.
	hor_sse4_unpack8_c7<7>(out, c7_alignr_rslt2_m128i); // Unpack 6th 8 values.

	__m128i c7_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c7_alignr_rslt3_m128i = _mm_alignr_epi8(c7_load_rslt4_m128i, c7_load_rslt3_m128i, 10);
	hor_sse4_unpack8_c7<0>(out, c7_alignr_rslt3_m128i); // Unpack 7th 8 values.
	hor_sse4_unpack8_c7<7>(out, c7_alignr_rslt3_m128i); // Unpack 8th 8 values.

	__m128i c7_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c7_alignr_rslt4_m128i = _mm_alignr_epi8(c7_load_rslt5_m128i, c7_load_rslt4_m128i, 8);
	hor_sse4_unpack8_c7<0>(out, c7_alignr_rslt4_m128i); // Unpack 9th 8 values.
	hor_sse4_unpack8_c7<7>(out, c7_alignr_rslt4_m128i); // Unpack 10th 8 values.

	__m128i c7_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c7_alignr_rslt5_m128i = _mm_alignr_epi8(c7_load_rslt6_m128i, c7_load_rslt5_m128i, 6);
	hor_sse4_unpack8_c7<0>(out, c7_alignr_rslt5_m128i); // Unpack 11th 8 values.
	hor_sse4_unpack8_c7<7>(out, c7_alignr_rslt5_m128i); // Unpack 12th 8 values.

	__m128i c7_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c7_alignr_rslt6_m128i = _mm_alignr_epi8(c7_load_rslt7_m128i, c7_load_rslt6_m128i, 4);
	hor_sse4_unpack8_c7<0>(out, c7_alignr_rslt6_m128i); // Unpack 13th 8 values.
	hor_sse4_unpack8_c7<7>(out, c7_alignr_rslt6_m128i); // Unpack 14th 8 values.

	hor_sse4_unpack8_c7<2>(out, c7_load_rslt7_m128i);   // Unpack 15th 8 values.
	hor_sse4_unpack8_c7<9>(out, c7_load_rslt7_m128i);   // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack8_c7(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 1st 4 values.
		const __m128i Hor_SSE4_c7_shfl_msk0_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 3, byte + 2,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0,
				0xFF, 0xFF, 0xFF, byte + 0);
		__m128i c7_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c7_shfl_msk0_m128i);
		__m128i c7_mul_rslt_m128i = _mm_mullo_epi32(c7_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[7][0]);
		__m128i c7_srli_rslt_m128i = _mm_srli_epi32(c7_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[7][0]);
		__m128i c7_rslt_m128i = _mm_and_si128(c7_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[7]);
		_mm_storeu_si128(out++, c7_rslt_m128i);

		// Unpack 2nd 4 values.
		const __m128i Hor_SSE4_c7_shfl_msk1_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 6,
				0xFF, 0xFF, byte + 6, byte + 5,
				0xFF, 0xFF, byte + 5, byte + 4,
				0xFF, 0xFF, byte + 4, byte + 3);
		c7_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c7_shfl_msk1_m128i);
		c7_mul_rslt_m128i = _mm_mullo_epi32(c7_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[7][1]);
		c7_srli_rslt_m128i = _mm_srli_epi32(c7_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[7][1]);
		c7_rslt_m128i = _mm_and_si128(c7_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[7]);
		_mm_storeu_si128(out++, c7_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 1st 4 values.
		const __m128i Hor_SSE4_c7_shfl_msk0_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 3, byte + 2,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0,
				0xFF, 0xFF, 0xFF, byte + 0);
		__m128i c7_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c7_shfl_msk0_m128i);
		__m128i c7_mul_rslt_m128i = _mm_mullo_epi32(c7_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[7][0]);
		__m128i c7_srli_rslt_m128i = _mm_srli_epi32(c7_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[7][0]);
		__m128i c7_and_rslt_m128i = _mm_and_si128(c7_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[7]);
		__m128i c7_rslt_m128i = _mm_or_si128(c7_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 7));
		_mm_storeu_si128(out++, c7_rslt_m128i);

		// Unpack 2nd 4 values.
		const __m128i Hor_SSE4_c7_shfl_msk1_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 6,
				0xFF, 0xFF, byte + 6, byte + 5,
				0xFF, 0xFF, byte + 5, byte + 4,
				0xFF, 0xFF, byte + 4, byte + 3);
		c7_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c7_shfl_msk1_m128i);
		c7_mul_rslt_m128i = _mm_mullo_epi32(c7_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[7][1]);
		c7_srli_rslt_m128i = _mm_srli_epi32(c7_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[7][1]);
		c7_and_rslt_m128i = _mm_and_si128(c7_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[7]);
		c7_rslt_m128i = _mm_or_si128(c7_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 7));
		_mm_storeu_si128(out++, c7_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 8-bit values.
 * Load 8 SSE vectors, each containing 16 8-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c8(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 16) {
		__m128i c8_load_rslt_m128i = _mm_loadu_si128(in++);
		hor_sse4_unpack4_c8<0>(out, c8_load_rslt_m128i);  // Unpack 1st 4 values.
		hor_sse4_unpack4_c8<4>(out, c8_load_rslt_m128i);  // Unpack 2nd 4 values.
		hor_sse4_unpack4_c8<8>(out, c8_load_rslt_m128i);  // Unpack 3rd 4 values.
		hor_sse4_unpack4_c8<12>(out, c8_load_rslt_m128i); // Unpack 4th 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c8(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c8_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 3,
				0xFF, 0xFF, 0xFF, byte + 2,
				0xFF, 0xFF, 0xFF, byte + 1,
				0xFF, 0xFF, 0xFF, byte + 0);
		__m128i c8_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c8_shfl_msk_m128i);
		_mm_storeu_si128(out++, c8_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c8_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 3,
				0xFF, 0xFF, 0xFF, byte + 2,
				0xFF, 0xFF, 0xFF, byte + 1,
				0xFF, 0xFF, 0xFF, byte + 0);
		__m128i c8_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c8_shfl_msk_m128i);
		__m128i c8_rslt_m128i = _mm_or_si128(c8_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 8));
		_mm_storeu_si128(out++, c8_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 9-bit values.
 * Load 9 SSE vectors, each containing 14 9-bit values. (15th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c9(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c9_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	hor_sse4_unpack8_c9<0>(out, c9_load_rslt1_m128i);   // Unpack 1st 8 values.

	__m128i c9_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c9_alignr_rslt1_m128i = _mm_alignr_epi8(c9_load_rslt2_m128i, c9_load_rslt1_m128i, 9);
	hor_sse4_unpack8_c9<0>(out, c9_alignr_rslt1_m128i); // Unpack 2nd 8 values.
	hor_sse4_unpack8_c9<2>(out, c9_load_rslt2_m128i);   // Unpack 3rd 8 values.

	__m128i c9_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c9_alignr_rslt2_m128i = _mm_alignr_epi8(c9_load_rslt3_m128i, c9_load_rslt2_m128i, 11);
	hor_sse4_unpack8_c9<0>(out, c9_alignr_rslt2_m128i); // Unpack 4th 8 values.
	hor_sse4_unpack8_c9<4>(out, c9_load_rslt3_m128i);   // Unpack 5th 8 values.

	__m128i c9_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c9_alignr_rslt3_m128i = _mm_alignr_epi8(c9_load_rslt4_m128i, c9_load_rslt3_m128i, 13);
	hor_sse4_unpack8_c9<0>(out, c9_alignr_rslt3_m128i); // Unpack 6th 8 values.
	hor_sse4_unpack8_c9<6>(out, c9_load_rslt4_m128i);   // Unpack 7th 8 values.

	__m128i c9_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c9_alignr_rslt4_m128i = _mm_alignr_epi8(c9_load_rslt5_m128i, c9_load_rslt4_m128i, 15);
	hor_sse4_unpack8_c9<0>(out, c9_alignr_rslt4_m128i); // Unpack 8th 8 values.

	__m128i c9_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c9_alignr_rslt5_m128i = _mm_alignr_epi8(c9_load_rslt6_m128i, c9_load_rslt5_m128i, 8);
	hor_sse4_unpack8_c9<0>(out, c9_alignr_rslt5_m128i); // Unpack 9th 8 values.
	hor_sse4_unpack8_c9<1>(out, c9_load_rslt6_m128i);   // Unpack 10th 8 values.

	__m128i c9_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c9_alignr_rslt6_m128i = _mm_alignr_epi8(c9_load_rslt7_m128i, c9_load_rslt6_m128i, 10);
	hor_sse4_unpack8_c9<0>(out, c9_alignr_rslt6_m128i); // Unpack 11th 8 values.
	hor_sse4_unpack8_c9<3>(out, c9_load_rslt7_m128i);   // Unpack 12th 8 values.

	__m128i c9_load_rslt8_m128i = _mm_loadu_si128(in + 7);
	__m128i c9_alignr_rslt7_m128i = _mm_alignr_epi8(c9_load_rslt8_m128i, c9_load_rslt7_m128i, 12);
	hor_sse4_unpack8_c9<0>(out, c9_alignr_rslt7_m128i); // Unpack 13th 8 values.
	hor_sse4_unpack8_c9<5>(out, c9_load_rslt8_m128i);   // Unpack 14th 8 values.

	__m128i c9_load_rslt9_m128i = _mm_loadu_si128(in + 8);
	__m128i c9_alignr_rslt8_m128i = _mm_alignr_epi8(c9_load_rslt9_m128i, c9_load_rslt8_m128i, 14);
	hor_sse4_unpack8_c9<0>(out, c9_alignr_rslt8_m128i); // Unpack 15th 8 values.
	hor_sse4_unpack8_c9<7>(out, c9_load_rslt9_m128i);   // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack8_c9(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 1st 4 values.
		const __m128i Hor_SSE4_c9_shfl_msk0_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, 0xFF, byte + 3, byte + 2,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c9_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c9_shfl_msk0_m128i);
		__m128i c9_mul_rslt_m128i = _mm_mullo_epi32(c9_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[9][0]);
		__m128i c9_srli_rslt_m128i = _mm_srli_epi32(c9_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[9][0]);
		__m128i c9_rslt_m128i = _mm_and_si128(c9_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[9]);
		_mm_storeu_si128(out++, c9_rslt_m128i);

		// Unpack 2nd 4 values.
		const __m128i Hor_SSE4_c9_shfl_msk1_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 8, byte + 7,
				0xFF, 0xFF, byte + 7, byte + 6,
				0xFF, 0xFF, byte + 6, byte + 5,
				0xFF, 0xFF, byte + 5, byte + 4);
		c9_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c9_shfl_msk1_m128i);
		c9_mul_rslt_m128i = _mm_mullo_epi32(c9_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[9][1]);
		c9_srli_rslt_m128i = _mm_srli_epi32(c9_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[9][1]);
		c9_rslt_m128i = _mm_and_si128(c9_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[9]);
		_mm_storeu_si128(out++, c9_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 1st 4 values.
		const __m128i Hor_SSE4_c9_shfl_msk0_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, 0xFF, byte + 3, byte + 2,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c9_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c9_shfl_msk0_m128i);
		__m128i c9_mul_rslt_m128i = _mm_mullo_epi32(c9_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[9][0]);
		__m128i c9_srli_rslt_m128i = _mm_srli_epi32(c9_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[9][0]);
		__m128i c9_and_rslt_m128i = _mm_and_si128(c9_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[9]);
		__m128i c9_rslt_m128i = _mm_or_si128(c9_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 9));
		_mm_storeu_si128(out++, c9_rslt_m128i);

		// Unpack 2nd 4 values.
		const __m128i Hor_SSE4_c9_shfl_msk1_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 8, byte + 7,
				0xFF, 0xFF, byte + 7, byte + 6,
				0xFF, 0xFF, byte + 6, byte + 5,
				0xFF, 0xFF, byte + 5, byte + 4);
		c9_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c9_shfl_msk1_m128i);
		c9_mul_rslt_m128i = _mm_mullo_epi32(c9_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[9][1]);
		c9_srli_rslt_m128i = _mm_srli_epi32(c9_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[9][1]);
		c9_and_rslt_m128i = _mm_and_si128(c9_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[9]);
		c9_rslt_m128i = _mm_or_si128(c9_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 9));
		_mm_storeu_si128(out++, c9_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 10-bit values.
 * Load 10 SSE vectors, each containing 12 11-bit values. (13th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c10(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
		__m128i c10_load_rslt1_m128i = _mm_loadu_si128(in++);
		hor_sse4_unpack4_c10<0>(out, c10_load_rslt1_m128i);    // Unpack 1st 4 values.
		hor_sse4_unpack4_c10<5>(out, c10_load_rslt1_m128i);    // Unpack 2nd 4 values.
		hor_sse4_unpack4_c10<10>(out, c10_load_rslt1_m128i);   // Unpack 3rd 4 values.

		__m128i c10_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c10_alignr_rslt1_m128i = _mm_alignr_epi8(c10_load_rslt2_m128i, c10_load_rslt1_m128i, 15);
		hor_sse4_unpack4_c10<0>(out, c10_alignr_rslt1_m128i);  // Unpack 4th 4 values.
		hor_sse4_unpack4_c10<5>(out, c10_alignr_rslt1_m128i);  // Unpack 5th 4 values.
		hor_sse4_unpack4_c10<10>(out, c10_alignr_rslt1_m128i); // Unpack 6th 4 values.

		__m128i c10_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c10_alignr_rslt2_m128i = _mm_alignr_epi8(c10_load_rslt3_m128i, c10_load_rslt2_m128i, 14);
		hor_sse4_unpack4_c10<0>(out, c10_alignr_rslt2_m128i);  // Unpack 7th 4 values.
		hor_sse4_unpack4_c10<5>(out, c10_alignr_rslt2_m128i);  // Unpack 8th 4 values.
		hor_sse4_unpack4_c10<10>(out, c10_alignr_rslt2_m128i); // Unpack 9th 4 values.

		__m128i c10_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c10_alignr_rslt3_m128i = _mm_alignr_epi8(c10_load_rslt4_m128i, c10_load_rslt3_m128i, 13);
		hor_sse4_unpack4_c10<0>(out, c10_alignr_rslt3_m128i);  // Unpack 10th 4 values.
		hor_sse4_unpack4_c10<5>(out, c10_alignr_rslt3_m128i);  // Unpack 11th 4 values.
		hor_sse4_unpack4_c10<10>(out, c10_alignr_rslt3_m128i); // Unpack 12th 4 values.

		__m128i c10_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c10_alignr_rslt4_m128i = _mm_alignr_epi8(c10_load_rslt5_m128i, c10_load_rslt4_m128i, 12);
		hor_sse4_unpack4_c10<0>(out, c10_alignr_rslt4_m128i);  // Unpack 13th 4 values.
		hor_sse4_unpack4_c10<5>(out, c10_alignr_rslt4_m128i);  // Unpack 14th 4 values.
		hor_sse4_unpack4_c10<10>(out, c10_alignr_rslt4_m128i); // Unpack 15th 4 values.
		hor_sse4_unpack4_c10<11>(out, c10_load_rslt5_m128i);   // Unpack 16th 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c10(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c10_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, 0xFF, byte + 3, byte + 2,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c10_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c10_shfl_msk_m128i);
		__m128i c10_mul_rslt_m128i = _mm_mullo_epi32(c10_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[10][0]);
		__m128i c10_srli_rslt_m128i = _mm_srli_epi32(c10_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[10][0]);
		__m128i c10_rslt_m128i = _mm_and_si128(c10_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[10]);
		_mm_storeu_si128(out++, c10_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c10_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, 0xFF, byte + 3, byte + 2,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c10_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c10_shfl_msk_m128i);
		__m128i c10_mul_rslt_m128i = _mm_mullo_epi32(c10_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[10][0]);
		__m128i c10_srli_rslt_m128i = _mm_srli_epi32(c10_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[10][0]);
		__m128i c10_and_rslt_m128i = _mm_and_si128(c10_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[10]);
		__m128i c10_rslt_m128i = _mm_or_si128(c10_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 10));
		_mm_storeu_si128(out++, c10_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 11-bit values.
 * Load 11 SSE vectors, each containing 11 11-bit values. (12th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c11(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c11_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	hor_sse4_unpack8_c11<0>(out, c11_load_rslt1_m128i);    // Unpack 1st 8 values.

	__m128i c11_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c11_alignr_rslt1_m128i = _mm_alignr_epi8(c11_load_rslt2_m128i, c11_load_rslt1_m128i, 11);
	hor_sse4_unpack8_c11<0>(out, c11_alignr_rslt1_m128i);  // Unpack 2nd 8 values.

	__m128i c11_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c11_alignr_rslt2_m128i = _mm_alignr_epi8(c11_load_rslt3_m128i, c11_load_rslt2_m128i, 6);
	hor_sse4_unpack8_c11<0>(out, c11_alignr_rslt2_m128i);  // Unpack 3rd 8 values.
	hor_sse4_unpack8_c11<1>(out, c11_load_rslt3_m128i);    // Unpack 4th 8 values.

	__m128i c11_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c11_alignr_rslt3_m128i = _mm_alignr_epi8(c11_load_rslt4_m128i, c11_load_rslt3_m128i, 12);
	hor_sse4_unpack8_c11<0>(out, c11_alignr_rslt3_m128i);  // Unpack 5th 8 values.

	__m128i c11_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c11_alignr_rslt4_m128i = _mm_alignr_epi8(c11_load_rslt5_m128i, c11_load_rslt4_m128i, 7);
	hor_sse4_unpack8_c11<0>(out, c11_alignr_rslt4_m128i);  // Unpack 6th 8 values.
	hor_sse4_unpack8_c11<2>(out, c11_load_rslt5_m128i);    // Unpack 7th 8 values.

	__m128i c11_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c11_alignr_rslt5_m128i = _mm_alignr_epi8(c11_load_rslt6_m128i, c11_load_rslt5_m128i, 13);
	hor_sse4_unpack8_c11<0>(out, c11_alignr_rslt5_m128i);  // Unpack 8th 8 values.

	__m128i c11_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c11_alignr_rslt6_m128i = _mm_alignr_epi8(c11_load_rslt7_m128i, c11_load_rslt6_m128i, 8);
	hor_sse4_unpack8_c11<0>(out, c11_alignr_rslt6_m128i);  // Unpack 9th 8 values.
	hor_sse4_unpack8_c11<3>(out, c11_load_rslt7_m128i);    // Unpack 10th 8 values.

	__m128i c11_load_rslt8_m128i = _mm_loadu_si128(in + 7);
	__m128i c11_alignr_rslt7_m128i = _mm_alignr_epi8(c11_load_rslt8_m128i, c11_load_rslt7_m128i, 14);
	hor_sse4_unpack8_c11<0>(out, c11_alignr_rslt7_m128i);  // Unpack 11th 8 values.

	__m128i c11_load_rslt9_m128i = _mm_loadu_si128(in + 8);
	__m128i c11_alignr_rslt8_m128i = _mm_alignr_epi8(c11_load_rslt9_m128i, c11_load_rslt8_m128i, 9);
	hor_sse4_unpack8_c11<0>(out, c11_alignr_rslt8_m128i);  // Unpack 12th 8 values.
	hor_sse4_unpack8_c11<4>(out, c11_load_rslt9_m128i);    // Unpack 13th 8 values.

	__m128i c11_load_rslt10_m128i = _mm_loadu_si128(in + 9);
	__m128i c11_alignr_rslt9_m128i = _mm_alignr_epi8(c11_load_rslt10_m128i, c11_load_rslt9_m128i, 15);
	hor_sse4_unpack8_c11<0>(out, c11_alignr_rslt9_m128i);  // Unpack 14th 8 values.

	__m128i c11_load_rslt11_m128i = _mm_loadu_si128(in + 10);
	__m128i c11_alignr_rslt10_m128i = _mm_alignr_epi8(c11_load_rslt11_m128i, c11_load_rslt10_m128i, 10);
	hor_sse4_unpack8_c11<0>(out, c11_alignr_rslt10_m128i); // Unpack 15th 8 values.
	hor_sse4_unpack8_c11<5>(out, c11_load_rslt11_m128i);   // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack8_c11(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 1st 4 values.
		const __m128i Hor_SSE4_c11_shfl_msk0_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 5, byte + 4,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c11_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c11_shfl_msk0_m128i);
		__m128i c11_mul_rslt_m128i = _mm_mullo_epi32(c11_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[11][0]);
		__m128i c11_srli_rslt_m128i = _mm_srli_epi32(c11_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[11][0]);
		__m128i c11_rslt_m128i = _mm_and_si128(c11_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[11]);
		_mm_storeu_si128(out++, c11_rslt_m128i);

		// Unpack 2nd 4 values.
		const __m128i Hor_SSE4_c11_shfl_msk1_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 10, byte + 9,
				0xFF, 0xFF, byte + 9, byte + 8,
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, 0xFF, byte + 6, byte + 5);
		c11_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c11_shfl_msk1_m128i);
		c11_mul_rslt_m128i = _mm_mullo_epi32(c11_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[11][1]);
		c11_srli_rslt_m128i = _mm_srli_epi32(c11_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[11][1]);
		c11_rslt_m128i = _mm_and_si128(c11_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[11]);
		_mm_storeu_si128(out++, c11_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 1st 4 values.
		const __m128i Hor_SSE4_c11_shfl_msk0_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 5, byte + 4,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c11_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c11_shfl_msk0_m128i);
		__m128i c11_mul_rslt_m128i = _mm_mullo_epi32(c11_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[11][0]);
		__m128i c11_srli_rslt_m128i = _mm_srli_epi32(c11_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[11][0]);
		__m128i c11_and_rslt_m128i = _mm_and_si128(c11_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[11]);
		__m128i c11_rslt_m128i = _mm_or_si128(c11_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 11));
		_mm_storeu_si128(out++, c11_rslt_m128i);

		// Unpack 2nd 4 values.
		const __m128i Hor_SSE4_c11_shfl_msk1_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 10, byte + 9,
				0xFF, 0xFF, byte + 9, byte + 8,
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, 0xFF, byte + 6, byte + 5);
		c11_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c11_shfl_msk1_m128i);
		c11_mul_rslt_m128i = _mm_mullo_epi32(c11_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[11][1]);
		c11_srli_rslt_m128i = _mm_srli_epi32(c11_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[11][1]);
		c11_and_rslt_m128i = _mm_and_si128(c11_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[11]);
		c11_rslt_m128i = _mm_or_si128(c11_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 11));
		_mm_storeu_si128(out++, c11_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 12-bit values.
 * Load 12 SSE vectors, each containing 10 12-bit values. (11th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c12(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
		__m128i c12_load_rslt1_m128i = _mm_loadu_si128(in++);
		hor_sse4_unpack4_c12<0>(out, c12_load_rslt1_m128i);   // Unpack 1st 4 values.
		hor_sse4_unpack4_c12<6>(out, c12_load_rslt1_m128i);   // Unpack 2nd 4 values.

		__m128i c12_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c12_alignr_rslt1_m128i = _mm_alignr_epi8(c12_load_rslt2_m128i, c12_load_rslt1_m128i, 12);
		hor_sse4_unpack4_c12<0>(out, c12_alignr_rslt1_m128i); // Unpack 3rd 4 values.
		hor_sse4_unpack4_c12<6>(out, c12_alignr_rslt1_m128i); // Unpack 4th 4 values.
		hor_sse4_unpack4_c12<8>(out, c12_load_rslt2_m128i);   // Unpack 5th 4 values.

		__m128i c12_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c12_alignr_rslt2_m128i = _mm_alignr_epi8(c12_load_rslt3_m128i, c12_load_rslt2_m128i, 14);
		hor_sse4_unpack4_c12<0>(out, c12_alignr_rslt2_m128i); // Unpack 6th 4 values.
		hor_sse4_unpack4_c12<6>(out, c12_alignr_rslt2_m128i); // Unpack 7th 4 values.
		hor_sse4_unpack4_c12<10>(out, c12_load_rslt3_m128i);  // Unpack 8th 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c12(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c12_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 5, byte + 4,
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c12_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c12_shfl_msk_m128i);
		__m128i c12_mul_rslt_m128i = _mm_mullo_epi32(c12_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[12][0]);
		__m128i c12_srli_rslt_m128i = _mm_srli_epi32(c12_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[12][0]);
		__m128i c12_rslt_m128i = _mm_and_si128(c12_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[12]);
		_mm_storeu_si128(out++, c12_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c12_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 5, byte + 4,
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c12_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c12_shfl_msk_m128i);
		__m128i c12_mul_rslt_m128i = _mm_mullo_epi32(c12_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[12][0]);
		__m128i c12_srli_rslt_m128i = _mm_srli_epi32(c12_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[12][0]);
		__m128i c12_and_rslt_m128i = _mm_and_si128(c12_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[12]);
		__m128i c12_rslt_m128i = _mm_or_si128(c12_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 12));
		_mm_storeu_si128(out++, c12_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 13-bit values.
 * Load 13 SSE vectors, each containing 9 13-bit values. (10th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c13(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c13_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	hor_sse4_unpack8_c13<0>(out, c13_load_rslt1_m128i);    // Unpack 1st 8 values.

	__m128i c13_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c13_alignr_rslt1_m128i = _mm_alignr_epi8(c13_load_rslt2_m128i, c13_load_rslt1_m128i, 13);
	hor_sse4_unpack8_c13<0>(out, c13_alignr_rslt1_m128i);  // Unpack 2nd 8 values.

	__m128i c13_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c13_alignr_rslt2_m128i = _mm_alignr_epi8(c13_load_rslt3_m128i, c13_load_rslt2_m128i, 10);
	hor_sse4_unpack8_c13<0>(out, c13_alignr_rslt2_m128i);  // Unpack 3rd 8 values.

    __m128i c13_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c13_alignr_rslt3_m128i = _mm_alignr_epi8(c13_load_rslt4_m128i, c13_load_rslt3_m128i, 7);
    hor_sse4_unpack8_c13<0>(out, c13_alignr_rslt3_m128i);  // Unpack 4th 8 values.

    __m128i c13_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c13_alignr_rslt4_m128i = _mm_alignr_epi8(c13_load_rslt5_m128i, c13_load_rslt4_m128i, 4);
    hor_sse4_unpack8_c13<0>(out, c13_alignr_rslt4_m128i);  // Unpack 5th 8 values.
    hor_sse4_unpack8_c13<1>(out, c13_load_rslt5_m128i);    // Unpack 6th 8 values.

    __m128i c13_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c13_alignr_rslt5_m128i = _mm_alignr_epi8(c13_load_rslt6_m128i, c13_load_rslt5_m128i, 14);
    hor_sse4_unpack8_c13<0>(out, c13_alignr_rslt5_m128i);  // Unpack 7th 8 values.

    __m128i c13_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c13_alignr_rslt6_m128i = _mm_alignr_epi8(c13_load_rslt7_m128i, c13_load_rslt6_m128i, 11);
    hor_sse4_unpack8_c13<0>(out, c13_alignr_rslt6_m128i);  // Unpack 8th 8 values.

    __m128i c13_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c13_alignr_rslt7_m128i = _mm_alignr_epi8(c13_load_rslt8_m128i, c13_load_rslt7_m128i, 8);
    hor_sse4_unpack8_c13<0>(out, c13_alignr_rslt7_m128i);  // Unpack 9th 8 values.

    __m128i c13_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c13_alignr_rslt8_m128i = _mm_alignr_epi8(c13_load_rslt9_m128i, c13_load_rslt8_m128i, 5);
    hor_sse4_unpack8_c13<0>(out, c13_alignr_rslt8_m128i);  // Unpack 10th 8 values.
    hor_sse4_unpack8_c13<2>(out, c13_load_rslt9_m128i);    // Unpack 11th 8 values.

    __m128i c13_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c13_alignr_rslt9_m128i = _mm_alignr_epi8(c13_load_rslt10_m128i, c13_load_rslt9_m128i, 15);
    hor_sse4_unpack8_c13<0>(out, c13_alignr_rslt9_m128i);  // Unpack 12th 8 values.

    __m128i c13_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c13_alignr_rslt10_m128i = _mm_alignr_epi8(c13_load_rslt11_m128i, c13_load_rslt10_m128i, 12);
    hor_sse4_unpack8_c13<0>(out, c13_alignr_rslt10_m128i); // Unpack 13th 8 values.

    __m128i c13_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c13_alignr_rslt11_m128i = _mm_alignr_epi8(c13_load_rslt12_m128i, c13_load_rslt11_m128i, 9);
    hor_sse4_unpack8_c13<0>(out, c13_alignr_rslt11_m128i); // Unpack 14th 8 values.

    __m128i c13_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c13_alignr_rslt12_m128i = _mm_alignr_epi8(c13_load_rslt13_m128i, c13_load_rslt12_m128i, 6);
    hor_sse4_unpack8_c13<0>(out, c13_alignr_rslt12_m128i); // Unpack 15th 8 values.
    hor_sse4_unpack8_c13<3>(out, c13_load_rslt13_m128i);   // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack8_c13(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 1st 4 values.
		const __m128i Hor_SSE4_c13_shfl_msk0_m128i = _mm_set_epi8(
				0xFF, byte + 6, byte + 5, byte + 4,
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, byte + 3, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c13_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c13_shfl_msk0_m128i);
		__m128i c13_mul_rslt_m128i = _mm_mullo_epi32(c13_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[13][0]);
		__m128i c13_srli_rslt_m128i = _mm_srli_epi32(c13_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[13][0]);
		__m128i c13_rslt_m128i = _mm_and_si128(c13_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[13]);
		_mm_storeu_si128(out++, c13_rslt_m128i);

		// Unpack 2nd 4 values.
		const __m128i Hor_SSE4_c13_shfl_msk1_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 12, byte + 11,
				0xFF, byte + 11, byte + 10, byte + 9,
				0xFF, 0xFF, byte + 9, byte + 8,
				0xFF, byte + 8, byte + 7, byte + 6);
		c13_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c13_shfl_msk1_m128i);
		c13_mul_rslt_m128i = _mm_mullo_epi32(c13_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[13][1]);
		c13_srli_rslt_m128i = _mm_srli_epi32(c13_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[13][1]);
		c13_rslt_m128i = _mm_and_si128(c13_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[13]);
		_mm_storeu_si128(out++, c13_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 1st 4 values.
		const __m128i Hor_SSE4_c13_shfl_msk0_m128i = _mm_set_epi8(
				0xFF, byte + 6, byte + 5, byte + 4,
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, byte + 3, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c13_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c13_shfl_msk0_m128i);
		__m128i c13_mul_rslt_m128i = _mm_mullo_epi32(c13_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[13][0]);
		__m128i c13_srli_rslt_m128i = _mm_srli_epi32(c13_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[13][0]);
		__m128i c13_and_rslt_m128i = _mm_and_si128(c13_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[13]);
		__m128i c13_rslt_m128i = _mm_or_si128(c13_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 13));
		_mm_storeu_si128(out++, c13_rslt_m128i);

		// Unpack 2nd 4 values.
		const __m128i Hor_SSE4_c13_shfl_msk1_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 12, byte + 11,
				0xFF, byte + 11, byte + 10, byte + 9,
				0xFF, 0xFF, byte + 9, byte + 8,
				0xFF, byte + 8, byte + 7, byte + 6);
		c13_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c13_shfl_msk1_m128i);
		c13_mul_rslt_m128i = _mm_mullo_epi32(c13_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[13][1]);
		c13_srli_rslt_m128i = _mm_srli_epi32(c13_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[13][1]);
		c13_and_rslt_m128i = _mm_and_si128(c13_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[13]);
		c13_rslt_m128i = _mm_or_si128(c13_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 13));
		_mm_storeu_si128(out++, c13_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 14-bit values.
 * Load 14 SSE vectors, each containing 9 14-bit values. (10th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c14(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
	     __m128i c14_load_rslt1_m128i = _mm_loadu_si128(in++);
	     hor_sse4_unpack4_c14<0>(out, c14_load_rslt1_m128i);  // Unpack 1st 4 values.
	     hor_sse4_unpack4_c14<7>(out, c14_load_rslt1_m128i);  // Unpack 2nd 4 values.

	     __m128i c14_load_rslt2_m128i = _mm_loadu_si128(in++);
	     __m128i c14_alignr_rslt1_m128i = _mm_alignr_epi8(c14_load_rslt2_m128i, c14_load_rslt1_m128i, 14);
	     hor_sse4_unpack4_c14<0>(out, c14_alignr_rslt1_m128i); // Unpack 3rd 4 values.
	     hor_sse4_unpack4_c14<7>(out, c14_alignr_rslt1_m128i); // Unpack 4th 4 values.

	     __m128i c14_load_rslt3_m128i = _mm_loadu_si128(in++);
	     __m128i c14_alignr_rslt2_m128i = _mm_alignr_epi8(c14_load_rslt3_m128i, c14_load_rslt2_m128i, 12);
	     hor_sse4_unpack4_c14<0>(out, c14_alignr_rslt2_m128i); // Unpack 5th 4 values.
	     hor_sse4_unpack4_c14<7>(out, c14_alignr_rslt2_m128i); // Unpack 6th 4 values.

	     __m128i c14_load_rslt4_m128i = _mm_loadu_si128(in++);
	     __m128i c14_alignr_rslt3_m128i = _mm_alignr_epi8(c14_load_rslt4_m128i, c14_load_rslt3_m128i, 10);
	     hor_sse4_unpack4_c14<0>(out, c14_alignr_rslt3_m128i); // Unpack 7th 4 values.
	     hor_sse4_unpack4_c14<7>(out, c14_alignr_rslt3_m128i); // Unpack 8th 4 values.
	     hor_sse4_unpack4_c14<8>(out, c14_load_rslt4_m128i);   // Unpack 9th 4 values.

	     __m128i c14_load_rslt5_m128i = _mm_loadu_si128(in++);
	     __m128i c14_alignr_rslt4_m128i = _mm_alignr_epi8(c14_load_rslt5_m128i, c14_load_rslt4_m128i, 15);
	     hor_sse4_unpack4_c14<0>(out, c14_alignr_rslt4_m128i); // Unpack 10th 4 values.
	     hor_sse4_unpack4_c14<7>(out, c14_alignr_rslt4_m128i); // Unpack 11th 4 values.

	     __m128i c14_load_rslt6_m128i = _mm_loadu_si128(in++);
	     __m128i c14_alignr_rslt5_m128i = _mm_alignr_epi8(c14_load_rslt6_m128i, c14_load_rslt5_m128i, 13);
	     hor_sse4_unpack4_c14<0>(out, c14_alignr_rslt5_m128i); // Unpack 12th 4 values.
	     hor_sse4_unpack4_c14<7>(out, c14_alignr_rslt5_m128i); // Unpack 13th 4 values.

	     __m128i c14_load_rslt7_m128i = _mm_loadu_si128(in++);
	     __m128i c14_alignr_rslt6_m128i = _mm_alignr_epi8(c14_load_rslt7_m128i, c14_load_rslt6_m128i, 11);
	     hor_sse4_unpack4_c14<0>(out, c14_alignr_rslt6_m128i); // Unpack 14th 4 values.
	     hor_sse4_unpack4_c14<7>(out, c14_alignr_rslt6_m128i); // Unpack 15th 4 values.
	     hor_sse4_unpack4_c14<9>(out, c14_load_rslt7_m128i);   // Unpack 16th 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c14(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i SSE4_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 6, byte + 5,
				0xFF, byte + 5, byte + 4, byte + 3,
				0xFF, byte + 3, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c14_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, SSE4_shfl_msk_m128i);
		__m128i c14_mul_rslt_m128i = _mm_mullo_epi32(c14_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[14][0]);
		__m128i c14_srli_rslt_m128i = _mm_srli_epi32(c14_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[14][0]);
		__m128i c14_rslt_m128i = _mm_and_si128(c14_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[14]);
		_mm_storeu_si128(out++, c14_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i SSE4_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 6, byte + 5,
				0xFF, byte + 5, byte + 4, byte + 3,
				0xFF, byte + 3, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c14_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, SSE4_shfl_msk_m128i);
		__m128i c14_mul_rslt_m128i = _mm_mullo_epi32(c14_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[14][0]);
		__m128i c14_srli_rslt_m128i = _mm_srli_epi32(c14_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[14][0]);
		__m128i c14_and_rslt_m128i = _mm_and_si128(c14_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[14]);
		__m128i c14_rslt_m128i = _mm_or_si128(c14_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 14));
		_mm_storeu_si128(out++, c14_rslt_m128i);

	}
}


/**
 * SSE4-based unpacking 128 15-bit values.
 * Load 15 SSE vectors, each containing 8 15-bit values. (9th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c15(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
    __m128i c15_load_rslt1_m128i = _mm_loadu_si128(in + 0);
    hor_sse4_unpack8_c15<0>(out, c15_load_rslt1_m128i);    // Unpack 1st 8 values.

    __m128i c15_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c15_alignr_rslt1_m128i = _mm_alignr_epi8(c15_load_rslt2_m128i, c15_load_rslt1_m128i, 15);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt1_m128i);  // Unpack 2nd 8 values.

    __m128i c15_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c15_alignr_rslt2_m128i = _mm_alignr_epi8(c15_load_rslt3_m128i, c15_load_rslt2_m128i, 14);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt2_m128i);  // Unpack 3rd 8 values.

    __m128i c15_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c15_alignr_rslt3_m128i = _mm_alignr_epi8(c15_load_rslt4_m128i, c15_load_rslt3_m128i, 13);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt3_m128i);  // Unpack 4th 8 values.

    __m128i c15_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c15_alignr_rslt4_m128i = _mm_alignr_epi8(c15_load_rslt5_m128i, c15_load_rslt4_m128i, 12);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt4_m128i);  // Unpack 5th 8 values.

    __m128i c15_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c15_alignr_rslt5_m128i = _mm_alignr_epi8(c15_load_rslt6_m128i, c15_load_rslt5_m128i, 11);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt5_m128i);  // Unpack 6th 8 values.

    __m128i c15_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c15_alignr_rslt6_m128i = _mm_alignr_epi8(c15_load_rslt7_m128i, c15_load_rslt6_m128i, 10);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt6_m128i);  // Unpack 7th 8 values.

    __m128i c15_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c15_alignr_rslt7_m128i = _mm_alignr_epi8(c15_load_rslt8_m128i, c15_load_rslt7_m128i, 9);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt7_m128i);  // Unpack 8th 8 values.

    __m128i c15_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c15_alignr_rslt8_m128i = _mm_alignr_epi8(c15_load_rslt9_m128i, c15_load_rslt8_m128i, 8);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt8_m128i);  // Unpack 9th 8 values.

    __m128i c15_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c15_alignr_rslt9_m128i = _mm_alignr_epi8(c15_load_rslt10_m128i, c15_load_rslt9_m128i, 7);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt9_m128i);  // Unpack 10th 8 values.

    __m128i c15_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c15_alignr_rslt10_m128i = _mm_alignr_epi8(c15_load_rslt11_m128i, c15_load_rslt10_m128i, 6);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt10_m128i); // Unpack 11th 8 values.

    __m128i c15_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c15_alignr_rslt11_m128i = _mm_alignr_epi8(c15_load_rslt12_m128i, c15_load_rslt11_m128i, 5);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt11_m128i); // Unpack 12th 8 values.

    __m128i c15_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c15_alignr_rslt12_m128i = _mm_alignr_epi8(c15_load_rslt13_m128i, c15_load_rslt12_m128i, 4);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt12_m128i); // Unpack 13th 8 values.

    __m128i c15_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c15_alignr_rslt13_m128i = _mm_alignr_epi8(c15_load_rslt14_m128i, c15_load_rslt13_m128i, 3);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt13_m128i); // Unpack 14th 8 values.

    __m128i c15_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c15_alignr_rslt14_m128i = _mm_alignr_epi8(c15_load_rslt15_m128i, c15_load_rslt14_m128i, 2);
    hor_sse4_unpack8_c15<0>(out, c15_alignr_rslt14_m128i); // Unpack 15th 8 values.
    hor_sse4_unpack8_c15<1>(out, c15_load_rslt15_m128i);   // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack8_c15(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 1st 4 values.
		const __m128i Hor_SSE4_c15_shfl_msk0_m128i = _mm_set_epi8(
				0xFF, byte + 7, byte + 6, byte + 5,
				0xFF, byte + 5, byte + 4, byte + 3,
				0xFF, byte + 3, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c15_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c15_shfl_msk0_m128i);
		__m128i c15_mul_rslt_m128i = _mm_mullo_epi32(c15_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[15][0]);
		__m128i c15_srli_rslt_m128i = _mm_srli_epi32(c15_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[15][0]);
		__m128i c15_rslt_m128i = _mm_and_si128(c15_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[15]);
		_mm_storeu_si128(out++, c15_rslt_m128i);

		// Unpack 2nd 4 values.
		const __m128i Hor_SSE4_c15_shfl_msk1_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 14, byte + 13,
				0xFF, byte + 13, byte + 12, byte + 11,
				0xFF, byte + 11, byte + 10, byte + 9,
				0xFF, byte + 9, byte + 8, byte + 7);
		c15_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c15_shfl_msk1_m128i);
		c15_mul_rslt_m128i = _mm_mullo_epi32(c15_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[15][1]);
		c15_srli_rslt_m128i = _mm_srli_epi32(c15_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[15][1]);
		c15_rslt_m128i = _mm_and_si128(c15_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[15]);
		_mm_storeu_si128(out++, c15_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 1st 4 values.
		const __m128i Hor_SSE4_c15_shfl_msk0_m128i = _mm_set_epi8(
				0xFF, byte + 7, byte + 6, byte + 5,
				0xFF, byte + 5, byte + 4, byte + 3,
				0xFF, byte + 3, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m128i c15_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c15_shfl_msk0_m128i);
		__m128i c15_mul_rslt_m128i = _mm_mullo_epi32(c15_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[15][0]);
		__m128i c15_srli_rslt_m128i = _mm_srli_epi32(c15_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[15][0]);
		__m128i c15_and_rslt_m128i = _mm_and_si128(c15_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[15]);
		__m128i c15_rslt_m128i = _mm_or_si128(c15_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 15));
		_mm_storeu_si128(out++, c15_rslt_m128i);

		// Unpack 2nd 4 values.
		const __m128i Hor_SSE4_c15_shfl_msk1_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 14, byte + 13,
				0xFF, byte + 13, byte + 12, byte + 11,
				0xFF, byte + 11, byte + 10, byte + 9,
				0xFF, byte + 9, byte + 8, byte + 7);
		c15_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c15_shfl_msk1_m128i);
		c15_mul_rslt_m128i = _mm_mullo_epi32(c15_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[15][1]);
		c15_srli_rslt_m128i = _mm_srli_epi32(c15_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[15][1]);
		c15_and_rslt_m128i = _mm_and_si128(c15_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[15]);
		c15_rslt_m128i = _mm_or_si128(c15_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 15));
		_mm_storeu_si128(out++, c15_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 16-bit values.
 * Load 16 SSE vectors, each containing 8 16-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c16(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 8) {
		__m128i c16_load_rslt_m128i = _mm_loadu_si128(in++);
		hor_sse4_unpack4_c16<0>(out, c16_load_rslt_m128i); // Unpack 1st 4 values.
		hor_sse4_unpack4_c16<8>(out, c16_load_rslt_m128i); // Unpack 2nd 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c16(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c16_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 7, byte + 6,
				0xFF, 0xFF, byte + 5, byte + 4,
				0xFF, 0xFF, byte + 3, byte + 2,
				0xFF, 0xFF, byte + 1, byte + 0);

		__m128i c16_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c16_shfl_msk_m128i);
		_mm_storeu_si128(out++, c16_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c16_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, byte + 7, byte + 6,
				0xFF, 0xFF, byte + 5, byte + 4,
				0xFF, 0xFF, byte + 3, byte + 2,
				0xFF, 0xFF, byte + 1, byte + 0);

		__m128i c16_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c16_shfl_msk_m128i);
		__m128i c16_rslt_m128i = _mm_or_si128(c16_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 16));
		_mm_storeu_si128(out++, c16_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 17-bit values.
 * Load 17 SSE vectors, each containing 7 17-bit values. (8th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c17(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c17_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	hor_sse4_unpack4_c17_f1<0>(out, c17_load_rslt1_m128i);    // Unpack 1st 4 values.

	__m128i c17_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c17_alignr_rslt1_m128i = _mm_alignr_epi8(c17_load_rslt2_m128i, c17_load_rslt1_m128i, 8);
	hor_sse4_unpack4_c17_f2<0>(out, c17_alignr_rslt1_m128i);  // Unpack 2nd 4 values.
	hor_sse4_unpack4_c17_f1<1>(out, c17_load_rslt2_m128i);    // Unpack 3rd 4 values.

	__m128i c17_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c17_alignr_rslt2_m128i = _mm_alignr_epi8(c17_load_rslt3_m128i, c17_load_rslt2_m128i, 9);
	hor_sse4_unpack4_c17_f2<0>(out, c17_alignr_rslt2_m128i);  // Unpack 4th 4 values.
	hor_sse4_unpack4_c17_f1<2>(out, c17_load_rslt3_m128i);    // Unpack 5th 4 values.

	__m128i c17_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c17_alignr_rslt3_m128i = _mm_alignr_epi8(c17_load_rslt4_m128i, c17_load_rslt3_m128i, 10);
	hor_sse4_unpack4_c17_f2<0>(out, c17_alignr_rslt3_m128i);  // Unpack 6th 4 values.
	hor_sse4_unpack4_c17_f1<3>(out, c17_load_rslt4_m128i);    // Unpack 7th 4 values.

	__m128i c17_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c17_alignr_rslt4_m128i = _mm_alignr_epi8(c17_load_rslt5_m128i, c17_load_rslt4_m128i, 11);
	hor_sse4_unpack4_c17_f2<0>(out, c17_alignr_rslt4_m128i);  // Unpack 8th 4 values.
	hor_sse4_unpack4_c17_f1<4>(out, c17_load_rslt5_m128i);    // Unpack 9th 4 values.

	__m128i c17_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c17_alignr_rslt5_m128i = _mm_alignr_epi8(c17_load_rslt6_m128i, c17_load_rslt5_m128i, 12);
	hor_sse4_unpack4_c17_f2<0>(out, c17_alignr_rslt5_m128i);  // Unpack 10th 4 values.
	hor_sse4_unpack4_c17_f1<5>(out, c17_load_rslt6_m128i);    // Unpack 11th 4 values.

	__m128i c17_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c17_alignr_rslt6_m128i = _mm_alignr_epi8(c17_load_rslt7_m128i, c17_load_rslt6_m128i, 13);
	hor_sse4_unpack4_c17_f2<0>(out, c17_alignr_rslt6_m128i);  // Unpack 12th 4 values.
	hor_sse4_unpack4_c17_f1<6>(out, c17_load_rslt7_m128i);    // Unpack 13th 4 values.

	__m128i c17_load_rslt8_m128i = _mm_loadu_si128(in + 7);
	__m128i c17_alignr_rslt7_m128i = _mm_alignr_epi8(c17_load_rslt8_m128i, c17_load_rslt7_m128i, 14);
	hor_sse4_unpack4_c17_f2<0>(out, c17_alignr_rslt7_m128i);  // Unpack 14th 4 values.
	hor_sse4_unpack4_c17_f1<7>(out, c17_load_rslt8_m128i);    // Unpack 15th 4 values.

	__m128i c17_load_rslt9_m128i = _mm_loadu_si128(in + 8);
	__m128i c17_alignr_rslt8_m128i = _mm_alignr_epi8(c17_load_rslt9_m128i, c17_load_rslt8_m128i, 15);
	hor_sse4_unpack4_c17_f2<0>(out, c17_alignr_rslt8_m128i);  // Unpack 16th 4 values.

	__m128i c17_load_rslt10_m128i = _mm_loadu_si128(in + 9);
	__m128i c17_alignr_rslt9_m128i = _mm_alignr_epi8(c17_load_rslt10_m128i, c17_load_rslt9_m128i, 8);
	hor_sse4_unpack4_c17_f1<0>(out, c17_alignr_rslt9_m128i);  // Unpack 17th 4 values.
	hor_sse4_unpack4_c17_f2<0>(out, c17_load_rslt10_m128i);   // Unpack 18th 4 values.

	__m128i c17_load_rslt11_m128i = _mm_loadu_si128(in + 10);
	__m128i c17_alignr_rslt10_m128i = _mm_alignr_epi8(c17_load_rslt11_m128i, c17_load_rslt10_m128i, 9);
	hor_sse4_unpack4_c17_f1<0>(out, c17_alignr_rslt10_m128i); // Unpack 19th 4 values.
	hor_sse4_unpack4_c17_f2<1>(out, c17_load_rslt11_m128i);   // Unpack 20th 4 values.

	__m128i c17_load_rslt12_m128i = _mm_loadu_si128(in + 11);
	__m128i c17_alignr_rslt11_m128i = _mm_alignr_epi8(c17_load_rslt12_m128i, c17_load_rslt11_m128i, 10);
	hor_sse4_unpack4_c17_f1<0>(out, c17_alignr_rslt11_m128i); // Unpack 21st 4 values.
	hor_sse4_unpack4_c17_f2<2>(out, c17_load_rslt12_m128i);   // Unpack 22nd 4 values.

	__m128i c17_load_rslt13_m128i = _mm_loadu_si128(in + 12);
	__m128i c17_alignr_rslt12_m128i = _mm_alignr_epi8(c17_load_rslt13_m128i, c17_load_rslt12_m128i, 11);
	hor_sse4_unpack4_c17_f1<0>(out, c17_alignr_rslt12_m128i); // Unpack 23rd 4 values.
	hor_sse4_unpack4_c17_f2<3>(out, c17_load_rslt13_m128i);   // Unpack 24th 4 values.

	__m128i c17_load_rslt14_m128i = _mm_loadu_si128(in + 13);
	__m128i c17_alignr_rslt13_m128i = _mm_alignr_epi8(c17_load_rslt14_m128i, c17_load_rslt13_m128i, 12);
	hor_sse4_unpack4_c17_f1<0>(out, c17_alignr_rslt13_m128i); // Unpack 25th 4 values.
	hor_sse4_unpack4_c17_f2<4>(out, c17_load_rslt14_m128i);   // Unpack 26th 4 values.

	__m128i c17_load_rslt15_m128i = _mm_loadu_si128(in + 14);
	__m128i c17_alignr_rslt14_m128i = _mm_alignr_epi8(c17_load_rslt15_m128i, c17_load_rslt14_m128i, 13);
	hor_sse4_unpack4_c17_f1<0>(out, c17_alignr_rslt14_m128i); // Unpack 27th 4 values.
	hor_sse4_unpack4_c17_f2<5>(out, c17_load_rslt15_m128i);   // Unpack 28th 4 values.

	__m128i c17_load_rslt16_m128i = _mm_loadu_si128(in + 15);
	__m128i c17_alignr_rslt15_m128i = _mm_alignr_epi8(c17_load_rslt16_m128i, c17_load_rslt15_m128i, 14);
	hor_sse4_unpack4_c17_f1<0>(out, c17_alignr_rslt15_m128i); // Unpack 29th 4 values.
	hor_sse4_unpack4_c17_f2<6>(out, c17_load_rslt16_m128i);   // Unpack 30th 4 values.

	__m128i c17_load_rslt17_m128i = _mm_loadu_si128(in + 16);
	__m128i c17_alignr_rslt16_m128i = _mm_alignr_epi8(c17_load_rslt17_m128i, c17_load_rslt16_m128i, 15);
	hor_sse4_unpack4_c17_f1<0>(out, c17_alignr_rslt16_m128i); // Unpack 31st 4 values.
	hor_sse4_unpack4_c17_f2<7>(out, c17_load_rslt17_m128i);   // Unpack 32nd 4 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c17_f1(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c17_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, byte + 6, byte + 5, byte + 4,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c17_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c17_shfl_msk_m128i);
		__m128i c17_mul_rslt_m128i = _mm_mullo_epi32(c17_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[17][0]);
		__m128i c17_srli_rslt_m128i = _mm_srli_epi32(c17_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[17][0]);
		__m128i c17_rslt_m128i = _mm_and_si128(c17_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[17]);
		_mm_storeu_si128(out++, c17_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c17_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, byte + 6, byte + 5, byte + 4,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c17_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c17_shfl_msk_m128i);
		__m128i c17_mul_rslt_m128i = _mm_mullo_epi32(c17_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[17][0]);
		__m128i c17_srli_rslt_m128i = _mm_srli_epi32(c17_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[17][0]);
		__m128i c17_and_rslt_m128i = _mm_and_si128(c17_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[17]);
		__m128i c17_rslt_m128i = _mm_or_si128(c17_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 17));
		_mm_storeu_si128(out++, c17_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c17_f2(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c17_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, byte + 6, byte + 5, byte + 4,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c17_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c17_shfl_msk_m128i);
		__m128i c17_mul_rslt_m128i = _mm_mullo_epi32(c17_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[17][1]);
		__m128i c17_srli_rslt_m128i = _mm_srli_epi32(c17_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[17][1]);
		__m128i c17_rslt_m128i = _mm_and_si128(c17_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[17]);
		_mm_storeu_si128(out++, c17_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c17_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, byte + 6, byte + 5, byte + 4,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c17_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c17_shfl_msk_m128i);
		__m128i c17_mul_rslt_m128i = _mm_mullo_epi32(c17_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[17][1]);
		__m128i c17_srli_rslt_m128i = _mm_srli_epi32(c17_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[17][1]);
		__m128i c17_and_rslt_m128i = _mm_and_si128(c17_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[17]);
		__m128i c17_rslt_m128i = _mm_or_si128(c17_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 17));
		_mm_storeu_si128(out++, c17_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 18-bit values.
 * Load 18 SSE vectors, each containing 7 18-bit values. (8th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c18(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
		__m128i c18_load_rslt1_m128i = _mm_loadu_si128(in++);
		hor_sse4_unpack4_c18<0>(out, c18_load_rslt1_m128i);   // Unpack 1st 4 values.

		__m128i c18_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt1_m128i = _mm_alignr_epi8(c18_load_rslt2_m128i, c18_load_rslt1_m128i, 9);
		hor_sse4_unpack4_c18<0>(out, c18_alignr_rslt1_m128i); // Unpack 2nd 4 values.
		hor_sse4_unpack4_c18<2>(out, c18_load_rslt2_m128i);   // Unpack 3rd 4 values.

		__m128i c18_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt2_m128i = _mm_alignr_epi8(c18_load_rslt3_m128i, c18_load_rslt2_m128i, 11);
		hor_sse4_unpack4_c18<0>(out, c18_alignr_rslt2_m128i); // Unpack 4th 4 values.
		hor_sse4_unpack4_c18<4>(out, c18_load_rslt3_m128i);   // Unpack 5th 4 values.

		__m128i c18_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt3_m128i = _mm_alignr_epi8(c18_load_rslt4_m128i, c18_load_rslt3_m128i, 13);
		hor_sse4_unpack4_c18<0>(out, c18_alignr_rslt3_m128i); // Unpack 6th 4 values.
		hor_sse4_unpack4_c18<6>(out, c18_load_rslt4_m128i);   // Unpack 7th 4 values.

		__m128i c18_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt4_m128i = _mm_alignr_epi8(c18_load_rslt5_m128i, c18_load_rslt4_m128i, 15);
		hor_sse4_unpack4_c18<0>(out, c18_alignr_rslt4_m128i); // Unpack 8th 4 values.

		__m128i c18_load_rslt6_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt5_m128i = _mm_alignr_epi8(c18_load_rslt6_m128i, c18_load_rslt5_m128i, 8);
		hor_sse4_unpack4_c18<0>(out, c18_alignr_rslt5_m128i); // Unpack 9th 4 values.
		hor_sse4_unpack4_c18<1>(out, c18_load_rslt6_m128i);   // Unpack 10th 4 values.

		__m128i c18_load_rslt7_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt6_m128i = _mm_alignr_epi8(c18_load_rslt7_m128i, c18_load_rslt6_m128i, 10);
		hor_sse4_unpack4_c18<0>(out, c18_alignr_rslt6_m128i); // Unpack 11th 4 values.
		hor_sse4_unpack4_c18<3>(out, c18_load_rslt7_m128i);   // Unpack 12th 4 values.

		__m128i c18_load_rslt8_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt7_m128i = _mm_alignr_epi8(c18_load_rslt8_m128i, c18_load_rslt7_m128i, 12);
		hor_sse4_unpack4_c18<0>(out, c18_alignr_rslt7_m128i); // Unpack 13th 4 values.
		hor_sse4_unpack4_c18<5>(out, c18_load_rslt8_m128i);   // Unpack 14th 4 values.

		__m128i c18_load_rslt9_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt8_m128i = _mm_alignr_epi8(c18_load_rslt9_m128i, c18_load_rslt8_m128i, 14);
		hor_sse4_unpack4_c18<0>(out, c18_alignr_rslt8_m128i); // Unpack 15th 4 values.
		hor_sse4_unpack4_c18<7>(out, c18_load_rslt9_m128i);   // Unpack 16th 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c18(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c18_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, byte + 6, byte + 5, byte + 4,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c18_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c18_shfl_msk_m128i);
		__m128i c18_mul_rslt_m128i = _mm_mullo_epi32(c18_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[18][0]);
		__m128i c18_srli_rslt_m128i = _mm_srli_epi32(c18_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[18][0]);
		__m128i c18_rslt_m128i = _mm_and_si128(c18_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[18]);
		_mm_storeu_si128(out++, c18_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c18_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, byte + 6, byte + 5, byte + 4,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c18_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c18_shfl_msk_m128i);
		__m128i c18_mul_rslt_m128i = _mm_mullo_epi32(c18_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[18][0]);
		__m128i c18_srli_rslt_m128i = _mm_srli_epi32(c18_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[18][0]);
		__m128i c18_and_rslt_m128i = _mm_and_si128(c18_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[18]);
		__m128i c18_rslt_m128i = _mm_or_si128(c18_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 18));
		_mm_storeu_si128(out++, c18_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 19-bit values.
 * Load 19 SSE vectors, each containing 6 19-bit values. (7th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c19(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
    __m128i c19_load_rslt1_m128i = _mm_loadu_si128(in + 0);
    hor_sse4_unpack4_c19_f1<0>(out, c19_load_rslt1_m128i);    // Unpack 1st 4 values.

    __m128i c19_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c19_alignr_rslt1_m128i = _mm_alignr_epi8(c19_load_rslt2_m128i, c19_load_rslt1_m128i, 9);
    hor_sse4_unpack4_c19_f2<0>(out, c19_alignr_rslt1_m128i);  // Unpack 2nd 4 values.
    hor_sse4_unpack4_c19_f1<3>(out, c19_load_rslt2_m128i);    // Unpack 3rd 4 values.

    __m128i c19_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c19_alignr_rslt2_m128i = _mm_alignr_epi8(c19_load_rslt3_m128i, c19_load_rslt2_m128i, 12);
    hor_sse4_unpack4_c19_f2<0>(out, c19_alignr_rslt2_m128i);  // Unpack 4th 4 values.
    hor_sse4_unpack4_c19_f1<6>(out, c19_load_rslt3_m128i);    // Unpack 5th 4 values.

    __m128i c19_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c19_alignr_rslt3_m128i = _mm_alignr_epi8(c19_load_rslt4_m128i, c19_load_rslt3_m128i, 15);
    hor_sse4_unpack4_c19_f2<0>(out, c19_alignr_rslt3_m128i);  // Unpack 6th 4 values.

    __m128i c19_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c19_alignr_rslt4_m128i = _mm_alignr_epi8(c19_load_rslt5_m128i, c19_load_rslt4_m128i, 9);
    hor_sse4_unpack4_c19_f1<0>(out, c19_alignr_rslt4_m128i);  // Unpack 7th 4 values.
    hor_sse4_unpack4_c19_f2<2>(out, c19_load_rslt5_m128i);    // Unpack 8th 4 values.

    __m128i c19_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c19_alignr_rslt5_m128i = _mm_alignr_epi8(c19_load_rslt6_m128i, c19_load_rslt5_m128i, 12);
    hor_sse4_unpack4_c19_f1<0>(out, c19_alignr_rslt5_m128i);  // Unpack 9th 4 values.
    hor_sse4_unpack4_c19_f2<5>(out, c19_load_rslt6_m128i);    // Unpack 10th 4 values.

    __m128i c19_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c19_alignr_rslt6_m128i = _mm_alignr_epi8(c19_load_rslt7_m128i, c19_load_rslt6_m128i, 15);
    hor_sse4_unpack4_c19_f1<0>(out, c19_alignr_rslt6_m128i);  // Unpack 11th 4 values.

    __m128i c19_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c19_alignr_rslt7_m128i = _mm_alignr_epi8(c19_load_rslt8_m128i, c19_load_rslt7_m128i, 8);
    hor_sse4_unpack4_c19_f2<0>(out, c19_alignr_rslt7_m128i);  // Unpack 12th 4 values.
    hor_sse4_unpack4_c19_f1<2>(out, c19_load_rslt8_m128i);    // Unpack 13th 4 values.

    __m128i c19_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c19_alignr_rslt8_m128i = _mm_alignr_epi8(c19_load_rslt9_m128i, c19_load_rslt8_m128i, 11);
    hor_sse4_unpack4_c19_f2<0>(out, c19_alignr_rslt8_m128i);  // Unpack 14th 4 values.
    hor_sse4_unpack4_c19_f1<5>(out, c19_load_rslt9_m128i);    // Unpack 15th 4 values.

    __m128i c19_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c19_alignr_rslt9_m128i = _mm_alignr_epi8(c19_load_rslt10_m128i, c19_load_rslt9_m128i, 14);
    hor_sse4_unpack4_c19_f2<0>(out, c19_alignr_rslt9_m128i);  // Unpack 16th 4 values.

    __m128i c19_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c19_alignr_rslt10_m128i = _mm_alignr_epi8(c19_load_rslt11_m128i, c19_load_rslt10_m128i, 8);
    hor_sse4_unpack4_c19_f1<0>(out, c19_alignr_rslt10_m128i); // Unpack 17th 4 values.
    hor_sse4_unpack4_c19_f2<1>(out, c19_load_rslt11_m128i);   // Unpack 18th 4 values.

    __m128i c19_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c19_alignr_rslt11_m128i = _mm_alignr_epi8(c19_load_rslt12_m128i, c19_load_rslt11_m128i, 11);
    hor_sse4_unpack4_c19_f1<0>(out, c19_alignr_rslt11_m128i); // Unpack 19th 4 values.
    hor_sse4_unpack4_c19_f2<4>(out, c19_load_rslt12_m128i);   // Unpack 20th 4 values.

    __m128i c19_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c19_alignr_rslt12_m128i = _mm_alignr_epi8(c19_load_rslt13_m128i, c19_load_rslt12_m128i, 14);
    hor_sse4_unpack4_c19_f1<0>(out, c19_alignr_rslt12_m128i); // Unpack 21st 4 values.

    __m128i c19_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c19_alignr_rslt13_m128i = _mm_alignr_epi8(c19_load_rslt14_m128i, c19_load_rslt13_m128i, 7);
    hor_sse4_unpack4_c19_f2<0>(out, c19_alignr_rslt13_m128i); // Unpack 22nd 4 values.
    hor_sse4_unpack4_c19_f1<1>(out, c19_load_rslt14_m128i);   // Unpack 23rd 4 values.

    __m128i c19_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c19_alignr_rslt14_m128i = _mm_alignr_epi8(c19_load_rslt15_m128i, c19_load_rslt14_m128i, 10);
    hor_sse4_unpack4_c19_f2<0>(out, c19_alignr_rslt14_m128i); // Unpack 24th 4 values.
    hor_sse4_unpack4_c19_f1<4>(out, c19_load_rslt15_m128i);   // Unpack 25th 4 values.

    __m128i c19_load_rslt16_m128i = _mm_loadu_si128(in + 15);
    __m128i c19_alignr_rslt15_m128i = _mm_alignr_epi8(c19_load_rslt16_m128i, c19_load_rslt15_m128i, 13);
    hor_sse4_unpack4_c19_f2<0>(out, c19_alignr_rslt15_m128i); // Unpack 26th 4 values.

    __m128i c19_load_rslt17_m128i = _mm_loadu_si128(in + 16);
    __m128i c19_alignr_rslt16_m128i = _mm_alignr_epi8(c19_load_rslt17_m128i, c19_load_rslt16_m128i, 7);
    hor_sse4_unpack4_c19_f1<0>(out, c19_alignr_rslt16_m128i); // Unpack 27th 4 values.
    hor_sse4_unpack4_c19_f2<0>(out, c19_load_rslt17_m128i);   // Unpack 28th 4 values.

    __m128i c19_load_rslt18_m128i = _mm_loadu_si128(in + 17);
    __m128i c19_alignr_rslt17_m128i = _mm_alignr_epi8(c19_load_rslt18_m128i, c19_load_rslt17_m128i, 10);
    hor_sse4_unpack4_c19_f1<0>(out, c19_alignr_rslt17_m128i); // Unpack 29th 4 values.
    hor_sse4_unpack4_c19_f2<3>(out, c19_load_rslt18_m128i);   // Unpack 30th 4 values.

    __m128i c19_load_rslt19_m128i = _mm_loadu_si128(in + 18);
    __m128i c19_alignr_rslt18_m128i = _mm_alignr_epi8(c19_load_rslt19_m128i, c19_load_rslt18_m128i, 13);
    hor_sse4_unpack4_c19_f1<0>(out, c19_alignr_rslt18_m128i); // Unpack 31th 4 values
    hor_sse4_unpack4_c19_f2<6>(out, c19_load_rslt19_m128i);   // Unpack 32th 4 values
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c19_f1(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c19_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c19_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c19_shfl_msk_m128i);
		__m128i c19_mul_rslt_m128i = _mm_mullo_epi32(c19_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[19][0]);
		__m128i c19_srli_rslt_m128i = _mm_srli_epi32(c19_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[19][0]);
		__m128i c19_rslt_m128i = _mm_and_si128(c19_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[19]);
		_mm_storeu_si128(out++, c19_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c19_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c19_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c19_shfl_msk_m128i);
		__m128i c19_mul_rslt_m128i = _mm_mullo_epi32(c19_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[19][0]);
		__m128i c19_srli_rslt_m128i = _mm_srli_epi32(c19_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[19][0]);
		__m128i c19_and_rslt_m128i = _mm_and_si128(c19_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[19]);
		__m128i c19_rslt_m128i = _mm_or_si128(c19_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 19));
		_mm_storeu_si128(out++, c19_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c19_f2(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c19_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 9, byte + 8, byte + 7,
				0xFF, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c19_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c19_shfl_msk_m128i);
		__m128i c19_mul_rslt_m128i = _mm_mullo_epi32(c19_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[19][1]);
		__m128i c19_srli_rslt_m128i = _mm_srli_epi32(c19_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[19][1]);
		__m128i c19_rslt_m128i = _mm_and_si128(c19_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[19]);
		_mm_storeu_si128(out++, c19_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c19_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 9, byte + 8, byte + 7,
				0xFF, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c19_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c19_shfl_msk_m128i);
		__m128i c19_mul_rslt_m128i = _mm_mullo_epi32(c19_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[19][1]);
		__m128i c19_srli_rslt_m128i = _mm_srli_epi32(c19_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[19][1]);
		__m128i c19_and_rslt_m128i = _mm_and_si128(c19_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[19]);
		__m128i c19_rslt_m128i = _mm_or_si128(c19_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 19));
		_mm_storeu_si128(out++, c19_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 20-bit values.
 * Load 20 SSE vectors, each containing 6 20-bit values. (7th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c20(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
		__m128i c20_load_rslt1_m128i = _mm_loadu_si128(in++);
		hor_sse4_unpack4_c20<0>(out, c20_load_rslt1_m128i);   // Unpack 1st 4 values.

		__m128i c20_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c20_alignr_rslt1_m128i = _mm_alignr_epi8(c20_load_rslt2_m128i, c20_load_rslt1_m128i, 10);
		hor_sse4_unpack4_c20<0>(out, c20_alignr_rslt1_m128i); // Unpack 2nd 4 values.
		hor_sse4_unpack4_c20<4>(out, c20_load_rslt2_m128i);   // Unpack 3rd 4 values.

		__m128i c20_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c20_alignr_rslt2_m128i = _mm_alignr_epi8(c20_load_rslt3_m128i, c20_load_rslt2_m128i, 14);
		hor_sse4_unpack4_c20<0>(out, c20_alignr_rslt2_m128i); // Unpack 4th 4 values.

		__m128i c20_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c20_alignr_rslt3_m128i = _mm_alignr_epi8(c20_load_rslt4_m128i, c20_load_rslt3_m128i, 8);
		hor_sse4_unpack4_c20<0>(out, c20_alignr_rslt3_m128i); // Unpack 5th 4 values.
		hor_sse4_unpack4_c20<2>(out, c20_load_rslt4_m128i);   // Unpack 6th 4 values.

		__m128i c20_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c20_alignr_rslt4_m128i = _mm_alignr_epi8(c20_load_rslt5_m128i, c20_load_rslt4_m128i, 12);
		hor_sse4_unpack4_c20<0>(out, c20_alignr_rslt4_m128i); // Unpack 7th 4 values.
		hor_sse4_unpack4_c20<6>(out, c20_load_rslt5_m128i);   // Unpack 8th 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c20(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
	// Unpack 4 values.
	const __m128i Hor_SSE4_c20_shfl_msk_m128i = _mm_set_epi8(
			0xFF, byte + 9, byte + 8, byte + 7,
			0xFF, byte + 7, byte + 6, byte + 5,
			0xFF, byte + 4, byte + 3, byte + 2,
			0xFF, byte + 2, byte + 1, byte + 0);

	__m128i c20_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c20_shfl_msk_m128i);
	__m128i c20_mul_rslt_m128i = _mm_mullo_epi32(c20_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[20][0]);
	__m128i c20_srli_rslt_m128i = _mm_srli_epi32(c20_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[20][0]);
	__m128i c20_rslt_m128i = _mm_and_si128(c20_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[20]);
	_mm_storeu_si128(out++, c20_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c20_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 9, byte + 8, byte + 7,
				0xFF, byte + 7, byte + 6, byte + 5,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c20_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c20_shfl_msk_m128i);
		__m128i c20_mul_rslt_m128i = _mm_mullo_epi32(c20_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[20][0]);
		__m128i c20_srli_rslt_m128i = _mm_srli_epi32(c20_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[20][0]);
		__m128i c20_and_rslt_m128i = _mm_and_si128(c20_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[20]);
		__m128i c20_rslt_m128i = _mm_or_si128(c20_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 20));
		_mm_storeu_si128(out++, c20_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 21-bit values.
 * Load 21 SSE vectors, each containing 6 21-bit values. (7th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c21(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
    __m128i c21_load_rslt1_m128i = _mm_loadu_si128(in + 0);
    hor_sse4_unpack4_c21_f1<0>(out, c21_load_rslt1_m128i);    // Unpack 1st 4 values.

    __m128i c21_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c21_alignr_rslt1_m128i = _mm_alignr_epi8(c21_load_rslt2_m128i, c21_load_rslt1_m128i, 10);
    hor_sse4_unpack4_c21_f2<0>(out, c21_alignr_rslt1_m128i);  // Unpack 2nd 4 values.
    hor_sse4_unpack4_c21_f1<5>(out, c21_load_rslt2_m128i);    // Unpack 3rd 4 values.

    __m128i c21_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c21_alignr_rslt2_m128i = _mm_alignr_epi8(c21_load_rslt3_m128i, c21_load_rslt2_m128i, 15);
    hor_sse4_unpack4_c21_f2<0>(out, c21_alignr_rslt2_m128i);  // Unpack 4th 4 values.

    __m128i c21_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c21_alignr_rslt3_m128i = _mm_alignr_epi8(c21_load_rslt4_m128i, c21_load_rslt3_m128i, 10);
    hor_sse4_unpack4_c21_f1<0>(out, c21_alignr_rslt3_m128i);  // Unpack 5th 4 values.
    hor_sse4_unpack4_c21_f2<4>(out, c21_load_rslt4_m128i);    // Unpack 6th 4 values.

    __m128i c21_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c21_alignr_rslt4_m128i = _mm_alignr_epi8(c21_load_rslt5_m128i, c21_load_rslt4_m128i, 15);
    hor_sse4_unpack4_c21_f1<0>(out, c21_alignr_rslt4_m128i);  // Unpack 7th 4 values.

    __m128i c21_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c21_alignr_rslt5_m128i = _mm_alignr_epi8(c21_load_rslt6_m128i, c21_load_rslt5_m128i, 9);
    hor_sse4_unpack4_c21_f2<0>(out, c21_alignr_rslt5_m128i);  // Unpack 8th 4 values.
    hor_sse4_unpack4_c21_f1<4>(out, c21_load_rslt6_m128i);    // Unpack 9th 4 values.

    __m128i c21_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c21_alignr_rslt6_m128i = _mm_alignr_epi8(c21_load_rslt7_m128i, c21_load_rslt6_m128i, 14);
    hor_sse4_unpack4_c21_f2<0>(out, c21_alignr_rslt6_m128i);  // Unpack 10th 4 values.

    __m128i c21_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c21_alignr_rslt7_m128i = _mm_alignr_epi8(c21_load_rslt8_m128i, c21_load_rslt7_m128i, 9);
    hor_sse4_unpack4_c21_f1<0>(out, c21_alignr_rslt7_m128i);  // Unpack 11th 4 values.
    hor_sse4_unpack4_c21_f2<3>(out, c21_load_rslt8_m128i);    // Unpack 12th 4 values.

    __m128i c21_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c21_alignr_rslt8_m128i = _mm_alignr_epi8(c21_load_rslt9_m128i, c21_load_rslt8_m128i, 14);
    hor_sse4_unpack4_c21_f1<0>(out, c21_alignr_rslt8_m128i);  // Unpack 13th 4 values.

    __m128i c21_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c21_alignr_rslt9_m128i = _mm_alignr_epi8(c21_load_rslt10_m128i, c21_load_rslt9_m128i, 8);
    hor_sse4_unpack4_c21_f2<0>(out, c21_alignr_rslt9_m128i);  // Unpack 14th 4 values.
    hor_sse4_unpack4_c21_f1<3>(out, c21_load_rslt10_m128i);   // Unpack 15th 4 values.

    __m128i c21_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c21_alignr_rslt10_m128i = _mm_alignr_epi8(c21_load_rslt11_m128i, c21_load_rslt10_m128i, 13);
    hor_sse4_unpack4_c21_f2<0>(out, c21_alignr_rslt10_m128i); // Unpack 16th 4 values.

    __m128i c21_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c21_alignr_rslt11_m128i = _mm_alignr_epi8(c21_load_rslt12_m128i, c21_load_rslt11_m128i, 8);
    hor_sse4_unpack4_c21_f1<0>(out, c21_alignr_rslt11_m128i); // Unpack 17th 4 values.
    hor_sse4_unpack4_c21_f2<2>(out, c21_load_rslt12_m128i);   // Unpack 18th 4 values.

    __m128i c21_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c21_alignr_rslt12_m128i = _mm_alignr_epi8(c21_load_rslt13_m128i, c21_load_rslt12_m128i, 13);
    hor_sse4_unpack4_c21_f1<0>(out, c21_alignr_rslt12_m128i); // Unpack 19th 4 values.

    __m128i c21_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c21_alignr_rslt13_m128i = _mm_alignr_epi8(c21_load_rslt14_m128i, c21_load_rslt13_m128i, 7);
    hor_sse4_unpack4_c21_f2<0>(out, c21_alignr_rslt13_m128i); // Unpack 20th 4 values.
    hor_sse4_unpack4_c21_f1<2>(out, c21_load_rslt14_m128i);   // Unpack 21st 4 values.

    __m128i c21_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c21_alignr_rslt14_m128i = _mm_alignr_epi8(c21_load_rslt15_m128i, c21_load_rslt14_m128i, 12);
    hor_sse4_unpack4_c21_f2<0>(out, c21_alignr_rslt14_m128i); // Unpack 22nd 4 values.

    __m128i c21_load_rslt16_m128i = _mm_loadu_si128(in + 15);
    __m128i c21_alignr_rslt15_m128i = _mm_alignr_epi8(c21_load_rslt16_m128i, c21_load_rslt15_m128i, 7);
    hor_sse4_unpack4_c21_f1<0>(out, c21_alignr_rslt15_m128i); // Unpack 23rd 4 values.
    hor_sse4_unpack4_c21_f2<1>(out, c21_load_rslt16_m128i);   // Unpack 24th 4 values.

    __m128i c21_load_rslt17_m128i = _mm_loadu_si128(in + 16);
    __m128i c21_alignr_rslt16_m128i = _mm_alignr_epi8(c21_load_rslt17_m128i, c21_load_rslt16_m128i, 12);
    hor_sse4_unpack4_c21_f1<0>(out, c21_alignr_rslt16_m128i); // Unpack 25th 4 values.

    __m128i c21_load_rslt18_m128i = _mm_loadu_si128(in + 17);
    __m128i c21_alignr_rslt17_m128i = _mm_alignr_epi8(c21_load_rslt18_m128i, c21_load_rslt17_m128i, 6);
    hor_sse4_unpack4_c21_f2<0>(out, c21_alignr_rslt17_m128i); // Unpack 26th 4 values.
    hor_sse4_unpack4_c21_f1<1>(out, c21_load_rslt18_m128i);   // Unpack 27th 4 values.

    __m128i c21_load_rslt19_m128i = _mm_loadu_si128(in + 18);
    __m128i c21_alignr_rslt18_m128i = _mm_alignr_epi8(c21_load_rslt19_m128i, c21_load_rslt18_m128i, 11);
    hor_sse4_unpack4_c21_f2<0>(out, c21_alignr_rslt18_m128i); // Unpack 28th 4 values.

    __m128i c21_load_rslt20_m128i = _mm_loadu_si128(in + 19);
    __m128i c21_alignr_rslt19_m128i = _mm_alignr_epi8(c21_load_rslt20_m128i, c21_load_rslt19_m128i, 6);
    hor_sse4_unpack4_c21_f1<0>(out, c21_alignr_rslt19_m128i); // Unpack 29th 4 values.
    hor_sse4_unpack4_c21_f2<0>(out, c21_load_rslt20_m128i);   // Unpack 30th 4 values.

    __m128i c21_load_rslt21_m128i = _mm_loadu_si128(in + 20);
    __m128i c21_alignr_rslt20_m128i = _mm_alignr_epi8(c21_load_rslt21_m128i, c21_load_rslt20_m128i, 11);
    hor_sse4_unpack4_c21_f1<0>(out, c21_alignr_rslt20_m128i); // Unpack 31st 4 values.
    hor_sse4_unpack4_c21_f2<5>(out, c21_load_rslt21_m128i);   // Unpack 32nd 4 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c21_f1(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c21_shfl_msk_m128i = _mm_set_epi8(
				byte + 10, byte + 9, byte + 8, byte + 7,
				0xFF, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c21_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c21_shfl_msk_m128i);
		__m128i c21_mul_rslt_m128i = _mm_mullo_epi32(c21_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[21][0]);
		__m128i c21_srli_rslt_m128i = _mm_srli_epi32(c21_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[21][0]);
		__m128i c21_rslt_m128i = _mm_and_si128(c21_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[21]);
		_mm_storeu_si128(out++, c21_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c21_shfl_msk_m128i = _mm_set_epi8(
				byte + 10, byte + 9, byte + 8, byte + 7,
				0xFF, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c21_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c21_shfl_msk_m128i);
		__m128i c21_mul_rslt_m128i = _mm_mullo_epi32(c21_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[21][0]);
		__m128i c21_srli_rslt_m128i = _mm_srli_epi32(c21_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[21][0]);
		__m128i c21_and_rslt_m128i = _mm_and_si128(c21_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[21]);
		__m128i c21_rslt_m128i = _mm_or_si128(c21_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 21));
		_mm_storeu_si128(out++, c21_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c21_f2(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c21_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 10, byte + 9, byte + 8,
				byte + 8, byte + 7, byte + 6, byte + 5,
				0xFF, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c21_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c21_shfl_msk_m128i);
		__m128i c21_mul_rslt_m128i = _mm_mullo_epi32(c21_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[21][1]);
		__m128i c21_srli_rslt_m128i = _mm_srli_epi32(c21_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[21][1]);
		__m128i c21_rslt_m128i = _mm_and_si128(c21_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[21]);
		_mm_storeu_si128(out++, c21_rslt_m128i);
	}
	else { // RiceCoding
		// Unpack 4 values.
		const __m128i Hor_SSE4_c21_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 10, byte + 9, byte + 8,
				byte + 8, byte + 7, byte + 6, byte + 5,
				0xFF, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c21_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c21_shfl_msk_m128i);
		__m128i c21_mul_rslt_m128i = _mm_mullo_epi32(c21_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[21][1]);
		__m128i c21_srli_rslt_m128i = _mm_srli_epi32(c21_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[21][1]);
		__m128i c21_and_rslt_m128i = _mm_and_si128(c21_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[21]);
		__m128i c21_rslt_m128i = _mm_or_si128(c21_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 21));
		_mm_storeu_si128(out++, c21_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 22-bit values.
 * Load 22 SSE vectors, each containing 5 22-bit values. (6th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c22(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
	     __m128i c22_load_rslt1_m128i = _mm_loadu_si128(in++);
	     hor_sse4_unpack4_c22<0>(out, c22_load_rslt1_m128i);    // Unpack 1st 4 values.

	     __m128i c22_load_rslt2_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt1_m128i = _mm_alignr_epi8(c22_load_rslt2_m128i, c22_load_rslt1_m128i, 11);
	     hor_sse4_unpack4_c22<0>(out, c22_alignr_rslt1_m128i);  // Unpack 2nd 4 values.

	     __m128i c22_load_rslt3_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt2_m128i = _mm_alignr_epi8(c22_load_rslt3_m128i, c22_load_rslt2_m128i, 6);
	     hor_sse4_unpack4_c22<0>(out, c22_alignr_rslt2_m128i);  // Unpack 3rd 4 values.
	     hor_sse4_unpack4_c22<1>(out, c22_load_rslt3_m128i);    // Unpack 4th 4 values.

	     __m128i c22_load_rslt4_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt3_m128i = _mm_alignr_epi8(c22_load_rslt4_m128i, c22_load_rslt3_m128i, 12);
	     hor_sse4_unpack4_c22<0>(out, c22_alignr_rslt3_m128i);  // Unpack 5th 4 values.

	     __m128i c22_load_rslt5_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt4_m128i = _mm_alignr_epi8(c22_load_rslt5_m128i, c22_load_rslt4_m128i, 7);
	     hor_sse4_unpack4_c22<0>(out, c22_alignr_rslt4_m128i);  // Unpack 6th 4 values.
	     hor_sse4_unpack4_c22<2>(out, c22_load_rslt5_m128i);    // Unpack 7th 4 values.

	     __m128i c22_load_rslt6_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt5_m128i = _mm_alignr_epi8(c22_load_rslt6_m128i, c22_load_rslt5_m128i, 13);
	     hor_sse4_unpack4_c22<0>(out, c22_alignr_rslt5_m128i);  // Unpack 8th 4 values.

	     __m128i c22_load_rslt7_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt6_m128i = _mm_alignr_epi8(c22_load_rslt7_m128i, c22_load_rslt6_m128i, 8);
	     hor_sse4_unpack4_c22<0>(out, c22_alignr_rslt6_m128i);  // Unpack 9th 4 values.
	     hor_sse4_unpack4_c22<3>(out, c22_load_rslt7_m128i);    // Unpack 10th 4 values.

	     __m128i c22_load_rslt8_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt7_m128i = _mm_alignr_epi8(c22_load_rslt8_m128i, c22_load_rslt7_m128i, 14);
	     hor_sse4_unpack4_c22<0>(out, c22_alignr_rslt7_m128i);  // Unpack 11th 4 values.

	     __m128i c22_load_rslt9_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt8_m128i = _mm_alignr_epi8(c22_load_rslt9_m128i, c22_load_rslt8_m128i, 9);
	     hor_sse4_unpack4_c22<0>(out, c22_alignr_rslt8_m128i);  // Unpack 12th 4 values.
	     hor_sse4_unpack4_c22<4>(out, c22_load_rslt9_m128i);    // Unpack 13th 4 values.

	     __m128i c22_load_rslt10_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt9_m128i = _mm_alignr_epi8(c22_load_rslt10_m128i, c22_load_rslt9_m128i, 15);
	     hor_sse4_unpack4_c22<0>(out, c22_alignr_rslt9_m128i);  // Unpack 14th 4 values.

	     __m128i c22_load_rslt11_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt10_m128i = _mm_alignr_epi8(c22_load_rslt11_m128i, c22_load_rslt10_m128i, 10);
	     hor_sse4_unpack4_c22<0>(out, c22_alignr_rslt10_m128i); // Unpack 15th 4 values.
	     hor_sse4_unpack4_c22<5>(out, c22_load_rslt11_m128i);   // Unpack 16th 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c22(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c22_shfl_msk_m128i = _mm_set_epi8(
				byte + 11, byte + 10, byte + 9, byte + 8,
				byte + 8, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);
		__m128i c22_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c22_shfl_msk_m128i);
		__m128i c22_mul_rslt_m128i = _mm_mullo_epi32(c22_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[22][0]);
		__m128i c22_srli_rslt_m128i = _mm_srli_epi32(c22_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[22][0]);
		__m128i c22_rslt_m128i = _mm_and_si128(c22_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[22]);
		_mm_storeu_si128(out++, c22_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c22_shfl_msk_m128i = _mm_set_epi8(
				byte + 11, byte + 10, byte + 9, byte + 8,
				byte + 8, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);
		__m128i c22_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c22_shfl_msk_m128i);
		__m128i c22_mul_rslt_m128i = _mm_mullo_epi32(c22_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[22][0]);
		__m128i c22_srli_rslt_m128i = _mm_srli_epi32(c22_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[22][0]);
		__m128i c22_and_rslt_m128i = _mm_and_si128(c22_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[22]);
		__m128i c22_rslt_m128i = _mm_or_si128(c22_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 22));
		_mm_storeu_si128(out++, c22_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 23-bit values.
 * Load 23 SSE vectors, each containing 5 23-bit values. (6th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c23(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
    __m128i c23_load_rslt1_m128i = _mm_loadu_si128(in + 0);
     hor_sse4_unpack4_c23_f1<0>(out, c23_load_rslt1_m128i);    // Unpack 1st 4 values.

     __m128i c23_load_rslt2_m128i = _mm_loadu_si128(in + 1);
     __m128i c23_alignr_rslt1_m128i = _mm_alignr_epi8(c23_load_rslt2_m128i, c23_load_rslt1_m128i, 11);
     hor_sse4_unpack4_c23_f2<0>(out, c23_alignr_rslt1_m128i);  // Unpack 2nd 4 values.

     __m128i c23_load_rslt3_m128i = _mm_loadu_si128(in + 2);
     __m128i c23_alignr_rslt2_m128i = _mm_alignr_epi8(c23_load_rslt3_m128i, c23_load_rslt2_m128i, 7);
     hor_sse4_unpack4_c23_f1<0>(out, c23_alignr_rslt2_m128i);  // Unpack 3rd 4 values.
     hor_sse4_unpack4_c23_f2<2>(out, c23_load_rslt3_m128i);    // Unpack 4th 4 values.

     __m128i c23_load_rslt4_m128i = _mm_loadu_si128(in + 3);
     __m128i c23_alignr_rslt3_m128i = _mm_alignr_epi8(c23_load_rslt4_m128i, c23_load_rslt3_m128i, 14);
     hor_sse4_unpack4_c23_f1<0>(out, c23_alignr_rslt3_m128i);  // Unpack 5th 4 values.

     __m128i c23_load_rslt5_m128i = _mm_loadu_si128(in + 4);
     __m128i c23_alignr_rslt4_m128i = _mm_alignr_epi8(c23_load_rslt5_m128i, c23_load_rslt4_m128i, 9);
     hor_sse4_unpack4_c23_f2<0>(out, c23_alignr_rslt4_m128i);  // Unpack 6th 4 values.

     __m128i c23_load_rslt6_m128i = _mm_loadu_si128(in + 5);
     __m128i c23_alignr_rslt5_m128i = _mm_alignr_epi8(c23_load_rslt6_m128i, c23_load_rslt5_m128i, 5);
     hor_sse4_unpack4_c23_f1<0>(out, c23_alignr_rslt5_m128i);  // Unpack 7th 4 values.
     hor_sse4_unpack4_c23_f2<0>(out, c23_load_rslt6_m128i);    // Unpack 8th 4 values.

     __m128i c23_load_rslt7_m128i = _mm_loadu_si128(in + 6);
     __m128i c23_alignr_rslt6_m128i = _mm_alignr_epi8(c23_load_rslt7_m128i, c23_load_rslt6_m128i, 12);
     hor_sse4_unpack4_c23_f1<0>(out, c23_alignr_rslt6_m128i);  // Unpack 9th 4 values.

     __m128i c23_load_rslt8_m128i = _mm_loadu_si128(in + 7);
     __m128i c23_alignr_rslt7_m128i = _mm_alignr_epi8(c23_load_rslt8_m128i, c23_load_rslt7_m128i, 7);
     hor_sse4_unpack4_c23_f2<0>(out, c23_alignr_rslt7_m128i);  // Unpack 10th 4 values.
     hor_sse4_unpack4_c23_f1<3>(out, c23_load_rslt8_m128i);    // Unpack 11th 4 values.

     __m128i c23_load_rslt9_m128i = _mm_loadu_si128(in + 8);
     __m128i c23_alignr_rslt8_m128i = _mm_alignr_epi8(c23_load_rslt9_m128i, c23_load_rslt8_m128i, 14);
     hor_sse4_unpack4_c23_f2<0>(out, c23_alignr_rslt8_m128i);  // Unpack 12th 4 values.

     __m128i c23_load_rslt10_m128i = _mm_loadu_si128(in + 9);
     __m128i c23_alignr_rslt9_m128i = _mm_alignr_epi8(c23_load_rslt10_m128i, c23_load_rslt9_m128i, 10);
     hor_sse4_unpack4_c23_f1<0>(out, c23_alignr_rslt9_m128i);  // Unpack 13th 4 values.

     __m128i c23_load_rslt11_m128i = _mm_loadu_si128(in + 10);
     __m128i c23_alignr_rslt10_m128i = _mm_alignr_epi8(c23_load_rslt11_m128i , c23_load_rslt10_m128i, 5);
     hor_sse4_unpack4_c23_f2<0>(out, c23_alignr_rslt10_m128i); // Unpack 14th 4 values.
     hor_sse4_unpack4_c23_f1<1>(out, c23_load_rslt11_m128i);   // Unpack 15th 4 values.

     __m128i c23_load_rslt12_m128i = _mm_loadu_si128(in + 11);
     __m128i c23_alignr_rslt11_m128i = _mm_alignr_epi8(c23_load_rslt12_m128i, c23_load_rslt11_m128i, 12);
     hor_sse4_unpack4_c23_f2<0>(out, c23_alignr_rslt11_m128i); // Unpack 16th 4 values.

     __m128i c23_load_rslt13_m128i = _mm_loadu_si128(in + 12);
     __m128i c23_alignr_rslt12_m128i = _mm_alignr_epi8(c23_load_rslt13_m128i, c23_load_rslt12_m128i, 8);
     hor_sse4_unpack4_c23_f1<0>(out, c23_alignr_rslt12_m128i); // Unpack 17th 4 values.
     hor_sse4_unpack4_c23_f2<3>(out, c23_load_rslt13_m128i);   // Unpack 18th 4 values.

     __m128i c23_load_rslt14_m128i = _mm_loadu_si128(in + 13);
     __m128i c23_alignr_rslt13_m128i = _mm_alignr_epi8(c23_load_rslt14_m128i, c23_load_rslt13_m128i, 15);
     hor_sse4_unpack4_c23_f1<0>(out, c23_alignr_rslt13_m128i); // Unpack 19th 4 values.

     __m128i c23_load_rslt15_m128i = _mm_loadu_si128(in + 14);
     __m128i c23_alignr_rslt14_m128i = _mm_alignr_epi8(c23_load_rslt15_m128i, c23_load_rslt14_m128i, 10);
     hor_sse4_unpack4_c23_f2<0>(out, c23_alignr_rslt14_m128i); // Unpack 20th 4 values.

     __m128i c23_load_rslt16_m128i = _mm_loadu_si128(in + 15);
     __m128i c23_alignr_rslt15_m128i = _mm_alignr_epi8(c23_load_rslt16_m128i, c23_load_rslt15_m128i, 6);
     hor_sse4_unpack4_c23_f1<0>(out, c23_alignr_rslt15_m128i); // Unpack 21st 4 values.
     hor_sse4_unpack4_c23_f2<1>(out, c23_load_rslt16_m128i);   // Unpack 22nd 4 values.

     __m128i c23_load_rslt17_m128i = _mm_loadu_si128(in + 16);
     __m128i c23_alignr_rslt16_m128i = _mm_alignr_epi8(c23_load_rslt17_m128i, c23_load_rslt16_m128i, 13);
     hor_sse4_unpack4_c23_f1<0>(out, c23_alignr_rslt16_m128i); // Unpack 23rd 4 values.

     __m128i c23_load_rslt18_m128i = _mm_loadu_si128(in + 17);
     __m128i c23_alignr_rslt17_m128i = _mm_alignr_epi8(c23_load_rslt18_m128i, c23_load_rslt17_m128i, 8);
     hor_sse4_unpack4_c23_f2<0>(out, c23_alignr_rslt17_m128i); // Unpack 24th 4 values.
     hor_sse4_unpack4_c23_f1<4>(out, c23_load_rslt18_m128i);   // Unpack 25th 4 values.

     __m128i c23_load_rslt19_m128i = _mm_loadu_si128(in + 18);
     __m128i c23_alignr_rslt18_m128i = _mm_alignr_epi8(c23_load_rslt19_m128i, c23_load_rslt18_m128i, 15);
     hor_sse4_unpack4_c23_f2<0>(out, c23_alignr_rslt18_m128i); // Unpack 26th 4 values.

     __m128i c23_load_rslt20_m128i = _mm_loadu_si128(in + 19);
     __m128i c23_alignr_rslt19_m128i = _mm_alignr_epi8(c23_load_rslt20_m128i, c23_load_rslt19_m128i, 11);
     hor_sse4_unpack4_c23_f1<0>(out, c23_alignr_rslt19_m128i); // Unpack 27th 4 values.

     __m128i c23_load_rslt21_m128i = _mm_loadu_si128(in + 20);
     __m128i c23_alignr_rslt20_m128i = _mm_alignr_epi8(c23_load_rslt21_m128i, c23_load_rslt20_m128i, 6);
     hor_sse4_unpack4_c23_f2<0>(out, c23_alignr_rslt20_m128i); // Unpack 28th 4 values.
     hor_sse4_unpack4_c23_f1<2>(out, c23_load_rslt21_m128i);   // Unpack 29th 4 values.

     __m128i c23_load_rslt22_m128i = _mm_loadu_si128(in + 21);
     __m128i c23_alignr_rslt21_m128i = _mm_alignr_epi8(c23_load_rslt22_m128i, c23_load_rslt21_m128i, 13);
     hor_sse4_unpack4_c23_f2<0>(out, c23_alignr_rslt21_m128i); // Unpack 30th 4 values.

     __m128i c23_load_rslt23_m128i = _mm_loadu_si128(in + 22);
     __m128i c23_alignr_rslt22_m128i = _mm_alignr_epi8(c23_load_rslt23_m128i, c23_load_rslt22_m128i, 9);
     hor_sse4_unpack4_c23_f1<0>(out, c23_alignr_rslt22_m128i); // Unpack 31st 4 values.
     hor_sse4_unpack4_c23_f2<4>(out, c23_load_rslt23_m128i);   // Unpack 32nd 4 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c23_f1(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c23_shfl_msk_m128i = _mm_set_epi8(
				byte + 11, byte + 10, byte + 9, byte + 8,
				byte + 8, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c23_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c23_shfl_msk_m128i);
		__m128i c23_mul_rslt_m128i = _mm_mullo_epi32(c23_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[23][0]);
		__m128i c23_srli_rslt_m128i = _mm_srli_epi32(c23_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[23][0]);
		__m128i c23_rslt_m128i = _mm_and_si128(c23_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[23]);
		_mm_storeu_si128(out++, c23_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c23_shfl_msk_m128i = _mm_set_epi8(
				byte + 11, byte + 10, byte + 9, byte + 8,
				byte + 8, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c23_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c23_shfl_msk_m128i);
		__m128i c23_mul_rslt_m128i = _mm_mullo_epi32(c23_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[23][0]);
		__m128i c23_srli_rslt_m128i = _mm_srli_epi32(c23_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[23][0]);
		__m128i c23_and_rslt_m128i = _mm_and_si128(c23_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[23]);
		__m128i c23_rslt_m128i = _mm_or_si128(c23_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 23));
		_mm_storeu_si128(out++, c23_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c23_f2(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c23_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c23_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c23_shfl_msk_m128i);
		__m128i c23_mul_rslt_m128i = _mm_mullo_epi32(c23_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[23][1]);
		__m128i c23_srli_rslt_m128i = _mm_srli_epi32(c23_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[23][1]);
		__m128i c23_rslt_m128i = _mm_and_si128(c23_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[23]);
		_mm_storeu_si128(out++, c23_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c23_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c23_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c23_shfl_msk_m128i);
		__m128i c23_mul_rslt_m128i = _mm_mullo_epi32(c23_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[23][1]);
		__m128i c23_srli_rslt_m128i = _mm_srli_epi32(c23_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[23][1]);
		__m128i c23_and_rslt_m128i = _mm_and_si128(c23_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[23]);
		__m128i c23_rslt_m128i = _mm_or_si128(c23_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 23));
		_mm_storeu_si128(out++, c23_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 24-bit values.
 * Load 24 SSE vectors, each containing 5 24-bit values. (6th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c24(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 16) {
		__m128i c24_load_rslt1_m128i = _mm_loadu_si128(in++);
		hor_sse4_unpack4_c24<0>(out, c24_load_rslt1_m128i);   // Unpack 1st 4 values.

		__m128i c24_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c24_alignr_rslt1_m128i = _mm_alignr_epi8(c24_load_rslt2_m128i, c24_load_rslt1_m128i, 12);
		hor_sse4_unpack4_c24<0>(out, c24_alignr_rslt1_m128i); // Unpack 2nd 4 values.

		__m128i c24_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c24_alignr_rslt2_m128i = _mm_alignr_epi8(c24_load_rslt3_m128i, c24_load_rslt2_m128i, 8);
		hor_sse4_unpack4_c24<0>(out, c24_alignr_rslt2_m128i); // Unpack 3rd 4 values.
		hor_sse4_unpack4_c24<4>(out, c24_load_rslt3_m128i);   // Unpack 4th 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c24(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c24_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 11, byte + 10, byte + 9,
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, byte + 5, byte + 4, byte + 3,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c24_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c24_shfl_msk_m128i);
		_mm_storeu_si128(out++, c24_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c24_shfl_msk_m128i = _mm_set_epi8(
				0xFF, byte + 11, byte + 10, byte + 9,
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, byte + 5, byte + 4, byte + 3,
				0xFF, byte + 2, byte + 1, byte + 0);

		__m128i c24_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c24_shfl_msk_m128i);
		__m128i c24_rslt_m128i = _mm_or_si128(c24_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 24));
		_mm_storeu_si128(out++, c24_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 25-bit values.
 * Load 25 SSE vectors, each containing 5 25-bit values. (6th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c25(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
    __m128i c25_load_rslt1_m128i = _mm_loadu_si128(in + 0);   // 16 bytes; contains 5 25-bit values. (6th is incomplete)
    hor_sse4_unpack4_c25_f1<0>(out, c25_load_rslt1_m128i);    // Unpack 1st 4 values.

    __m128i c25_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c25_alignr_rslt1_m128i = _mm_alignr_epi8(c25_load_rslt2_m128i, c25_load_rslt1_m128i, 12); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f2<0>(out, c25_alignr_rslt1_m128i);  // Unpack 2nd 4 values.

    __m128i c25_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c25_alignr_rslt2_m128i = _mm_alignr_epi8(c25_load_rslt3_m128i, c25_load_rslt2_m128i, 9); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f1<0>(out, c25_alignr_rslt2_m128i);  // Unpack 3rd 4 values.

    __m128i c25_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c25_alignr_rslt3_m128i = _mm_alignr_epi8(c25_load_rslt4_m128i, c25_load_rslt3_m128i, 5); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f2<0>(out, c25_alignr_rslt3_m128i);  // Unpack 4th 4 values.
    hor_sse4_unpack4_c25_f1<2>(out, c25_load_rslt4_m128i);    // Unpack 5th 4 values.

    __m128i c25_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c25_alignr_rslt4_m128i = _mm_alignr_epi8(c25_load_rslt5_m128i, c25_load_rslt4_m128i, 14); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f2<0>(out, c25_alignr_rslt4_m128i);  // Unpack 6th 4 values.

    __m128i c25_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c25_alignr_rslt5_m128i = _mm_alignr_epi8(c25_load_rslt6_m128i, c25_load_rslt5_m128i, 11); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f1<0>(out, c25_alignr_rslt5_m128i);  // Unpack 7th 4 values.

    __m128i c25_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c25_alignr_rslt6_m128i = _mm_alignr_epi8(c25_load_rslt7_m128i, c25_load_rslt6_m128i, 7); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f2<0>(out, c25_alignr_rslt6_m128i);  // Unpack 8th 4 values.

    __m128i c25_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c25_alignr_rslt7_m128i = _mm_alignr_epi8(c25_load_rslt8_m128i, c25_load_rslt7_m128i, 4); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f1<0>(out, c25_alignr_rslt7_m128i);  // Unpack 9th 4 values.
    hor_sse4_unpack4_c25_f2<0>(out, c25_load_rslt8_m128i);    // Unpack 10th 4 values.

    __m128i c25_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c25_alignr_rslt8_m128i = _mm_alignr_epi8(c25_load_rslt9_m128i, c25_load_rslt8_m128i, 13); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f1<0>(out, c25_alignr_rslt8_m128i);  // Unpack 11th 4 values.

    __m128i c25_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c25_alignr_rslt9_m128i = _mm_alignr_epi8(c25_load_rslt10_m128i, c25_load_rslt9_m128i, 9); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f2<0>(out, c25_alignr_rslt9_m128i);  // Unpack 12th 4 values.

    __m128i c25_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c25_alignr_rslt10_m128i = _mm_alignr_epi8(c25_load_rslt11_m128i, c25_load_rslt10_m128i, 6); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f1<0>(out, c25_alignr_rslt10_m128i); // Unpack 13th 4 values.
    hor_sse4_unpack4_c25_f2<2>(out, c25_load_rslt11_m128i);   // Unpack 14th 4 values.

    __m128i c25_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c25_alignr_rslt11_m128i = _mm_alignr_epi8(c25_load_rslt12_m128i, c25_load_rslt11_m128i, 15); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f1<0>(out, c25_alignr_rslt11_m128i); // Unpack 15th 4 values.

    __m128i c25_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c25_alignr_rslt12_m128i = _mm_alignr_epi8(c25_load_rslt13_m128i, c25_load_rslt12_m128i, 11); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f2<0>(out, c25_alignr_rslt12_m128i); // Unpack 16th 4 values.

    __m128i c25_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c25_alignr_rslt13_m128i = _mm_alignr_epi8(c25_load_rslt14_m128i, c25_load_rslt13_m128i, 8); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f1<0>(out, c25_alignr_rslt13_m128i); // Unpack 17th 4 values.

    __m128i c25_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c25_alignr_rslt14_m128i = _mm_alignr_epi8(c25_load_rslt15_m128i, c25_load_rslt14_m128i, 4); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f2<0>(out, c25_alignr_rslt14_m128i); // Unpack 18th 4 values.
    hor_sse4_unpack4_c25_f1<1>(out, c25_load_rslt15_m128i);   // Unpack 19th 4 values.

    __m128i c25_load_rslt16_m128i = _mm_loadu_si128(in + 15);
    __m128i c25_alignr_rslt15_m128i = _mm_alignr_epi8(c25_load_rslt16_m128i, c25_load_rslt15_m128i, 13); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f2<0>(out, c25_alignr_rslt15_m128i); // Unpack 20th 4 values.

    __m128i c25_load_rslt17_m128i = _mm_loadu_si128(in + 16);
    __m128i c25_alignr_rslt16_m128i = _mm_alignr_epi8(c25_load_rslt17_m128i, c25_load_rslt16_m128i, 10); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f1<0>(out, c25_alignr_rslt16_m128i); // Unpack 21st 4 values.

    __m128i c25_load_rslt18_m128i = _mm_loadu_si128(in + 17);
    __m128i c25_alignr_rslt17_m128i = _mm_alignr_epi8(c25_load_rslt18_m128i, c25_load_rslt17_m128i, 6); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f2<0>(out, c25_alignr_rslt17_m128i); // Unpack 22nd 4 values.
    hor_sse4_unpack4_c25_f1<3>(out, c25_load_rslt18_m128i);   // Unpack 23rd 4 values.

    __m128i c25_load_rslt19_m128i = _mm_loadu_si128(in + 18);
    __m128i c25_alignr_rslt18_m128i = _mm_alignr_epi8(c25_load_rslt19_m128i, c25_load_rslt18_m128i, 15); // 16 bytes; contains 5 values.

    hor_sse4_unpack4_c25_f2<0>(out, c25_alignr_rslt18_m128i); // Unpack 24th 4 values.

    __m128i c25_load_rslt20_m128i = _mm_loadu_si128(in + 19);
    __m128i c25_alignr_rslt19_m128i = _mm_alignr_epi8(c25_load_rslt20_m128i, c25_load_rslt19_m128i, 12); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f1<0>(out, c25_alignr_rslt19_m128i); // Unpack 25th 4 values.

    __m128i c25_load_rslt21_m128i = _mm_loadu_si128(in + 20);
    __m128i c25_alignr_rslt20_m128i = _mm_alignr_epi8(c25_load_rslt21_m128i, c25_load_rslt20_m128i, 8); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f2<0>(out, c25_alignr_rslt20_m128i); // Unpack 26th 4 values.

    __m128i c25_load_rslt22_m128i = _mm_loadu_si128(in + 21);
    __m128i c25_alignr_rslt21_m128i = _mm_alignr_epi8(c25_load_rslt22_m128i, c25_load_rslt21_m128i, 5); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f1<0>(out, c25_alignr_rslt21_m128i); // Unpack 27th 4 values.
    hor_sse4_unpack4_c25_f2<1>(out, c25_load_rslt22_m128i);   // Unpack 28th 4 values.

    __m128i c25_load_rslt23_m128i = _mm_loadu_si128(in + 22);
    __m128i c25_alignr_rslt22_m128i = _mm_alignr_epi8(c25_load_rslt23_m128i, c25_load_rslt22_m128i, 14); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f1<0>(out, c25_alignr_rslt22_m128i); // Unpack 29th 4 values.

    __m128i c25_load_rslt24_m128i = _mm_loadu_si128(in + 23);
    __m128i c25_alignr_rslt23_m128i = _mm_alignr_epi8(c25_load_rslt24_m128i, c25_load_rslt23_m128i, 10); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f2<0>(out, c25_alignr_rslt23_m128i); // Unpack 30th 4 values.

    __m128i c25_load_rslt25_m128i = _mm_loadu_si128(in + 24);
    __m128i c25_alignr_rslt24_m128i = _mm_alignr_epi8(c25_load_rslt25_m128i, c25_load_rslt24_m128i, 7); // 16 bytes; contains 5 values.
    hor_sse4_unpack4_c25_f1<0>(out, c25_alignr_rslt24_m128i); // Unpack 31st 4 values.
    hor_sse4_unpack4_c25_f2<3>(out, c25_load_rslt25_m128i);   // Unpack 32nd 4 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c25_f1(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c25_shfl_msk_m128i = _mm_set_epi8(
				byte + 12, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c25_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c25_shfl_msk_m128i);
		__m128i c25_mul_rslt_m128i = _mm_mullo_epi32(c25_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[25][0]);
		__m128i c25_srli_rslt_m128i = _mm_srli_epi32(c25_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[25][0]);
		__m128i c25_rslt_m128i = _mm_and_si128(c25_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[25]);
		_mm_storeu_si128(out++, c25_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c25_shfl_msk_m128i = _mm_set_epi8(
				byte + 12, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c25_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c25_shfl_msk_m128i);
		__m128i c25_mul_rslt_m128i = _mm_mullo_epi32(c25_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[25][0]);
		__m128i c25_srli_rslt_m128i = _mm_srli_epi32(c25_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[25][0]);
		__m128i c25_and_rslt_m128i = _mm_and_si128(c25_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[25]);
		__m128i c25_rslt_m128i = _mm_or_si128(c25_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 25));
		_mm_storeu_si128(out++, c25_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c25_f2(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c25_shfl_msk_m128i = _mm_set_epi8(
				byte + 12, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c25_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c25_shfl_msk_m128i);
		__m128i c25_mul_rslt_m128i = _mm_mullo_epi32(c25_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[25][1]);
		__m128i c25_srli_rslt_m128i = _mm_srli_epi32(c25_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[25][1]);
		__m128i c25_rslt_m128i = _mm_and_si128(c25_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[25]);
		_mm_storeu_si128(out++, c25_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c25_shfl_msk_m128i = _mm_set_epi8(
				byte + 12, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c25_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c25_shfl_msk_m128i);
		__m128i c25_mul_rslt_m128i = _mm_mullo_epi32(c25_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[25][1]);
		__m128i c25_srli_rslt_m128i = _mm_srli_epi32(c25_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[25][1]);
		__m128i c25_and_rslt_m128i = _mm_and_si128(c25_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[25]);
		__m128i c25_rslt_m128i = _mm_or_si128(c25_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 25));
		_mm_storeu_si128(out++, c25_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 26-bit values.
 * Load 26 SSE vectors, each containing 4 26-bit values. (5th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c26(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
		__m128i c26_load_rslt1_m128i = _mm_loadu_si128(in++);
		hor_sse4_unpack4_c26<0>(out, c26_load_rslt1_m128i);    // Unpack 1st 4 values.

		__m128i c26_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt1_m128i = _mm_alignr_epi8(c26_load_rslt2_m128i, c26_load_rslt1_m128i, 13);
		hor_sse4_unpack4_c26<0>(out, c26_alignr_rslt1_m128i);  // Unpack 2nd 4 values.

		__m128i c26_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt2_m128i = _mm_alignr_epi8(c26_load_rslt3_m128i, c26_load_rslt2_m128i, 10);
		hor_sse4_unpack4_c26<0>(out, c26_alignr_rslt2_m128i);  // Unpack 3rd 4 values.

		__m128i c26_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt3_m128i = _mm_alignr_epi8(c26_load_rslt4_m128i, c26_load_rslt3_m128i, 7);
		hor_sse4_unpack4_c26<0>(out, c26_alignr_rslt3_m128i);  // Unpack 4th 4 values.

		__m128i c26_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt4_m128i = _mm_alignr_epi8(c26_load_rslt5_m128i, c26_load_rslt4_m128i, 4);
		hor_sse4_unpack4_c26<0>(out, c26_alignr_rslt4_m128i);  // Unpack 5th 4 values.
		hor_sse4_unpack4_c26<1>(out, c26_load_rslt5_m128i);    // Unpack 6th 4 values.

		__m128i c26_load_rslt6_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt5_m128i = _mm_alignr_epi8(c26_load_rslt6_m128i, c26_load_rslt5_m128i, 14);
		hor_sse4_unpack4_c26<0>(out, c26_alignr_rslt5_m128i);  // Unpack 7th 4 values.

		__m128i c26_load_rslt7_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt6_m128i = _mm_alignr_epi8(c26_load_rslt7_m128i, c26_load_rslt6_m128i, 11);
		hor_sse4_unpack4_c26<0>(out, c26_alignr_rslt6_m128i);  // Unpack 8th 4 values.

		__m128i c26_load_rslt8_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt7_m128i = _mm_alignr_epi8(c26_load_rslt8_m128i, c26_load_rslt7_m128i, 8);
		hor_sse4_unpack4_c26<0>(out, c26_alignr_rslt7_m128i);  // Unpack 9th 4 values.

		__m128i c26_load_rslt9_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt8_m128i = _mm_alignr_epi8(c26_load_rslt9_m128i, c26_load_rslt8_m128i, 5);
		hor_sse4_unpack4_c26<0>(out, c26_alignr_rslt8_m128i);  // Unpack 10th 4 values.
		hor_sse4_unpack4_c26<2>(out, c26_load_rslt9_m128i);    // Unpack 11th 4 values.

		__m128i c26_load_rslt10_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt9_m128i = _mm_alignr_epi8(c26_load_rslt10_m128i, c26_load_rslt9_m128i, 15);
		hor_sse4_unpack4_c26<0>(out, c26_alignr_rslt9_m128i);  // Unpack 12th 4 values.

		__m128i c26_load_rslt11_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt10_m128i = _mm_alignr_epi8(c26_load_rslt11_m128i, c26_load_rslt10_m128i, 12);
		hor_sse4_unpack4_c26<0>(out, c26_alignr_rslt10_m128i); // Unpack 13th 4 values.

		__m128i c26_load_rslt12_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt11_m128i = _mm_alignr_epi8(c26_load_rslt12_m128i, c26_load_rslt11_m128i, 9);
		hor_sse4_unpack4_c26<0>(out, c26_alignr_rslt11_m128i); // Unpack 14th 4 values.

		__m128i c26_load_rslt13_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt12_m128i = _mm_alignr_epi8(c26_load_rslt13_m128i, c26_load_rslt12_m128i, 6);
		hor_sse4_unpack4_c26<0>(out, c26_alignr_rslt12_m128i); // Unpack 15th 4 values.
		hor_sse4_unpack4_c26<3>(out, c26_load_rslt13_m128i);   // Unpack 16th 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c26(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c26_shfl_msk_m128i = _mm_set_epi8(
				byte + 12, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c26_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c26_shfl_msk_m128i);
		__m128i c26_mul_rslt_m128i = _mm_mullo_epi32(c26_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[26][0]);
		__m128i c26_srli_rslt_m128i = _mm_srli_epi32(c26_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[26][0]);
		__m128i c26_rslt_m128i = _mm_and_si128(c26_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[26]);
		_mm_storeu_si128(out++, c26_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c26_shfl_msk_m128i = _mm_set_epi8(
				byte + 12, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c26_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c26_shfl_msk_m128i);
		__m128i c26_mul_rslt_m128i = _mm_mullo_epi32(c26_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[26][0]);
		__m128i c26_srli_rslt_m128i = _mm_srli_epi32(c26_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[26][0]);
		__m128i c26_and_rslt_m128i = _mm_and_si128(c26_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[26]);
		__m128i c26_rslt_m128i = _mm_or_si128(c26_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 26));
		_mm_storeu_si128(out++, c26_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 27-bit values.
 * Load 27 SSE vectors, each containing 4 27-bit values. (5th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c27(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c27_load_rslt1_m128i = _mm_loadu_si128(in + 0);
    hor_sse4_unpack4_c27_f1<0>(out, c27_load_rslt1_m128i);    // Unpack 1st 4 values.

    __m128i c27_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c27_alignr_rslt1_m128i = _mm_alignr_epi8(c27_load_rslt2_m128i, c27_load_rslt1_m128i, 13);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt1_m128i);  // Unpack 2nd 4 values.

    __m128i c27_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c27_alignr_rslt2_m128i = _mm_alignr_epi8(c27_load_rslt3_m128i, c27_load_rslt2_m128i, 11);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt2_m128i);  // Unpack 3rd 4 values.

    __m128i c27_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c27_alignr_rslt3_m128i = _mm_alignr_epi8(c27_load_rslt4_m128i, c27_load_rslt3_m128i, 8);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt3_m128i);  // Unpack 4th 4 values.

    __m128i c27_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c27_alignr_rslt4_m128i = _mm_alignr_epi8(c27_load_rslt5_m128i, c27_load_rslt4_m128i, 6);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt4_m128i);  // Unpack 5th 4 values.

    __m128i c27_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c27_alignr_rslt5_m128i = _mm_alignr_epi8(c27_load_rslt6_m128i, c27_load_rslt5_m128i, 3);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt5_m128i);  // Unpack 6th 4 values.
    hor_sse4_unpack4_c27_f1<1>(out, c27_load_rslt6_m128i);    // Unpack 7th 4 values.

    __m128i c27_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c27_alignr_rslt6_m128i = _mm_alignr_epi8(c27_load_rslt7_m128i, c27_load_rslt6_m128i, 14);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt6_m128i);  // Unpack 8th 4 values.

    __m128i c27_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c27_alignr_rslt7_m128i = _mm_alignr_epi8(c27_load_rslt8_m128i, c27_load_rslt7_m128i, 12);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt7_m128i);  // Unpack 9th 4 values.

    __m128i c27_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c27_alignr_rslt8_m128i = _mm_alignr_epi8(c27_load_rslt9_m128i, c27_load_rslt8_m128i, 9);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt8_m128i);  // Unpack 10th 4 values.

    __m128i c27_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c27_alignr_rslt9_m128i = _mm_alignr_epi8(c27_load_rslt10_m128i, c27_load_rslt9_m128i, 7);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt9_m128i);  // Unpack 11th 4 values.

    __m128i c27_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c27_alignr_rslt10_m128i = _mm_alignr_epi8(c27_load_rslt11_m128i, c27_load_rslt10_m128i, 4);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt10_m128i); // Unpack 12th 4 values.
    hor_sse4_unpack4_c27_f1<2>(out, c27_load_rslt11_m128i);   // Unpack 13th 4 values.

    __m128i c27_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c27_alignr_rslt11_m128i = _mm_alignr_epi8(c27_load_rslt12_m128i, c27_load_rslt11_m128i, 15);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt11_m128i); // Unpack 14th 4 values.

    __m128i c27_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c27_alignr_rslt12_m128i = _mm_alignr_epi8(c27_load_rslt13_m128i, c27_load_rslt12_m128i, 13);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt12_m128i); // Unpack 15th 4 values.

    __m128i c27_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c27_alignr_rslt13_m128i = _mm_alignr_epi8(c27_load_rslt14_m128i, c27_load_rslt13_m128i, 10);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt13_m128i); // Unpack 16th 4 values.

    __m128i c27_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c27_alignr_rslt14_m128i = _mm_alignr_epi8(c27_load_rslt15_m128i, c27_load_rslt14_m128i, 8);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt14_m128i); // Unpack 17th 4 values.

    __m128i c27_load_rslt16_m128i = _mm_loadu_si128(in + 15);
    __m128i c27_alignr_rslt15_m128i = _mm_alignr_epi8(c27_load_rslt16_m128i, c27_load_rslt15_m128i, 5);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt15_m128i); // Unpack 18th 4 values.

    __m128i c27_load_rslt17_m128i = _mm_loadu_si128(in + 16);
    __m128i c27_alignr_rslt16_m128i = _mm_alignr_epi8(c27_load_rslt17_m128i, c27_load_rslt16_m128i, 3);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt16_m128i); // Unpack 19th 4 values.
    hor_sse4_unpack4_c27_f2<0>(out, c27_load_rslt17_m128i);   // Unpack 20th 4 values.

    __m128i c27_load_rslt18_m128i = _mm_loadu_si128(in + 17);
    __m128i c27_alignr_rslt17_m128i = _mm_alignr_epi8(c27_load_rslt18_m128i, c27_load_rslt17_m128i, 14);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt17_m128i); // Unpack 21st 4 values.

    __m128i c27_load_rslt19_m128i = _mm_loadu_si128(in + 18);
    __m128i c27_alignr_rslt18_m128i = _mm_alignr_epi8(c27_load_rslt19_m128i, c27_load_rslt18_m128i, 11);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt18_m128i); // Unpack 22nd 4 values.

    __m128i c27_load_rslt20_m128i = _mm_loadu_si128(in + 19);
    __m128i c27_alignr_rslt19_m128i = _mm_alignr_epi8(c27_load_rslt20_m128i, c27_load_rslt19_m128i, 9);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt19_m128i); // Unpack 23th 4 values.

    __m128i c27_load_rslt21_m128i = _mm_loadu_si128(in + 20);
    __m128i c27_alignr_rslt20_m128i = _mm_alignr_epi8(c27_load_rslt21_m128i, c27_load_rslt20_m128i, 6);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt20_m128i); // Unpack 24th 4 values.

    __m128i c27_load_rslt22_m128i = _mm_loadu_si128(in + 21);
    __m128i c27_alignr_rslt21_m128i = _mm_alignr_epi8(c27_load_rslt22_m128i, c27_load_rslt21_m128i, 4);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt21_m128i); // Unpack 25th 4 values.
    hor_sse4_unpack4_c27_f2<1>(out, c27_load_rslt22_m128i);   // Unpack 26th 4 values.

    __m128i c27_load_rslt23_m128i = _mm_loadu_si128(in + 22);
    __m128i c27_alignr_rslt22_m128i = _mm_alignr_epi8(c27_load_rslt23_m128i, c27_load_rslt22_m128i, 15);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt22_m128i); // Unpack 27th 4 values.

    __m128i c27_load_rslt24_m128i = _mm_loadu_si128(in + 23);
    __m128i c27_alignr_rslt23_m128i = _mm_alignr_epi8(c27_load_rslt24_m128i, c27_load_rslt23_m128i, 12);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt23_m128i); // Unpack 28th 4 values.

    __m128i c27_load_rslt25_m128i = _mm_loadu_si128(in + 24);
    __m128i c27_alignr_rslt24_m128i = _mm_alignr_epi8(c27_load_rslt25_m128i, c27_load_rslt24_m128i, 10);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt24_m128i); // Unpack 29th 4 values.

    __m128i c27_load_rslt26_m128i = _mm_loadu_si128(in + 25);
    __m128i c27_alignr_rslt25_m128i = _mm_alignr_epi8(c27_load_rslt26_m128i, c27_load_rslt25_m128i, 7);
    hor_sse4_unpack4_c27_f2<0>(out, c27_alignr_rslt25_m128i); // Unpack 30th 4 values.

    __m128i c27_load_rslt27_m128i = _mm_loadu_si128(in + 26);
    __m128i c27_alignr_rslt26_m128i = _mm_alignr_epi8(c27_load_rslt27_m128i, c27_load_rslt26_m128i, 5);
    hor_sse4_unpack4_c27_f1<0>(out, c27_alignr_rslt26_m128i); // Unpack 31st 4 values.
    hor_sse4_unpack4_c27_f2<2>(out, c27_load_rslt27_m128i);   // Unpack 32nd 4 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c27_f1(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m128i Hor_SSE4_c27_shfl_msk_m128i = _mm_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c27_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c27_shfl_msk_m128i);
		__m128i c27_srli_rslt1_m128i = _mm_srli_epi64(c27_shfl_rslt_m128i, 6);

		__m128i c27_blend_rslt_m128i = _mm_blend_epi16(c27_shfl_rslt_m128i, c27_srli_rslt1_m128i, 0x30);

		__m128i c27_mul_rslt_m128i = _mm_mullo_epi32(c27_blend_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[27][0]);
		__m128i c27_srli_rslt2_m128i = _mm_srli_epi32(c27_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[27][0]);
		__m128i c27_rslt_m128i = _mm_and_si128(c27_srli_rslt2_m128i, SIMDMasks::SSE2_and_msk_m128i[27]);
		_mm_storeu_si128(out++, c27_rslt_m128i);
	}
	else { // For Rice and OptRice.
		const __m128i Hor_SSE4_c27_shfl_msk_m128i = _mm_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c27_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c27_shfl_msk_m128i);
		__m128i c27_srli_rslt1_m128i = _mm_srli_epi64(c27_shfl_rslt_m128i, 6);

		__m128i c27_blend_rslt_m128i = _mm_blend_epi16(c27_shfl_rslt_m128i, c27_srli_rslt1_m128i, 0x30);

		__m128i c27_mul_rslt_m128i = _mm_mullo_epi32(c27_blend_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[27][0]);
		__m128i c27_srli_rslt2_m128i = _mm_srli_epi32(c27_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[27][0]);
		__m128i c27_and_rslt_m128i = _mm_and_si128(c27_srli_rslt2_m128i, SIMDMasks::SSE2_and_msk_m128i[27]);
		__m128i c27_rslt_m128i = _mm_or_si128(c27_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 27));
		_mm_storeu_si128(out++, c27_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c27_f2(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m128i Hor_SSE4_c27_shfl_msk_m128i = _mm_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c27_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c27_shfl_msk_m128i);
		__m128i c27_slli_rslt_m128i = _mm_slli_epi64(c27_shfl_rslt_m128i, 1);

		__m128i c27_blend_rslt_m128i = _mm_blend_epi16(c27_shfl_rslt_m128i, c27_slli_rslt_m128i, 0x0C);

		__m128i c27_mul_rslt_m128i = _mm_mullo_epi32(c27_blend_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[27][1]);
		__m128i c27_srli_rslt_m128i = _mm_srli_epi32(c27_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[27][1]);
		__m128i c27_rslt_m128i = _mm_and_si128(c27_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[27]);
		_mm_storeu_si128(out++, c27_rslt_m128i);
	}
	else {
		const __m128i Hor_SSE4_c27_shfl_msk_m128i = _mm_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c27_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c27_shfl_msk_m128i);
		__m128i c27_slli_rslt_m128i = _mm_slli_epi64(c27_shfl_rslt_m128i, 1);

		__m128i c27_blend_rslt_m128i = _mm_blend_epi16(c27_shfl_rslt_m128i, c27_slli_rslt_m128i, 0x0C);

		__m128i c27_mul_rslt_m128i = _mm_mullo_epi32(c27_blend_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[27][1]);
		__m128i c27_srli_rslt_m128i = _mm_srli_epi32(c27_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[27][1]);
		__m128i c27_and_rslt_m128i = _mm_and_si128(c27_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[27]);
		__m128i c27_rslt_m128i = _mm_or_si128(c27_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 27));
		_mm_storeu_si128(out++, c27_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 28-bit values.
 * Load 28 SSE vectors, each containing 4 28-bit values. (5th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c28(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
		__m128i c28_load_rslt1_m128i = _mm_loadu_si128(in++);
		hor_sse4_unpack4_c28<0>(out, c28_load_rslt1_m128i);   // Unpack 1st 4 values.

		__m128i c28_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt1_m128i = _mm_alignr_epi8(c28_load_rslt2_m128i, c28_load_rslt1_m128i, 14);
		hor_sse4_unpack4_c28<0>(out, c28_alignr_rslt1_m128i); // Unpack 2nd 4 values.

		__m128i c28_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt2_m128i = _mm_alignr_epi8(c28_load_rslt3_m128i, c28_load_rslt2_m128i, 12);
		hor_sse4_unpack4_c28<0>(out, c28_alignr_rslt2_m128i); // Unpack 3rd 4 values.

		__m128i c28_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt3_m128i = _mm_alignr_epi8(c28_load_rslt4_m128i, c28_load_rslt3_m128i, 10);
		hor_sse4_unpack4_c28<0>(out, c28_alignr_rslt3_m128i); // Unpack 4th 4 values.

		__m128i c28_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt4_m128i = _mm_alignr_epi8(c28_load_rslt5_m128i, c28_load_rslt4_m128i, 8);
		hor_sse4_unpack4_c28<0>(out, c28_alignr_rslt4_m128i); // Unpack 5th 4 values.

		__m128i c28_load_rslt6_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt5_m128i = _mm_alignr_epi8(c28_load_rslt6_m128i, c28_load_rslt5_m128i, 6);
		hor_sse4_unpack4_c28<0>(out, c28_alignr_rslt5_m128i); // Unpack 6th 4 values.

		__m128i c28_load_rslt7_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt6_m128i = _mm_alignr_epi8(c28_load_rslt7_m128i, c28_load_rslt6_m128i, 4);
		hor_sse4_unpack4_c28<0>(out, c28_alignr_rslt6_m128i); // Unpack 7th 4 values.
		hor_sse4_unpack4_c28<2>(out, c28_load_rslt7_m128i);   // Unpack 8th 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c28(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c28_shfl_msk_m128i = _mm_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c28_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c28_shfl_msk_m128i);
		__m128i c28_mul_rslt_m128i = _mm_mullo_epi32(c28_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[28][0]);
		__m128i c28_srli_rslt_m128i = _mm_srli_epi32(c28_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[28][0]);
		__m128i c28_rslt_m128i = _mm_and_si128(c28_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[28]);
		_mm_storeu_si128(out++, c28_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c28_shfl_msk_m128i = _mm_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c28_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c28_shfl_msk_m128i);
		__m128i c28_mul_rslt_m128i = _mm_mullo_epi32(c28_shfl_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[28][0]);
		__m128i c28_srli_rslt_m128i = _mm_srli_epi32(c28_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[28][0]);
		__m128i c28_and_rslt_m128i = _mm_and_si128(c28_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[28]);
		__m128i c28_rslt_m128i = _mm_or_si128(c28_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 28));
		_mm_storeu_si128(out++, c28_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 29-bit values.
 * Load 29 SSE vectors, each containing 4 29-bit values. (5th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c29(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
     __m128i c29_load_rslt1_m128i = _mm_loadu_si128(in + 0);
     hor_sse4_unpack4_c29_f1<0>(out, c29_load_rslt1_m128i);    // Unpack 1st 4 values.

     __m128i c29_load_rslt2_m128i = _mm_loadu_si128(in + 1);
     __m128i c29_alignr_rslt1_m128i = _mm_alignr_epi8(c29_load_rslt2_m128i, c29_load_rslt1_m128i, 14);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt1_m128i);  // Unpack 2nd 4 values.

     __m128i c29_load_rslt3_m128i = _mm_loadu_si128(in + 2);
     __m128i c29_alignr_rslt2_m128i = _mm_alignr_epi8(c29_load_rslt3_m128i, c29_load_rslt2_m128i, 13);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt2_m128i);  // Unpack 3rd 4 values.

     __m128i c29_load_rslt4_m128i = _mm_loadu_si128(in + 3);
     __m128i c29_alignr_rslt3_m128i = _mm_alignr_epi8(c29_load_rslt4_m128i, c29_load_rslt3_m128i, 11);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt3_m128i);  // Unpack 4th 4 values.

     __m128i c29_load_rslt5_m128i = _mm_loadu_si128(in + 4);
     __m128i c29_alignr_rslt4_m128i = _mm_alignr_epi8(c29_load_rslt5_m128i, c29_load_rslt4_m128i, 10);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt4_m128i);  // Unpack 5th 4 values.

     __m128i c29_load_rslt6_m128i = _mm_loadu_si128(in + 5);
     __m128i c29_alignr_rslt5_m128i = _mm_alignr_epi8(c29_load_rslt6_m128i, c29_load_rslt5_m128i, 8);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt5_m128i);  // Unpack 6th 4 values.

     __m128i c29_load_rslt7_m128i = _mm_loadu_si128(in + 6);
     __m128i c29_alignr_rslt6_m128i = _mm_alignr_epi8(c29_load_rslt7_m128i, c29_load_rslt6_m128i, 7);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt6_m128i);  // Unpack 7th 4 values.

     __m128i c29_load_rslt8_m128i = _mm_loadu_si128(in + 7);
     __m128i c29_alignr_rslt7_m128i = _mm_alignr_epi8(c29_load_rslt8_m128i, c29_load_rslt7_m128i, 5);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt7_m128i);  // Unpack 8th 4 values.

     __m128i c29_load_rslt9_m128i = _mm_loadu_si128(in + 8);
     __m128i c29_alignr_rslt8_m128i = _mm_alignr_epi8(c29_load_rslt9_m128i, c29_load_rslt8_m128i, 4);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt8_m128i);  // Unpack 9th 4 values.

     __m128i c29_load_rslt10_m128i = _mm_loadu_si128(in + 9);
     __m128i c29_alignr_rslt9_m128i = _mm_alignr_epi8(c29_load_rslt10_m128i, c29_load_rslt9_m128i, 2);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt9_m128i);  // Unpack 10th 4 values.
     hor_sse4_unpack4_c29_f1<1>(out, c29_load_rslt10_m128i);   // Unpack 11th 4 values.

     __m128i c29_load_rslt11_m128i = _mm_loadu_si128(in + 10);
     __m128i c29_alignr_rslt10_m128i = _mm_alignr_epi8(c29_load_rslt11_m128i, c29_load_rslt10_m128i, 15);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt10_m128i); // Unpack 12th 4 values.

     __m128i c29_load_rslt12_m128i = _mm_loadu_si128(in + 11);
     __m128i c29_alignr_rslt11_m128i = _mm_alignr_epi8(c29_load_rslt12_m128i, c29_load_rslt11_m128i, 14);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt11_m128i); // Unpack 13th 4 values.

     __m128i c29_load_rslt13_m128i = _mm_loadu_si128(in + 12);
     __m128i c29_alignr_rslt12_m128i = _mm_alignr_epi8(c29_load_rslt13_m128i, c29_load_rslt12_m128i, 12);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt12_m128i); // Unpack 14th 4 values.

     __m128i c29_load_rslt14_m128i = _mm_loadu_si128(in + 13);
     __m128i c29_alignr_rslt13_m128i = _mm_alignr_epi8(c29_load_rslt14_m128i, c29_load_rslt13_m128i, 11);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt13_m128i); // Unpack 15th 4 values.

     __m128i c29_load_rslt15_m128i = _mm_loadu_si128(in + 14);
     __m128i c29_alignr_rslt14_m128i = _mm_alignr_epi8(c29_load_rslt15_m128i, c29_load_rslt14_m128i, 9);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt14_m128i); // Unpack 16th 4 values.

     __m128i c29_load_rslt16_m128i = _mm_loadu_si128(in + 15);
     __m128i c29_alignr_rslt15_m128i = _mm_alignr_epi8(c29_load_rslt16_m128i, c29_load_rslt15_m128i, 8);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt15_m128i); // Unpack 17th 4 values.

     __m128i c29_load_rslt17_m128i = _mm_loadu_si128(in + 16);
     __m128i c29_alignr_rslt16_m128i = _mm_alignr_epi8(c29_load_rslt17_m128i, c29_load_rslt16_m128i, 6);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt16_m128i); // Unpack 18th 4 values.

     __m128i c29_load_rslt18_m128i = _mm_loadu_si128(in + 17);
     __m128i c29_alignr_rslt17_m128i = _mm_alignr_epi8(c29_load_rslt18_m128i, c29_load_rslt17_m128i, 5);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt17_m128i); // Unpack 19th 4 values.

     __m128i c29_load_rslt19_m128i = _mm_loadu_si128(in + 18);
     __m128i c29_alignr_rslt18_m128i = _mm_alignr_epi8(c29_load_rslt19_m128i, c29_load_rslt18_m128i, 3);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt18_m128i); // Unpack 20th 4 values.

     __m128i c29_load_rslt20_m128i = _mm_loadu_si128(in + 19);
     __m128i c29_alignr_rslt19_m128i = _mm_alignr_epi8(c29_load_rslt20_m128i, c29_load_rslt19_m128i, 2);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt19_m128i); // Unpack 21st 4 values.
     hor_sse4_unpack4_c29_f2<0>(out, c29_load_rslt20_m128i);   // Unpack 22nd 4 values.

     __m128i c29_load_rslt21_m128i = _mm_loadu_si128(in + 20);
     __m128i c29_alignr_rslt20_m128i = _mm_alignr_epi8(c29_load_rslt21_m128i, c29_load_rslt20_m128i, 15);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt20_m128i); // Unpack 23rd 4 values.

     __m128i c29_load_rslt22_m128i = _mm_loadu_si128(in + 21);
     __m128i c29_alignr_rslt21_m128i = _mm_alignr_epi8(c29_load_rslt22_m128i, c29_load_rslt21_m128i, 13);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt21_m128i); // Unpack 24th 4 values.

     __m128i c29_load_rslt23_m128i = _mm_loadu_si128(in + 22);
     __m128i c29_alignr_rslt22_m128i = _mm_alignr_epi8(c29_load_rslt23_m128i, c29_load_rslt22_m128i, 12);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt22_m128i); // Unpack 25th 4 values.

     __m128i c29_load_rslt24_m128i = _mm_loadu_si128(in + 23);
     __m128i c29_alignr_rslt23_m128i = _mm_alignr_epi8(c29_load_rslt24_m128i, c29_load_rslt23_m128i, 10);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt23_m128i); // Unpack 26th 4 values.

     __m128i c29_load_rslt25_m128i = _mm_loadu_si128(in + 24);
     __m128i c29_alignr_rslt24_m128i = _mm_alignr_epi8(c29_load_rslt25_m128i, c29_load_rslt24_m128i, 9);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt24_m128i); // Unpack 27th 4 values.

     __m128i c29_load_rslt26_m128i = _mm_loadu_si128(in + 25);
     __m128i c29_alignr_rslt25_m128i = _mm_alignr_epi8(c29_load_rslt26_m128i, c29_load_rslt25_m128i, 7);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt25_m128i); // Unpack 28th 4 values.

     __m128i c29_load_rslt27_m128i = _mm_loadu_si128(in + 26);
     __m128i c29_alignr_rslt26_m128i = _mm_alignr_epi8(c29_load_rslt27_m128i, c29_load_rslt26_m128i, 6);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt26_m128i); // Unpack 29th 4 values.

     __m128i c29_load_rslt28_m128i = _mm_loadu_si128(in + 27);
     __m128i c29_alignr_rslt27_m128i = _mm_alignr_epi8(c29_load_rslt28_m128i, c29_load_rslt27_m128i, 4);
     hor_sse4_unpack4_c29_f2<0>(out, c29_alignr_rslt27_m128i); // Unpack 30th 4 values.

     __m128i c29_load_rslt29_m128i = _mm_loadu_si128(in + 28);
     __m128i c29_alignr_rslt28_m128i = _mm_alignr_epi8(c29_load_rslt29_m128i, c29_load_rslt28_m128i, 3);
     hor_sse4_unpack4_c29_f1<0>(out, c29_alignr_rslt28_m128i); // Unpack 31st 4 values.
     hor_sse4_unpack4_c29_f2<1>(out, c29_load_rslt29_m128i);   // Unpack 32nd 4 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c29_f1(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c29_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c29_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c29_shfl_msk_m128i);
		__m128i c29_slli_rslt_m128i = _mm_slli_epi64(c29_shfl_rslt_m128i, 3);

		__m128i c29_blend_rslt_m128i = _mm_blend_epi16(c29_shfl_rslt_m128i, c29_slli_rslt_m128i, 0xCC);

		__m128i c29_mul_rslt_m128i = _mm_mullo_epi32(c29_blend_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[29][0]);
		__m128i c29_srli_rslt_m128i = _mm_srli_epi32(c29_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[29][0]);
		__m128i c29_rslt_m128i = _mm_and_si128(c29_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[29]);
		_mm_storeu_si128(out++, c29_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c29_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c29_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c29_shfl_msk_m128i);
		__m128i c29_slli_rslt_m128i = _mm_slli_epi64(c29_shfl_rslt_m128i, 3);

		__m128i c29_blend_rslt_m128i = _mm_blend_epi16(c29_shfl_rslt_m128i, c29_slli_rslt_m128i, 0xCC);

		__m128i c29_mul_rslt_m128i = _mm_mullo_epi32(c29_blend_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[29][0]);
		__m128i c29_srli_rslt_m128i = _mm_srli_epi32(c29_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[29][0]);
		__m128i c29_and_rslt_m128i = _mm_and_si128(c29_srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[29]);
		__m128i c29_rslt_m128i = _mm_or_si128(c29_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 29));
		_mm_storeu_si128(out++, c29_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c29_f2(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c29_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c29_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c29_shfl_msk_m128i);
		__m128i c29_srli_rslt1_m128i = _mm_srli_epi64(c29_shfl_rslt_m128i, 3);

		__m128i c29_blend_rslt_m128i = _mm_blend_epi16(c29_shfl_rslt_m128i, c29_srli_rslt1_m128i, 0x33);

		__m128i c29_mul_rslt_m128i = _mm_mullo_epi32(c29_blend_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[29][1]);
		__m128i c29_srli_rslt2_m128i = _mm_srli_epi32(c29_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[29][1]);
		__m128i c29_rslt_m128i = _mm_and_si128(c29_srli_rslt2_m128i, SIMDMasks::SSE2_and_msk_m128i[29]);
		_mm_storeu_si128(out++, c29_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Unpack 4 values.
		const __m128i Hor_SSE4_c29_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c29_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c29_shfl_msk_m128i);
		__m128i c29_srli_rslt1_m128i = _mm_srli_epi64(c29_shfl_rslt_m128i, 3);

		__m128i c29_blend_rslt_m128i = _mm_blend_epi16(c29_shfl_rslt_m128i, c29_srli_rslt1_m128i, 0x33);

		__m128i c29_mul_rslt_m128i = _mm_mullo_epi32(c29_blend_rslt_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[29][1]);
		__m128i c29_srli_rslt2_m128i = _mm_srli_epi32(c29_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[29][1]);
		__m128i c29_and_rslt_m128i = _mm_and_si128(c29_srli_rslt2_m128i, SIMDMasks::SSE2_and_msk_m128i[29]);
		__m128i c29_rslt_m128i = _mm_or_si128(c29_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 29));
		_mm_storeu_si128(out++, c29_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 30-bit values.
 * Load 30 SSE vectors, each containing 4 30-bit values. (5th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c30(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
	     __m128i c30_load_rslt1_m128i = _mm_loadu_si128(in++);
	     hor_sse4_unpack4_c30<0>(out, c30_load_rslt1_m128i);   // Unpack 1st 4 values.

	     __m128i c30_load_rslt2_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt1_m128i = _mm_alignr_epi8(c30_load_rslt2_m128i, c30_load_rslt1_m128i, 15);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt1_m128i);  // Unpack 2nd 4 values.

	     __m128i c30_load_rslt3_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt2_m128i = _mm_alignr_epi8(c30_load_rslt3_m128i, c30_load_rslt2_m128i, 14);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt2_m128i);  // Unpack 3rd 4 values.

	     __m128i c30_load_rslt4_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt3_m128i = _mm_alignr_epi8(c30_load_rslt4_m128i, c30_load_rslt3_m128i, 13);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt3_m128i);  // Unpack 4th 4 values.

	     __m128i c30_load_rslt5_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt4_m128i = _mm_alignr_epi8(c30_load_rslt5_m128i, c30_load_rslt4_m128i, 12);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt4_m128i);  // Unpack 5th 4 values.

	     __m128i c30_load_rslt6_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt5_m128i = _mm_alignr_epi8(c30_load_rslt6_m128i, c30_load_rslt5_m128i, 11);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt5_m128i);  // Unpack 6th 4 values.

	     __m128i c30_load_rslt7_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt6_m128i = _mm_alignr_epi8(c30_load_rslt7_m128i, c30_load_rslt6_m128i, 10);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt6_m128i);  // Unpack 7th 4 values.

	     __m128i c30_load_rslt8_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt7_m128i = _mm_alignr_epi8(c30_load_rslt8_m128i, c30_load_rslt7_m128i, 9);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt7_m128i);  // Unpack 8th 4 values.

	     __m128i c30_load_rslt9_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt8_m128i = _mm_alignr_epi8(c30_load_rslt9_m128i, c30_load_rslt8_m128i, 8);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt8_m128i);  // Unpack 9th 4 values.

	     __m128i c30_load_rslt10_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt9_m128i = _mm_alignr_epi8(c30_load_rslt10_m128i, c30_load_rslt9_m128i, 7);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt9_m128i);  // Unpack 10th 4 values.

	     __m128i c30_load_rslt11_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt10_m128i = _mm_alignr_epi8(c30_load_rslt11_m128i, c30_load_rslt10_m128i, 6);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt10_m128i); // Unpack 11th 4 values.

	     __m128i c30_load_rslt12_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt11_m128i = _mm_alignr_epi8(c30_load_rslt12_m128i, c30_load_rslt11_m128i, 5);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt11_m128i); // Unpack 12th 4 values.

	     __m128i c30_load_rslt13_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt12_m128i = _mm_alignr_epi8(c30_load_rslt13_m128i, c30_load_rslt12_m128i, 4);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt12_m128i); // Unpack 13th 4 values.

	     __m128i c30_load_rslt14_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt13_m128i = _mm_alignr_epi8(c30_load_rslt14_m128i, c30_load_rslt13_m128i, 3);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt13_m128i); // Unpack 14th 4 values.

	     __m128i c30_load_rslt15_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt14_m128i = _mm_alignr_epi8(c30_load_rslt15_m128i, c30_load_rslt14_m128i, 2);
	     hor_sse4_unpack4_c30<0>(out, c30_alignr_rslt14_m128i); // Unpack 15th 4 values.
	     hor_sse4_unpack4_c30<1>(out, c30_load_rslt15_m128i);   // Unpack 16th 4 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c30(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Note that the 1st value's codeword is already in place. (aligned)
		//           the 4th value's codeword is already in place. (unaligned; 2 bits to the left)
		const __m128i Hor_SSE4_c30_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);
		__m128i c30_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c30_shfl_msk_m128i);

		// Shift the 2nd value's codeword in place. (aligned)
		__m128i c30_slli_rslt_m128i = _mm_slli_epi64(c30_shfl_rslt_m128i, 2);
		// Shift the 3rd value's codeword in place. (unaligned; 2 bits to the left)
		__m128i c30_srli_rslt1_m128i = _mm_srli_epi64(c30_shfl_rslt_m128i, 2);

		// Concatenate the 4 values's codewords.
		__m128i c30_blend_rslt1_m128i = _mm_blend_epi16(c30_shfl_rslt_m128i, c30_slli_rslt_m128i, 0x0C);
		__m128i c30_blend_rslt2_m128i = _mm_blend_epi16(c30_blend_rslt1_m128i, c30_srli_rslt1_m128i, 0x30);

		// Note that at this point the 4 values's codewords are already in place but unaligned.
		__m128i c30_mul_rslt_m128i = _mm_mullo_epi32(c30_blend_rslt2_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[30][0]);
		__m128i c30_srli_rslt2_m128i = _mm_srli_epi32(c30_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[30][0]);
		__m128i c30_rslt_m128i = _mm_and_si128(c30_srli_rslt2_m128i, SIMDMasks::SSE2_and_msk_m128i[30]);
		_mm_storeu_si128(out++, c30_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Note that the 1st value's codeword is already in place. (aligned)
		//           the 4th value's codeword is already in place. (unaligned; 2 bits to the left)
		const __m128i Hor_SSE4_c30_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);
		__m128i c30_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c30_shfl_msk_m128i);

		// Shift the 2nd value's codeword in place. (aligned)
		__m128i c30_slli_rslt_m128i = _mm_slli_epi64(c30_shfl_rslt_m128i, 2);
		// Shift the 3rd value's codeword in place. (unaligned; 2 bits to the left)
		__m128i c30_srli_rslt1_m128i = _mm_srli_epi64(c30_shfl_rslt_m128i, 2);

		// Concatenate the 4 values's codewords.
		__m128i c30_blend_rslt1_m128i = _mm_blend_epi16(c30_shfl_rslt_m128i, c30_slli_rslt_m128i, 0x0C);
		__m128i c30_blend_rslt2_m128i = _mm_blend_epi16(c30_blend_rslt1_m128i, c30_srli_rslt1_m128i, 0x30);

		// Note that at this point the 4 values's codewords are already in place but unaligned.
		__m128i c30_mul_rslt_m128i = _mm_mullo_epi32(c30_blend_rslt2_m128i, SIMDMasks::Hor_SSE4_mul_msk_m128i[30][0]);
		__m128i c30_srli_rslt2_m128i = _mm_srli_epi32(c30_mul_rslt_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[30][0]);
		__m128i c30_and_rslt_m128i = _mm_and_si128(c30_srli_rslt2_m128i, SIMDMasks::SSE2_and_msk_m128i[30]);
		__m128i c30_rslt_m128i = _mm_or_si128(c30_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 30));
		_mm_storeu_si128(out++, c30_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 31-bit values.
 * Load 31 SSE vectors, each containing 4 31-bit values. (5th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c31(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c31_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	hor_sse4_unpack4_c31_f1<0>(out, c31_load_rslt1_m128i);    // Unpack 1st 4 values.

	__m128i c31_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c31_alignr_rslt1_m128i = _mm_alignr_epi8(c31_load_rslt2_m128i, c31_load_rslt1_m128i, 15);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt1_m128i);  // Unpack 2nd 4 values.

	__m128i c31_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c31_alignr_rslt2_m128i = _mm_alignr_epi8(c31_load_rslt3_m128i, c31_load_rslt2_m128i, 15);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt2_m128i);  // Unpack 3rd 4 values.

	__m128i c31_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c31_alignr_rslt3_m128i = _mm_alignr_epi8(c31_load_rslt4_m128i, c31_load_rslt3_m128i, 14);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt3_m128i);  // Unpack 4th 4 values.

	__m128i c31_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c31_alignr_rslt4_m128i = _mm_alignr_epi8(c31_load_rslt5_m128i, c31_load_rslt4_m128i, 14);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt4_m128i);  // Unpack 5th 4 values.

	__m128i c31_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c31_alignr_rslt5_m128i = _mm_alignr_epi8(c31_load_rslt6_m128i, c31_load_rslt5_m128i, 13);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt5_m128i);  // Unpack 6th 4 values.

	__m128i c31_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c31_alignr_rslt6_m128i = _mm_alignr_epi8(c31_load_rslt7_m128i, c31_load_rslt6_m128i, 13);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt6_m128i);  // Unpack 7th 4 values.

	__m128i c31_load_rslt8_m128i = _mm_loadu_si128(in + 7);
	__m128i c31_alignr_rslt7_m128i = _mm_alignr_epi8(c31_load_rslt8_m128i, c31_load_rslt7_m128i, 12);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt7_m128i);  // Unpack 8th 4 values.

	__m128i c31_load_rslt9_m128i = _mm_loadu_si128(in + 8);
	__m128i c31_alignr_rslt8_m128i = _mm_alignr_epi8(c31_load_rslt9_m128i, c31_load_rslt8_m128i, 12);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt8_m128i);  // Unpack 9th 4 values.

	__m128i c31_load_rslt10_m128i = _mm_loadu_si128(in + 9);
	__m128i c31_alignr_rslt9_m128i = _mm_alignr_epi8(c31_load_rslt10_m128i, c31_load_rslt9_m128i, 11);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt9_m128i);  // Unpack 10th 4 values.

	__m128i c31_load_rslt11_m128i = _mm_loadu_si128(in + 10);
	__m128i c31_alignr_rslt10_m128i = _mm_alignr_epi8(c31_load_rslt11_m128i, c31_load_rslt10_m128i, 11);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt10_m128i); // Unpack 11th 4 values.

	__m128i c31_load_rslt12_m128i = _mm_loadu_si128(in + 11);
	__m128i c31_alignr_rslt11_m128i = _mm_alignr_epi8(c31_load_rslt12_m128i, c31_load_rslt11_m128i, 10);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt11_m128i); // Unpack 12th 4 values.

	__m128i c31_load_rslt13_m128i = _mm_loadu_si128(in + 12);
	__m128i c31_alignr_rslt12_m128i = _mm_alignr_epi8(c31_load_rslt13_m128i, c31_load_rslt12_m128i, 10);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt12_m128i); // Unpack 13th 4 values.

	__m128i c31_load_rslt14_m128i = _mm_loadu_si128(in + 13);
	__m128i c31_alignr_rslt13_m128i = _mm_alignr_epi8(c31_load_rslt14_m128i, c31_load_rslt13_m128i, 9);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt13_m128i); // Unpack 14th 4 values.

	__m128i c31_load_rslt15_m128i = _mm_loadu_si128(in + 14);
	__m128i c31_alignr_rslt14_m128i = _mm_alignr_epi8(c31_load_rslt15_m128i, c31_load_rslt14_m128i, 9);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt14_m128i); // Unpack 15th 4 values.

	__m128i c31_load_rslt16_m128i = _mm_loadu_si128(in + 15);
	__m128i c31_alignr_rslt15_m128i = _mm_alignr_epi8(c31_load_rslt16_m128i, c31_load_rslt15_m128i, 8);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt15_m128i); // Unpack 16th 4 values.

	__m128i c31_load_rslt17_m128i = _mm_loadu_si128(in + 16);
	__m128i c31_alignr_rslt16_m128i = _mm_alignr_epi8(c31_load_rslt17_m128i, c31_load_rslt16_m128i, 8);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt16_m128i); // Unpack 17th 4 values.

	__m128i c31_load_rslt18_m128i = _mm_loadu_si128(in + 17);
	__m128i c31_alignr_rslt17_m128i = _mm_alignr_epi8(c31_load_rslt18_m128i, c31_load_rslt17_m128i, 7);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt17_m128i); // Unpack 18th 4 values.

	__m128i c31_load_rslt19_m128i = _mm_loadu_si128(in + 18);
	__m128i c31_alignr_rslt18_m128i = _mm_alignr_epi8(c31_load_rslt19_m128i, c31_load_rslt18_m128i, 7);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt18_m128i); // Unpack 19th 4 values.

	__m128i c31_load_rslt20_m128i = _mm_loadu_si128(in + 19);
	__m128i c31_alignr_rslt19_m128i = _mm_alignr_epi8(c31_load_rslt20_m128i, c31_load_rslt19_m128i, 6);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt19_m128i); // Unpack 20th 4 values.

	__m128i c31_load_rslt21_m128i = _mm_loadu_si128(in + 20);
	__m128i c31_alignr_rslt20_m128i = _mm_alignr_epi8(c31_load_rslt21_m128i, c31_load_rslt20_m128i, 6);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt20_m128i); // Unpack 21st 4 values.

	__m128i c31_load_rslt22_m128i = _mm_loadu_si128(in + 21);
	__m128i c31_alignr_rslt21_m128i = _mm_alignr_epi8(c31_load_rslt22_m128i, c31_load_rslt21_m128i, 5);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt21_m128i); // Unpack 22nd 4 values.

	__m128i c31_load_rslt23_m128i = _mm_loadu_si128(in + 22);
	__m128i c31_alignr_rslt22_m128i = _mm_alignr_epi8(c31_load_rslt23_m128i, c31_load_rslt22_m128i, 5);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt22_m128i); // Unpack 23rd 4 values.

	__m128i c31_load_rslt24_m128i = _mm_loadu_si128(in + 23);
	__m128i c31_alignr_rslt23_m128i = _mm_alignr_epi8(c31_load_rslt24_m128i, c31_load_rslt23_m128i, 4);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt23_m128i); // Unpack 24th 4 values.

	__m128i c31_load_rslt25_m128i = _mm_loadu_si128(in + 24);
	__m128i c31_alignr_rslt24_m128i = _mm_alignr_epi8(c31_load_rslt25_m128i, c31_load_rslt24_m128i, 4);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt24_m128i); // Unpack 25th 4 values.

	__m128i c31_load_rslt26_m128i = _mm_loadu_si128(in + 25);
	__m128i c31_alignr_rslt25_m128i = _mm_alignr_epi8(c31_load_rslt26_m128i, c31_load_rslt25_m128i, 3);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt25_m128i); // Unpack 26th 4 values.

	__m128i c31_load_rslt27_m128i = _mm_loadu_si128(in + 26);
	__m128i c31_alignr_rslt26_m128i = _mm_alignr_epi8(c31_load_rslt27_m128i, c31_load_rslt26_m128i, 3);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt26_m128i); // Unpack 27th 4 values.

	__m128i c31_load_rslt28_m128i = _mm_loadu_si128(in + 27);
	__m128i c31_alignr_rslt27_m128i = _mm_alignr_epi8(c31_load_rslt28_m128i, c31_load_rslt27_m128i, 2);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt27_m128i); // Unpack 28th 4 values.

	__m128i c31_load_rslt29_m128i = _mm_loadu_si128(in + 28);
	__m128i c31_alignr_rslt28_m128i = _mm_alignr_epi8(c31_load_rslt29_m128i, c31_load_rslt28_m128i, 2);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt28_m128i); // Unpack 29th 4 values.

	__m128i c31_load_rslt30_m128i = _mm_loadu_si128(in + 29);
	__m128i c31_alignr_rslt29_m128i = _mm_alignr_epi8(c31_load_rslt30_m128i, c31_load_rslt29_m128i, 1);
	hor_sse4_unpack4_c31_f2<0>(out, c31_alignr_rslt29_m128i); // Unpack 30th 4 values.

	__m128i c31_load_rslt31_m128i = _mm_loadu_si128(in + 30);
	__m128i c31_alignr_rslt30_m128i = _mm_alignr_epi8(c31_load_rslt31_m128i, c31_load_rslt30_m128i, 1);
	hor_sse4_unpack4_c31_f1<0>(out, c31_alignr_rslt30_m128i); // Unpack 31st 4 values.
	hor_sse4_unpack4_c31_f2<0>(out, c31_load_rslt31_m128i);   // Unpack 32nd 4 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c31_f1(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Note that the 1st value's codeword is already in place (aligned).
		const __m128i Hor_SSE4_c31_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				0xFF, 0xFF, 0xFF, 0xFF,
				0xFF, 0xFF, 0xFF, 0xFF);
		__m128i c31_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c31_shfl_msk_m128i);

		// Shift the 3rd value's codeword in place (aligned).
		__m128i c31_srli_rslt_m128i = _mm_srli_epi64(c31_shfl_rslt_m128i, 6);
		// Shift the 2nd value's codeword in place (aligned).
		__m128i c31_slli_rslt1_m128i = _mm_slli_epi64(InReg, 1);
		// Shift the 4th value's codeword in place (aligned).
		__m128i c31_slli_rslt2_m128i = _mm_slli_epi64(InReg, 3);

		// Concatenate the 4 values's codewords.
		__m128i c31_blend_rslt1_m128i = _mm_blend_epi16(InReg, c31_srli_rslt_m128i, 0x30);
		__m128i c31_blend_rslt2_m128i = _mm_blend_epi16(c31_blend_rslt1_m128i, c31_slli_rslt1_m128i, 0x0C);
		__m128i c31_blend_rslt3_m128i = _mm_blend_epi16(c31_blend_rslt2_m128i, c31_slli_rslt2_m128i, 0xC0);

		// Note that at this point the 4 values's codewords are in place and aligned.
		__m128i c31_rslt_m128i = _mm_and_si128(c31_blend_rslt3_m128i, SIMDMasks::SSE2_and_msk_m128i[31]);
		_mm_storeu_si128(out++, c31_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Note that the 1st value's codeword is already in place (aligned).
		const __m128i Hor_SSE4_c31_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				0xFF, 0xFF, 0xFF, 0xFF,
				0xFF, 0xFF, 0xFF, 0xFF);
		__m128i c31_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c31_shfl_msk_m128i);

		// Shift the 3rd value's codeword in place (aligned).
		__m128i c31_srli_rslt_m128i = _mm_srli_epi64(c31_shfl_rslt_m128i, 6);
		// Shift the 2nd value's codeword in place (aligned).
		__m128i c31_slli_rslt1_m128i = _mm_slli_epi64(InReg, 1);
		// Shift the 4th value's codeword in place (aligned).
		__m128i c31_slli_rslt2_m128i = _mm_slli_epi64(InReg, 3);

		// Concatenate the 4 values's codewords.
		__m128i c31_blend_rslt1_m128i = _mm_blend_epi16(InReg, c31_srli_rslt_m128i, 0x30);
		__m128i c31_blend_rslt2_m128i = _mm_blend_epi16(c31_blend_rslt1_m128i, c31_slli_rslt1_m128i, 0x0C);
		__m128i c31_blend_rslt3_m128i = _mm_blend_epi16(c31_blend_rslt2_m128i, c31_slli_rslt2_m128i, 0xC0);

		// Note that at this point the 4 values's codewords are in place and aligned.
		__m128i c31_and_rslt_m128i = _mm_and_si128(c31_blend_rslt3_m128i, SIMDMasks::SSE2_and_msk_m128i[31]);
		__m128i c31_rslt_m128i = _mm_or_si128(c31_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 31));
		_mm_storeu_si128(out++, c31_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<SSE, IsRiceCoding>::hor_sse4_unpack4_c31_f2(__m128i *  __restrict__  &out,
		const __m128i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		// Note that the 4th value's codeword is already in place but unaligned. (1 bit to the left)
		const __m128i Hor_SSE4_c31_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, 0xFF,
				0xFF, 0xFF, 0xFF, 0xFF,
				byte + 8, byte + 7, byte + 6, byte + 5,
				byte + 4, byte + 3, byte + 2, byte + 1);
		__m128i c31_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c31_shfl_msk_m128i);

		// Shift the 2nd value's codeword in place. (unaligned, 1 bit to the left)
		__m128i c31_slli_rslt_m128i = _mm_slli_epi64(c31_shfl_rslt_m128i, 6);
		// Shift the 1st value's codeword in place. (unaligned, 1 bit to the left)
		__m128i c31_srli_rslt1_m128i = _mm_srli_epi64(InReg, 3);
		// Shift the 3rd value's codeword in place. (unaligned, 1 bit to the left)
		__m128i c31_srli_rslt2_m128i = _mm_srli_epi64(InReg, 1);

		// Concatenate the 4 values's codewords.
		__m128i c31_blend_rslt1_m128i = _mm_blend_epi16(InReg, c31_slli_rslt_m128i, 0x0C);
		__m128i c31_blend_rslt2_m128i = _mm_blend_epi16(c31_blend_rslt1_m128i, c31_srli_rslt1_m128i, 0x03);
		__m128i c31_blend_rslt3_m128i = _mm_blend_epi16(c31_blend_rslt2_m128i, c31_srli_rslt2_m128i, 0x30);

		// Note that at this point the 4 values's codewords are in place but unaligned. (1 bit to the left)
		__m128i c31_srli_rslt3_m128i = _mm_srli_epi32(c31_blend_rslt3_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[31][1]);
		__m128i c31_rslt_m128i = _mm_and_si128(c31_srli_rslt3_m128i, SIMDMasks::SSE2_and_msk_m128i[31]);
		_mm_storeu_si128(out++, c31_rslt_m128i);
	}
	else { // For Rice and OptRice.
		// Note that the 4th value's codeword is already in place but unaligned. (1 bit to the left)
		const __m128i Hor_SSE4_c31_shfl_msk_m128i = _mm_set_epi8(
				0xFF, 0xFF, 0xFF, 0xFF,
				0xFF, 0xFF, 0xFF, 0xFF,
				byte + 8, byte + 7, byte + 6, byte + 5,
				byte + 4, byte + 3, byte + 2, byte + 1);
		__m128i c31_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Hor_SSE4_c31_shfl_msk_m128i);

		// Shift the 2nd value's codeword in place. (unaligned; 1 bit to the left)
		__m128i c31_slli_rslt_m128i = _mm_slli_epi64(c31_shfl_rslt_m128i, 6);
		// Shift the 1st value's codeword in place. (unaligned; 1 bit to the left)
		__m128i c31_srli_rslt1_m128i = _mm_srli_epi64(InReg, 3);
		// Shift the 3rd value's codeword in place. (unaligned; 1 bit to the left)
		__m128i c31_srli_rslt2_m128i = _mm_srli_epi64(InReg, 1);

		// Concatenate the 4 values's codewords.
		__m128i c31_blend_rslt1_m128i = _mm_blend_epi16(InReg, c31_slli_rslt_m128i, 0x0C);
		__m128i c31_blend_rslt2_m128i = _mm_blend_epi16(c31_blend_rslt1_m128i, c31_srli_rslt1_m128i, 0x03);
		__m128i c31_blend_rslt3_m128i = _mm_blend_epi16(c31_blend_rslt2_m128i, c31_srli_rslt2_m128i, 0x30);

		// Note that at this point the 4 values's codewords are in place but unaligned. (1 bit to the left)
		__m128i c31_srli_rslt3_m128i = _mm_srli_epi32(c31_blend_rslt3_m128i, SIMDMasks::Hor_SSE4_srli_imm_int[31][1]);
		__m128i c31_and_rslt_m128i = _mm_and_si128(c31_srli_rslt3_m128i, SIMDMasks::SSE2_and_msk_m128i[31]);
		__m128i c31_rslt_m128i = _mm_or_si128(c31_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 31));
		_mm_storeu_si128(out++, c31_rslt_m128i);
	}
}


/**
 * SSE4-based unpacking 128 32-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<SSE, IsRiceCoding>::horizontalunpack_c32(__m128i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	uint32_t *outPtr = reinterpret_cast<uint32_t *>(out);
	const uint32_t *inPtr = reinterpret_cast<const uint32_t *>(in);
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
		memcpy32(outPtr, inPtr);
		outPtr += 32;
		inPtr += 32;
	}
}


#endif // CODECS_HORSSEUNPACKER_H_ 
