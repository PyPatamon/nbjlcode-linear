/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Mar 23, 2016
 */

#ifndef CODECS_HORAVXUNPACKER_H_ 
#define CODECS_HORAVXUNPACKER_H_ 


/**
 * AVX2-based unpacking 128 0-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c0(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
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
 * AVX2-based unpacking 128 1-bit values.
 * Load 1 SSE vector, containing 128 1-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c1(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c1_load_rslt_m128i = _mm_loadu_si128(in);
	__m256i c1_broadcast_rslt_m256i = _mm256_broadcastsi128_si256(c1_load_rslt_m128i);
	hor_avx2_unpack32_c1<0x00>(out, c1_broadcast_rslt_m256i); // Unpack 1st 32 values.
	hor_avx2_unpack32_c1<0x55>(out, c1_broadcast_rslt_m256i); // Unpack 1st 32 values.
	hor_avx2_unpack32_c1<0xAA>(out, c1_broadcast_rslt_m256i); // Unpack 3rd 32 values.
	hor_avx2_unpack32_c1<0xFF>(out, c1_broadcast_rslt_m256i); // Unpack 4th 32 values.
}

template <bool IsRiceCoding>
template <int imm8>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack32_c1(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	// We shuffle in a way to have a 1-bit codeword in every byte, and order the codewords appropriately:
	// v31v23v15v7|v30v22v14v6|v29v21v13v5|v28v20v12v4|v27v19v11v3|v26v18v10v2|v25v17v9v1|v24v16v8v0
    __m256i c1_shfl_rslt_m256i = _mm256_shuffle_epi32(InReg, imm8);
    hor_avx2_unpack8_c1<0>(out, c1_shfl_rslt_m256i);  // Unpack 1st 8 values.
    hor_avx2_unpack8_c1<8>(out, c1_shfl_rslt_m256i);  // Unpack 2nd 8 values.
    hor_avx2_unpack8_c1<16>(out, c1_shfl_rslt_m256i); // Unpack 3rd 8 values.
    hor_avx2_unpack8_c1<24>(out, c1_shfl_rslt_m256i); // Unpack 4th 8 values.
}

template <bool IsRiceCoding>
template <int bit>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c1(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c1_srlv_msk_m256i = _mm256_set_epi32(
				bit + 7, bit + 6, bit + 5, bit + 4,
				bit + 3, bit + 2, bit + 1, bit + 0);
		__m256i c1_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c1_srlv_msk_m256i);
		__m256i c1_rslt_m256i = _mm256_and_si256(c1_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[1]);
		_mm256_storeu_si256(out++, c1_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c1_srlv_msk_m256i = _mm256_set_epi32(
				bit + 7, bit + 6, bit + 5, bit + 4,
				bit + 3, bit + 2, bit + 1, bit + 0);
		__m256i c1_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c1_srlv_msk_m256i);
		__m256i c1_and_rslt_m256i = _mm256_and_si256(c1_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[1]);
		__m256i c1_rslt_m256i = _mm256_or_si256(c1_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 1));
		_mm256_storeu_si256(out++, c1_rslt_m256i);
	}
}


/**
 * AVX2-based unpacking 128 2-bit values.
 * Load 2 SSE vectors, each containing 64 2-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c2(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
		__m128i c2_load_rslt_m128i = _mm_loadu_si128(in++);
		__m256i c2_broadcast_rslt_m256i = _mm256_broadcastsi128_si256(c2_load_rslt_m128i);
		hor_avx2_unpack32_c2<0>(out, c2_broadcast_rslt_m256i); // Unpack 1st 32 values.
		hor_avx2_unpack32_c2<8>(out, c2_broadcast_rslt_m256i); // Unpack 2nd 32 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack32_c2(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	// We shuffle in a way to have a 2-bit codeword in every byte, and order the codewords appropriately:
	// v31v23v15v7|v30v22v14v6|v29v21v13v5|v28v20v12v4|v27v19v11v3|v26v18v10v2|v25v17v9v1|v24v16v8v0
	const __m256i Hor_AVX2_c2_shfl_msk_m256i = _mm256_set_epi8(
			byte + 7, byte + 5, byte + 3, byte + 1,
			byte + 7, byte + 5, byte + 3, byte + 1,
			byte + 7, byte + 5, byte + 3, byte + 1,
			byte + 7, byte + 5, byte + 3, byte + 1,
			byte + 6, byte + 4, byte + 2, byte + 0,
			byte + 6, byte + 4, byte + 2, byte + 0,
			byte + 6, byte + 4, byte + 2, byte + 0,
			byte + 6, byte + 4, byte + 2, byte + 0);
	__m256i c2_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c2_shfl_msk_m256i);
	hor_avx2_unpack8_c2<0>(out, c2_shfl_rslt_m256i);  // Unpack 1st 8 values.
	hor_avx2_unpack8_c2<8>(out, c2_shfl_rslt_m256i);  // Unpack 2nd 8 values.
	hor_avx2_unpack8_c2<16>(out, c2_shfl_rslt_m256i); // Unpack 3rd 8 values.
	hor_avx2_unpack8_c2<24>(out, c2_shfl_rslt_m256i); // Unpack 4th 8 values.
}

template <bool IsRiceCoding>
template <int bit>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c2(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c2_srlv_msk_m256i = _mm256_set_epi32(
				bit + 6, bit + 4, bit + 2, bit + 0,
				bit + 6, bit + 4, bit + 2, bit + 0);
		__m256i c2_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c2_srlv_msk_m256i);
		__m256i c2_rslt_m256i = _mm256_and_si256(c2_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[2]);
		_mm256_storeu_si256(out++, c2_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c2_srlv_msk_m256i = _mm256_set_epi32(
				bit + 6, bit + 4, bit + 2, bit + 0,
				bit + 6, bit + 4, bit + 2, bit + 0);
		__m256i c2_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c2_srlv_msk_m256i);
		__m256i c2_and_rslt_m256i = _mm256_and_si256(c2_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[2]);
		__m256i c2_rslt_m256i = _mm256_or_si256(c2_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 2));
		_mm256_storeu_si256(out++, c2_rslt_m256i);
	}
}


/**
 * AVX2-based unpacking 128 3-bit values.
 * Load 3 SSE vectors, each containing 42 3-bit values. (43th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c3(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c3_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	__m256i c3_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c3_load_rslt1_m128i);
	hor_avx2_unpack16_c3<0>(out, c3_broadcast_rslt1_m256i); // Unpack 1st 16 values.
	hor_avx2_unpack16_c3<6>(out, c3_broadcast_rslt1_m256i); // Unpack 2nd 16 values.

	__m128i c3_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c3_alignr_rslt1_m128i = _mm_alignr_epi8(c3_load_rslt2_m128i, c3_load_rslt1_m128i, 12);
	__m256i c3_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c3_alignr_rslt1_m128i);
	hor_avx2_unpack16_c3<0>(out, c3_broadcast_rslt2_m256i);  // Unpack 3rd 16 values.
	hor_avx2_unpack16_c3<6>(out, c3_broadcast_rslt2_m256i);  // Unpack 4th 16 values.
	__m256i c3_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c3_load_rslt2_m128i);
	hor_avx2_unpack16_c3<8>(out, c3_broadcast_rslt3_m256i);  // Unpack 5th 16 values.

	__m128i c3_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c3_alignr_rslt2_m128i = _mm_alignr_epi8(c3_load_rslt3_m128i, c3_load_rslt2_m128i, 14);
	__m256i c3_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c3_alignr_rslt2_m128i);
	hor_avx2_unpack16_c3<0>(out, c3_broadcast_rslt4_m256i);  // Unpack 6th 16 values.
	hor_avx2_unpack16_c3<6>(out, c3_broadcast_rslt4_m256i);  // Unpack 7th 16 values.
	__m256i c3_broadcast_rslt5_m256i = _mm256_broadcastsi128_si256(c3_load_rslt3_m128i);
	hor_avx2_unpack16_c3<10>(out, c3_broadcast_rslt5_m256i); // Unpack 8th 16 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack16_c3(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	// We shuffle in a way to have a 3-bit codeword in every 16-bit word,
	// and order the codewords appropriately:
	// -v15-v7|-v14-v6|-v13-v5|-v12-v4|-v11-v3|-v10-v2|-v9-v1|-v8-v0
	const __m256i Hor_AVX2_c3_shfl_msk_m256i = _mm256_set_epi8(
			0xFF, byte + 5, 0xFF, byte + 2,
			0xFF, byte + 5, 0xFF, byte + 2,
			byte + 5, byte + 4, byte + 2, byte + 1,
			0xFF, byte + 4, 0xFF, byte + 1,
			0xFF, byte + 4, 0xFF, byte + 1,
			byte + 4, byte + 3, byte + 1, byte + 0,
			0xFF, byte + 3, 0xFF, byte + 0,
			0xFF, byte + 3, 0xFF, byte + 0);
	__m256i c3_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c3_shfl_msk_m256i);
	hor_avx2_unpack8_c3<0>(out, c3_shfl_rslt_m256i);  // Unpack 1st 8 values.
	hor_avx2_unpack8_c3<16>(out, c3_shfl_rslt_m256i); // Unpack 2nd 8 values.
}

template <bool IsRiceCoding>
template <int bit>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c3(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c3_srlv_msk_m256i = _mm256_set_epi32(
				bit + 5, bit + 2, bit + 7, bit + 4,
				bit + 1, bit + 6, bit + 3, bit + 0);
		__m256i c3_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c3_srlv_msk_m256i);
		__m256i c3_rslt_m256i = _mm256_and_si256(c3_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[3]);
		_mm256_storeu_si256(out++, c3_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c3_srlv_msk_m256i = _mm256_set_epi32(
				bit + 5, bit + 2, bit + 7, bit + 4,
				bit + 1, bit + 6, bit + 3, bit + 0);
		__m256i c3_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c3_srlv_msk_m256i);
		__m256i c3_and_rslt_m256i = _mm256_and_si256(c3_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[3]);
		__m256i c3_rslt_m256i = _mm256_or_si256(c3_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 3));
		_mm256_storeu_si256(out++, c3_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 3-bit values.
// * Load 3 SSE vectors, each containing 42 3-bit values. (43th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c3(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	__m128i c3_load_rslt1_m128i = _mm_loadu_si128(in + 0);
//	__m256i c3_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c3_load_rslt1_m128i);
//	hor_avx2_unpack8_c3<0>(out, c3_broadcast_rslt1_m256i);  // Unpack 1st 8 values.
//	hor_avx2_unpack8_c3<3>(out, c3_broadcast_rslt1_m256i);  // Unpack 2nd 8 values.
//	hor_avx2_unpack8_c3<6>(out, c3_broadcast_rslt1_m256i);  // Unpack 3rd 8 values.
//	hor_avx2_unpack8_c3<9>(out, c3_broadcast_rslt1_m256i);  // Unpack 4th 8 values.
//	hor_avx2_unpack8_c3<12>(out, c3_broadcast_rslt1_m256i); // Unpack 5th 8 values.
//
//	__m128i c3_load_rslt2_m128i = _mm_loadu_si128(in + 1);
//	__m128i c3_alignr_rslt1_m128i = _mm_alignr_epi8(c3_load_rslt2_m128i, c3_load_rslt1_m128i, 15);
//	__m256i c3_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c3_alignr_rslt1_m128i);
//	hor_avx2_unpack8_c3<0>(out, c3_broadcast_rslt2_m256i);  // Unpack 6th 8 values.
//	hor_avx2_unpack8_c3<3>(out, c3_broadcast_rslt2_m256i);  // Unpack 7th 8 values.
//	hor_avx2_unpack8_c3<6>(out, c3_broadcast_rslt2_m256i);  // Unpack 8th 8 values.
//	hor_avx2_unpack8_c3<9>(out, c3_broadcast_rslt2_m256i);  // Unpack 9th 8 values.
//	hor_avx2_unpack8_c3<12>(out, c3_broadcast_rslt2_m256i); // Unpack 10th 8 values.
//
//	__m128i c3_load_rslt3_m128i = _mm_loadu_si128(in + 2);
//	__m128i c3_alignr_rslt2_m128i = _mm_alignr_epi8(c3_load_rslt3_m128i, c3_load_rslt2_m128i, 14);
//	__m256i c3_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c3_alignr_rslt2_m128i);
//	hor_avx2_unpack8_c3<0>(out, c3_broadcast_rslt3_m256i);  // Unpack 11th 8 values.
//	hor_avx2_unpack8_c3<3>(out, c3_broadcast_rslt3_m256i);  // Unpack 12th 8 values.
//	hor_avx2_unpack8_c3<6>(out, c3_broadcast_rslt3_m256i);  // Unpack 13th 8 values.
//	hor_avx2_unpack8_c3<9>(out, c3_broadcast_rslt3_m256i);  // Unpack 14th 8 values.
//	hor_avx2_unpack8_c3<12>(out, c3_broadcast_rslt3_m256i); // Unpack 15th 8 values.
//	__m256i c3_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c3_load_rslt3_m128i);
//	hor_avx2_unpack8_c3<13>(out, c3_broadcast_rslt4_m256i); // Unpack 16th 8 values.
//}
//
//template <bool IsRiceCoding>
//template <int byte>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c3(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		const __m256i Hor_AVX2_c3_shfl_msk_m256i = _mm256_set_epi8(
//				0xFF, 0xFF, 0xFF, byte + 2,
//				0xFF, 0xFF, 0xFF, byte + 2,
//				0xFF, 0xFF, byte + 2, byte + 1,
//				0xFF, 0xFF, 0xFF, byte + 1,
//				0xFF, 0xFF, 0xFF, byte + 1,
//				0xFF, 0xFF, byte + 1, byte + 0,
//				0xFF, 0xFF, 0xFF, byte + 0,
//				0xFF, 0xFF, 0xFF, byte + 0);
//		__m256i c3_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c3_shfl_msk_m256i);
//		__m256i c3_srlv_rslt_m256i = _mm256_srlv_epi32(c3_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[3]);
//		__m256i c3_rslt_m256i = _mm256_and_si256(c3_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[3]);
//		_mm256_storeu_si256(out++, c3_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		const __m256i Hor_AVX2_c3_shfl_msk_m256i = _mm256_set_epi8(
//				0xFF, 0xFF, 0xFF, byte + 2,
//				0xFF, 0xFF, 0xFF, byte + 2,
//				0xFF, 0xFF, byte + 2, byte + 1,
//				0xFF, 0xFF, 0xFF, byte + 1,
//				0xFF, 0xFF, 0xFF, byte + 1,
//				0xFF, 0xFF, byte + 1, byte + 0,
//				0xFF, 0xFF, 0xFF, byte + 0,
//				0xFF, 0xFF, 0xFF, byte + 0);
//		__m256i c3_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c3_shfl_msk_m256i);
//		__m256i c3_srlv_rslt_m256i = _mm256_srlv_epi32(c3_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[3]);
//		__m256i c3_and_rslt_m256i = _mm256_and_si256(c3_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[3]);
//		__m256i c3_rslt_m256i = _mm256_or_si256(c3_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 3));
//		_mm256_storeu_si256(out++, c3_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 4-bit values.
 * Load 4 SSE vectors, each containing 32 4-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c4(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
		__m128i c4_load_rslt_m128i = _mm_loadu_si128(in++);
		__m256i c4_broadcast_rslt_m256i = _mm256_broadcastsi128_si256(c4_load_rslt_m128i);
		// We shuffle in a way to have a 4-bit codeword in every byte, and order the codewords appropriately:
		// v31v23v15v7|v30v22v14v6|v29v21v13v5|v28v20v12v4|v27v19v11v3|v26v18v10v2|v25v17v9v1|v24v16v8v0
		const __m256i Hor_AVX2_c4_shfl_msk_m256i = _mm256_set_epi8(
				15, 11, 7, 3,
				15, 11, 7, 3,
				14, 10, 6, 2,
				14, 10, 6, 2,
				13, 9, 5, 1,
				13, 9, 5, 1,
				12, 8, 4, 0,
				12, 8, 4, 0);
		__m256i c4_shfl_rslt_m256i = _mm256_shuffle_epi8(c4_broadcast_rslt_m256i, Hor_AVX2_c4_shfl_msk_m256i);
		hor_avx2_unpack8_c4<0>(out, c4_shfl_rslt_m256i);  // Unpack 1st 8 values.
		hor_avx2_unpack8_c4<8>(out, c4_shfl_rslt_m256i);  // Unpack 2nd 8 values.
		hor_avx2_unpack8_c4<16>(out, c4_shfl_rslt_m256i); // Unpack 3rd 8 values.
		hor_avx2_unpack8_c4<24>(out, c4_shfl_rslt_m256i); // Unpack 4th 8 values.
	}
}

template <bool IsRiceCoding>
template <int bit>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c4(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c4_srlv_msk_m256i = _mm256_set_epi32(
				bit + 4, bit + 0, bit + 4, bit + 0,
				bit + 4, bit + 0, bit + 4, bit + 0);
		__m256i c4_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c4_srlv_msk_m256i);
		__m256i c4_rslt_m256i = _mm256_and_si256(c4_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[4]);
		_mm256_storeu_si256(out++, c4_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c4_srlv_msk_m256i = _mm256_set_epi32(
				bit + 4, bit + 0, bit + 4, bit + 0,
				bit + 4, bit + 0, bit + 4, bit + 0);
		__m256i c4_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c4_srlv_msk_m256i);
		__m256i c4_and_rslt_m256i = _mm256_and_si256(c4_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[4]);
		__m256i c4_rslt_m256i = _mm256_or_si256(c4_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 4));
		_mm256_storeu_si256(out++, c4_rslt_m256i);
	}
}


/**
 * AVX2-based unpacking 128 5-bit values.
 * Load 5 SSE vectors, each containing 25 5-bit values. (26th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c5(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c5_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	__m256i c5_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c5_load_rslt1_m128i);
	hor_avx2_unpack16_c5<0>(out, c5_broadcast_rslt1_m256i); // Unpack 1st 16 values.

	__m128i c5_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c5_alignr_rslt1_m128i = _mm_alignr_epi8(c5_load_rslt2_m128i, c5_load_rslt1_m128i, 10);
	__m256i c5_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c5_alignr_rslt1_m128i);
	hor_avx2_unpack16_c5<0>(out, c5_broadcast_rslt2_m256i); // Unpack 2nd 16 values.
	__m256i c5_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c5_load_rslt2_m128i);
	hor_avx2_unpack16_c5<4>(out, c5_broadcast_rslt3_m256i); // Unpack 3rd 16 values.

	__m128i c5_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c5_alignr_rslt2_m128i = _mm_alignr_epi8(c5_load_rslt3_m128i, c5_load_rslt2_m128i, 14);
	__m256i c5_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c5_alignr_rslt2_m128i);
	hor_avx2_unpack16_c5<0>(out, c5_broadcast_rslt4_m256i); // Unpack 4th 16 values.

	__m128i c5_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c5_alignr_rslt3_m128i = _mm_alignr_epi8(c5_load_rslt4_m128i, c5_load_rslt3_m128i, 8);
	__m256i c5_broadcast_rslt5_m256i = _mm256_broadcastsi128_si256(c5_alignr_rslt3_m128i);
	hor_avx2_unpack16_c5<0>(out, c5_broadcast_rslt5_m256i); // Unpack 5th 16 values.
	__m256i c5_broadcast_rslt6_m256i = _mm256_broadcastsi128_si256(c5_load_rslt4_m128i);
	hor_avx2_unpack16_c5<2>(out, c5_broadcast_rslt6_m256i); // Unpack 6th 16 values.

	__m128i c5_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c5_alignr_rslt4_m128i = _mm_alignr_epi8(c5_load_rslt5_m128i, c5_load_rslt4_m128i, 12);
	__m256i c5_broadcast_rslt7_m256i = _mm256_broadcastsi128_si256(c5_alignr_rslt4_m128i);
	hor_avx2_unpack16_c5<0>(out, c5_broadcast_rslt7_m256i); // Unpack 7th 16 values.
	__m256i c5_broadcast_rslt8_m256i = _mm256_broadcastsi128_si256(c5_load_rslt5_m128i);
	hor_avx2_unpack16_c5<6>(out, c5_broadcast_rslt8_m256i); // Unpack 8th 16 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack16_c5(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	// We shuffle in a way to have a 5-bit codeword in every 16-bit word,
	// and order the codewords appropriately:
	// -v15-v7|-v14-v6|-v13-v5|-v12-v4|-v11-v3|-v10-v2|-v9-v1|-v8-v0
	const __m256i Hor_AVX2_c5_shfl_msk_m256i = _mm256_set_epi8(
			0xFF, byte + 9, 0xFF, byte + 4,
			byte + 9, byte + 8, byte + 4, byte + 3,
			0xFF, byte + 8, 0xFF, byte + 3,
			byte + 8, byte + 7, byte + 3, byte + 2,
			byte + 7, byte + 6, byte + 2, byte + 1,
			0xFF, byte + 6, 0xFF, byte + 1,
			byte + 6, byte + 5, byte + 1, byte + 0,
			0xFF, byte + 5, 0xFF, byte + 0);
	__m256i c5_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c5_shfl_msk_m256i);
	hor_avx2_unpack8_c5<0>(out, c5_shfl_rslt_m256i);  // Unpack 1st 8 values.
	hor_avx2_unpack8_c5<16>(out, c5_shfl_rslt_m256i); // Unpack 2nd 8 values.
}

template <bool IsRiceCoding>
template <int bit>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c5(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c5_srlv_msk_m256i = _mm256_set_epi32(
				bit + 3, bit + 6, bit + 1, bit + 4,
				bit + 7, bit + 2, bit + 5, bit + 0);
		__m256i c5_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c5_srlv_msk_m256i);
		__m256i c5_rslt_m256i = _mm256_and_si256(c5_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[5]);
		_mm256_storeu_si256(out++, c5_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c5_srlv_msk_m256i = _mm256_set_epi32(
				bit + 3, bit + 6, bit + 1, bit + 4,
				bit + 7, bit + 2, bit + 5, bit + 0);
		__m256i c5_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c5_srlv_msk_m256i);
		__m256i c5_and_rslt_m256i = _mm256_and_si256(c5_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[5]);
		__m256i c5_rslt_m256i = _mm256_or_si256(c5_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 5));
		_mm256_storeu_si256(out++, c5_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 5-bit values.
// * Load 5 SSE vectors, each containing 25 5-bit values. (26th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c5(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	__m128i c5_load_rslt1_m128i = _mm_loadu_si128(in + 0);
//	__m256i c5_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c5_load_rslt1_m128i);
//	hor_avx2_unpack8_c5<0>(out, c5_broadcast_rslt1_m256i);  // Unpack 1st 8 values.
//	hor_avx2_unpack8_c5<5>(out, c5_broadcast_rslt1_m256i);  // Unpack 2nd 8 values.
//	hor_avx2_unpack8_c5<10>(out, c5_broadcast_rslt1_m256i); // Unpack 3rd 8 values.
//
//	__m128i c5_load_rslt2_m128i = _mm_loadu_si128(in + 1);
//	__m128i c5_alignr_rslt1_m128i = _mm_alignr_epi8(c5_load_rslt2_m128i, c5_load_rslt1_m128i, 15);
//	__m256i c5_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c5_alignr_rslt1_m128i);
//	hor_avx2_unpack8_c5<0>(out, c5_broadcast_rslt2_m256i);  // Unpack 4th 8 values.
//	hor_avx2_unpack8_c5<5>(out, c5_broadcast_rslt2_m256i);  // Unpack 5th 8 values.
//	hor_avx2_unpack8_c5<10>(out, c5_broadcast_rslt2_m256i); // Unpack 6th 8 values.
//
//	__m128i c5_load_rslt3_m128i = _mm_loadu_si128(in + 2);
//	__m128i c5_alignr_rslt2_m128i = _mm_alignr_epi8(c5_load_rslt3_m128i, c5_load_rslt2_m128i, 14);
//	__m256i c5_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c5_alignr_rslt2_m128i);
//	hor_avx2_unpack8_c5<0>(out, c5_broadcast_rslt3_m256i);  // Unpack 7th 8 values.
//	hor_avx2_unpack8_c5<5>(out, c5_broadcast_rslt3_m256i);  // Unpack 8th 8 values.
//	hor_avx2_unpack8_c5<10>(out, c5_broadcast_rslt3_m256i); // Unpack 9th 8 values.
//
//	__m128i c5_load_rslt4_m128i = _mm_loadu_si128(in + 3);
//	__m128i c5_alignr_rslt3_m128i = _mm_alignr_epi8(c5_load_rslt4_m128i, c5_load_rslt3_m128i, 13);
//	__m256i c5_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c5_alignr_rslt3_m128i);
//	hor_avx2_unpack8_c5<0>(out, c5_broadcast_rslt4_m256i);  // Unpack 10th 8 values.
//	hor_avx2_unpack8_c5<5>(out, c5_broadcast_rslt4_m256i);  // Unpack 11th 8 values.
//	hor_avx2_unpack8_c5<10>(out, c5_broadcast_rslt4_m256i); // Unpack 12th 8 values.
//
//	__m128i c5_load_rslt5_m128i = _mm_loadu_si128(in + 4);
//	__m128i c5_alignr_rslt4_m128i = _mm_alignr_epi8(c5_load_rslt5_m128i, c5_load_rslt4_m128i, 12);
//	__m256i c5_broadcast_rslt5_m256i = _mm256_broadcastsi128_si256(c5_alignr_rslt4_m128i);
//	hor_avx2_unpack8_c5<0>(out, c5_broadcast_rslt5_m256i);  // Unpack 13th 8 values.
//	hor_avx2_unpack8_c5<5>(out, c5_broadcast_rslt5_m256i);  // Unpack 14th 8 values.
//	hor_avx2_unpack8_c5<10>(out, c5_broadcast_rslt5_m256i); // Unpack 15th 8 values.
//	__m256i c5_broadcast_rslt6_m256i = _mm256_broadcastsi128_si256(c5_load_rslt5_m128i);
//	hor_avx2_unpack8_c5<11>(out, c5_broadcast_rslt6_m256i); // Unpack 16th 8 values.
//}
//
//template <bool IsRiceCoding>
//template <int byte>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c5(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		const __m256i Hor_AVX2_c5_shfl_msk_m256i = _mm256_set_epi8(
//				0xFF, 0xFF, 0xFF, byte + 4,
//				0xFF, 0xFF, byte + 4, byte + 3,
//				0xFF, 0xFF, 0xFF, byte + 3,
//				0xFF, 0xFF, byte + 3, byte + 2,
//				0xFF, 0xFF, byte + 2, byte + 1,
//				0xFF, 0xFF, 0xFF, byte + 1,
//				0xFF, 0xFF, byte + 1, byte + 0,
//				0xFF, 0xFF, 0xFF, byte + 0);
//		__m256i c5_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c5_shfl_msk_m256i);
//		__m256i c5_srlv_rslt_m256i = _mm256_srlv_epi32(c5_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[5]);
//		__m256i c5_rslt_m256i = _mm256_and_si256(c5_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[5]);
//		_mm256_storeu_si256(out++, c5_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		const __m256i Hor_AVX2_c5_shfl_msk_m256i = _mm256_set_epi8(
//				0xFF, 0xFF, 0xFF, byte + 4,
//				0xFF, 0xFF, byte + 4, byte + 3,
//				0xFF, 0xFF, 0xFF, byte + 3,
//				0xFF, 0xFF, byte + 3, byte + 2,
//				0xFF, 0xFF, byte + 2, byte + 1,
//				0xFF, 0xFF, 0xFF, byte + 1,
//				0xFF, 0xFF, byte + 1, byte + 0,
//				0xFF, 0xFF, 0xFF, byte + 0);
//		__m256i c5_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c5_shfl_msk_m256i);
//		__m256i c5_srlv_rslt_m256i = _mm256_srlv_epi32(c5_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[5]);
//		__m256i c5_and_rslt_m256i = _mm256_and_si256(c5_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[5]);
//		__m256i c5_rslt_m256i = _mm256_or_si256(c5_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 5));
//		_mm256_storeu_si256(out++, c5_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 6-bit values.
 * Load 6 SSE vectors, each containing 21 6-bit values. (22nd is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c6(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
		__m128i c6_load_rslt1_m128i = _mm_loadu_si128(in++);
		__m256i c6_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c6_load_rslt1_m128i);
		hor_avx2_unpack16_c6<0>(out, c6_broadcast_rslt1_m256i); // Unpack 1st 16 values

		__m128i c6_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c6_alignr_rslt1_m128i = _mm_alignr_epi8(c6_load_rslt2_m128i, c6_load_rslt1_m128i, 12);
		__m256i c6_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c6_alignr_rslt1_m128i);
		hor_avx2_unpack16_c6<0>(out, c6_broadcast_rslt2_m256i); // Unpack 2nd 16 values

		__m128i c6_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c6_alignr_rslt2_m128i = _mm_alignr_epi8(c6_load_rslt3_m128i, c6_load_rslt2_m128i, 8);
		__m256i c6_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c6_alignr_rslt2_m128i);
		hor_avx2_unpack16_c6<0>(out, c6_broadcast_rslt3_m256i); // Unpack 3rd 16 values.
		__m256i c6_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c6_load_rslt3_m128i);
		hor_avx2_unpack16_c6<4>(out, c6_broadcast_rslt4_m256i); // Unpack 4th 16 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack16_c6(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	// We shuffle in a way to have a 6-bit codeword in every 16-bit word,
	// and order the codewords appropriately:
	// -v15-v7|-v14-v6|-v13-v5|-v12-v4|-v11-v3|-v10-v2|-v9-v1|-v8-v0
	const __m256i Hor_AVX2_c6_shfl_msk_m256i = _mm256_set_epi8(
			0xFF, byte + 11, 0xFF, byte + 5,
			byte + 11, byte + 10, byte + 5, byte + 4,
			byte + 10, byte + 9, byte + 4, byte + 3,
			0xFF, byte + 9, 0xFF, byte + 3,
			0xFF, byte + 8, 0xFF, byte + 2,
			byte + 8, byte + 7, byte + 2, byte + 1,
			byte + 7, byte + 6, byte + 1, byte + 0,
			0xFF, byte + 6, 0xFF, byte + 0);
	__m256i c6_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c6_shfl_msk_m256i);
	hor_avx2_unpack8_c6<0>(out, c6_shfl_rslt_m256i);  // Unpack 1st 8 values.
	hor_avx2_unpack8_c6<16>(out, c6_shfl_rslt_m256i); // Unpack 2nd 8 values.
}

template <bool IsRiceCoding>
template <int bit>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c6(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c6_srlv_msk_m256i = _mm256_set_epi32(
				bit + 2, bit + 4, bit + 6, bit + 0,
				bit + 2, bit + 4, bit + 6, bit + 0);
		__m256i c6_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c6_srlv_msk_m256i);
		__m256i c6_rslt_m256i = _mm256_and_si256(c6_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[6]);
		_mm256_storeu_si256(out++, c6_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c6_srlv_msk_m256i = _mm256_set_epi32(
				bit + 2, bit + 4, bit + 6, bit + 0,
				bit + 2, bit + 4, bit + 6, bit + 0);
		__m256i c6_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c6_srlv_msk_m256i);
		__m256i c6_and_rslt_m256i = _mm256_and_si256(c6_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[6]);
		__m256i c6_rslt_m256i = _mm256_or_si256(c6_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 6));
		_mm256_storeu_si256(out++, c6_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 6-bit values.
// * Load 6 SSE vectors, each containing 21 6-bit values. (22nd is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c6(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
//		__m128i c6_load_rslt1_m128i = _mm_loadu_si128(in++);
//		__m256i c6_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c6_load_rslt1_m128i);
//		hor_avx2_unpack8_c6<0>(out, c6_broadcast_rslt1_m256i);  // Unpack 1st 8 values
//		hor_avx2_unpack8_c6<6>(out, c6_broadcast_rslt1_m256i);  // Unpack 2nd 8 values
//
//		__m128i c6_load_rslt2_m128i = _mm_loadu_si128(in++);
//		__m128i c6_alignr_rslt1_m128i = _mm_alignr_epi8(c6_load_rslt2_m128i, c6_load_rslt1_m128i, 12);
//		__m256i c6_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c6_alignr_rslt1_m128i);
//		hor_avx2_unpack8_c6<0>(out, c6_broadcast_rslt2_m256i);  // Unpack 3rd 8 values
//		hor_avx2_unpack8_c6<6>(out, c6_broadcast_rslt2_m256i);  // Unpack 4th 8 values
//		__m256i c6_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c6_load_rslt2_m128i);
//		hor_avx2_unpack8_c6<8>(out, c6_broadcast_rslt3_m256i);  // Unpack 5th 8 values
//
//		__m128i c6_load_rslt3_m128i = _mm_loadu_si128(in++);
//		__m128i c6_alignr_rslt2_m128i = _mm_alignr_epi8(c6_load_rslt3_m128i, c6_load_rslt2_m128i, 14);
//		__m256i c6_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c6_alignr_rslt2_m128i);
//		hor_avx2_unpack8_c6<0>(out, c6_broadcast_rslt4_m256i);  // Unpack 6th 8 values.
//		hor_avx2_unpack8_c6<6>(out, c6_broadcast_rslt4_m256i);  // Unpack 7th 8 values.
//		__m256i c6_broadcast_rslt5_m256i = _mm256_broadcastsi128_si256(c6_load_rslt3_m128i);
//		hor_avx2_unpack8_c6<10>(out, c6_broadcast_rslt5_m256i); // Unpack 8th 8 values.
//	}
//}
//
//template <bool IsRiceCoding>
//template <int byte>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c6(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		const __m256i Hor_AVX2_c6_shfl_msk_m256i = _mm256_set_epi8(
//				0xFF, 0xFF, 0xFF, byte + 5,
//				0xFF, 0xFF, byte + 5, byte + 4,
//				0xFF, 0xFF, byte + 4, byte + 3,
//				0xFF, 0xFF, 0xFF, byte + 3,
//				0xFF, 0xFF, 0xFF, byte + 2,
//				0xFF, 0xFF, byte + 2, byte + 1,
//				0xFF, 0xFF, byte + 1, byte + 0,
//				0xFF, 0xFF, 0xFF, byte + 0);
//		__m256i c6_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c6_shfl_msk_m256i);
//		__m256i c6_srlv_rslt_m256i = _mm256_srlv_epi32(c6_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[6]);
//		__m256i c6_rslt_m256i = _mm256_and_si256(c6_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[6]);
//		_mm256_storeu_si256(out++, c6_rslt_m256i);
//
//	}
//	else { // For Rice and OptRice.
//		const __m256i Hor_AVX2_c6_shfl_msk_m256i = _mm256_set_epi8(
//				0xFF, 0xFF, 0xFF, byte + 5,
//				0xFF, 0xFF, byte + 5, byte + 4,
//				0xFF, 0xFF, byte + 4, byte + 3,
//				0xFF, 0xFF, 0xFF, byte + 3,
//				0xFF, 0xFF, 0xFF, byte + 2,
//				0xFF, 0xFF, byte + 2, byte + 1,
//				0xFF, 0xFF, byte + 1, byte + 0,
//				0xFF, 0xFF, 0xFF, byte + 0);
//		__m256i c6_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c6_shfl_msk_m256i);
//		__m256i c6_srlv_rslt_m256i = _mm256_srlv_epi32(c6_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[6]);
//		__m256i c6_and_rslt_m256i = _mm256_and_si256(c6_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[6]);
//		__m256i c6_rslt_m256i = _mm256_or_si256(c6_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 6));
//		_mm256_storeu_si256(out++, c6_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 7-bit values.
 * Load 7 SSE vectors, each containing 18 7-bit values. (19th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c7(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c7_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	__m256i c7_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c7_load_rslt1_m128i);
	hor_avx2_unpack16_c7<0>(out, c7_broadcast_rslt1_m256i); // Unpack 1st 16 values.

	__m128i c7_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c7_alignr_rslt1_m128i = _mm_alignr_epi8(c7_load_rslt2_m128i, c7_load_rslt1_m128i, 14);
	__m256i c7_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c7_alignr_rslt1_m128i);
	hor_avx2_unpack16_c7<0>(out, c7_broadcast_rslt2_m256i); // Unpack 2nd 16 values.

	__m128i c7_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c7_alignr_rslt2_m128i = _mm_alignr_epi8(c7_load_rslt3_m128i, c7_load_rslt2_m128i, 12);
	__m256i c7_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c7_alignr_rslt2_m128i);
	hor_avx2_unpack16_c7<0>(out, c7_broadcast_rslt3_m256i); // Unpack 3rd 16 values.

	__m128i c7_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c7_alignr_rslt3_m128i = _mm_alignr_epi8(c7_load_rslt4_m128i, c7_load_rslt3_m128i, 10);
	__m256i c7_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c7_alignr_rslt3_m128i);
	hor_avx2_unpack16_c7<0>(out, c7_broadcast_rslt4_m256i); // Unpack 4th 16 values.

	__m128i c7_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c7_alignr_rslt4_m128i = _mm_alignr_epi8(c7_load_rslt5_m128i, c7_load_rslt4_m128i, 8);
	__m256i c7_broadcast_rslt5_m256i = _mm256_broadcastsi128_si256(c7_alignr_rslt4_m128i);
	hor_avx2_unpack16_c7<0>(out, c7_broadcast_rslt5_m256i); // Unpack 5th 16 values.

	__m128i c7_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c7_alignr_rslt5_m128i = _mm_alignr_epi8(c7_load_rslt6_m128i, c7_load_rslt5_m128i, 6);
	__m256i c7_broadcast_rslt6_m256i = _mm256_broadcastsi128_si256(c7_alignr_rslt5_m128i);
	hor_avx2_unpack16_c7<0>(out, c7_broadcast_rslt6_m256i); // Unpack 6th 16 values.

	__m128i c7_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c7_alignr_rslt6_m128i = _mm_alignr_epi8(c7_load_rslt7_m128i, c7_load_rslt6_m128i, 4);
	__m256i c7_broadcast_rslt7_m256i = _mm256_broadcastsi128_si256(c7_alignr_rslt6_m128i);
	hor_avx2_unpack16_c7<0>(out, c7_broadcast_rslt7_m256i); // Unpack 7th 16 values.
	__m256i c7_broadcast_rslt8_m256i = _mm256_broadcastsi128_si256(c7_load_rslt7_m128i);
	hor_avx2_unpack16_c7<2>(out, c7_broadcast_rslt8_m256i); // Unpack 8th 16 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack16_c7(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	// We shuffle in a way to have a 7-bit codeword in every 16-bit word,
	// and order the codewords appropriately:
	// -v15-v7|-v14-v6|-v13-v5|-v12-v4|-v11-v3|-v10-v2|-v9-v1|-v8-v0
	const __m256i Hor_AVX2_c7_shfl_msk_m256i = _mm256_set_epi8(
			0xFF, byte + 13, 0xFF, byte + 6,
			byte + 13, byte + 12, byte + 6, byte + 5,
			byte + 12, byte + 11, byte + 5, byte + 4,
			byte + 11, byte + 10, byte + 4, byte + 3,
			byte + 10, byte + 9, byte + 3, byte + 2,
			byte + 9, byte + 8, byte + 2, byte + 1,
			byte + 8, byte + 7, byte + 1, byte + 0,
			0xFF, byte + 7, 0xFF, byte + 0);
	__m256i c7_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c7_shfl_msk_m256i);
	hor_avx2_unpack8_c7<0>(out, c7_shfl_rslt_m256i);  // Unpack 1st 8 values.
	hor_avx2_unpack8_c7<16>(out, c7_shfl_rslt_m256i); // Unpack 2nd 8 values.
}

template <bool IsRiceCoding>
template <int bit>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c7(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c7_srlv_msk_m256i = _mm256_set_epi32(
				bit + 1, bit + 2, bit + 3, bit + 4,
				bit + 5, bit + 6, bit + 7, bit + 0);
		__m256i c7_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c7_srlv_msk_m256i);
		__m256i c7_rslt_m256i = _mm256_and_si256(c7_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[7]);
		_mm256_storeu_si256(out++, c7_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c7_srlv_msk_m256i = _mm256_set_epi32(
				bit + 1, bit + 2, bit + 3, bit + 4,
				bit + 5, bit + 6, bit + 7, bit + 0);
		__m256i c7_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c7_srlv_msk_m256i);
		__m256i c7_and_rslt_m256i = _mm256_and_si256(c7_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[7]);
		__m256i c7_rslt_m256i = _mm256_or_si256(c7_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 7));
		_mm256_storeu_si256(out++, c7_rslt_m256i);
	}

}

///**
// * AVX2-based unpacking 128 7-bit values.
// * Load 7 SSE vectors, each containing 18 7-bit values. (19th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c7(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	__m128i c7_load_rslt1_m128i = _mm_loadu_si128(in + 0);
//	__m256i c7_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c7_load_rslt1_m128i);
//	hor_avx2_unpack8_c7<0>(out, c7_broadcast_rslt1_m256i); // Unpack 1st 8 values.
//	hor_avx2_unpack8_c7<7>(out, c7_broadcast_rslt1_m256i); // Unpack 2nd 8 values.
//
//	__m128i c7_load_rslt2_m128i = _mm_loadu_si128(in + 1);
//	__m128i c7_alignr_rslt1_m128i = _mm_alignr_epi8(c7_load_rslt2_m128i, c7_load_rslt1_m128i, 14);
//	__m256i c7_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c7_alignr_rslt1_m128i);
//	hor_avx2_unpack8_c7<0>(out, c7_broadcast_rslt2_m256i); // Unpack 3rd 8 values.
//	hor_avx2_unpack8_c7<7>(out, c7_broadcast_rslt2_m256i); // Unpack 4th 8 values.
//
//	__m128i c7_load_rslt3_m128i = _mm_loadu_si128(in + 2);
//	__m128i c7_alignr_rslt2_m128i = _mm_alignr_epi8(c7_load_rslt3_m128i, c7_load_rslt2_m128i, 12);
//	__m256i c7_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c7_alignr_rslt2_m128i);
//	hor_avx2_unpack8_c7<0>(out, c7_broadcast_rslt3_m256i); // Unpack 5th 8 values.
//	hor_avx2_unpack8_c7<7>(out, c7_broadcast_rslt3_m256i); // Unpack 6th 8 values.
//
//	__m128i c7_load_rslt4_m128i = _mm_loadu_si128(in + 3);
//	__m128i c7_alignr_rslt3_m128i = _mm_alignr_epi8(c7_load_rslt4_m128i, c7_load_rslt3_m128i, 10);
//	__m256i c7_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c7_alignr_rslt3_m128i);
//	hor_avx2_unpack8_c7<0>(out, c7_broadcast_rslt4_m256i); // Unpack 7th 8 values.
//	hor_avx2_unpack8_c7<7>(out, c7_broadcast_rslt4_m256i); // Unpack 8th 8 values.
//
//	__m128i c7_load_rslt5_m128i = _mm_loadu_si128(in + 4);
//	__m128i c7_alignr_rslt4_m128i = _mm_alignr_epi8(c7_load_rslt5_m128i, c7_load_rslt4_m128i, 8);
//	__m256i c7_broadcast_rslt5_m256i = _mm256_broadcastsi128_si256(c7_alignr_rslt4_m128i);
//	hor_avx2_unpack8_c7<0>(out, c7_broadcast_rslt5_m256i); // Unpack 9th 8 values.
//	hor_avx2_unpack8_c7<7>(out, c7_broadcast_rslt5_m256i); // Unpack 10th 8 values.
//
//	__m128i c7_load_rslt6_m128i = _mm_loadu_si128(in + 5);
//	__m128i c7_alignr_rslt5_m128i = _mm_alignr_epi8(c7_load_rslt6_m128i, c7_load_rslt5_m128i, 6);
//	__m256i c7_broadcast_rslt6_m256i = _mm256_broadcastsi128_si256(c7_alignr_rslt5_m128i);
//	hor_avx2_unpack8_c7<0>(out, c7_broadcast_rslt6_m256i); // Unpack 11th 8 values.
//	hor_avx2_unpack8_c7<7>(out, c7_broadcast_rslt6_m256i); // Unpack 12th 8 values.
//
//	__m128i c7_load_rslt7_m128i = _mm_loadu_si128(in + 6);
//	__m128i c7_alignr_rslt6_m128i = _mm_alignr_epi8(c7_load_rslt7_m128i, c7_load_rslt6_m128i, 4);
//	__m256i c7_broadcast_rslt7_m256i = _mm256_broadcastsi128_si256(c7_alignr_rslt6_m128i);
//	hor_avx2_unpack8_c7<0>(out, c7_broadcast_rslt7_m256i); // Unpack 13th 8 values.
//	hor_avx2_unpack8_c7<7>(out, c7_broadcast_rslt7_m256i); // Unpack 14th 8 values.
//	__m256i c7_broadcast_rslt8_m256i = _mm256_broadcastsi128_si256(c7_load_rslt7_m128i);
//	hor_avx2_unpack8_c7<2>(out, c7_broadcast_rslt8_m256i); // Unpack 15th 8 values.
//	hor_avx2_unpack8_c7<9>(out, c7_broadcast_rslt8_m256i); // Unpack 16th 8 values.
//}
//
//template <bool IsRiceCoding>
//template <int byte>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c7(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		const __m256i Hor_AVX2_c7_shfl_msk_m256i = _mm256_set_epi8(
//				0xFF, 0xFF, 0xFF, byte + 6,
//				0xFF, 0xFF, byte + 6, byte + 5,
//				0xFF, 0xFF, byte + 5, byte + 4,
//				0xFF, 0xFF, byte + 4, byte + 3,
//				0xFF, 0xFF, byte + 3, byte + 2,
//				0xFF, 0xFF, byte + 2, byte + 1,
//				0xFF, 0xFF, byte + 1, byte + 0,
//				0xFF, 0xFF, 0xFF, byte + 0);
//		__m256i c7_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c7_shfl_msk_m256i);
//		__m256i c7_srlv_rslt_m256i = _mm256_srlv_epi32(c7_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[7]);
//		__m256i c7_rslt_m256i = _mm256_and_si256(c7_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[7]);
//		_mm256_storeu_si256(out++, c7_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		const __m256i Hor_AVX2_c7_shfl_msk_m256i = _mm256_set_epi8(
//				0xFF, 0xFF, 0xFF, byte + 6,
//				0xFF, 0xFF, byte + 6, byte + 5,
//				0xFF, 0xFF, byte + 5, byte + 4,
//				0xFF, 0xFF, byte + 4, byte + 3,
//				0xFF, 0xFF, byte + 3, byte + 2,
//				0xFF, 0xFF, byte + 2, byte + 1,
//				0xFF, 0xFF, byte + 1, byte + 0,
//				0xFF, 0xFF, 0xFF, byte + 0);
//		__m256i c7_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c7_shfl_msk_m256i);
//		__m256i c7_srlv_rslt_m256i = _mm256_srlv_epi32(c7_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[7]);
//		__m256i c7_and_rslt_m256i = _mm256_and_si256(c7_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[7]);
//		__m256i c7_rslt_m256i = _mm256_or_si256(c7_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 7));
//		_mm256_storeu_si256(out++, c7_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 8-bit values.
 * Load 8 SSE vectors, each containing 16 8-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c8(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 16) {
		__m128i c8_load_rslt_m128i = _mm_loadu_si128(in++);
		__m256i c8_broadcast_rslt_m256i = _mm256_broadcastsi128_si256(c8_load_rslt_m128i);
		hor_avx2_unpack8_c8<0>(out, c8_broadcast_rslt_m256i); // Unpack 1st 8 values.
		hor_avx2_unpack8_c8<8>(out, c8_broadcast_rslt_m256i); // Unpack 2nd 8 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c8(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c8_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 7,
				0xFF, 0xFF, 0xFF, byte + 6,
				0xFF, 0xFF, 0xFF, byte + 5,
				0xFF, 0xFF, 0xFF, byte + 4,
				0xFF, 0xFF, 0xFF, byte + 3,
				0xFF, 0xFF, 0xFF, byte + 2,
				0xFF, 0xFF, 0xFF, byte + 1,
				0xFF, 0xFF, 0xFF, byte + 0);
		__m256i c8_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c8_shfl_msk_m256i);
		_mm256_storeu_si256(out++, c8_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c8_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, 0xFF, 0xFF, byte + 7,
				0xFF, 0xFF, 0xFF, byte + 6,
				0xFF, 0xFF, 0xFF, byte + 5,
				0xFF, 0xFF, 0xFF, byte + 4,
				0xFF, 0xFF, 0xFF, byte + 3,
				0xFF, 0xFF, 0xFF, byte + 2,
				0xFF, 0xFF, 0xFF, byte + 1,
				0xFF, 0xFF, 0xFF, byte + 0);
		__m256i c8_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c8_shfl_msk_m256i);
		__m256i c8_rslt_m256i = _mm256_or_si256(c8_shfl_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 8));
		_mm256_storeu_si256(out++, c8_rslt_m256i);
	}
}


/**
 * AVX2-based unpacking 128 9-bit values.
 * Load 9 SSE vectors, each containing 14 9-bit values. (15th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c9(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c9_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	__m128i c9_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c9_alignr_rslt1_m128i = _mm_alignr_epi8(c9_load_rslt2_m128i, c9_load_rslt1_m128i, 4);
	__m256i c9_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c9_load_rslt1_m128i), c9_alignr_rslt1_m128i, 1);
	hor_avx2_unpack16_c9<0, 0>(out, c9_insert_rslt1_m256i); // Unpack 1st 16 values.

	__m128i c9_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c9_alignr_rslt2_m128i = _mm_alignr_epi8(c9_load_rslt3_m128i, c9_load_rslt2_m128i, 6);
	__m256i c9_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c9_load_rslt2_m128i), c9_alignr_rslt2_m128i, 1);
	hor_avx2_unpack16_c9<2, 0>(out, c9_insert_rslt2_m256i); // Unpack 2nd 16 values.

	__m128i c9_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c9_alignr_rslt3_1_m128i = _mm_alignr_epi8(c9_load_rslt4_m128i, c9_load_rslt3_m128i, 4);
	__m128i c9_alignr_rslt3_2_m128i = _mm_alignr_epi8(c9_load_rslt4_m128i, c9_load_rslt3_m128i, 8);
	__m256i c9_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c9_alignr_rslt3_1_m128i), c9_alignr_rslt3_2_m128i, 1);
	hor_avx2_unpack16_c9<0, 0>(out, c9_insert_rslt3_m256i); // Unpack 3rd 16 values.

	__m128i c9_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c9_alignr_rslt4_1_m128i = _mm_alignr_epi8(c9_load_rslt5_m128i, c9_load_rslt4_m128i, 6);
	__m128i c9_alignr_rslt4_2_m128i = _mm_alignr_epi8(c9_load_rslt5_m128i, c9_load_rslt4_m128i, 10);
	__m256i c9_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c9_alignr_rslt4_1_m128i), c9_alignr_rslt4_2_m128i, 1);
	hor_avx2_unpack16_c9<0, 0>(out, c9_insert_rslt4_m256i); // Unpack 4th 16 values.

	__m128i c9_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c9_alignr_rslt5_1_m128i = _mm_alignr_epi8(c9_load_rslt6_m128i, c9_load_rslt5_m128i, 8);
	__m128i c9_alignr_rslt5_2_m128i = _mm_alignr_epi8(c9_load_rslt6_m128i, c9_load_rslt5_m128i, 12);
	__m256i c9_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c9_alignr_rslt5_1_m128i), c9_alignr_rslt5_2_m128i, 1);
	hor_avx2_unpack16_c9<0, 0>(out, c9_insert_rslt5_m256i); // Unpack 5th 16 values.

	__m128i c9_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c9_alignr_rslt6_1_m128i = _mm_alignr_epi8(c9_load_rslt7_m128i, c9_load_rslt6_m128i, 10);
	__m128i c9_alignr_rslt6_2_m128i = _mm_alignr_epi8(c9_load_rslt7_m128i, c9_load_rslt6_m128i, 14);
	__m256i c9_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c9_alignr_rslt6_1_m128i), c9_alignr_rslt6_2_m128i, 1);
	hor_avx2_unpack16_c9<0, 0>(out, c9_insert_rslt6_m256i); // Unpack 6th 16 values.

	__m128i c9_load_rslt8_m128i = _mm_loadu_si128(in + 7);
	__m128i c9_alignr_rslt7_m128i = _mm_alignr_epi8(c9_load_rslt8_m128i, c9_load_rslt7_m128i, 12);
	__m256i c9_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c9_alignr_rslt7_m128i), c9_load_rslt8_m128i, 1);
	hor_avx2_unpack16_c9<0, 0>(out, c9_insert_rslt7_m256i); // Unpack 7th 16 values.

	__m128i c9_load_rslt9_m128i = _mm_loadu_si128(in + 8);
	__m128i c9_alignr_rslt8_m128i = _mm_alignr_epi8(c9_load_rslt9_m128i, c9_load_rslt8_m128i, 14);
	__m256i c9_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c9_alignr_rslt8_m128i), c9_load_rslt9_m128i, 1);
	hor_avx2_unpack16_c9<0, 2>(out, c9_insert_rslt8_m256i); // Unpack 8th 16 values.
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack16_c9(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	// We shuffle in a way to have a 9-bit codeword in every 16-bit word,
	// and order the codewords appropriately:
	// -v15-v7|-v14-v6|-v13-v5|-v12-v4|-v11-v3|-v10-v2|-v9-v1|-v8-v0
	const __m256i Hor_AVX2_c9_shfl_msk_m256i = _mm256_set_epi8(
			byte2 + 13, byte2 + 12, byte2 + 4, byte2 + 3,
			byte2 + 12, byte2 + 11, byte2 + 3, byte2 + 2,
			byte2 + 11, byte2 + 10, byte2 + 2, byte2 + 1,
			byte2 + 10, byte2 + 9, byte2 + 1, byte2 + 0,
			byte1 + 13, byte1 + 12, byte1 + 4, byte1 + 3,
			byte1 + 12, byte1 + 11, byte1 + 3, byte1 + 2,
			byte1 + 11, byte1 + 10, byte1 + 2, byte1 + 1,
			byte1 + 10, byte1 + 9, byte1 + 1, byte1 + 0);
	__m256i c9_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c9_shfl_msk_m256i);
	hor_avx2_unpack8_c9<0>(out, c9_shfl_rslt_m256i);  // Unpack 1st 8 values.
	hor_avx2_unpack8_c9<16>(out, c9_shfl_rslt_m256i); // Unpack 2nd 8 values.
}

template <bool IsRiceCoding>
template <int bit>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c9(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c9_srlv_msk_m256i = _mm256_set_epi32(
				bit + 7, bit + 6, bit + 5, bit + 4,
				bit + 3, bit + 2, bit + 1, bit + 0);
		__m256i c9_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c9_srlv_msk_m256i);
		__m256i c9_rslt_m256i = _mm256_and_si256(c9_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[9]);
		_mm256_storeu_si256(out++, c9_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c9_srlv_msk_m256i = _mm256_set_epi32(
				bit + 7, bit + 6, bit + 5, bit + 4,
				bit + 3, bit + 2, bit + 1, bit + 0);
		__m256i c9_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c9_srlv_msk_m256i);
		__m256i c9_and_rslt_m256i = _mm256_and_si256(c9_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[9]);
		__m256i c9_rslt_m256i = _mm256_or_si256(c9_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 9));
		_mm256_storeu_si256(out++, c9_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 9-bit values.
// * Load 9 SSE vectors, each containing 14 9-bit values. (15th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c9(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	__m128i c9_load_rslt1_m128i = _mm_loadu_si128(in + 0);
//	__m256i c9_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c9_load_rslt1_m128i);
//	hor_avx2_unpack8_c9<0>(out, c9_broadcast_rslt1_m256i);  // Unpack 1st 8 values.
//
//	__m128i c9_load_rslt2_m128i = _mm_loadu_si128(in + 1);
//	__m128i c9_alignr_rslt1_m128i = _mm_alignr_epi8(c9_load_rslt2_m128i, c9_load_rslt1_m128i, 9);
//	__m256i c9_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c9_alignr_rslt1_m128i);
//	hor_avx2_unpack8_c9<0>(out, c9_broadcast_rslt2_m256i);  // Unpack 2nd 8 values.
//	__m256i c9_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c9_load_rslt2_m128i);
//	hor_avx2_unpack8_c9<2>(out, c9_broadcast_rslt3_m256i);  // Unpack 3rd 8 values.
//
//	__m128i c9_load_rslt3_m128i = _mm_loadu_si128(in + 2);
//	__m128i c9_alignr_rslt2_m128i = _mm_alignr_epi8(c9_load_rslt3_m128i, c9_load_rslt2_m128i, 11);
//	__m256i c9_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c9_alignr_rslt2_m128i);
//	hor_avx2_unpack8_c9<0>(out, c9_broadcast_rslt4_m256i);  // Unpack 4th 8 values.
//	__m256i c9_broadcast_rslt5_m256i = _mm256_broadcastsi128_si256(c9_load_rslt3_m128i);
//	hor_avx2_unpack8_c9<4>(out, c9_broadcast_rslt5_m256i);  // Unpack 5th 8 values.
//
//	__m128i c9_load_rslt4_m128i = _mm_loadu_si128(in + 3);
//	__m128i c9_alignr_rslt3_m128i = _mm_alignr_epi8(c9_load_rslt4_m128i, c9_load_rslt3_m128i, 13);
//	__m256i c9_broadcast_rslt6_m256i = _mm256_broadcastsi128_si256(c9_alignr_rslt3_m128i);
//	hor_avx2_unpack8_c9<0>(out, c9_broadcast_rslt6_m256i);  // Unpack 6th 8 values.
//	__m256i c9_broadcast_rslt7_m256i = _mm256_broadcastsi128_si256(c9_load_rslt4_m128i);
//	hor_avx2_unpack8_c9<6>(out, c9_broadcast_rslt7_m256i);  // Unpack 7th 8 values.
//
//	__m128i c9_load_rslt5_m128i = _mm_loadu_si128(in + 4);
//	__m128i c9_alignr_rslt4_m128i = _mm_alignr_epi8(c9_load_rslt5_m128i, c9_load_rslt4_m128i, 15);
//	__m256i c9_broadcast_rslt8_m256i = _mm256_broadcastsi128_si256(c9_alignr_rslt4_m128i);
//	hor_avx2_unpack8_c9<0>(out, c9_broadcast_rslt8_m256i);  // Unpack 8th 8 values.
//
//	__m128i c9_load_rslt6_m128i = _mm_loadu_si128(in + 5);
//	__m128i c9_alignr_rslt5_m128i = _mm_alignr_epi8(c9_load_rslt6_m128i, c9_load_rslt5_m128i, 8);
//	__m256i c9_broadcast_rslt9_m256i = _mm256_broadcastsi128_si256(c9_alignr_rslt5_m128i);
//	hor_avx2_unpack8_c9<0>(out, c9_broadcast_rslt9_m256i);  // Unpack 9th 8 values.
//	__m256i c9_broadcast_rslt10_m256i = _mm256_broadcastsi128_si256(c9_load_rslt6_m128i);
//	hor_avx2_unpack8_c9<1>(out, c9_broadcast_rslt10_m256i); // Unpack 10th 8 values.
//
//	__m128i c9_load_rslt7_m128i = _mm_loadu_si128(in + 6);
//	__m128i c9_alignr_rslt6_m128i = _mm_alignr_epi8(c9_load_rslt7_m128i, c9_load_rslt6_m128i, 10);
//	__m256i c9_broadcast_rslt11_m256i = _mm256_broadcastsi128_si256(c9_alignr_rslt6_m128i);
//	hor_avx2_unpack8_c9<0>(out, c9_broadcast_rslt11_m256i); // Unpack 11th 8 values.
//	__m256i c9_broadcast_rslt12_m256i = _mm256_broadcastsi128_si256(c9_load_rslt7_m128i);
//	hor_avx2_unpack8_c9<3>(out, c9_broadcast_rslt12_m256i); // Unpack 12th 8 values.
//
//	__m128i c9_load_rslt8_m128i = _mm_loadu_si128(in + 7);
//	__m128i c9_alignr_rslt7_m128i = _mm_alignr_epi8(c9_load_rslt8_m128i, c9_load_rslt7_m128i, 12);
//	__m256i c9_broadcast_rslt13_m256i = _mm256_broadcastsi128_si256(c9_alignr_rslt7_m128i);
//	hor_avx2_unpack8_c9<0>(out, c9_broadcast_rslt13_m256i); // Unpack 13th 8 values.
//	__m256i c9_broadcast_rslt14_m256i = _mm256_broadcastsi128_si256(c9_load_rslt8_m128i);
//	hor_avx2_unpack8_c9<5>(out, c9_broadcast_rslt14_m256i); // Unpack 14th 8 values.
//
//	__m128i c9_load_rslt9_m128i = _mm_loadu_si128(in + 8);
//	__m128i c9_alignr_rslt8_m128i = _mm_alignr_epi8(c9_load_rslt9_m128i, c9_load_rslt8_m128i, 14);
//	__m256i c9_broadcast_rslt15_m256i = _mm256_broadcastsi128_si256(c9_alignr_rslt8_m128i);
//	hor_avx2_unpack8_c9<0>(out, c9_broadcast_rslt15_m256i); // Unpack 15th 8 values.
//	__m256i c9_broadcast_rslt16_m256i = _mm256_broadcastsi128_si256(c9_load_rslt9_m128i);
//	hor_avx2_unpack8_c9<7>(out, c9_broadcast_rslt16_m256i); // Unpack 16th 8 values.
//}
//
//template <bool IsRiceCoding>
//template <int byte>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c9(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		const __m256i Hor_AVX2_c9_shfl_msk_m256i = _mm256_set_epi8(
//				0xFF, 0xFF, byte + 8, byte + 7,
//				0xFF, 0xFF, byte + 7, byte + 6,
//				0xFF, 0xFF, byte + 6, byte + 5,
//				0xFF, 0xFF, byte + 5, byte + 4,
//				0xFF, 0xFF, byte + 4, byte + 3,
//				0xFF, 0xFF, byte + 3, byte + 2,
//				0xFF, 0xFF, byte + 2, byte + 1,
//				0xFF, 0xFF, byte + 1, byte + 0);
//		__m256i c9_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c9_shfl_msk_m256i);
//		__m256i c9_srlv_rslt_m256i = _mm256_srlv_epi32(c9_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[9]);
//		__m256i c9_rslt_m256i = _mm256_and_si256(c9_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[9]);
//		_mm256_storeu_si256(out++, c9_rslt_m256i);
//	}
//	else  { // For Rice and OptRice.
//		const __m256i Hor_AVX2_c9_shfl_msk_m256i = _mm256_set_epi8(
//				0xFF, 0xFF, byte + 8, byte + 7,
//				0xFF, 0xFF, byte + 7, byte + 6,
//				0xFF, 0xFF, byte + 6, byte + 5,
//				0xFF, 0xFF, byte + 5, byte + 4,
//				0xFF, 0xFF, byte + 4, byte + 3,
//				0xFF, 0xFF, byte + 3, byte + 2,
//				0xFF, 0xFF, byte + 2, byte + 1,
//				0xFF, 0xFF, byte + 1, byte + 0);
//		__m256i c9_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c9_shfl_msk_m256i);
//		__m256i c9_srlv_rslt_m256i = _mm256_srlv_epi32(c9_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[9]);
//		__m256i c9_and_rslt_m256i = _mm256_and_si256(c9_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[9]);
//		__m256i c9_rslt_m256i = _mm256_or_si256(c9_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 9));
//		_mm256_storeu_si256(out++, c9_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 10-bit values.
 * Load 10 SSE vectors, each containing 12 10-bit values. (13th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c10(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
		__m128i c10_load_rslt1_m128i = _mm_loadu_si128(in++);
		__m128i c10_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c10_alignr_rslt1_m128i = _mm_alignr_epi8(c10_load_rslt2_m128i, c10_load_rslt1_m128i, 5);
		__m256i c10_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c10_load_rslt1_m128i), c10_alignr_rslt1_m128i, 1);
		hor_avx2_unpack16_c10<0, 0>(out, c10_insert_rslt1_m256i); // Unpack 1st 16 values.

		__m128i c10_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c10_alignr_rslt2_1_m128i = _mm_alignr_epi8(c10_load_rslt3_m128i, c10_load_rslt2_m128i, 4);
		__m128i c10_alignr_rslt2_2_m128i = _mm_alignr_epi8(c10_load_rslt3_m128i, c10_load_rslt2_m128i, 9);
		__m256i c10_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c10_alignr_rslt2_1_m128i), c10_alignr_rslt2_2_m128i, 1);
		hor_avx2_unpack16_c10<0, 0>(out, c10_insert_rslt2_m256i); // Unpack 2nd 16 values.

		__m128i c10_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c10_alignr_rslt3_1_m128i = _mm_alignr_epi8(c10_load_rslt4_m128i, c10_load_rslt3_m128i, 8);
		__m128i c10_alignr_rslt3_2_m128i = _mm_alignr_epi8(c10_load_rslt4_m128i, c10_load_rslt3_m128i, 13);
		__m256i c10_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c10_alignr_rslt3_1_m128i), c10_alignr_rslt3_2_m128i, 1);
		hor_avx2_unpack16_c10<0, 0>(out, c10_insert_rslt3_m256i); // Unpack 3rd 16 values.

		__m128i c10_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c10_alignr_rslt4_m128i = _mm_alignr_epi8(c10_load_rslt5_m128i, c10_load_rslt4_m128i, 12);
		__m256i c10_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c10_alignr_rslt4_m128i), c10_load_rslt5_m128i, 1);
		hor_avx2_unpack16_c10<0, 1>(out, c10_insert_rslt4_m256i); // Unpack 4th 16 values.
	}
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack16_c10(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	// We shuffle in a way to have a 10-bit codeword in every 16-bit word,
	// and order the codewords appropriately:
	// -v15-v7|-v14-v6|-v13-v5|-v12-v4|-v11-v3|-v10-v2|-v9-v1|-v8-v0
	const __m256i Hor_AVX2_c10_shfl_msk_m256i = _mm256_set_epi8(
			byte2 + 14, byte2 + 13, byte2 + 4, byte2 + 3,
			byte2 + 13, byte2 + 12, byte2 + 3, byte2 + 2,
			byte2 + 12, byte2 + 11, byte2 + 2, byte2 + 1,
			byte2 + 11, byte2 + 10, byte2 + 1, byte2 + 0,
			byte1 + 14, byte1 + 13, byte1 + 4, byte1 + 3,
			byte1 + 13, byte1 + 12, byte1 + 3, byte1 + 2,
			byte1 + 12, byte1 + 11, byte1 + 2, byte1 + 1,
			byte1 + 11, byte1 + 10, byte1 + 1, byte1 + 0);
	__m256i c10_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c10_shfl_msk_m256i);
	hor_avx2_unpack8_c10<0>(out, c10_shfl_rslt_m256i);  // Unpack 1st 8 values.
	hor_avx2_unpack8_c10<16>(out, c10_shfl_rslt_m256i); // Unpack 2nd 8 values.
}

template <bool IsRiceCoding>
template <int bit>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c10(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c10_srlv_msk_m256i = _mm256_set_epi32(
				bit + 6, bit + 4, bit + 2, bit + 0,
				bit + 6, bit + 4, bit + 2, bit + 0);
		__m256i c10_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c10_srlv_msk_m256i);
		__m256i c10_rslt_m256i = _mm256_and_si256(c10_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[10]);
		_mm256_storeu_si256(out++, c10_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c10_srlv_msk_m256i = _mm256_set_epi32(
				bit + 6, bit + 4, bit + 2, bit + 0,
				bit + 6, bit + 4, bit + 2, bit + 0);
		__m256i c10_srlv_rslt_m256i = _mm256_srlv_epi32(InReg, Hor_AVX2_c10_srlv_msk_m256i);
		__m256i c10_and_rslt_m256i = _mm256_and_si256(c10_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[10]);
		__m256i c10_rslt_m256i = _mm256_or_si256(c10_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 10));
		_mm256_storeu_si256(out++, c10_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 10-bit values.
// * Load 10 SSE vectors, each containing 12 10-bit values. (13th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c10(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
//		__m128i c10_load_rslt1_m128i = _mm_loadu_si128(in++);
//		__m256i c10_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c10_load_rslt1_m128i);
//		hor_avx2_unpack8_c10<0>(out, c10_broadcast_rslt1_m256i); // Unpack 1st 8 values.
//
//		__m128i c10_load_rslt2_m128i = _mm_loadu_si128(in++);
//		__m128i c10_alignr_rslt1_m128i = _mm_alignr_epi8(c10_load_rslt2_m128i, c10_load_rslt1_m128i, 10);
//		__m256i c10_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c10_alignr_rslt1_m128i);
//		hor_avx2_unpack8_c10<0>(out, c10_broadcast_rslt2_m256i); // Unpack 2nd 8 values.
//		__m256i c10_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c10_load_rslt2_m128i);
//		hor_avx2_unpack8_c10<4>(out, c10_broadcast_rslt3_m256i); // Unpack 3rd 8 values.
//
//		__m128i c10_load_rslt3_m128i = _mm_loadu_si128(in++);
//		__m128i c10_alignr_rslt2_m128i = _mm_alignr_epi8(c10_load_rslt3_m128i, c10_load_rslt2_m128i, 14);
//		__m256i c10_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c10_alignr_rslt2_m128i);
//		hor_avx2_unpack8_c10<0>(out, c10_broadcast_rslt4_m256i); // Unpack 4th 8 values.
//
//		__m128i c10_load_rslt4_m128i = _mm_loadu_si128(in++);
//		__m128i c10_alignr_rslt3_m128i = _mm_alignr_epi8(c10_load_rslt4_m128i, c10_load_rslt3_m128i, 8);
//		__m256i c10_broadcast_rslt5_m256i = _mm256_broadcastsi128_si256(c10_alignr_rslt3_m128i);
//		hor_avx2_unpack8_c10<0>(out, c10_broadcast_rslt5_m256i); // Unpack 5th 8 values.
//		__m256i c10_broadcast_rslt6_m256i = _mm256_broadcastsi128_si256(c10_load_rslt4_m128i);
//		hor_avx2_unpack8_c10<2>(out, c10_broadcast_rslt6_m256i); // Unpack 6th 8 values.
//
//		__m128i c10_load_rslt5_m128i = _mm_loadu_si128(in++);
//		__m128i c10_alignr_rslt4_m128i = _mm_alignr_epi8(c10_load_rslt5_m128i, c10_load_rslt4_m128i, 12);
//		__m256i c10_broadcast_rslt7_m256i = _mm256_broadcastsi128_si256(c10_alignr_rslt4_m128i);
//		hor_avx2_unpack8_c10<0>(out, c10_broadcast_rslt7_m256i); // Unpack 7th 8 values.
//		__m256i c10_broadcast_rslt8_m256i = _mm256_broadcastsi128_si256(c10_load_rslt5_m128i);
//		hor_avx2_unpack8_c10<6>(out, c10_broadcast_rslt8_m256i); // Unpack 8th 8 values.
//	}
//}
//
//template <bool IsRiceCoding>
//template <int byte>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c10(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		const __m256i Hor_AVX2_c10_shfl_msk_m256i = _mm256_set_epi8(
//				0xFF, 0xFF, byte + 9, byte + 8,
//				0xFF, 0xFF, byte + 8, byte + 7,
//				0xFF, 0xFF, byte + 7, byte + 6,
//				0xFF, 0xFF, byte + 6, byte + 5,
//				0xFF, 0xFF, byte + 4, byte + 3,
//				0xFF, 0xFF, byte + 3, byte + 2,
//				0xFF, 0xFF, byte + 2, byte + 1,
//				0xFF, 0xFF, byte + 1, byte + 0);
//		__m256i c10_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c10_shfl_msk_m256i);
//		__m256i c10_srlv_rslt_m256i = _mm256_srlv_epi32(c10_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[10]);
//		__m256i c10_rslt_m256i = _mm256_and_si256(c10_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[10]);
//		_mm256_storeu_si256(out++, c10_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		const __m256i Hor_AVX2_c10_shfl_msk_m256i = _mm256_set_epi8(
//				0xFF, 0xFF, byte + 9, byte + 8,
//				0xFF, 0xFF, byte + 8, byte + 7,
//				0xFF, 0xFF, byte + 7, byte + 6,
//				0xFF, 0xFF, byte + 6, byte + 5,
//				0xFF, 0xFF, byte + 4, byte + 3,
//				0xFF, 0xFF, byte + 3, byte + 2,
//				0xFF, 0xFF, byte + 2, byte + 1,
//				0xFF, 0xFF, byte + 1, byte + 0);
//		__m256i c10_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c10_shfl_msk_m256i);
//		__m256i c10_srlv_rslt_m256i = _mm256_srlv_epi32(c10_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[10]);
//		__m256i c10_and_rslt_m256i = _mm256_and_si256(c10_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[10]);
//		__m256i c10_rslt_m256i = _mm256_or_si256(c10_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 10));
//		_mm256_storeu_si256(out++, c10_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 11-bit values.
 * Load 11 SSE vectors, each containing 11 11-bit values. (12th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c11(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c11_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	__m256i c11_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c11_load_rslt1_m128i);
	hor_avx2_unpack8_c11<0>(out, c11_broadcast_rslt1_m256i);  // Unpack 1st 8 values.

	__m128i c11_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c11_alignr_rslt1_m128i = _mm_alignr_epi8(c11_load_rslt2_m128i, c11_load_rslt1_m128i, 11);
	__m256i c11_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c11_alignr_rslt1_m128i);
	hor_avx2_unpack8_c11<0>(out, c11_broadcast_rslt2_m256i);  // Unpack 2nd 8 values.

	__m128i c11_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c11_alignr_rslt2_m128i = _mm_alignr_epi8(c11_load_rslt3_m128i, c11_load_rslt2_m128i, 6);
	__m256i c11_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c11_alignr_rslt2_m128i);
	hor_avx2_unpack8_c11<0>(out, c11_broadcast_rslt3_m256i);  // Unpack 3rd 8 values.
	__m256i c11_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c11_load_rslt3_m128i);
	hor_avx2_unpack8_c11<1>(out, c11_broadcast_rslt4_m256i);  // Unpack 4th 8 values.

	__m128i c11_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c11_alignr_rslt3_m128i = _mm_alignr_epi8(c11_load_rslt4_m128i, c11_load_rslt3_m128i, 12);
	__m256i c11_broadcast_rslt5_m256i = _mm256_broadcastsi128_si256(c11_alignr_rslt3_m128i);
	hor_avx2_unpack8_c11<0>(out, c11_broadcast_rslt5_m256i);  // Unpack 5th 8 values.

	__m128i c11_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c11_alignr_rslt4_m128i = _mm_alignr_epi8(c11_load_rslt5_m128i, c11_load_rslt4_m128i, 7);
	__m256i c11_broadcast_rslt6_m256i = _mm256_broadcastsi128_si256(c11_alignr_rslt4_m128i);
	hor_avx2_unpack8_c11<0>(out, c11_broadcast_rslt6_m256i);  // Unpack 6th 8 values.
	__m256i c11_broadcast_rslt7_m256i = _mm256_broadcastsi128_si256(c11_load_rslt5_m128i);
	hor_avx2_unpack8_c11<2>(out, c11_broadcast_rslt7_m256i);  // Unpack 7th 8 values.

	__m128i c11_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c11_alignr_rslt5_m128i = _mm_alignr_epi8(c11_load_rslt6_m128i, c11_load_rslt5_m128i, 13);
	__m256i c11_broadcast_rslt8_m256i = _mm256_broadcastsi128_si256(c11_alignr_rslt5_m128i);
	hor_avx2_unpack8_c11<0>(out, c11_broadcast_rslt8_m256i);  // Unpack 8th 8 values.

	__m128i c11_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c11_alignr_rslt6_m128i = _mm_alignr_epi8(c11_load_rslt7_m128i, c11_load_rslt6_m128i, 8);
	__m256i c11_broadcast_rslt9_m256i = _mm256_broadcastsi128_si256(c11_alignr_rslt6_m128i);
	hor_avx2_unpack8_c11<0>(out, c11_broadcast_rslt9_m256i);  // Unpack 9th 8 values.
	__m256i c11_broadcast_rslt10_m256i = _mm256_broadcastsi128_si256(c11_load_rslt7_m128i);
	hor_avx2_unpack8_c11<3>(out, c11_broadcast_rslt10_m256i); // Unpack 10th 8 values.

	__m128i c11_load_rslt8_m128i = _mm_loadu_si128(in + 7);
	__m128i c11_alignr_rslt7_m128i = _mm_alignr_epi8(c11_load_rslt8_m128i, c11_load_rslt7_m128i, 14);
	__m256i c11_broadcast_rslt11_m256i = _mm256_broadcastsi128_si256(c11_alignr_rslt7_m128i);
	hor_avx2_unpack8_c11<0>(out, c11_broadcast_rslt11_m256i); // Unpack 11th 8 values.

	__m128i c11_load_rslt9_m128i = _mm_loadu_si128(in + 8);
	__m128i c11_alignr_rslt8_m128i = _mm_alignr_epi8(c11_load_rslt9_m128i, c11_load_rslt8_m128i, 9);
	__m256i c11_broadcast_rslt12_m256i = _mm256_broadcastsi128_si256(c11_alignr_rslt8_m128i);
	hor_avx2_unpack8_c11<0>(out, c11_broadcast_rslt12_m256i); // Unpack 12th 8 values.
	__m256i c11_broadcast_rslt13_m256i = _mm256_broadcastsi128_si256(c11_load_rslt9_m128i);
	hor_avx2_unpack8_c11<4>(out, c11_broadcast_rslt13_m256i); // Unpack 13th 8 values.

	__m128i c11_load_rslt10_m128i = _mm_loadu_si128(in + 9);
	__m128i c11_alignr_rslt9_m128i = _mm_alignr_epi8(c11_load_rslt10_m128i, c11_load_rslt9_m128i, 15);
	__m256i c11_broadcast_rslt14_m256i = _mm256_broadcastsi128_si256(c11_alignr_rslt9_m128i);
	hor_avx2_unpack8_c11<0>(out, c11_broadcast_rslt14_m256i); // Unpack 14th 8 values.

	__m128i c11_load_rslt11_m128i = _mm_loadu_si128(in + 10);
	__m128i c11_alignr_rslt10_m128i = _mm_alignr_epi8(c11_load_rslt11_m128i, c11_load_rslt10_m128i, 10);
	__m256i c11_broadcast_rslt15_m256i = _mm256_broadcastsi128_si256(c11_alignr_rslt10_m128i);
	hor_avx2_unpack8_c11<0>(out, c11_broadcast_rslt15_m256i); // Unpack 15th 8 values.
	__m256i c11_broadcast_rslt16_m256i = _mm256_broadcastsi128_si256(c11_load_rslt11_m128i);
	hor_avx2_unpack8_c11<5>(out, c11_broadcast_rslt16_m256i); // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c11(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c11_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, 0xFF, byte + 10, byte + 9,
				0xFF, 0xFF, byte + 9, byte + 8,
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, 0xFF, byte + 6, byte + 5,
				0xFF, 0xFF, byte + 5, byte + 4,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m256i c11_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c11_shfl_msk_m256i);
		__m256i c11_srlv_rslt_m256i = _mm256_srlv_epi32(c11_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[11]);
		__m256i c11_rslt_m256i = _mm256_and_si256(c11_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[11]);
		_mm256_storeu_si256(out++, c11_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c11_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, 0xFF, byte + 10, byte + 9,
				0xFF, 0xFF, byte + 9, byte + 8,
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, 0xFF, byte + 6, byte + 5,
				0xFF, 0xFF, byte + 5, byte + 4,
				0xFF, byte + 4, byte + 3, byte + 2,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m256i c11_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c11_shfl_msk_m256i);
		__m256i c11_srlv_rslt_m256i = _mm256_srlv_epi32(c11_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[11]);
		__m256i c11_and_rslt_m256i = _mm256_and_si256(c11_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[11]);
		__m256i c11_rslt_m256i = _mm256_or_si256(c11_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 11));
		_mm256_storeu_si256(out++, c11_rslt_m256i);
	}
}


/**
 * AVX2-based unpacking 128 12-bit values.
 * Load 12 SSE vectors, each containing 10 12-bit values. (11th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c12(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
		__m128i c12_load_rslt1_m128i = _mm_loadu_si128(in++);
		__m256i c12_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c12_load_rslt1_m128i);
		hor_avx2_unpack8_c12<0>(out, c12_broadcast_rslt1_m256i); // Unpack 1st 8 values.

		__m128i c12_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c12_alignr_rslt1_m128i = _mm_alignr_epi8(c12_load_rslt2_m128i, c12_load_rslt1_m128i, 12);
		__m256i c12_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c12_alignr_rslt1_m128i);
		hor_avx2_unpack8_c12<0>(out, c12_broadcast_rslt2_m256i); // Unpack 2nd 8 values.

		__m128i c12_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c12_alignr_rslt2_m128i = _mm_alignr_epi8(c12_load_rslt3_m128i, c12_load_rslt2_m128i, 8);
		__m256i c12_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c12_alignr_rslt2_m128i);
		hor_avx2_unpack8_c12<0>(out, c12_broadcast_rslt3_m256i); // Unpack 3rd 8 values.
		__m256i c12_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c12_load_rslt3_m128i);
		hor_avx2_unpack8_c12<4>(out, c12_broadcast_rslt4_m256i); // Unpack 4th 8 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c12(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c12_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, 0xFF, byte + 11, byte + 10,
				0xFF, 0xFF, byte + 10, byte + 9,
				0xFF, 0xFF, byte + 8, byte + 7,
				0xFF, 0xFF, byte + 7, byte + 6,
				0xFF, 0xFF, byte + 5, byte + 4,
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m256i c12_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c12_shfl_msk_m256i);
		__m256i c12_srlv_rslt_m256i = _mm256_srlv_epi32(c12_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[12]);
		__m256i c12_rslt_m256i = _mm256_and_si256(c12_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[12]);
		_mm256_storeu_si256(out++, c12_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c12_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, 0xFF, byte + 11, byte + 10,
				0xFF, 0xFF, byte + 10, byte + 9,
				0xFF, 0xFF, byte + 8, byte + 7,
				0xFF, 0xFF, byte + 7, byte + 6,
				0xFF, 0xFF, byte + 5, byte + 4,
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, 0xFF, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m256i c12_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c12_shfl_msk_m256i);
		__m256i c12_srlv_rslt_m256i = _mm256_srlv_epi32(c12_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[12]);
		__m256i c12_and_rslt_m256i = _mm256_and_si256(c12_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[12]);
		__m256i c12_rslt_m256i = _mm256_or_si256(c12_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 12));
		_mm256_storeu_si256(out++, c12_rslt_m256i);
	}
}


/**
 * AVX2-based unpacking 128 13-bit values.
 * Load 13 SSE vectors, each containing 9 13-bit values. (10th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c13(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c13_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	__m256i c13_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c13_load_rslt1_m128i);
	hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt1_m256i); // Unpack 1st 8 values.

	__m128i c13_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c13_alignr_rslt1_m128i = _mm_alignr_epi8(c13_load_rslt2_m128i, c13_load_rslt1_m128i, 13);
	__m256i c13_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c13_alignr_rslt1_m128i);
	hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt2_m256i); // Unpack 2nd 8 values.

	__m128i c13_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c13_alignr_rslt2_m128i = _mm_alignr_epi8(c13_load_rslt3_m128i, c13_load_rslt2_m128i, 10);
	__m256i c13_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c13_alignr_rslt2_m128i);
	hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt3_m256i); // Unpack 3rd 8 values.

    __m128i c13_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c13_alignr_rslt3_m128i = _mm_alignr_epi8(c13_load_rslt4_m128i, c13_load_rslt3_m128i, 7);
	__m256i c13_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c13_alignr_rslt3_m128i);
    hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt4_m256i); // Unpack 4th 8 values.

    __m128i c13_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c13_alignr_rslt4_m128i = _mm_alignr_epi8(c13_load_rslt5_m128i, c13_load_rslt4_m128i, 4);
	__m256i c13_broadcast_rslt5_m256i = _mm256_broadcastsi128_si256(c13_alignr_rslt4_m128i);
    hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt5_m256i); // Unpack 5th 8 values.
	__m256i c13_broadcast_rslt6_m256i = _mm256_broadcastsi128_si256(c13_load_rslt5_m128i);
    hor_avx2_unpack8_c13<1>(out, c13_broadcast_rslt6_m256i); // Unpack 6th 8 values.

    __m128i c13_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c13_alignr_rslt5_m128i = _mm_alignr_epi8(c13_load_rslt6_m128i, c13_load_rslt5_m128i, 14);
	__m256i c13_broadcast_rslt7_m256i = _mm256_broadcastsi128_si256(c13_alignr_rslt5_m128i);
    hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt7_m256i); // Unpack 7th 8 values.

    __m128i c13_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c13_alignr_rslt6_m128i = _mm_alignr_epi8(c13_load_rslt7_m128i, c13_load_rslt6_m128i, 11);
	__m256i c13_broadcast_rslt8_m256i = _mm256_broadcastsi128_si256(c13_alignr_rslt6_m128i);
    hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt8_m256i); // Unpack 8th 8 values.

    __m128i c13_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c13_alignr_rslt7_m128i = _mm_alignr_epi8(c13_load_rslt8_m128i, c13_load_rslt7_m128i, 8);
	__m256i c13_broadcast_rslt9_m256i = _mm256_broadcastsi128_si256(c13_alignr_rslt7_m128i);
    hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt9_m256i); // Unpack 9th 8 values.

    __m128i c13_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c13_alignr_rslt8_m128i = _mm_alignr_epi8(c13_load_rslt9_m128i, c13_load_rslt8_m128i, 5);
	__m256i c13_broadcast_rslt10_m256i = _mm256_broadcastsi128_si256(c13_alignr_rslt8_m128i);
    hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt10_m256i); // Unpack 10th 8 values.
	__m256i c13_broadcast_rslt11_m256i = _mm256_broadcastsi128_si256(c13_load_rslt9_m128i);
    hor_avx2_unpack8_c13<2>(out, c13_broadcast_rslt11_m256i); // Unpack 11th 8 values.

    __m128i c13_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c13_alignr_rslt9_m128i = _mm_alignr_epi8(c13_load_rslt10_m128i, c13_load_rslt9_m128i, 15);
	__m256i c13_broadcast_rslt12_m256i = _mm256_broadcastsi128_si256(c13_alignr_rslt9_m128i);
    hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt12_m256i); // Unpack 12th 8 values.

    __m128i c13_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c13_alignr_rslt10_m128i = _mm_alignr_epi8(c13_load_rslt11_m128i, c13_load_rslt10_m128i, 12);
	__m256i c13_broadcast_rslt13_m256i = _mm256_broadcastsi128_si256(c13_alignr_rslt10_m128i);
    hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt13_m256i); // Unpack 13th 8 values.

    __m128i c13_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c13_alignr_rslt11_m128i = _mm_alignr_epi8(c13_load_rslt12_m128i, c13_load_rslt11_m128i, 9);
	__m256i c13_broadcast_rslt14_m256i = _mm256_broadcastsi128_si256(c13_alignr_rslt11_m128i);
    hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt14_m256i); // Unpack 14th 8 values.

    __m128i c13_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c13_alignr_rslt12_m128i = _mm_alignr_epi8(c13_load_rslt13_m128i, c13_load_rslt12_m128i, 6);
	__m256i c13_broadcast_rslt15_m256i = _mm256_broadcastsi128_si256(c13_alignr_rslt12_m128i);
    hor_avx2_unpack8_c13<0>(out, c13_broadcast_rslt15_m256i); // Unpack 15th 8 values.
	__m256i c13_broadcast_rslt16_m256i = _mm256_broadcastsi128_si256(c13_load_rslt13_m128i);
    hor_avx2_unpack8_c13<3>(out, c13_broadcast_rslt16_m256i); // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c13(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c13_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, 0xFF, byte + 12, byte + 11,
				0xFF, byte + 11, byte + 10, byte + 9,
				0xFF, 0xFF, byte + 9, byte + 8,
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, byte + 6, byte + 5, byte + 4,
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, byte + 3, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m256i c13_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c13_shfl_msk_m256i);
		__m256i c13_srlv_rslt_m256i = _mm256_srlv_epi32(c13_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[13]);
		__m256i c13_rslt_m256i = _mm256_and_si256(c13_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[13]);
		_mm256_storeu_si256(out++, c13_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c13_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, 0xFF, byte + 12, byte + 11,
				0xFF, byte + 11, byte + 10, byte + 9,
				0xFF, 0xFF, byte + 9, byte + 8,
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, byte + 6, byte + 5, byte + 4,
				0xFF, 0xFF, byte + 4, byte + 3,
				0xFF, byte + 3, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m256i c13_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c13_shfl_msk_m256i);
		__m256i c13_srlv_rslt_m256i = _mm256_srlv_epi32(c13_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[13]);
		__m256i c13_and_rslt_m256i = _mm256_and_si256(c13_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[13]);
		__m256i c13_rslt_m256i = _mm256_or_si256(c13_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 13));
		_mm256_storeu_si256(out++, c13_rslt_m256i);
	}
}


/**
 * AVX2-based unpacking 128 14-bit values.
 * Load 14 SSE vectors, each containing 9 14-bit values. (10th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c14(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
	     __m128i c14_load_rslt1_m128i = _mm_loadu_si128(in++);
	     __m256i c14_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c14_load_rslt1_m128i);
	     hor_avx2_unpack8_c14<0>(out, c14_broadcast_rslt1_m256i); // Unpack 1st 8 values.

	     __m128i c14_load_rslt2_m128i = _mm_loadu_si128(in++);
	     __m128i c14_alignr_rslt1_m128i = _mm_alignr_epi8(c14_load_rslt2_m128i, c14_load_rslt1_m128i, 14);
	     __m256i c14_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c14_alignr_rslt1_m128i);
	     hor_avx2_unpack8_c14<0>(out, c14_broadcast_rslt2_m256i); // Unpack 2nd 8 values.

	     __m128i c14_load_rslt3_m128i = _mm_loadu_si128(in++);
	     __m128i c14_alignr_rslt2_m128i = _mm_alignr_epi8(c14_load_rslt3_m128i, c14_load_rslt2_m128i, 12);
	     __m256i c14_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c14_alignr_rslt2_m128i);
	     hor_avx2_unpack8_c14<0>(out, c14_broadcast_rslt3_m256i); // Unpack 3rd 8 values.

	     __m128i c14_load_rslt4_m128i = _mm_loadu_si128(in++);
	     __m128i c14_alignr_rslt3_m128i = _mm_alignr_epi8(c14_load_rslt4_m128i, c14_load_rslt3_m128i, 10);
	     __m256i c14_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c14_alignr_rslt3_m128i);
	     hor_avx2_unpack8_c14<0>(out, c14_broadcast_rslt4_m256i); // Unpack 4th 8 values.

	     __m128i c14_load_rslt5_m128i = _mm_loadu_si128(in++);
	     __m128i c14_alignr_rslt4_m128i = _mm_alignr_epi8(c14_load_rslt5_m128i, c14_load_rslt4_m128i, 8);
	     __m256i c14_broadcast_rslt5_m256i = _mm256_broadcastsi128_si256(c14_alignr_rslt4_m128i);
	     hor_avx2_unpack8_c14<0>(out, c14_broadcast_rslt5_m256i); // Unpack 5th 8 values.

	     __m128i c14_load_rslt6_m128i = _mm_loadu_si128(in++);
	     __m128i c14_alignr_rslt5_m128i = _mm_alignr_epi8(c14_load_rslt6_m128i, c14_load_rslt5_m128i, 6);
	     __m256i c14_broadcast_rslt6_m256i = _mm256_broadcastsi128_si256(c14_alignr_rslt5_m128i);
	     hor_avx2_unpack8_c14<0>(out, c14_broadcast_rslt6_m256i); // Unpack 6th 8 values.

	     __m128i c14_load_rslt7_m128i = _mm_loadu_si128(in++);
	     __m128i c14_alignr_rslt6_m128i = _mm_alignr_epi8(c14_load_rslt7_m128i, c14_load_rslt6_m128i, 4);
	     __m256i c14_broadcast_rslt7_m256i = _mm256_broadcastsi128_si256(c14_alignr_rslt6_m128i);
	     hor_avx2_unpack8_c14<0>(out, c14_broadcast_rslt7_m256i); // Unpack 7th 8 values.
	     __m256i c14_broadcast_rslt8_m256i = _mm256_broadcastsi128_si256(c14_load_rslt7_m128i);
	     hor_avx2_unpack8_c14<2>(out, c14_broadcast_rslt8_m256i); // Unpack 8th 8 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c14(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c14_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, 0xFF, byte + 13, byte + 12,
				0xFF, byte + 12, byte + 11, byte + 10,
				0xFF, byte + 10, byte + 9, byte + 8,
				0xFF, 0xFF, byte + 8, byte + 7,
				0xFF, 0xFF, byte + 6, byte + 5,
				0xFF, byte + 5, byte + 4, byte + 3,
				0xFF, byte + 3, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m256i c14_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c14_shfl_msk_m256i);
		__m256i c14_srlv_rslt_m256i = _mm256_srlv_epi32(c14_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[14]);
		__m256i c14_rslt_m256i = _mm256_and_si256(c14_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[14]);
		_mm256_storeu_si256(out++, c14_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c14_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, 0xFF, byte + 13, byte + 12,
				0xFF, byte + 12, byte + 11, byte + 10,
				0xFF, byte + 10, byte + 9, byte + 8,
				0xFF, 0xFF, byte + 8, byte + 7,
				0xFF, 0xFF, byte + 6, byte + 5,
				0xFF, byte + 5, byte + 4, byte + 3,
				0xFF, byte + 3, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m256i c14_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c14_shfl_msk_m256i);
		__m256i c14_srlv_rslt_m256i = _mm256_srlv_epi32(c14_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[14]);
		__m256i c14_and_rslt_m256i = _mm256_and_si256(c14_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[14]);
		__m256i c14_rslt_m256i = _mm256_or_si256(c14_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 14));
		_mm256_storeu_si256(out++, c14_rslt_m256i);
	}
}


/**
 * AVX2-based unpacking 128 15-bit values.
 * Load 15 SSE vectors, each containing 8 15-bit values. (9th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c15(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
    __m128i c15_load_rslt1_m128i = _mm_loadu_si128(in + 0);
    __m256i c15_broadcast_rslt1_m256i = _mm256_broadcastsi128_si256(c15_load_rslt1_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt1_m256i); // Unpack 1st 8 values.

    __m128i c15_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c15_alignr_rslt1_m128i = _mm_alignr_epi8(c15_load_rslt2_m128i, c15_load_rslt1_m128i, 15);
    __m256i c15_broadcast_rslt2_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt1_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt2_m256i); // Unpack 2nd 8 values.

    __m128i c15_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c15_alignr_rslt2_m128i = _mm_alignr_epi8(c15_load_rslt3_m128i, c15_load_rslt2_m128i, 14);
    __m256i c15_broadcast_rslt3_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt2_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt3_m256i); // Unpack 3rd 8 values.

    __m128i c15_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c15_alignr_rslt3_m128i = _mm_alignr_epi8(c15_load_rslt4_m128i, c15_load_rslt3_m128i, 13);
    __m256i c15_broadcast_rslt4_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt3_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt4_m256i); // Unpack 4th 8 values.

    __m128i c15_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c15_alignr_rslt4_m128i = _mm_alignr_epi8(c15_load_rslt5_m128i, c15_load_rslt4_m128i, 12);
    __m256i c15_broadcast_rslt5_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt4_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt5_m256i); // Unpack 5th 8 values.

    __m128i c15_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c15_alignr_rslt5_m128i = _mm_alignr_epi8(c15_load_rslt6_m128i, c15_load_rslt5_m128i, 11);
    __m256i c15_broadcast_rslt6_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt5_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt6_m256i); // Unpack 6th 8 values.

    __m128i c15_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c15_alignr_rslt6_m128i = _mm_alignr_epi8(c15_load_rslt7_m128i, c15_load_rslt6_m128i, 10);
    __m256i c15_broadcast_rslt7_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt6_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt7_m256i); // Unpack 7th 8 values.

    __m128i c15_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c15_alignr_rslt7_m128i = _mm_alignr_epi8(c15_load_rslt8_m128i, c15_load_rslt7_m128i, 9);
    __m256i c15_broadcast_rslt8_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt7_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt8_m256i); // Unpack 8th 8 values.

    __m128i c15_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c15_alignr_rslt8_m128i = _mm_alignr_epi8(c15_load_rslt9_m128i, c15_load_rslt8_m128i, 8);
    __m256i c15_broadcast_rslt9_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt8_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt9_m256i); // Unpack 9th 8 values.

    __m128i c15_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c15_alignr_rslt9_m128i = _mm_alignr_epi8(c15_load_rslt10_m128i, c15_load_rslt9_m128i, 7);
    __m256i c15_broadcast_rslt10_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt9_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt10_m256i); // Unpack 10th 8 values.

    __m128i c15_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c15_alignr_rslt10_m128i = _mm_alignr_epi8(c15_load_rslt11_m128i, c15_load_rslt10_m128i, 6);
    __m256i c15_broadcast_rslt11_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt10_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt11_m256i); // Unpack 11th 8 values.

    __m128i c15_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c15_alignr_rslt11_m128i = _mm_alignr_epi8(c15_load_rslt12_m128i, c15_load_rslt11_m128i, 5);
    __m256i c15_broadcast_rslt12_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt11_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt12_m256i); // Unpack 12th 8 values.

    __m128i c15_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c15_alignr_rslt12_m128i = _mm_alignr_epi8(c15_load_rslt13_m128i, c15_load_rslt12_m128i, 4);
    __m256i c15_broadcast_rslt13_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt12_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt13_m256i); // Unpack 13th 8 values.

    __m128i c15_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c15_alignr_rslt13_m128i = _mm_alignr_epi8(c15_load_rslt14_m128i, c15_load_rslt13_m128i, 3);
    __m256i c15_broadcast_rslt14_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt13_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt14_m256i); // Unpack 14th 8 values.

    __m128i c15_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c15_alignr_rslt14_m128i = _mm_alignr_epi8(c15_load_rslt15_m128i, c15_load_rslt14_m128i, 2);
    __m256i c15_broadcast_rslt15_m256i = _mm256_broadcastsi128_si256(c15_alignr_rslt14_m128i);
    hor_avx2_unpack8_c15<0>(out, c15_broadcast_rslt15_m256i); // Unpack 15th 8 values.
    __m256i c15_broadcast_rslt16_m256i = _mm256_broadcastsi128_si256(c15_load_rslt15_m128i);
    hor_avx2_unpack8_c15<1>(out, c15_broadcast_rslt16_m256i); // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c15(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c15_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, 0xFF, byte + 14, byte + 13,
				0xFF, byte + 13, byte + 12, byte + 11,
				0xFF, byte + 11, byte + 10, byte + 9,
				0xFF, byte + 9, byte + 8, byte + 7,
				0xFF, byte + 7, byte + 6, byte + 5,
				0xFF, byte + 5, byte + 4, byte + 3,
				0xFF, byte + 3, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m256i c15_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c15_shfl_msk_m256i);
		__m256i c15_srlv_rslt_m256i = _mm256_srlv_epi32(c15_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[15]);
		__m256i c15_rslt_m256i = _mm256_and_si256(c15_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[15]);
		_mm256_storeu_si256(out++, c15_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c15_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, 0xFF, byte + 14, byte + 13,
				0xFF, byte + 13, byte + 12, byte + 11,
				0xFF, byte + 11, byte + 10, byte + 9,
				0xFF, byte + 9, byte + 8, byte + 7,
				0xFF, byte + 7, byte + 6, byte + 5,
				0xFF, byte + 5, byte + 4, byte + 3,
				0xFF, byte + 3, byte + 2, byte + 1,
				0xFF, 0xFF, byte + 1, byte + 0);
		__m256i c15_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c15_shfl_msk_m256i);
		__m256i c15_srlv_rslt_m256i = _mm256_srlv_epi32(c15_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[15]);
		__m256i c15_and_rslt_m256i = _mm256_and_si256(c15_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[15]);
		__m256i c15_rslt_m256i = _mm256_or_si256(c15_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 15));
		_mm256_storeu_si256(out++, c15_rslt_m256i);
	}
}


/**
 * AVX2-based unpacking 128 16-bit values.
 * Load 16 SSE vectors, each containing 8 16-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c16(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 8) {
		__m128i c16_load_rslt_m128i = _mm_loadu_si128(in++);
		__m256i c16_broadcast_rslt_m256i = _mm256_broadcastsi128_si256(c16_load_rslt_m128i);
		hor_avx2_unpack8_c16(out, c16_broadcast_rslt_m256i); // Unpack 8 values.
	}
}

template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c16(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		__m256i c16_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[16]);
		_mm256_storeu_si256(out++, c16_rslt_m256i);
	}
	else { // For Rice and OptRice.
		__m256i c16_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[16]);
		__m256i c16_rslt_m256i = _mm256_or_si256(c16_shfl_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 16));
		_mm256_storeu_si256(out++, c16_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 16-bit values.
// * Load 16 SSE vectors, each containing 8 16-bit values.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c16(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 8) {
//		__m128i c16_load_rslt_m128i = _mm_loadu_si128(in++);
//		__m256i c16_cvt_rslt_m256i = _mm256_cvtepu16_epi32(c16_load_rslt_m128i);
//		_mm256_storeu_si256(out++, c16_cvt_rslt_m256i); // Unpack 8 values.
//	}
//}


/**
 * AVX2-based unpacking 128 17-bit values.
 * Load 17 SSE vectors, each containing 7 17-bit values. (8th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c17(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c17_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	__m128i c17_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c17_alignr_rslt1_m128i = _mm_alignr_epi8(c17_load_rslt2_m128i, c17_load_rslt1_m128i, 8);
	__m256i c17_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_load_rslt1_m128i), c17_alignr_rslt1_m128i, 1);
	hor_avx2_unpack8_c17<0, 0>(out, c17_insert_rslt1_m256i);  // Unpack 1st 8 values.

	__m128i c17_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c17_alignr_rslt2_m128i = _mm_alignr_epi8(c17_load_rslt3_m128i, c17_load_rslt2_m128i, 9);
	__m256i c17_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_load_rslt2_m128i), c17_alignr_rslt2_m128i, 1);
	hor_avx2_unpack8_c17<1, 0>(out, c17_insert_rslt2_m256i);  // Unpack 2nd 8 values.

	__m128i c17_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c17_alignr_rslt3_m128i = _mm_alignr_epi8(c17_load_rslt4_m128i, c17_load_rslt3_m128i, 10);
	__m256i c17_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_load_rslt3_m128i), c17_alignr_rslt3_m128i, 1);
	hor_avx2_unpack8_c17<2, 0>(out, c17_insert_rslt3_m256i);  // Unpack 3rd 8 values.

	__m128i c17_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c17_alignr_rslt4_m128i = _mm_alignr_epi8(c17_load_rslt5_m128i, c17_load_rslt4_m128i, 11);
	__m256i c17_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_load_rslt4_m128i), c17_alignr_rslt4_m128i, 1);
	hor_avx2_unpack8_c17<3, 0>(out, c17_insert_rslt4_m256i);  // Unpack 4th 8 values.

	__m128i c17_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c17_alignr_rslt5_m128i = _mm_alignr_epi8(c17_load_rslt6_m128i, c17_load_rslt5_m128i, 12);
	__m256i c17_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_load_rslt5_m128i), c17_alignr_rslt5_m128i, 1);
	hor_avx2_unpack8_c17<4, 0>(out, c17_insert_rslt5_m256i);  // Unpack 5th 8 values.

	__m128i c17_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c17_alignr_rslt6_m128i = _mm_alignr_epi8(c17_load_rslt7_m128i, c17_load_rslt6_m128i, 13);
	__m256i c17_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_load_rslt6_m128i), c17_alignr_rslt6_m128i, 1);
	hor_avx2_unpack8_c17<5, 0>(out, c17_insert_rslt6_m256i);  // Unpack 6th 8 values.

	__m128i c17_load_rslt8_m128i = _mm_loadu_si128(in + 7);
	__m128i c17_alignr_rslt7_m128i = _mm_alignr_epi8(c17_load_rslt8_m128i, c17_load_rslt7_m128i, 14);
	__m256i c17_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_load_rslt7_m128i), c17_alignr_rslt7_m128i, 1);
	hor_avx2_unpack8_c17<6, 0>(out, c17_insert_rslt7_m256i);  // Unpack 7th 8 values.

	__m128i c17_load_rslt9_m128i = _mm_loadu_si128(in + 8);
	__m128i c17_alignr_rslt8_m128i = _mm_alignr_epi8(c17_load_rslt9_m128i, c17_load_rslt8_m128i, 15);
	__m256i c17_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_load_rslt8_m128i), c17_alignr_rslt8_m128i, 1);
	hor_avx2_unpack8_c17<7, 0>(out, c17_insert_rslt8_m256i);  // Unpack 8th 8 values.

	__m128i c17_load_rslt10_m128i = _mm_loadu_si128(in + 9);
	__m128i c17_alignr_rslt9_m128i = _mm_alignr_epi8(c17_load_rslt10_m128i, c17_load_rslt9_m128i, 8);
	__m256i c17_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt9_m128i), c17_load_rslt10_m128i, 1);
	hor_avx2_unpack8_c17<0, 0>(out, c17_insert_rslt9_m256i);  // Unpack 9th 8 values.

	__m128i c17_load_rslt11_m128i = _mm_loadu_si128(in + 10);
	__m128i c17_alignr_rslt10_m128i = _mm_alignr_epi8(c17_load_rslt11_m128i, c17_load_rslt10_m128i, 9);
	__m256i c17_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt10_m128i), c17_load_rslt11_m128i, 1);
	hor_avx2_unpack8_c17<0, 1>(out, c17_insert_rslt10_m256i); // Unpack 10th 8 values.

	__m128i c17_load_rslt12_m128i = _mm_loadu_si128(in + 11);
	__m128i c17_alignr_rslt11_m128i = _mm_alignr_epi8(c17_load_rslt12_m128i, c17_load_rslt11_m128i, 10);
	__m256i c17_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt11_m128i), c17_load_rslt12_m128i, 1);
	hor_avx2_unpack8_c17<0, 2>(out, c17_insert_rslt11_m256i); // Unpack 11th 8 values.

	__m128i c17_load_rslt13_m128i = _mm_loadu_si128(in + 12);
	__m128i c17_alignr_rslt12_m128i = _mm_alignr_epi8(c17_load_rslt13_m128i, c17_load_rslt12_m128i, 11);
	__m256i c17_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt12_m128i), c17_load_rslt13_m128i, 1);
	hor_avx2_unpack8_c17<0, 3>(out, c17_insert_rslt12_m256i); // Unpack 12th 8 values.

	__m128i c17_load_rslt14_m128i = _mm_loadu_si128(in + 13);
	__m128i c17_alignr_rslt13_m128i = _mm_alignr_epi8(c17_load_rslt14_m128i, c17_load_rslt13_m128i, 12);
	__m256i c17_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt13_m128i), c17_load_rslt14_m128i, 1);
	hor_avx2_unpack8_c17<0, 4>(out, c17_insert_rslt13_m256i); // Unpack 13th 8 values.

	__m128i c17_load_rslt15_m128i = _mm_loadu_si128(in + 14);
	__m128i c17_alignr_rslt14_m128i = _mm_alignr_epi8(c17_load_rslt15_m128i, c17_load_rslt14_m128i, 13);
	__m256i c17_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt14_m128i), c17_load_rslt15_m128i, 1);
	hor_avx2_unpack8_c17<0, 5>(out, c17_insert_rslt14_m256i); // Unpack 14th 8 values.

	__m128i c17_load_rslt16_m128i = _mm_loadu_si128(in + 15);
	__m128i c17_alignr_rslt15_m128i = _mm_alignr_epi8(c17_load_rslt16_m128i, c17_load_rslt15_m128i, 14);
	__m256i c17_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt15_m128i), c17_load_rslt16_m128i, 1);
	hor_avx2_unpack8_c17<0, 6>(out, c17_insert_rslt15_m256i); // Unpack 15th 8 values.

	__m128i c17_load_rslt17_m128i = _mm_loadu_si128(in + 16);
	__m128i c17_alignr_rslt16_m128i = _mm_alignr_epi8(c17_load_rslt17_m128i, c17_load_rslt16_m128i, 15);
	__m256i c17_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt16_m128i), c17_load_rslt17_m128i, 1);
	hor_avx2_unpack8_c17<0, 7>(out, c17_insert_rslt16_m256i); // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c17(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c17_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte2 + 8, byte2 + 7, byte2 + 6,
				0xFF, byte2 + 6, byte2 + 5, byte2 + 4,
				0xFF, byte2 + 4, byte2 + 3, byte2 + 2,
				0xFF, byte2 + 2, byte2 + 1, byte2 + 0,
				0xFF, byte1 + 8, byte1 + 7, byte1 + 6,
				0xFF, byte1 + 6, byte1 + 5, byte1 + 4,
				0xFF, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c17_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c17_shfl_msk_m256i);
		__m256i c17_srlv_rslt_m256i = _mm256_srlv_epi32(c17_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[17]);
		__m256i c17_rslt_m256i = _mm256_and_si256(c17_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[17]);
		_mm256_storeu_si256(out++, c17_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c17_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte2 + 8, byte2 + 7, byte2 + 6,
				0xFF, byte2 + 6, byte2 + 5, byte2 + 4,
				0xFF, byte2 + 4, byte2 + 3, byte2 + 2,
				0xFF, byte2 + 2, byte2 + 1, byte2 + 0,
				0xFF, byte1 + 8, byte1 + 7, byte1 + 6,
				0xFF, byte1 + 6, byte1 + 5, byte1 + 4,
				0xFF, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c17_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c17_shfl_msk_m256i);
		__m256i c17_srlv_rslt_m256i = _mm256_srlv_epi32(c17_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[17]);
		__m256i c17_and_rslt_m256i = _mm256_and_si256(c17_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[17]);
		__m256i c17_rslt_m256i = _mm256_or_si256(c17_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 17));
		_mm256_storeu_si256(out++, c17_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 17-bit values.
// * Load 17 SSE vectors, each containing 7 17-bit values. (8th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c17(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	__m128i c17_load_rslt1_m128i = _mm_loadu_si128(in + 0);
//	__m128i c17_load_rslt2_m128i = _mm_loadu_si128(in + 1);
//	__m128i c17_alignr_rslt1_m128i = _mm_alignr_epi8(c17_load_rslt2_m128i, c17_load_rslt1_m128i, 8);
//	__m256i c17_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_load_rslt1_m128i), c17_alignr_rslt1_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt1_m256i);      // Unpack 1st 8 values.
//
//	__m128i c17_load_rslt3_m128i = _mm_loadu_si128(in + 2);
//	__m128i c17_alignr_rslt2_1_m128i = _mm_alignr_epi8(c17_load_rslt3_m128i, c17_load_rslt2_m128i, 1);
//	__m128i c17_alignr_rslt2_2_m128i = _mm_alignr_epi8(c17_load_rslt3_m128i, c17_load_rslt2_m128i, 9);
//	__m256i c17_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt2_1_m128i), c17_alignr_rslt2_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt2_m256i);      // Unpack 2nd 8 values.
//
//	__m128i c17_load_rslt4_m128i = _mm_loadu_si128(in + 3);
//	__m128i c17_alignr_rslt3_1_m128i = _mm_alignr_epi8(c17_load_rslt4_m128i, c17_load_rslt3_m128i, 2);
//	__m128i c17_alignr_rslt3_2_m128i = _mm_alignr_epi8(c17_load_rslt4_m128i, c17_load_rslt3_m128i, 10);
//	__m256i c17_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt3_1_m128i), c17_alignr_rslt3_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt3_m256i);      // Unpack 3rd 8 values.
//
//	__m128i c17_load_rslt5_m128i = _mm_loadu_si128(in + 4);
//	__m128i c17_alignr_rslt4_1_m128i = _mm_alignr_epi8(c17_load_rslt5_m128i, c17_load_rslt4_m128i, 3);
//	__m128i c17_alignr_rslt4_2_m128i = _mm_alignr_epi8(c17_load_rslt5_m128i, c17_load_rslt4_m128i, 11);
//	__m256i c17_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt4_1_m128i), c17_alignr_rslt4_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt4_m256i);      // Unpack 4th 8 values.
//
//	__m128i c17_load_rslt6_m128i = _mm_loadu_si128(in + 5);
//	__m128i c17_alignr_rslt5_1_m128i = _mm_alignr_epi8(c17_load_rslt6_m128i, c17_load_rslt5_m128i, 4);
//	__m128i c17_alignr_rslt5_2_m128i = _mm_alignr_epi8(c17_load_rslt6_m128i, c17_load_rslt5_m128i, 12);
//	__m256i c17_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt5_1_m128i), c17_alignr_rslt5_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt5_m256i);      // Unpack 5th 8 values.
//
//	__m128i c17_load_rslt7_m128i = _mm_loadu_si128(in + 6);
//	__m128i c17_alignr_rslt6_1_m128i = _mm_alignr_epi8(c17_load_rslt7_m128i, c17_load_rslt6_m128i, 5);
//	__m128i c17_alignr_rslt6_2_m128i = _mm_alignr_epi8(c17_load_rslt7_m128i, c17_load_rslt6_m128i, 13);
//	__m256i c17_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt6_1_m128i), c17_alignr_rslt6_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt6_m256i);      // Unpack 6th 8 values.
//
//	__m128i c17_load_rslt8_m128i = _mm_loadu_si128(in + 7);
//	__m128i c17_alignr_rslt7_1_m128i = _mm_alignr_epi8(c17_load_rslt8_m128i, c17_load_rslt7_m128i, 6);
//	__m128i c17_alignr_rslt7_2_m128i = _mm_alignr_epi8(c17_load_rslt8_m128i, c17_load_rslt7_m128i, 14);
//	__m256i c17_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt7_1_m128i), c17_alignr_rslt7_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt7_m256i);      // Unpack 7th 8 values.
//
//	__m128i c17_load_rslt9_m128i = _mm_loadu_si128(in + 8);
//	__m128i c17_alignr_rslt8_1_m128i = _mm_alignr_epi8(c17_load_rslt9_m128i, c17_load_rslt8_m128i, 7);
//	__m128i c17_alignr_rslt8_2_m128i = _mm_alignr_epi8(c17_load_rslt9_m128i, c17_load_rslt8_m128i, 15);
//	__m256i c17_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt8_1_m128i), c17_alignr_rslt8_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt8_m256i);      // Unpack 8th 8 values.
//
//	__m128i c17_load_rslt10_m128i = _mm_loadu_si128(in + 9);
//	__m128i c17_alignr_rslt9_m128i = _mm_alignr_epi8(c17_load_rslt10_m128i, c17_load_rslt9_m128i, 8);
//	__m256i c17_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt9_m128i), c17_load_rslt10_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt9_m256i);      // Unpack 9th 8 values.
//
//	__m128i c17_load_rslt11_m128i = _mm_loadu_si128(in + 10);
//	__m128i c17_alignr_rslt10_1_m128i = _mm_alignr_epi8(c17_load_rslt11_m128i, c17_load_rslt10_m128i, 9);
//	__m128i c17_alignr_rslt10_2_m128i = _mm_alignr_epi8(c17_load_rslt11_m128i, c17_load_rslt10_m128i, 17);
//	__m256i c17_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt10_1_m128i), c17_alignr_rslt10_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt10_m256i);     // Unpack 10th 8 values.
//
//	__m128i c17_load_rslt12_m128i = _mm_loadu_si128(in + 11);
//	__m128i c17_alignr_rslt11_1_m128i = _mm_alignr_epi8(c17_load_rslt12_m128i, c17_load_rslt11_m128i, 10);
//	__m128i c17_alignr_rslt11_2_m128i = _mm_alignr_epi8(c17_load_rslt12_m128i, c17_load_rslt11_m128i, 18);
//	__m256i c17_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt11_1_m128i), c17_alignr_rslt11_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt11_m256i);     // Unpack 11th 8 values.
//
//	__m128i c17_load_rslt13_m128i = _mm_loadu_si128(in + 12);
//	__m128i c17_alignr_rslt12_1_m128i = _mm_alignr_epi8(c17_load_rslt13_m128i, c17_load_rslt12_m128i, 11);
//	__m128i c17_alignr_rslt12_2_m128i = _mm_alignr_epi8(c17_load_rslt13_m128i, c17_load_rslt12_m128i, 19);
//	__m256i c17_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt12_1_m128i), c17_alignr_rslt12_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt12_m256i);     // Unpack 12th 8 values.
//
//	__m128i c17_load_rslt14_m128i = _mm_loadu_si128(in + 13);
//	__m128i c17_alignr_rslt13_1_m128i = _mm_alignr_epi8(c17_load_rslt14_m128i, c17_load_rslt13_m128i, 12);
//	__m128i c17_alignr_rslt13_2_m128i = _mm_alignr_epi8(c17_load_rslt14_m128i, c17_load_rslt13_m128i, 20);
//	__m256i c17_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt13_1_m128i), c17_alignr_rslt13_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt13_m256i);     // Unpack 13th 8 values.
//
//	__m128i c17_load_rslt15_m128i = _mm_loadu_si128(in + 14);
//	__m128i c17_alignr_rslt14_1_m128i = _mm_alignr_epi8(c17_load_rslt15_m128i, c17_load_rslt14_m128i, 13);
//	__m128i c17_alignr_rslt14_2_m128i = _mm_alignr_epi8(c17_load_rslt15_m128i, c17_load_rslt14_m128i, 21);
//	__m256i c17_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt14_1_m128i), c17_alignr_rslt14_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt14_m256i);     // Unpack 14th 8 values.
//
//	__m128i c17_load_rslt16_m128i = _mm_loadu_si128(in + 15);
//	__m128i c17_alignr_rslt15_1_m128i = _mm_alignr_epi8(c17_load_rslt16_m128i, c17_load_rslt15_m128i, 14);
//	__m128i c17_alignr_rslt15_2_m128i = _mm_alignr_epi8(c17_load_rslt16_m128i, c17_load_rslt15_m128i, 22);
//	__m256i c17_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt15_1_m128i), c17_alignr_rslt15_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt15_m256i);     // Unpack 15th 8 values.
//
//	__m128i c17_load_rslt17_m128i = _mm_loadu_si128(in + 16);
//	__m128i c17_alignr_rslt16_1_m128i = _mm_alignr_epi8(c17_load_rslt17_m128i, c17_load_rslt16_m128i, 15);
//	__m128i c17_alignr_rslt16_2_m128i = _mm_alignr_epi8(c17_load_rslt17_m128i, c17_load_rslt16_m128i, 23);
//	__m256i c17_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c17_alignr_rslt16_1_m128i), c17_alignr_rslt16_2_m128i, 1);
//	hor_avx2_unpack8_c17(out, c17_insert_rslt16_m256i);     // Unpack 16th 8 values.
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c17(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c17_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[17]);
//		__m256i c17_srlv_rslt_m256i = _mm256_srlv_epi32(c17_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[17]);
//		__m256i c17_rslt_m256i = _mm256_and_si256(c17_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[17]);
//		_mm256_storeu_si256(out++, c17_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c17_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[17]);
//		__m256i c17_srlv_rslt_m256i = _mm256_srlv_epi32(c17_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[17]);
//		__m256i c17_and_rslt_m256i = _mm256_and_si256(c17_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[17]);
//		__m256i c17_rslt_m256i = _mm256_or_si256(c17_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 17));
//		_mm256_storeu_si256(out++, c17_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 18-bit values.
 * Load 18 SSE vectors, each containing 7 18-bit values. (8th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c18(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
		__m128i c18_load_rslt1_m128i = _mm_loadu_si128(in++);
		__m128i c18_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt1_m128i = _mm_alignr_epi8(c18_load_rslt2_m128i, c18_load_rslt1_m128i, 9);
		__m256i c18_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_load_rslt1_m128i), c18_alignr_rslt1_m128i, 1);
		hor_avx2_unpack8_c18<0, 0>(out, c18_insert_rslt1_m256i); // Unpack 1st 8 values.

		__m128i c18_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt2_m128i = _mm_alignr_epi8(c18_load_rslt3_m128i, c18_load_rslt2_m128i, 11);
		__m256i c18_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_load_rslt2_m128i), c18_alignr_rslt2_m128i, 1);
		hor_avx2_unpack8_c18<2, 0>(out, c18_insert_rslt2_m256i); // Unpack 2nd 8 values.

		__m128i c18_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt3_m128i = _mm_alignr_epi8(c18_load_rslt4_m128i, c18_load_rslt3_m128i, 13);
		__m256i c18_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_load_rslt3_m128i), c18_alignr_rslt3_m128i, 1);
		hor_avx2_unpack8_c18<4, 0>(out, c18_insert_rslt3_m256i); // Unpack 3rd 8 values.

		__m128i c18_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt4_m128i = _mm_alignr_epi8(c18_load_rslt5_m128i, c18_load_rslt4_m128i, 15);
		__m256i c18_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_load_rslt4_m128i), c18_alignr_rslt4_m128i, 1);
		hor_avx2_unpack8_c18<6, 0>(out, c18_insert_rslt4_m256i); // Unpack 4th 8 values.

		__m128i c18_load_rslt6_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt5_m128i = _mm_alignr_epi8(c18_load_rslt6_m128i, c18_load_rslt5_m128i, 8);
		__m256i c18_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_alignr_rslt5_m128i), c18_load_rslt6_m128i, 1);
		hor_avx2_unpack8_c18<0, 1>(out, c18_insert_rslt5_m256i); // Unpack 5th 8 values.

		__m128i c18_load_rslt7_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt6_m128i = _mm_alignr_epi8(c18_load_rslt7_m128i, c18_load_rslt6_m128i, 10);
		__m256i c18_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_alignr_rslt6_m128i), c18_load_rslt7_m128i, 1);
		hor_avx2_unpack8_c18<0, 3>(out, c18_insert_rslt6_m256i); // Unpack 6th 8 values.

		__m128i c18_load_rslt8_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt7_m128i = _mm_alignr_epi8(c18_load_rslt8_m128i, c18_load_rslt7_m128i, 12);
		__m256i c18_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_alignr_rslt7_m128i), c18_load_rslt8_m128i, 1);
		hor_avx2_unpack8_c18<0, 5>(out, c18_insert_rslt7_m256i); // Unpack 7th 8 values.

		__m128i c18_load_rslt9_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt8_m128i = _mm_alignr_epi8(c18_load_rslt9_m128i, c18_load_rslt8_m128i, 14);
		__m256i c18_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_alignr_rslt8_m128i), c18_load_rslt9_m128i, 1);
		hor_avx2_unpack8_c18<0, 7>(out, c18_insert_rslt8_m256i); // Unpack 8th 8 values.
	}
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c18(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c18_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte2 + 8, byte2 + 7, byte2 + 6,
				0xFF, byte2 + 6, byte2 + 5, byte2 + 4,
				0xFF, byte2 + 4, byte2 + 3, byte2 + 2,
				0xFF, byte2 + 2, byte2 + 1, byte2 + 0,
				0xFF, byte1 + 8, byte1 + 7, byte1 + 6,
				0xFF, byte1 + 6, byte1 + 5, byte1 + 4,
				0xFF, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c18_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c18_shfl_msk_m256i);
		__m256i c18_srlv_rslt_m256i = _mm256_srlv_epi32(c18_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[18]);
		__m256i c18_rslt_m256i = _mm256_and_si256(c18_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[18]);
		_mm256_storeu_si256(out++, c18_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c18_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte2 + 8, byte2 + 7, byte2 + 6,
				0xFF, byte2 + 6, byte2 + 5, byte2 + 4,
				0xFF, byte2 + 4, byte2 + 3, byte2 + 2,
				0xFF, byte2 + 2, byte2 + 1, byte2 + 0,
				0xFF, byte1 + 8, byte1 + 7, byte1 + 6,
				0xFF, byte1 + 6, byte1 + 5, byte1 + 4,
				0xFF, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c18_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c18_shfl_msk_m256i);
		__m256i c18_srlv_rslt_m256i = _mm256_srlv_epi32(c18_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[18]);
		__m256i c18_and_rslt_m256i = _mm256_and_si256(c18_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[18]);
		__m256i c18_rslt_m256i = _mm256_or_si256(c18_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 18));
		_mm256_storeu_si256(out++, c18_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 18-bit values.
// * Load 18 SSE vectors, each containing 7 18-bit values. (8th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c18(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
//		__m128i c18_load_rslt1_m128i = _mm_loadu_si128(in++);
//		__m128i c18_load_rslt2_m128i = _mm_loadu_si128(in++);
//		__m128i c18_alignr_rslt1_m128i = _mm_alignr_epi8(c18_load_rslt2_m128i, c18_load_rslt1_m128i, 9);
//		__m256i c18_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_load_rslt1_m128i), c18_alignr_rslt1_m128i, 1);
//		hor_avx2_unpack8_c18(out, c18_insert_rslt1_m256i); // Unpack 1st 8 values.
//
//		__m128i c18_load_rslt3_m128i = _mm_loadu_si128(in++);
//		__m128i c18_alignr_rslt2_1_m128i = _mm_alignr_epi8(c18_load_rslt3_m128i, c18_load_rslt2_m128i, 2);
//		__m128i c18_alignr_rslt2_2_m128i = _mm_alignr_epi8(c18_load_rslt3_m128i, c18_load_rslt2_m128i, 11);
//		__m256i c18_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_alignr_rslt2_1_m128i), c18_alignr_rslt2_2_m128i, 1);
//		hor_avx2_unpack8_c18(out, c18_insert_rslt2_m256i); // Unpack 2nd 8 values.
//
//		__m128i c18_load_rslt4_m128i = _mm_loadu_si128(in++);
//		__m128i c18_alignr_rslt3_1_m128i = _mm_alignr_epi8(c18_load_rslt4_m128i, c18_load_rslt3_m128i, 4);
//		__m128i c18_alignr_rslt3_2_m128i = _mm_alignr_epi8(c18_load_rslt4_m128i, c18_load_rslt3_m128i, 13);
//		__m256i c18_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_alignr_rslt3_1_m128i), c18_alignr_rslt3_2_m128i, 1);
//		hor_avx2_unpack8_c18(out, c18_insert_rslt3_m256i); // Unpack 3rd 8 values.
//
//		__m128i c18_load_rslt5_m128i = _mm_loadu_si128(in++);
//		__m128i c18_alignr_rslt4_1_m128i = _mm_alignr_epi8(c18_load_rslt5_m128i, c18_load_rslt4_m128i, 6);
//		__m128i c18_alignr_rslt4_2_m128i = _mm_alignr_epi8(c18_load_rslt5_m128i, c18_load_rslt4_m128i, 15);
//		__m256i c18_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_alignr_rslt4_1_m128i), c18_alignr_rslt4_2_m128i, 1);
//		hor_avx2_unpack8_c18(out, c18_insert_rslt4_m256i); // Unpack 4th 8 values.
//
//		__m128i c18_load_rslt6_m128i = _mm_loadu_si128(in++);
//		__m128i c18_alignr_rslt5_1_m128i = _mm_alignr_epi8(c18_load_rslt6_m128i, c18_load_rslt5_m128i, 8);
//		__m128i c18_alignr_rslt5_2_m128i = _mm_alignr_epi8(c18_load_rslt6_m128i, c18_load_rslt5_m128i, 17);
//		__m256i c18_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_alignr_rslt5_1_m128i), c18_alignr_rslt5_2_m128i, 1);
//		hor_avx2_unpack8_c18(out, c18_insert_rslt5_m256i); // Unpack 5th 8 values.
//
//		__m128i c18_load_rslt7_m128i = _mm_loadu_si128(in++);
//		__m128i c18_alignr_rslt6_1_m128i = _mm_alignr_epi8(c18_load_rslt7_m128i, c18_load_rslt6_m128i, 10);
//		__m128i c18_alignr_rslt6_2_m128i = _mm_alignr_epi8(c18_load_rslt7_m128i, c18_load_rslt6_m128i, 19);
//		__m256i c18_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_alignr_rslt6_1_m128i), c18_alignr_rslt6_2_m128i, 1);
//		hor_avx2_unpack8_c18(out, c18_insert_rslt6_m256i); // Unpack 6th 8 values.
//
//		__m128i c18_load_rslt8_m128i = _mm_loadu_si128(in++);
//		__m128i c18_alignr_rslt7_1_m128i = _mm_alignr_epi8(c18_load_rslt8_m128i, c18_load_rslt7_m128i, 12);
//		__m128i c18_alignr_rslt7_2_m128i = _mm_alignr_epi8(c18_load_rslt8_m128i, c18_load_rslt7_m128i, 21);
//		__m256i c18_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_alignr_rslt7_1_m128i), c18_alignr_rslt7_2_m128i, 1);
//		hor_avx2_unpack8_c18(out, c18_insert_rslt7_m256i); // Unpack 7th 8 values.
//
//		__m128i c18_load_rslt9_m128i = _mm_loadu_si128(in++);
//		__m128i c18_alignr_rslt8_1_m128i = _mm_alignr_epi8(c18_load_rslt9_m128i, c18_load_rslt8_m128i, 14);
//		__m128i c18_alignr_rslt8_2_m128i = _mm_alignr_epi8(c18_load_rslt9_m128i, c18_load_rslt8_m128i, 23);
//		__m256i c18_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c18_alignr_rslt8_1_m128i), c18_alignr_rslt8_2_m128i, 1);
//		hor_avx2_unpack8_c18(out, c18_insert_rslt8_m256i); // Unpack 8th 8 values.
//	}
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c18(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c18_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[18]);
//		__m256i c18_srlv_rslt_m256i = _mm256_srlv_epi32(c18_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[18]);
//		__m256i c18_rslt_m256i = _mm256_and_si256(c18_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[18]);
//		_mm256_storeu_si256(out++, c18_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c18_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[18]);
//		__m256i c18_srlv_rslt_m256i = _mm256_srlv_epi32(c18_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[18]);
//		__m256i c18_and_rslt_m256i = _mm256_and_si256(c18_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[18]);
//		__m256i c18_rslt_m256i = _mm256_or_si256(c18_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 18));
//		_mm256_storeu_si256(out++, c18_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 19-bit values.
 * Load 19 SSE vectors, each containing 6 19-bit values. (7th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c19(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
    __m128i c19_load_rslt1_m128i = _mm_loadu_si128(in + 0);
    __m128i c19_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c19_alignr_rslt1_m128i = _mm_alignr_epi8(c19_load_rslt2_m128i, c19_load_rslt1_m128i, 9);
    __m256i c19_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_load_rslt1_m128i), c19_alignr_rslt1_m128i, 1);
    hor_avx2_unpack8_c19<0, 0>(out, c19_insert_rslt1_m256i); // Unpack 1st 8 values.

    __m128i c19_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c19_alignr_rslt2_m128i = _mm_alignr_epi8(c19_load_rslt3_m128i, c19_load_rslt2_m128i, 12);
    __m256i c19_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_load_rslt2_m128i), c19_alignr_rslt2_m128i, 1);
    hor_avx2_unpack8_c19<3, 0>(out, c19_insert_rslt2_m256i); // Unpack 2nd 8 values.

    __m128i c19_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c19_alignr_rslt3_m128i = _mm_alignr_epi8(c19_load_rslt4_m128i, c19_load_rslt3_m128i, 15);
    __m256i c19_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_load_rslt3_m128i), c19_alignr_rslt3_m128i, 1);
    hor_avx2_unpack8_c19<6, 0>(out, c19_insert_rslt3_m256i); // Unpack 3rd 8 values.

    __m128i c19_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c19_alignr_rslt4_m128i = _mm_alignr_epi8(c19_load_rslt5_m128i, c19_load_rslt4_m128i, 9);
    __m256i c19_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt4_m128i), c19_load_rslt5_m128i, 1);
    hor_avx2_unpack8_c19<0, 2>(out, c19_insert_rslt4_m256i); // Unpack 4th 8 values.

    __m128i c19_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c19_alignr_rslt5_m128i = _mm_alignr_epi8(c19_load_rslt6_m128i, c19_load_rslt5_m128i, 12);
    __m256i c19_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt5_m128i), c19_load_rslt6_m128i, 1);
    hor_avx2_unpack8_c19<0, 5>(out, c19_insert_rslt5_m256i); // Unpack 5h 8 values.

    __m128i c19_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c19_alignr_rslt6_1_m128i = _mm_alignr_epi8(c19_load_rslt7_m128i, c19_load_rslt6_m128i, 15);
    __m128i c19_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c19_alignr_rslt6_2_m128i = _mm_alignr_epi8(c19_load_rslt8_m128i, c19_load_rslt7_m128i, 8);
    __m256i c19_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt6_1_m128i), c19_alignr_rslt6_2_m128i, 1);
    hor_avx2_unpack8_c19<0, 0>(out, c19_insert_rslt6_m256i); // Unpack 6th 8 values.

    __m128i c19_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c19_alignr_rslt7_m128i = _mm_alignr_epi8(c19_load_rslt9_m128i, c19_load_rslt8_m128i, 11);
    __m256i c19_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_load_rslt8_m128i), c19_alignr_rslt7_m128i, 1);
    hor_avx2_unpack8_c19<2, 0>(out, c19_insert_rslt7_m256i); // Unpack 7th 8 values.

    __m128i c19_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c19_alignr_rslt8_m128i = _mm_alignr_epi8(c19_load_rslt10_m128i, c19_load_rslt9_m128i, 14);
    __m256i c19_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_load_rslt9_m128i), c19_alignr_rslt8_m128i, 1);
    hor_avx2_unpack8_c19<5, 0>(out, c19_insert_rslt8_m256i); // Unpack 8th 8 values.

    __m128i c19_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c19_alignr_rslt9_m128i = _mm_alignr_epi8(c19_load_rslt11_m128i, c19_load_rslt10_m128i, 8);
    __m256i c19_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt9_m128i), c19_load_rslt11_m128i, 1);
    hor_avx2_unpack8_c19<0, 1>(out, c19_insert_rslt9_m256i); // Unpack 9th 8 values.

    __m128i c19_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c19_alignr_rslt10_m128i = _mm_alignr_epi8(c19_load_rslt12_m128i, c19_load_rslt11_m128i, 11);
    __m256i c19_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt10_m128i), c19_load_rslt12_m128i, 1);
    hor_avx2_unpack8_c19<0, 4>(out, c19_insert_rslt10_m256i); // Unpack 10th 8 values.

    __m128i c19_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c19_alignr_rslt11_1_m128i = _mm_alignr_epi8(c19_load_rslt13_m128i, c19_load_rslt12_m128i, 14);
    __m128i c19_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c19_alignr_rslt11_2_m128i = _mm_alignr_epi8(c19_load_rslt14_m128i, c19_load_rslt13_m128i, 7);
    __m256i c19_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt11_1_m128i), c19_alignr_rslt11_2_m128i, 1);
    hor_avx2_unpack8_c19<0, 0>(out, c19_insert_rslt11_m256i); // Unpack 11th 8 values.

    __m128i c19_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c19_alignr_rslt12_m128i = _mm_alignr_epi8(c19_load_rslt15_m128i, c19_load_rslt14_m128i, 10);
    __m256i c19_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_load_rslt14_m128i), c19_alignr_rslt12_m128i, 1);
    hor_avx2_unpack8_c19<1, 0>(out, c19_insert_rslt12_m256i); // Unpack 12th 8 values.

    __m128i c19_load_rslt16_m128i = _mm_loadu_si128(in + 15);
    __m128i c19_alignr_rslt13_m128i = _mm_alignr_epi8(c19_load_rslt16_m128i, c19_load_rslt15_m128i, 13);
    __m256i c19_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_load_rslt15_m128i), c19_alignr_rslt13_m128i, 1);
    hor_avx2_unpack8_c19<4, 0>(out, c19_insert_rslt13_m256i); // Unpack 13th 8 values.

    __m128i c19_load_rslt17_m128i = _mm_loadu_si128(in + 16);
    __m128i c19_alignr_rslt14_m128i = _mm_alignr_epi8(c19_load_rslt17_m128i, c19_load_rslt16_m128i, 7);
    __m256i c19_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt14_m128i), c19_load_rslt17_m128i, 1);
    hor_avx2_unpack8_c19<0, 0>(out, c19_insert_rslt14_m256i); // Unpack 14th 8 values.

    __m128i c19_load_rslt18_m128i = _mm_loadu_si128(in + 17);
    __m128i c19_alignr_rslt15_m128i = _mm_alignr_epi8(c19_load_rslt18_m128i, c19_load_rslt17_m128i, 10);
    __m256i c19_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt15_m128i), c19_load_rslt18_m128i, 1);
    hor_avx2_unpack8_c19<0, 3>(out, c19_insert_rslt15_m256i); // Unpack 15th 8 values.

    __m128i c19_load_rslt19_m128i = _mm_loadu_si128(in + 18);
    __m128i c19_alignr_rslt16_m128i = _mm_alignr_epi8(c19_load_rslt19_m128i, c19_load_rslt18_m128i, 13);
    __m256i c19_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt16_m128i), c19_load_rslt19_m128i, 1);
    hor_avx2_unpack8_c19<0, 6>(out, c19_insert_rslt16_m256i); // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c19(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c19_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte2 + 9, byte2 + 8, byte2 + 7,
				0xFF, byte2 + 7, byte2 + 6, byte2 + 5,
				byte2 + 5, byte2 + 4, byte2 + 3, byte2 + 2,
				0xFF, byte2 + 2, byte2 + 1, byte2 + 0,
				0xFF, byte1 + 9, byte1 + 8, byte1 + 7,
				byte1 + 7, byte1 + 6, byte1 + 5, byte1 + 4,
				0xFF, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c19_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c19_shfl_msk_m256i);
		__m256i c19_srlv_rslt_m256i = _mm256_srlv_epi32(c19_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[19]);
		__m256i c19_rslt_m256i = _mm256_and_si256(c19_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[19]);
		_mm256_storeu_si256(out++, c19_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c19_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte2 + 9, byte2 + 8, byte2 + 7,
				0xFF, byte2 + 7, byte2 + 6, byte2 + 5,
				byte2 + 5, byte2 + 4, byte2 + 3, byte2 + 2,
				0xFF, byte2 + 2, byte2 + 1, byte2 + 0,
				0xFF, byte1 + 9, byte1 + 8, byte1 + 7,
				byte1 + 7, byte1 + 6, byte1 + 5, byte1 + 4,
				0xFF, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c19_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c19_shfl_msk_m256i);
		__m256i c19_srlv_rslt_m256i = _mm256_srlv_epi32(c19_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[19]);
		__m256i c19_and_rslt_m256i = _mm256_and_si256(c19_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[19]);
		__m256i c19_rslt_m256i = _mm256_or_si256(c19_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 19));
		_mm256_storeu_si256(out++, c19_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 19-bit values.
// * Load 19 SSE vectors, each containing 6 19-bit values. (7th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c19(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//    __m128i c19_load_rslt1_m128i = _mm_loadu_si128(in + 0);
//    __m128i c19_load_rslt2_m128i = _mm_loadu_si128(in + 1);
//    __m128i c19_alignr_rslt1_m128i = _mm_alignr_epi8(c19_load_rslt2_m128i, c19_load_rslt1_m128i, 9);
//    __m256i c19_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_load_rslt1_m128i), c19_alignr_rslt1_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt1_m256i); // Unpack 1st 8 values.
//
//    __m128i c19_load_rslt3_m128i = _mm_loadu_si128(in + 2);
//    __m128i c19_alignr_rslt2_1_m128i = _mm_alignr_epi8(c19_load_rslt3_m128i, c19_load_rslt2_m128i, 3);
//    __m128i c19_alignr_rslt2_2_m128i = _mm_alignr_epi8(c19_load_rslt3_m128i, c19_load_rslt2_m128i, 12);
//    __m256i c19_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt2_1_m128i), c19_alignr_rslt2_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt2_m256i); // Unpack 2nd 8 values.
//
//    __m128i c19_load_rslt4_m128i = _mm_loadu_si128(in + 3);
//    __m128i c19_alignr_rslt3_1_m128i = _mm_alignr_epi8(c19_load_rslt4_m128i, c19_load_rslt3_m128i, 6);
//    __m128i c19_alignr_rslt3_2_m128i = _mm_alignr_epi8(c19_load_rslt4_m128i, c19_load_rslt3_m128i, 15);
//    __m256i c19_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt3_1_m128i), c19_alignr_rslt3_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt3_m256i); // Unpack 3rd 8 values.
//
//    __m128i c19_load_rslt5_m128i = _mm_loadu_si128(in + 4);
//    __m128i c19_alignr_rslt4_1_m128i = _mm_alignr_epi8(c19_load_rslt5_m128i, c19_load_rslt4_m128i, 9);
//    __m128i c19_alignr_rslt4_2_m128i = _mm_alignr_epi8(c19_load_rslt5_m128i, c19_load_rslt4_m128i, 18);
//    __m256i c19_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt4_1_m128i), c19_alignr_rslt4_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt4_m256i); // Unpack 4th 8 values.
//
//    __m128i c19_load_rslt6_m128i = _mm_loadu_si128(in + 5);
//    __m128i c19_alignr_rslt5_1_m128i = _mm_alignr_epi8(c19_load_rslt6_m128i, c19_load_rslt5_m128i, 12);
//    __m128i c19_alignr_rslt5_2_m128i = _mm_alignr_epi8(c19_load_rslt6_m128i, c19_load_rslt5_m128i, 21);
//    __m256i c19_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt5_1_m128i), c19_alignr_rslt5_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt5_m256i); // Unpack 5h 8 values.
//
//    __m128i c19_load_rslt7_m128i = _mm_loadu_si128(in + 6);
//    __m128i c19_alignr_rslt6_1_m128i = _mm_alignr_epi8(c19_load_rslt7_m128i, c19_load_rslt6_m128i, 15);
//    __m128i c19_load_rslt8_m128i = _mm_loadu_si128(in + 7);
//    __m128i c19_alignr_rslt6_2_m128i = _mm_alignr_epi8(c19_load_rslt8_m128i, c19_load_rslt7_m128i, 8);
//    __m256i c19_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt6_1_m128i), c19_alignr_rslt6_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt6_m256i); // Unpack 6th 8 values.
//
//    __m128i c19_load_rslt9_m128i = _mm_loadu_si128(in + 8);
//    __m128i c19_alignr_rslt7_1_m128i = _mm_alignr_epi8(c19_load_rslt9_m128i, c19_load_rslt8_m128i, 2);
//    __m128i c19_alignr_rslt7_2_m128i = _mm_alignr_epi8(c19_load_rslt9_m128i, c19_load_rslt8_m128i, 11);
//    __m256i c19_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt7_1_m128i), c19_alignr_rslt7_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt7_m256i); // Unpack 7th 8 values.
//
//    __m128i c19_load_rslt10_m128i = _mm_loadu_si128(in + 9);
//    __m128i c19_alignr_rslt8_1_m128i = _mm_alignr_epi8(c19_load_rslt10_m128i, c19_load_rslt9_m128i, 5);
//    __m128i c19_alignr_rslt8_2_m128i = _mm_alignr_epi8(c19_load_rslt10_m128i, c19_load_rslt9_m128i, 14);
//    __m256i c19_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt8_1_m128i), c19_alignr_rslt8_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt8_m256i); // Unpack 8th 8 values.
//
//    __m128i c19_load_rslt11_m128i = _mm_loadu_si128(in + 10);
//    __m128i c19_alignr_rslt9_1_m128i = _mm_alignr_epi8(c19_load_rslt11_m128i, c19_load_rslt10_m128i, 8);
//    __m128i c19_alignr_rslt9_2_m128i = _mm_alignr_epi8(c19_load_rslt11_m128i, c19_load_rslt10_m128i, 17);
//    __m256i c19_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt9_1_m128i), c19_alignr_rslt9_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt9_m256i); // Unpack 9th 8 values.
//
//    __m128i c19_load_rslt12_m128i = _mm_loadu_si128(in + 11);
//    __m128i c19_alignr_rslt10_1_m128i = _mm_alignr_epi8(c19_load_rslt12_m128i, c19_load_rslt11_m128i, 11);
//    __m128i c19_alignr_rslt10_2_m128i = _mm_alignr_epi8(c19_load_rslt12_m128i, c19_load_rslt11_m128i, 20);
//    __m256i c19_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt10_1_m128i), c19_alignr_rslt10_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt10_m256i); // Unpack 10th 8 values.
//
//    __m128i c19_load_rslt13_m128i = _mm_loadu_si128(in + 12);
//    __m128i c19_alignr_rslt11_1_m128i = _mm_alignr_epi8(c19_load_rslt13_m128i, c19_load_rslt12_m128i, 14);
//    __m128i c19_load_rslt14_m128i = _mm_loadu_si128(in + 13);
//    __m128i c19_alignr_rslt11_2_m128i = _mm_alignr_epi8(c19_load_rslt14_m128i, c19_load_rslt13_m128i, 7);
//    __m256i c19_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt11_1_m128i), c19_alignr_rslt11_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt11_m256i); // Unpack 11th 8 values.
//
//    __m128i c19_load_rslt15_m128i = _mm_loadu_si128(in + 14);
//    __m128i c19_alignr_rslt12_1_m128i = _mm_alignr_epi8(c19_load_rslt15_m128i, c19_load_rslt14_m128i, 1);
//    __m128i c19_alignr_rslt12_2_m128i = _mm_alignr_epi8(c19_load_rslt15_m128i, c19_load_rslt14_m128i, 10);
//    __m256i c19_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt12_1_m128i), c19_alignr_rslt12_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt12_m256i); // Unpack 12th 8 values.
//
//    __m128i c19_load_rslt16_m128i = _mm_loadu_si128(in + 15);
//    __m128i c19_alignr_rslt13_1_m128i = _mm_alignr_epi8(c19_load_rslt16_m128i, c19_load_rslt15_m128i, 4);
//    __m128i c19_alignr_rslt13_2_m128i = _mm_alignr_epi8(c19_load_rslt16_m128i, c19_load_rslt15_m128i, 13);
//    __m256i c19_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt13_1_m128i), c19_alignr_rslt13_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt13_m256i); // Unpack 13th 8 values.
//
//    __m128i c19_load_rslt17_m128i = _mm_loadu_si128(in + 16);
//    __m128i c19_alignr_rslt14_1_m128i = _mm_alignr_epi8(c19_load_rslt17_m128i, c19_load_rslt16_m128i, 7);
//    __m128i c19_alignr_rslt14_2_m128i = _mm_alignr_epi8(c19_load_rslt17_m128i, c19_load_rslt16_m128i, 16);
//    __m256i c19_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt14_1_m128i), c19_alignr_rslt14_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt14_m256i); // Unpack 14th 8 values.
//
//    __m128i c19_load_rslt18_m128i = _mm_loadu_si128(in + 17);
//    __m128i c19_alignr_rslt15_1_m128i = _mm_alignr_epi8(c19_load_rslt18_m128i, c19_load_rslt17_m128i, 10);
//    __m128i c19_alignr_rslt15_2_m128i = _mm_alignr_epi8(c19_load_rslt18_m128i, c19_load_rslt17_m128i, 19);
//    __m256i c19_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt15_1_m128i), c19_alignr_rslt15_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt15_m256i); // Unpack 15th 8 values.
//
//    __m128i c19_load_rslt19_m128i = _mm_loadu_si128(in + 18);
//    __m128i c19_alignr_rslt16_1_m128i = _mm_alignr_epi8(c19_load_rslt19_m128i, c19_load_rslt18_m128i, 13);
//    __m128i c19_alignr_rslt16_2_m128i = _mm_alignr_epi8(c19_load_rslt19_m128i, c19_load_rslt18_m128i, 22);
//    __m256i c19_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c19_alignr_rslt16_1_m128i), c19_alignr_rslt16_2_m128i, 1);
//    hor_avx2_unpack8_c19(out, c19_insert_rslt16_m256i); // Unpack 16th 8 values.
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c19(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c19_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[19]);
//		__m256i c19_srlv_rslt_m256i = _mm256_srlv_epi32(c19_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[19]);
//		__m256i c19_rslt_m256i = _mm256_and_si256(c19_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[19]);
//		_mm256_storeu_si256(out++, c19_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c19_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[19]);
//		__m256i c19_srlv_rslt_m256i = _mm256_srlv_epi32(c19_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[19]);
//		__m256i c19_and_rslt_m256i = _mm256_and_si256(c19_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[19]);
//		__m256i c19_rslt_m256i = _mm256_or_si256(c19_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 19));
//		_mm256_storeu_si256(out++, c19_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 20-bit values.
 * Load 20 SSE vectors, each containing 6 20-bit values. (7th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c20(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
		__m128i c20_load_rslt1_m128i = _mm_loadu_si128(in++);
		__m128i c20_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c20_alignr_rslt1_m128i = _mm_alignr_epi8(c20_load_rslt2_m128i, c20_load_rslt1_m128i, 10);
		__m256i c20_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c20_load_rslt1_m128i), c20_alignr_rslt1_m128i, 1);
		hor_avx2_unpack8_c20<0, 0>(out, c20_insert_rslt1_m256i); // Unpack 1st 8 values.

		__m128i c20_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c20_alignr_rslt2_m128i = _mm_alignr_epi8(c20_load_rslt3_m128i, c20_load_rslt2_m128i, 14);
		__m256i c20_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c20_load_rslt2_m128i), c20_alignr_rslt2_m128i, 1);
		hor_avx2_unpack8_c20<4, 0>(out, c20_insert_rslt2_m256i); // Unpack 2nd 8 values.

		__m128i c20_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c20_alignr_rslt3_m128i = _mm_alignr_epi8(c20_load_rslt4_m128i, c20_load_rslt3_m128i, 8);
		__m256i c20_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c20_alignr_rslt3_m128i), c20_load_rslt4_m128i, 1);
		hor_avx2_unpack8_c20<0, 2>(out, c20_insert_rslt3_m256i); // Unpack 3rd 8 values.

		__m128i c20_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c20_alignr_rslt4_m128i = _mm_alignr_epi8(c20_load_rslt5_m128i, c20_load_rslt4_m128i, 12);
		__m256i c20_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c20_alignr_rslt4_m128i), c20_load_rslt5_m128i, 1);
		hor_avx2_unpack8_c20<0, 6>(out, c20_insert_rslt4_m256i); // Unpack 3rd 8 values.
	}
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c20(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c20_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte2 + 9, byte2 + 8, byte2 + 7,
				0xFF, byte2 + 7, byte2 + 6, byte2 + 5,
				0xFF, byte2 + 4, byte2 + 3, byte2 + 2,
				0xFF, byte2 + 2, byte2 + 1, byte2 + 0,
				0xFF, byte1 + 9, byte1 + 8, byte1 + 7,
				0xFF, byte1 + 7, byte1 + 6, byte1 + 5,
				0xFF, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c20_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c20_shfl_msk_m256i);
		__m256i c20_srlv_rslt_m256i = _mm256_srlv_epi32(c20_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[20]);
		__m256i c20_rslt_m256i = _mm256_and_si256(c20_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[20]);
		_mm256_storeu_si256(out++, c20_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c20_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte2 + 9, byte2 + 8, byte2 + 7,
				0xFF, byte2 + 7, byte2 + 6, byte2 + 5,
				0xFF, byte2 + 4, byte2 + 3, byte2 + 2,
				0xFF, byte2 + 2, byte2 + 1, byte2 + 0,
				0xFF, byte1 + 9, byte1 + 8, byte1 + 7,
				0xFF, byte1 + 7, byte1 + 6, byte1 + 5,
				0xFF, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c20_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c20_shfl_msk_m256i);
		__m256i c20_srlv_rslt_m256i = _mm256_srlv_epi32(c20_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[20]);
		__m256i c20_and_rslt_m256i = _mm256_and_si256(c20_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[20]);
		__m256i c20_rslt_m256i = _mm256_or_si256(c20_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 20));
		_mm256_storeu_si256(out++, c20_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 20-bit values.
// * Load 20 SSE vectors, each containing 6 20-bit values. (7th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c20(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
//		__m128i c20_load_rslt1_m128i = _mm_loadu_si128(in++);
//		__m128i c20_load_rslt2_m128i = _mm_loadu_si128(in++);
//		__m128i c20_alignr_rslt1_m128i = _mm_alignr_epi8(c20_load_rslt2_m128i, c20_load_rslt1_m128i, 10);
//		__m256i c20_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c20_load_rslt1_m128i), c20_alignr_rslt1_m128i, 1);
//		hor_avx2_unpack8_c20(out, c20_insert_rslt1_m256i);   // Unpack 1st 8 values.
//
//		__m128i c20_load_rslt3_m128i = _mm_loadu_si128(in++);
//		__m128i c20_alignr_rslt2_1_m128i = _mm_alignr_epi8(c20_load_rslt3_m128i, c20_load_rslt2_m128i, 4);
//		__m128i c20_alignr_rslt2_2_m128i = _mm_alignr_epi8(c20_load_rslt3_m128i, c20_load_rslt2_m128i, 14);
//		__m256i c20_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c20_alignr_rslt2_1_m128i), c20_alignr_rslt2_2_m128i, 1);
//		hor_avx2_unpack8_c20(out, c20_insert_rslt2_m256i);   // Unpack 2nd 8 values.
//
//		__m128i c20_load_rslt4_m128i = _mm_loadu_si128(in++);
//		__m128i c20_alignr_rslt3_1_m128i = _mm_alignr_epi8(c20_load_rslt4_m128i, c20_load_rslt3_m128i, 8);
//		__m128i c20_alignr_rslt3_2_m128i = _mm_alignr_epi8(c20_load_rslt4_m128i, c20_load_rslt3_m128i, 18);
//		__m256i c20_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c20_alignr_rslt3_1_m128i), c20_alignr_rslt3_2_m128i, 1);
//		hor_avx2_unpack8_c20(out, c20_insert_rslt3_m256i);   // Unpack 3rd 8 values.
//
//		__m128i c20_load_rslt5_m128i = _mm_loadu_si128(in++);
//		__m128i c20_alignr_rslt4_1_m128i = _mm_alignr_epi8(c20_load_rslt5_m128i, c20_load_rslt4_m128i, 12);
//		__m128i c20_alignr_rslt4_2_m128i = _mm_alignr_epi8(c20_load_rslt5_m128i, c20_load_rslt4_m128i, 22);
//		__m256i c20_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c20_alignr_rslt4_1_m128i), c20_alignr_rslt4_2_m128i, 1);
//		hor_avx2_unpack8_c20(out, c20_insert_rslt4_m256i);   // Unpack 3rd 8 values.
//	}
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c20(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c20_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[20]);
//		__m256i c20_srlv_rslt_m256i = _mm256_srlv_epi32(c20_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[20]);
//		__m256i c20_rslt_m256i = _mm256_and_si256(c20_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[20]);
//		_mm256_storeu_si256(out++, c20_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c20_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[20]);
//		__m256i c20_srlv_rslt_m256i = _mm256_srlv_epi32(c20_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[20]);
//		__m256i c20_and_rslt_m256i = _mm256_and_si256(c20_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[20]);
//		__m256i c20_rslt_m256i = _mm256_or_si256(c20_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 20));
//		_mm256_storeu_si256(out++, c20_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 21-bit values.
 * Load 21 SSE vectors, each containing 6 21-bit values. (7th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c21(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
    __m128i c21_load_rslt1_m128i = _mm_loadu_si128(in + 0);
    __m128i c21_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c21_alignr_rslt1_m128i = _mm_alignr_epi8(c21_load_rslt2_m128i, c21_load_rslt1_m128i, 10);
    __m256i c21_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_load_rslt1_m128i), c21_alignr_rslt1_m128i, 1);
    hor_avx2_unpack8_c21<0, 0>(out, c21_insert_rslt1_m256i); // Unpack 1st 8 values.

    __m128i c21_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c21_alignr_rslt2_m128i = _mm_alignr_epi8(c21_load_rslt3_m128i, c21_load_rslt2_m128i, 15);
    __m256i c21_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_load_rslt2_m128i), c21_alignr_rslt2_m128i, 1);
    hor_avx2_unpack8_c21<5, 0>(out, c21_insert_rslt2_m256i); // Unpack 2nd 8 values.

    __m128i c21_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c21_alignr_rslt3_m128i = _mm_alignr_epi8(c21_load_rslt4_m128i, c21_load_rslt3_m128i, 10);
    __m256i c21_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt3_m128i), c21_load_rslt4_m128i, 1);
    hor_avx2_unpack8_c21<0, 4>(out, c21_insert_rslt3_m256i); // Unpack 3rd 8 values.

    __m128i c21_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c21_alignr_rslt4_1_m128i = _mm_alignr_epi8(c21_load_rslt5_m128i, c21_load_rslt4_m128i, 15);
    __m128i c21_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c21_alignr_rslt4_2_m128i = _mm_alignr_epi8(c21_load_rslt6_m128i, c21_load_rslt5_m128i, 9);
    __m256i c21_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt4_1_m128i), c21_alignr_rslt4_2_m128i, 1);
    hor_avx2_unpack8_c21<0, 0>(out, c21_insert_rslt4_m256i); // Unpack 4th 8 values.

    __m128i c21_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c21_alignr_rslt5_m128i = _mm_alignr_epi8(c21_load_rslt7_m128i, c21_load_rslt6_m128i, 14);
    __m256i c21_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_load_rslt6_m128i), c21_alignr_rslt5_m128i, 1);
    hor_avx2_unpack8_c21<4, 0>(out, c21_insert_rslt5_m256i); // Unpack 5th 8 values.

    __m128i c21_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c21_alignr_rslt6_m128i = _mm_alignr_epi8(c21_load_rslt8_m128i, c21_load_rslt7_m128i, 9);
    __m256i c21_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt6_m128i), c21_load_rslt8_m128i, 1);
    hor_avx2_unpack8_c21<0, 3>(out, c21_insert_rslt6_m256i); // Unpack 6th 8 values.

    __m128i c21_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c21_alignr_rslt7_1_m128i = _mm_alignr_epi8(c21_load_rslt9_m128i, c21_load_rslt8_m128i, 14);
    __m128i c21_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c21_alignr_rslt7_2_m128i = _mm_alignr_epi8(c21_load_rslt10_m128i, c21_load_rslt9_m128i, 8);
    __m256i c21_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt7_1_m128i), c21_alignr_rslt7_2_m128i, 1);
    hor_avx2_unpack8_c21<0, 0>(out, c21_insert_rslt7_m256i); // Unpack 7th 8 values.

    __m128i c21_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c21_alignr_rslt8_m128i = _mm_alignr_epi8(c21_load_rslt11_m128i, c21_load_rslt10_m128i, 13);
    __m256i c21_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_load_rslt10_m128i), c21_alignr_rslt8_m128i, 1);
    hor_avx2_unpack8_c21<3, 0>(out, c21_insert_rslt8_m256i); // Unpack 8th 8 values.

    __m128i c21_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c21_alignr_rslt9_m128i = _mm_alignr_epi8(c21_load_rslt12_m128i, c21_load_rslt11_m128i, 8);
    __m256i c21_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt9_m128i), c21_load_rslt12_m128i, 1);
    hor_avx2_unpack8_c21<0, 2>(out, c21_insert_rslt9_m256i); // Unpack 9th 8 values.

    __m128i c21_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c21_alignr_rslt10_1_m128i = _mm_alignr_epi8(c21_load_rslt13_m128i, c21_load_rslt12_m128i, 13);
    __m128i c21_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c21_alignr_rslt10_2_m128i = _mm_alignr_epi8(c21_load_rslt14_m128i, c21_load_rslt13_m128i, 7);
    __m256i c21_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt10_1_m128i), c21_alignr_rslt10_2_m128i, 1);
    hor_avx2_unpack8_c21<0, 0>(out, c21_insert_rslt10_m256i); // Unpack 10th 8 values.

    __m128i c21_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c21_alignr_rslt11_m128i = _mm_alignr_epi8(c21_load_rslt15_m128i, c21_load_rslt14_m128i, 12);
    __m256i c21_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_load_rslt14_m128i), c21_alignr_rslt11_m128i, 1);
    hor_avx2_unpack8_c21<2, 0>(out, c21_insert_rslt11_m256i); // Unpack 11th 8 values.

    __m128i c21_load_rslt16_m128i = _mm_loadu_si128(in + 15);
    __m128i c21_alignr_rslt12_m128i = _mm_alignr_epi8(c21_load_rslt16_m128i, c21_load_rslt15_m128i, 7);
    __m256i c21_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt12_m128i), c21_load_rslt16_m128i, 1);
    hor_avx2_unpack8_c21<0, 1>(out, c21_insert_rslt12_m256i); // Unpack 12th 8 values.

    __m128i c21_load_rslt17_m128i = _mm_loadu_si128(in + 16);
    __m128i c21_alignr_rslt13_1_m128i = _mm_alignr_epi8(c21_load_rslt17_m128i, c21_load_rslt16_m128i, 12);
    __m128i c21_load_rslt18_m128i = _mm_loadu_si128(in + 17);
    __m128i c21_alignr_rslt13_2_m128i = _mm_alignr_epi8(c21_load_rslt18_m128i, c21_load_rslt17_m128i, 6);
    __m256i c21_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt13_1_m128i), c21_alignr_rslt13_2_m128i, 1);
    hor_avx2_unpack8_c21<0, 0>(out, c21_insert_rslt13_m256i); // Unpack 13th 8 values.

    __m128i c21_load_rslt19_m128i = _mm_loadu_si128(in + 18);
    __m128i c21_alignr_rslt14_m128i = _mm_alignr_epi8(c21_load_rslt19_m128i, c21_load_rslt18_m128i, 11);
    __m256i c21_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_load_rslt18_m128i), c21_alignr_rslt14_m128i, 1);
    hor_avx2_unpack8_c21<1, 0>(out, c21_insert_rslt14_m256i); // Unpack 14th 8 values.

    __m128i c21_load_rslt20_m128i = _mm_loadu_si128(in + 19);
    __m128i c21_alignr_rslt15_m128i = _mm_alignr_epi8(c21_load_rslt20_m128i, c21_load_rslt19_m128i, 6);
    __m256i c21_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt15_m128i), c21_load_rslt20_m128i, 1);
    hor_avx2_unpack8_c21<0, 0>(out, c21_insert_rslt15_m256i); // Unpack 15th 8 values.

    __m128i c21_load_rslt21_m128i = _mm_loadu_si128(in + 20);
    __m128i c21_alignr_rslt16_m128i = _mm_alignr_epi8(c21_load_rslt21_m128i, c21_load_rslt20_m128i, 11);
    __m256i c21_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt16_m128i), c21_load_rslt21_m128i, 1);
    hor_avx2_unpack8_c21<0, 5>(out, c21_insert_rslt16_m256i); // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c21(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c21_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte2 + 10, byte2 + 9, byte2 + 8,
				byte2 + 8, byte2 + 7, byte2 + 6, byte2 + 5,
				0xFF, byte2 + 5, byte2 + 4, byte2 + 3,
				byte2 + 3, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 10, byte1 + 9, byte1 + 8, byte1 + 7,
				0xFF, byte1 + 7, byte1 + 6, byte1 + 5,
				byte1 + 5, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c21_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c21_shfl_msk_m256i);
		__m256i c21_srlv_rslt_m256i = _mm256_srlv_epi32(c21_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[21]);
		__m256i c21_rslt_m256i = _mm256_and_si256(c21_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[21]);
		_mm256_storeu_si256(out++, c21_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c21_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte2 + 10, byte2 + 9, byte2 + 8,
				byte2 + 8, byte2 + 7, byte2 + 6, byte2 + 5,
				0xFF, byte2 + 5, byte2 + 4, byte2 + 3,
				byte2 + 3, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 10, byte1 + 9, byte1 + 8, byte1 + 7,
				0xFF, byte1 + 7, byte1 + 6, byte1 + 5,
				byte1 + 5, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c21_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c21_shfl_msk_m256i);
		__m256i c21_srlv_rslt_m256i = _mm256_srlv_epi32(c21_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[21]);
		__m256i c21_and_rslt_m256i = _mm256_and_si256(c21_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[21]);
		__m256i c21_rslt_m256i = _mm256_or_si256(c21_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 21));
		_mm256_storeu_si256(out++, c21_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 21-bit values.
// * Load 21 SSE vectors, each containing 6 21-bit values. (7th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c21(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//    __m128i c21_load_rslt1_m128i = _mm_loadu_si128(in + 0);
//    __m128i c21_load_rslt2_m128i = _mm_loadu_si128(in + 1);
//    __m128i c21_alignr_rslt1_m128i = _mm_alignr_epi8(c21_load_rslt2_m128i, c21_load_rslt1_m128i, 10);
//    __m256i c21_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_load_rslt1_m128i), c21_alignr_rslt1_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt1_m256i); // Unpack 1st 8 values.
//
//    __m128i c21_load_rslt3_m128i = _mm_loadu_si128(in + 2);
//    __m128i c21_alignr_rslt2_1_m128i = _mm_alignr_epi8(c21_load_rslt3_m128i, c21_load_rslt2_m128i, 5);
//    __m128i c21_alignr_rslt2_2_m128i = _mm_alignr_epi8(c21_load_rslt3_m128i, c21_load_rslt2_m128i, 15);
//    __m256i c21_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt2_1_m128i), c21_alignr_rslt2_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt2_m256i); // Unpack 2nd 8 values.
//
//    __m128i c21_load_rslt4_m128i = _mm_loadu_si128(in + 3);
//    __m128i c21_alignr_rslt3_1_m128i = _mm_alignr_epi8(c21_load_rslt4_m128i, c21_load_rslt3_m128i, 10);
//    __m128i c21_alignr_rslt3_2_m128i = _mm_alignr_epi8(c21_load_rslt4_m128i, c21_load_rslt3_m128i, 20);
//    __m256i c21_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt3_1_m128i), c21_alignr_rslt3_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt3_m256i); // Unpack 3rd 8 values.
//
//    __m128i c21_load_rslt5_m128i = _mm_loadu_si128(in + 4);
//    __m128i c21_alignr_rslt4_1_m128i = _mm_alignr_epi8(c21_load_rslt5_m128i, c21_load_rslt4_m128i, 15);
//    __m128i c21_load_rslt6_m128i = _mm_loadu_si128(in + 5);
//    __m128i c21_alignr_rslt4_2_m128i = _mm_alignr_epi8(c21_load_rslt6_m128i, c21_load_rslt5_m128i, 9);
//    __m256i c21_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt4_1_m128i), c21_alignr_rslt4_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt4_m256i); // Unpack 4th 8 values.
//
//    __m128i c21_load_rslt7_m128i = _mm_loadu_si128(in + 6);
//    __m128i c21_alignr_rslt5_1_m128i = _mm_alignr_epi8(c21_load_rslt7_m128i, c21_load_rslt6_m128i, 4);
//    __m128i c21_alignr_rslt5_2_m128i = _mm_alignr_epi8(c21_load_rslt7_m128i, c21_load_rslt6_m128i, 14);
//    __m256i c21_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt5_1_m128i), c21_alignr_rslt5_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt5_m256i); // Unpack 5th 8 values.
//
//    __m128i c21_load_rslt8_m128i = _mm_loadu_si128(in + 7);
//    __m128i c21_alignr_rslt6_1_m128i = _mm_alignr_epi8(c21_load_rslt8_m128i, c21_load_rslt7_m128i, 9);
//    __m128i c21_alignr_rslt6_2_m128i = _mm_alignr_epi8(c21_load_rslt8_m128i, c21_load_rslt7_m128i, 19);
//    __m256i c21_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt6_1_m128i), c21_alignr_rslt6_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt6_m256i); // Unpack 6th 8 values.
//
//    __m128i c21_load_rslt9_m128i = _mm_loadu_si128(in + 8);
//    __m128i c21_alignr_rslt7_1_m128i = _mm_alignr_epi8(c21_load_rslt9_m128i, c21_load_rslt8_m128i, 14);
//    __m128i c21_load_rslt10_m128i = _mm_loadu_si128(in + 9);
//    __m128i c21_alignr_rslt7_2_m128i = _mm_alignr_epi8(c21_load_rslt10_m128i, c21_load_rslt9_m128i, 8);
//    __m256i c21_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt7_1_m128i), c21_alignr_rslt7_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt7_m256i); // Unpack 7th 8 values.
//
//    __m128i c21_load_rslt11_m128i = _mm_loadu_si128(in + 10);
//    __m128i c21_alignr_rslt8_1_m128i = _mm_alignr_epi8(c21_load_rslt11_m128i, c21_load_rslt10_m128i, 3);
//    __m128i c21_alignr_rslt8_2_m128i = _mm_alignr_epi8(c21_load_rslt11_m128i, c21_load_rslt10_m128i, 13);
//    __m256i c21_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt8_1_m128i), c21_alignr_rslt8_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt8_m256i); // Unpack 8th 8 values.
//
//    __m128i c21_load_rslt12_m128i = _mm_loadu_si128(in + 11);
//    __m128i c21_alignr_rslt9_1_m128i = _mm_alignr_epi8(c21_load_rslt12_m128i, c21_load_rslt11_m128i, 8);
//    __m128i c21_alignr_rslt9_2_m128i = _mm_alignr_epi8(c21_load_rslt12_m128i, c21_load_rslt11_m128i, 18);
//    __m256i c21_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt9_1_m128i), c21_alignr_rslt9_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt9_m256i); // Unpack 9th 8 values.
//
//    __m128i c21_load_rslt13_m128i = _mm_loadu_si128(in + 12);
//    __m128i c21_alignr_rslt10_1_m128i = _mm_alignr_epi8(c21_load_rslt13_m128i, c21_load_rslt12_m128i, 13);
//    __m128i c21_load_rslt14_m128i = _mm_loadu_si128(in + 13);
//    __m128i c21_alignr_rslt10_2_m128i = _mm_alignr_epi8(c21_load_rslt14_m128i, c21_load_rslt13_m128i, 7);
//    __m256i c21_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt10_1_m128i), c21_alignr_rslt10_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt10_m256i); // Unpack 10th 8 values.
//
//    __m128i c21_load_rslt15_m128i = _mm_loadu_si128(in + 14);
//    __m128i c21_alignr_rslt11_1_m128i = _mm_alignr_epi8(c21_load_rslt15_m128i, c21_load_rslt14_m128i, 2);
//    __m128i c21_alignr_rslt11_2_m128i = _mm_alignr_epi8(c21_load_rslt15_m128i, c21_load_rslt14_m128i, 12);
//    __m256i c21_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt11_1_m128i), c21_alignr_rslt11_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt11_m256i); // Unpack 11th 8 values.
//
//    __m128i c21_load_rslt16_m128i = _mm_loadu_si128(in + 15);
//    __m128i c21_alignr_rslt12_1_m128i = _mm_alignr_epi8(c21_load_rslt16_m128i, c21_load_rslt15_m128i, 7);
//    __m128i c21_alignr_rslt12_2_m128i = _mm_alignr_epi8(c21_load_rslt16_m128i, c21_load_rslt15_m128i, 17);
//    __m256i c21_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt12_1_m128i), c21_alignr_rslt12_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt12_m256i); // Unpack 12th 8 values.
//
//    __m128i c21_load_rslt17_m128i = _mm_loadu_si128(in + 16);
//    __m128i c21_alignr_rslt13_1_m128i = _mm_alignr_epi8(c21_load_rslt17_m128i, c21_load_rslt16_m128i, 12);
//    __m128i c21_load_rslt18_m128i = _mm_loadu_si128(in + 17);
//    __m128i c21_alignr_rslt13_2_m128i = _mm_alignr_epi8(c21_load_rslt18_m128i, c21_load_rslt17_m128i, 6);
//    __m256i c21_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt13_1_m128i), c21_alignr_rslt13_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt13_m256i); // Unpack 13th 8 values.
//
//    __m128i c21_load_rslt19_m128i = _mm_loadu_si128(in + 18);
//    __m128i c21_alignr_rslt14_1_m128i = _mm_alignr_epi8(c21_load_rslt19_m128i, c21_load_rslt18_m128i, 1);
//    __m128i c21_alignr_rslt14_2_m128i = _mm_alignr_epi8(c21_load_rslt19_m128i, c21_load_rslt18_m128i, 11);
//    __m256i c21_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt14_1_m128i), c21_alignr_rslt14_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt14_m256i); // Unpack 14th 8 values.
//
//    __m128i c21_load_rslt20_m128i = _mm_loadu_si128(in + 19);
//    __m128i c21_alignr_rslt15_1_m128i = _mm_alignr_epi8(c21_load_rslt20_m128i, c21_load_rslt19_m128i, 6);
//    __m128i c21_alignr_rslt15_2_m128i = _mm_alignr_epi8(c21_load_rslt20_m128i, c21_load_rslt19_m128i, 16);
//    __m256i c21_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt15_1_m128i), c21_alignr_rslt15_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt15_m256i); // Unpack 15th 8 values.
//
//    __m128i c21_load_rslt21_m128i = _mm_loadu_si128(in + 20);
//    __m128i c21_alignr_rslt16_1_m128i = _mm_alignr_epi8(c21_load_rslt21_m128i, c21_load_rslt20_m128i, 11);
//    __m128i c21_alignr_rslt16_2_m128i = _mm_alignr_epi8(c21_load_rslt21_m128i, c21_load_rslt20_m128i, 21);
//    __m256i c21_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c21_alignr_rslt16_1_m128i), c21_alignr_rslt16_2_m128i, 1);
//    hor_avx2_unpack8_c21(out, c21_insert_rslt16_m256i); // Unpack 16th 8 values.
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c21(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c21_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[21]);
//		__m256i c21_srlv_rslt_m256i = _mm256_srlv_epi32(c21_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[21]);
//		__m256i c21_rslt_m256i = _mm256_and_si256(c21_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[21]);
//		_mm256_storeu_si256(out++, c21_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c21_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[21]);
//		__m256i c21_srlv_rslt_m256i = _mm256_srlv_epi32(c21_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[21]);
//		__m256i c21_and_rslt_m256i = _mm256_and_si256(c21_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[21]);
//		__m256i c21_rslt_m256i = _mm256_or_si256(c21_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 21));
//		_mm256_storeu_si256(out++, c21_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 22-bit values.
 * Load 22 SSE vectors, each containing 5 22-bit values. (6th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c22(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
	     __m128i c22_load_rslt1_m128i = _mm_loadu_si128(in++);
	     __m128i c22_load_rslt2_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt1_m128i = _mm_alignr_epi8(c22_load_rslt2_m128i, c22_load_rslt1_m128i, 11);
	     __m256i c22_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_load_rslt1_m128i), c22_alignr_rslt1_m128i, 1);
	     hor_avx2_unpack8_c22<0, 0>(out, c22_insert_rslt1_m256i); // Unpack 1st 8 values.

	     __m128i c22_load_rslt3_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt2_m128i = _mm_alignr_epi8(c22_load_rslt3_m128i, c22_load_rslt2_m128i, 6);
	     __m256i c22_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_alignr_rslt2_m128i), c22_load_rslt3_m128i, 1);
	     hor_avx2_unpack8_c22<0, 1>(out, c22_insert_rslt2_m256i); // Unpack 2nd 8 values.

	     __m128i c22_load_rslt4_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt3_1_m128i = _mm_alignr_epi8(c22_load_rslt4_m128i, c22_load_rslt3_m128i, 12);
	     __m128i c22_load_rslt5_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt3_2_m128i = _mm_alignr_epi8(c22_load_rslt5_m128i, c22_load_rslt4_m128i, 7);
	     __m256i c22_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_alignr_rslt3_1_m128i), c22_alignr_rslt3_2_m128i, 1);
	     hor_avx2_unpack8_c22<0, 0>(out, c22_insert_rslt3_m256i); // Unpack 3rd 8 values.

	     __m128i c22_load_rslt6_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt4_m128i = _mm_alignr_epi8(c22_load_rslt6_m128i, c22_load_rslt5_m128i, 13);
	     __m256i c22_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_load_rslt5_m128i), c22_alignr_rslt4_m128i, 1);
	     hor_avx2_unpack8_c22<2, 0>(out, c22_insert_rslt4_m256i); // Unpack 4th 8 values.

	     __m128i c22_load_rslt7_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt5_m128i = _mm_alignr_epi8(c22_load_rslt7_m128i, c22_load_rslt6_m128i, 8);
	     __m256i c22_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_alignr_rslt5_m128i), c22_load_rslt7_m128i, 1);
	     hor_avx2_unpack8_c22<0, 3>(out, c22_insert_rslt5_m256i); // Unpack 5th 8 values.

	     __m128i c22_load_rslt8_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt6_1_m128i = _mm_alignr_epi8(c22_load_rslt8_m128i, c22_load_rslt7_m128i, 14);
	     __m128i c22_load_rslt9_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt6_2_m128i = _mm_alignr_epi8(c22_load_rslt9_m128i, c22_load_rslt8_m128i, 9);
	     __m256i c22_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_alignr_rslt6_1_m128i), c22_alignr_rslt6_2_m128i, 1);
	     hor_avx2_unpack8_c22<0, 0>(out, c22_insert_rslt6_m256i); // Unpack 6th 8 values.

	     __m128i c22_load_rslt10_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt7_m128i = _mm_alignr_epi8(c22_load_rslt10_m128i, c22_load_rslt9_m128i, 15);
	     __m256i c22_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_load_rslt9_m128i), c22_alignr_rslt7_m128i, 1);
	     hor_avx2_unpack8_c22<4, 0>(out, c22_insert_rslt7_m256i); // Unpack 6th 8 values.

	     __m128i c22_load_rslt11_m128i = _mm_loadu_si128(in++);
	     __m128i c22_alignr_rslt8_m128i = _mm_alignr_epi8(c22_load_rslt11_m128i, c22_load_rslt10_m128i, 10);
	     __m256i c22_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_alignr_rslt8_m128i), c22_load_rslt11_m128i, 1);
	     hor_avx2_unpack8_c22<0, 5>(out, c22_insert_rslt8_m256i); // Unpack 6th 8 values.
	}
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c22(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c22_shfl_msk_m256i = _mm256_set_epi8(
				byte2 + 11, byte2 + 10, byte2 + 9, byte2 + 8,
				byte2 + 8, byte2 + 7, byte2 + 6, byte2 + 5,
				byte2 + 5, byte2 + 4, byte2 + 3, byte2 + 2,
				0xFF, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 11, byte1 + 10, byte1 + 9, byte1 + 8,
				byte1 + 8, byte1 + 7, byte1 + 6, byte1 + 5,
				byte1 + 5, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c22_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c22_shfl_msk_m256i);
		__m256i c22_srlv_rslt_m256i = _mm256_srlv_epi32(c22_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[22]);
		__m256i c22_rslt_m256i = _mm256_and_si256(c22_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[22]);
		_mm256_storeu_si256(out++, c22_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c22_shfl_msk_m256i = _mm256_set_epi8(
				byte2 + 11, byte2 + 10, byte2 + 9, byte2 + 8,
				byte2 + 8, byte2 + 7, byte2 + 6, byte2 + 5,
				byte2 + 5, byte2 + 4, byte2 + 3, byte2 + 2,
				0xFF, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 11, byte1 + 10, byte1 + 9, byte1 + 8,
				byte1 + 8, byte1 + 7, byte1 + 6, byte1 + 5,
				byte1 + 5, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c22_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c22_shfl_msk_m256i);
		__m256i c22_srlv_rslt_m256i = _mm256_srlv_epi32(c22_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[22]);
		__m256i c22_and_rslt_m256i = _mm256_and_si256(c22_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[22]);
		__m256i c22_rslt_m256i = _mm256_or_si256(c22_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 22));
		_mm256_storeu_si256(out++, c22_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 22-bit values.
// * Load 22 SSE vectors, each containing 5 22-bit values. (6th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c22(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
//	     __m128i c22_load_rslt1_m128i = _mm_loadu_si128(in++);
//	     __m128i c22_load_rslt2_m128i = _mm_loadu_si128(in++);
//	     __m128i c22_alignr_rslt1_m128i = _mm_alignr_epi8(c22_load_rslt2_m128i, c22_load_rslt1_m128i, 11);
//	     __m256i c22_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_load_rslt1_m128i), c22_alignr_rslt1_m128i, 1);
//	     hor_avx2_unpack8_c22(out, c22_insert_rslt1_m256i); // Unpack 1st 8 values.
//
//	     __m128i c22_load_rslt3_m128i = _mm_loadu_si128(in++);
//	     __m128i c22_alignr_rslt2_1_m128i = _mm_alignr_epi8(c22_load_rslt3_m128i, c22_load_rslt2_m128i, 6);
//	     __m128i c22_alignr_rslt2_2_m128i = _mm_alignr_epi8(c22_load_rslt3_m128i, c22_load_rslt2_m128i, 17);
//	     __m256i c22_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_alignr_rslt2_1_m128i), c22_alignr_rslt2_2_m128i, 1);
//	     hor_avx2_unpack8_c22(out, c22_insert_rslt2_m256i); // Unpack 2nd 8 values.
//
//	     __m128i c22_load_rslt4_m128i = _mm_loadu_si128(in++);
//	     __m128i c22_alignr_rslt3_1_m128i = _mm_alignr_epi8(c22_load_rslt4_m128i, c22_load_rslt3_m128i, 12);
//	     __m128i c22_load_rslt5_m128i = _mm_loadu_si128(in++);
//	     __m128i c22_alignr_rslt3_2_m128i = _mm_alignr_epi8(c22_load_rslt5_m128i, c22_load_rslt4_m128i, 7);
//	     __m256i c22_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_alignr_rslt3_1_m128i), c22_alignr_rslt3_2_m128i, 1);
//	     hor_avx2_unpack8_c22(out, c22_insert_rslt3_m256i); // Unpack 3rd 8 values.
//
//	     __m128i c22_load_rslt6_m128i = _mm_loadu_si128(in++);
//	     __m128i c22_alignr_rslt4_1_m128i = _mm_alignr_epi8(c22_load_rslt6_m128i, c22_load_rslt5_m128i, 2);
//	     __m128i c22_alignr_rslt4_2_m128i = _mm_alignr_epi8(c22_load_rslt6_m128i, c22_load_rslt5_m128i, 13);
//	     __m256i c22_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_alignr_rslt4_1_m128i), c22_alignr_rslt4_2_m128i, 1);
//	     hor_avx2_unpack8_c22(out, c22_insert_rslt4_m256i); // Unpack 4th 8 values.
//
//	     __m128i c22_load_rslt7_m128i = _mm_loadu_si128(in++);
//	     __m128i c22_alignr_rslt5_1_m128i = _mm_alignr_epi8(c22_load_rslt7_m128i, c22_load_rslt6_m128i, 8);
//	     __m128i c22_alignr_rslt5_2_m128i = _mm_alignr_epi8(c22_load_rslt7_m128i, c22_load_rslt6_m128i, 19);
//	     __m256i c22_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_alignr_rslt5_1_m128i), c22_alignr_rslt5_2_m128i, 1);
//	     hor_avx2_unpack8_c22(out, c22_insert_rslt5_m256i); // Unpack 5th 8 values.
//
//	     __m128i c22_load_rslt8_m128i = _mm_loadu_si128(in++);
//	     __m128i c22_alignr_rslt6_1_m128i = _mm_alignr_epi8(c22_load_rslt8_m128i, c22_load_rslt7_m128i, 14);
//	     __m128i c22_load_rslt9_m128i = _mm_loadu_si128(in++);
//	     __m128i c22_alignr_rslt6_2_m128i = _mm_alignr_epi8(c22_load_rslt9_m128i, c22_load_rslt8_m128i, 9);
//	     __m256i c22_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_alignr_rslt6_1_m128i), c22_alignr_rslt6_2_m128i, 1);
//	     hor_avx2_unpack8_c22(out, c22_insert_rslt6_m256i); // Unpack 6th 8 values.
//
//	     __m128i c22_load_rslt10_m128i = _mm_loadu_si128(in++);
//	     __m128i c22_alignr_rslt7_1_m128i = _mm_alignr_epi8(c22_load_rslt10_m128i, c22_load_rslt9_m128i, 4);
//	     __m128i c22_alignr_rslt7_2_m128i = _mm_alignr_epi8(c22_load_rslt10_m128i, c22_load_rslt9_m128i, 15);
//	     __m256i c22_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_alignr_rslt7_1_m128i), c22_alignr_rslt7_2_m128i, 1);
//	     hor_avx2_unpack8_c22(out, c22_insert_rslt7_m256i); // Unpack 6th 8 values.
//
//	     __m128i c22_load_rslt11_m128i = _mm_loadu_si128(in++);
//	     __m128i c22_alignr_rslt8_1_m128i = _mm_alignr_epi8(c22_load_rslt11_m128i, c22_load_rslt10_m128i, 10);
//	     __m128i c22_alignr_rslt8_2_m128i = _mm_alignr_epi8(c22_load_rslt11_m128i, c22_load_rslt10_m128i, 21);
//	     __m256i c22_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c22_alignr_rslt8_1_m128i), c22_alignr_rslt8_2_m128i, 1);
//	     hor_avx2_unpack8_c22(out, c22_insert_rslt8_m256i); // Unpack 6th 8 values.
//	}
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c22(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c22_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[22]);
//		__m256i c22_srlv_rslt_m256i = _mm256_srlv_epi32(c22_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[22]);
//		__m256i c22_rslt_m256i = _mm256_and_si256(c22_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[22]);
//		_mm256_storeu_si256(out++, c22_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c22_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[22]);
//		__m256i c22_srlv_rslt_m256i = _mm256_srlv_epi32(c22_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[22]);
//		__m256i c22_and_rslt_m256i = _mm256_and_si256(c22_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[22]);
//		__m256i c22_rslt_m256i = _mm256_or_si256(c22_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 22));
//		_mm256_storeu_si256(out++, c22_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 23-bit values.
 * Load 23 SSE vectors, each containing 5 23-bit values. (6th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c23(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
    __m128i c23_load_rslt1_m128i = _mm_loadu_si128(in + 0);
    __m128i c23_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c23_alignr_rslt1_m128i = _mm_alignr_epi8(c23_load_rslt2_m128i, c23_load_rslt1_m128i, 11);
    __m256i c23_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_load_rslt1_m128i), c23_alignr_rslt1_m128i, 1);
     hor_avx2_unpack8_c23<0, 0>(out, c23_insert_rslt1_m256i); // Unpack 1st 8 values.

     __m128i c23_load_rslt3_m128i = _mm_loadu_si128(in + 2);
     __m128i c23_alignr_rslt2_m128i = _mm_alignr_epi8(c23_load_rslt3_m128i, c23_load_rslt2_m128i, 7);
     __m256i c23_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt2_m128i), c23_load_rslt3_m128i, 1);
     hor_avx2_unpack8_c23<0, 2>(out, c23_insert_rslt2_m256i); // Unpack 2nd 8 values.

     __m128i c23_load_rslt4_m128i = _mm_loadu_si128(in + 3);
     __m128i c23_alignr_rslt3_1_m128i = _mm_alignr_epi8(c23_load_rslt4_m128i, c23_load_rslt3_m128i, 14);
     __m128i c23_load_rslt5_m128i = _mm_loadu_si128(in + 4);
     __m128i c23_alignr_rslt3_2_m128i = _mm_alignr_epi8(c23_load_rslt5_m128i, c23_load_rslt4_m128i, 9);
     __m256i c23_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt3_1_m128i), c23_alignr_rslt3_2_m128i, 1);
     hor_avx2_unpack8_c23<0, 0>(out, c23_insert_rslt3_m256i); // Unpack 3rd 8 values.

     __m128i c23_load_rslt6_m128i = _mm_loadu_si128(in + 5);
     __m128i c23_alignr_rslt4_m128i = _mm_alignr_epi8(c23_load_rslt6_m128i, c23_load_rslt5_m128i, 5);
     __m256i c23_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt4_m128i), c23_load_rslt6_m128i, 1);
     hor_avx2_unpack8_c23<0, 0>(out, c23_insert_rslt4_m256i); // Unpack 4th 8 values.

     __m128i c23_load_rslt7_m128i = _mm_loadu_si128(in + 6);
     __m128i c23_alignr_rslt5_1_m128i = _mm_alignr_epi8(c23_load_rslt7_m128i, c23_load_rslt6_m128i, 12);
     __m128i c23_load_rslt8_m128i = _mm_loadu_si128(in + 7);
     __m128i c23_alignr_rslt5_2_m128i = _mm_alignr_epi8(c23_load_rslt8_m128i, c23_load_rslt7_m128i, 7);
     __m256i c23_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt5_1_m128i), c23_alignr_rslt5_2_m128i, 1);
     hor_avx2_unpack8_c23<0, 0>(out, c23_insert_rslt5_m256i); // Unpack 5th 8 values.

     __m128i c23_load_rslt9_m128i = _mm_loadu_si128(in + 8);
     __m128i c23_alignr_rslt6_m128i = _mm_alignr_epi8(c23_load_rslt9_m128i, c23_load_rslt8_m128i, 14);
     __m256i c23_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_load_rslt8_m128i), c23_alignr_rslt6_m128i, 1);
     hor_avx2_unpack8_c23<3, 0>(out, c23_insert_rslt6_m256i); // Unpack 6th 8 values.

     __m128i c23_load_rslt10_m128i = _mm_loadu_si128(in + 9);
     __m128i c23_alignr_rslt7_1_m128i = _mm_alignr_epi8(c23_load_rslt10_m128i, c23_load_rslt9_m128i, 10);
     __m128i c23_load_rslt11_m128i = _mm_loadu_si128(in + 10);
     __m128i c23_alignr_rslt7_2_m128i = _mm_alignr_epi8(c23_load_rslt11_m128i, c23_load_rslt10_m128i, 5);
     __m256i c23_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt7_1_m128i), c23_alignr_rslt7_2_m128i, 1);
     hor_avx2_unpack8_c23<0, 0>(out, c23_insert_rslt7_m256i); // Unpack 7th 8 values.

     __m128i c23_load_rslt12_m128i = _mm_loadu_si128(in + 11);
     __m128i c23_alignr_rslt8_m128i = _mm_alignr_epi8(c23_load_rslt12_m128i, c23_load_rslt11_m128i, 12);
     __m256i c23_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_load_rslt11_m128i), c23_alignr_rslt8_m128i, 1);
     hor_avx2_unpack8_c23<1, 0>(out, c23_insert_rslt8_m256i); // Unpack 8th 8 values.

     __m128i c23_load_rslt13_m128i = _mm_loadu_si128(in + 12);
     __m128i c23_alignr_rslt9_m128i = _mm_alignr_epi8(c23_load_rslt13_m128i, c23_load_rslt12_m128i, 8);
     __m256i c23_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt9_m128i), c23_load_rslt13_m128i, 1);
     hor_avx2_unpack8_c23<0, 3>(out, c23_insert_rslt9_m256i); // Unpack 9th 8 values.

     __m128i c23_load_rslt14_m128i = _mm_loadu_si128(in + 13);
     __m128i c23_alignr_rslt10_1_m128i = _mm_alignr_epi8(c23_load_rslt14_m128i, c23_load_rslt13_m128i, 15);
     __m128i c23_load_rslt15_m128i = _mm_loadu_si128(in + 14);
     __m128i c23_alignr_rslt10_2_m128i = _mm_alignr_epi8(c23_load_rslt15_m128i, c23_load_rslt14_m128i, 10);
     __m256i c23_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt10_1_m128i), c23_alignr_rslt10_2_m128i, 1);
     hor_avx2_unpack8_c23<0, 0>(out, c23_insert_rslt10_m256i); // Unpack 10th 8 values.

     __m128i c23_load_rslt16_m128i = _mm_loadu_si128(in + 15);
     __m128i c23_alignr_rslt11_m128i = _mm_alignr_epi8(c23_load_rslt16_m128i, c23_load_rslt15_m128i, 6);
     __m256i c23_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt11_m128i), c23_load_rslt16_m128i, 1);
     hor_avx2_unpack8_c23<0, 1>(out, c23_insert_rslt11_m256i); // Unpack 11th 8 values.

     __m128i c23_load_rslt17_m128i = _mm_loadu_si128(in + 16);
     __m128i c23_alignr_rslt12_1_m128i = _mm_alignr_epi8(c23_load_rslt17_m128i, c23_load_rslt16_m128i, 13);
     __m128i c23_load_rslt18_m128i = _mm_loadu_si128(in + 17);
     __m128i c23_alignr_rslt12_2_m128i = _mm_alignr_epi8(c23_load_rslt18_m128i, c23_load_rslt17_m128i, 8);
     __m256i c23_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt12_1_m128i), c23_alignr_rslt12_2_m128i, 1);
     hor_avx2_unpack8_c23<0, 0>(out, c23_insert_rslt12_m256i); // Unpack 12th 8 values.

     __m128i c23_load_rslt19_m128i = _mm_loadu_si128(in + 18);
     __m128i c23_alignr_rslt13_m128i = _mm_alignr_epi8(c23_load_rslt19_m128i, c23_load_rslt18_m128i, 15);
     __m256i c23_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_load_rslt18_m128i), c23_alignr_rslt13_m128i, 1);
     hor_avx2_unpack8_c23<4, 0>(out, c23_insert_rslt13_m256i); // Unpack 13th 8 values.

     __m128i c23_load_rslt20_m128i = _mm_loadu_si128(in + 19);
     __m128i c23_alignr_rslt14_1_m128i = _mm_alignr_epi8(c23_load_rslt20_m128i, c23_load_rslt19_m128i, 11);
     __m128i c23_load_rslt21_m128i = _mm_loadu_si128(in + 20);
     __m128i c23_alignr_rslt14_2_m128i = _mm_alignr_epi8(c23_load_rslt21_m128i, c23_load_rslt20_m128i, 6);
     __m256i c23_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt14_1_m128i), c23_alignr_rslt14_2_m128i, 1);
     hor_avx2_unpack8_c23<0, 0>(out, c23_insert_rslt14_m256i); // Unpack 14th 8 values.

     __m128i c23_load_rslt22_m128i = _mm_loadu_si128(in + 21);
     __m128i c23_alignr_rslt15_m128i = _mm_alignr_epi8(c23_load_rslt22_m128i, c23_load_rslt21_m128i, 13);
     __m256i c23_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_load_rslt21_m128i), c23_alignr_rslt15_m128i, 1);
     hor_avx2_unpack8_c23<2, 0>(out, c23_insert_rslt15_m256i); // Unpack 15th 8 values.

     __m128i c23_load_rslt23_m128i = _mm_loadu_si128(in + 22);
     __m128i c23_alignr_rslt16_m128i = _mm_alignr_epi8(c23_load_rslt23_m128i, c23_load_rslt22_m128i, 9);
     __m256i c23_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt16_m128i), c23_load_rslt23_m128i, 1);
     hor_avx2_unpack8_c23<0, 4>(out, c23_insert_rslt16_m256i); // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c23(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c23_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte2 + 11, byte2 + 10, byte2 + 9,
				byte2 + 9, byte2 + 8, byte2 + 7, byte2 + 6,
				byte2 + 6, byte2 + 5, byte2 + 4, byte2 + 3,
				byte2 + 3, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 11, byte1 + 10, byte1 + 9, byte1 + 8,
				byte1 + 8, byte1 + 7, byte1 + 6, byte1 + 5,
				byte1 + 5, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c23_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c23_shfl_msk_m256i);
		__m256i c23_srlv_rslt_m256i = _mm256_srlv_epi32(c23_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[23]);
		__m256i c23_rslt_m256i = _mm256_and_si256(c23_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[23]);
		_mm256_storeu_si256(out++, c23_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c23_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte2 + 11, byte2 + 10, byte2 + 9,
				byte2 + 9, byte2 + 8, byte2 + 7, byte2 + 6,
				byte2 + 6, byte2 + 5, byte2 + 4, byte2 + 3,
				byte2 + 3, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 11, byte1 + 10, byte1 + 9, byte1 + 8,
				byte1 + 8, byte1 + 7, byte1 + 6, byte1 + 5,
				byte1 + 5, byte1 + 4, byte1 + 3, byte1 + 2,
				0xFF, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c23_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c23_shfl_msk_m256i);
		__m256i c23_srlv_rslt_m256i = _mm256_srlv_epi32(c23_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[23]);
		__m256i c23_and_rslt_m256i = _mm256_and_si256(c23_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[23]);
		__m256i c23_rslt_m256i = _mm256_or_si256(c23_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 23));
		_mm256_storeu_si256(out++, c23_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 23-bit values.
// * Load 23 SSE vectors, each containing 5 23-bit values. (6th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c23(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//    __m128i c23_load_rslt1_m128i = _mm_loadu_si128(in + 0);
//    __m128i c23_load_rslt2_m128i = _mm_loadu_si128(in + 1);
//    __m128i c23_alignr_rslt1_m128i = _mm_alignr_epi8(c23_load_rslt2_m128i, c23_load_rslt1_m128i, 11);
//    __m256i c23_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_load_rslt1_m128i), c23_alignr_rslt1_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt1_m256i); // Unpack 1st 8 values.
//
//     __m128i c23_load_rslt3_m128i = _mm_loadu_si128(in + 2);
//     __m128i c23_alignr_rslt2_1_m128i = _mm_alignr_epi8(c23_load_rslt3_m128i, c23_load_rslt2_m128i, 7);
//     __m128i c23_alignr_rslt2_2_m128i = _mm_alignr_epi8(c23_load_rslt3_m128i, c23_load_rslt2_m128i, 18);
//     __m256i c23_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt2_1_m128i), c23_alignr_rslt2_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt2_m256i); // Unpack 2nd 8 values.
//
//     __m128i c23_load_rslt4_m128i = _mm_loadu_si128(in + 3);
//     __m128i c23_alignr_rslt3_1_m128i = _mm_alignr_epi8(c23_load_rslt4_m128i, c23_load_rslt3_m128i, 14);
//     __m128i c23_load_rslt5_m128i = _mm_loadu_si128(in + 4);
//     __m128i c23_alignr_rslt3_2_m128i = _mm_alignr_epi8(c23_load_rslt5_m128i, c23_load_rslt4_m128i, 9);
//     __m256i c23_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt3_1_m128i), c23_alignr_rslt3_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt3_m256i); // Unpack 3rd 8 values.
//
//     __m128i c23_load_rslt6_m128i = _mm_loadu_si128(in + 5);
//     __m128i c23_alignr_rslt4_1_m128i = _mm_alignr_epi8(c23_load_rslt6_m128i, c23_load_rslt5_m128i, 5);
//     __m128i c23_alignr_rslt4_2_m128i = _mm_alignr_epi8(c23_load_rslt6_m128i, c23_load_rslt5_m128i, 16);
//     __m256i c23_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt4_1_m128i), c23_alignr_rslt4_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt4_m256i); // Unpack 4th 8 values.
//
//     __m128i c23_load_rslt7_m128i = _mm_loadu_si128(in + 6);
//     __m128i c23_alignr_rslt5_1_m128i = _mm_alignr_epi8(c23_load_rslt7_m128i, c23_load_rslt6_m128i, 12);
//     __m128i c23_load_rslt8_m128i = _mm_loadu_si128(in + 7);
//     __m128i c23_alignr_rslt5_2_m128i = _mm_alignr_epi8(c23_load_rslt8_m128i, c23_load_rslt7_m128i, 7);
//     __m256i c23_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt5_1_m128i), c23_alignr_rslt5_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt5_m256i); // Unpack 5th 8 values.
//
//     __m128i c23_load_rslt9_m128i = _mm_loadu_si128(in + 8);
//     __m128i c23_alignr_rslt6_1_m128i = _mm_alignr_epi8(c23_load_rslt9_m128i, c23_load_rslt8_m128i, 3);
//     __m128i c23_alignr_rslt6_2_m128i = _mm_alignr_epi8(c23_load_rslt9_m128i, c23_load_rslt8_m128i, 14);
//     __m256i c23_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt6_1_m128i), c23_alignr_rslt6_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt6_m256i); // Unpack 6th 8 values.
//
//     __m128i c23_load_rslt10_m128i = _mm_loadu_si128(in + 9);
//     __m128i c23_alignr_rslt7_1_m128i = _mm_alignr_epi8(c23_load_rslt10_m128i, c23_load_rslt9_m128i, 10);
//     __m128i c23_load_rslt11_m128i = _mm_loadu_si128(in + 10);
//     __m128i c23_alignr_rslt7_2_m128i = _mm_alignr_epi8(c23_load_rslt11_m128i, c23_load_rslt10_m128i, 5);
//     __m256i c23_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt7_1_m128i), c23_alignr_rslt7_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt7_m256i); // Unpack 7th 8 values.
//
//     __m128i c23_load_rslt12_m128i = _mm_loadu_si128(in + 11);
//     __m128i c23_alignr_rslt8_1_m128i = _mm_alignr_epi8(c23_load_rslt12_m128i, c23_load_rslt11_m128i, 1);
//     __m128i c23_alignr_rslt8_2_m128i = _mm_alignr_epi8(c23_load_rslt12_m128i, c23_load_rslt11_m128i, 12);
//     __m256i c23_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt8_1_m128i), c23_alignr_rslt8_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt8_m256i); // Unpack 8th 8 values.
//
//     __m128i c23_load_rslt13_m128i = _mm_loadu_si128(in + 12);
//     __m128i c23_alignr_rslt9_1_m128i = _mm_alignr_epi8(c23_load_rslt13_m128i, c23_load_rslt12_m128i, 8);
//     __m128i c23_alignr_rslt9_2_m128i = _mm_alignr_epi8(c23_load_rslt13_m128i, c23_load_rslt12_m128i, 19);
//     __m256i c23_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt9_1_m128i), c23_alignr_rslt9_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt9_m256i); // Unpack 9th 8 values.
//
//     __m128i c23_load_rslt14_m128i = _mm_loadu_si128(in + 13);
//     __m128i c23_alignr_rslt10_1_m128i = _mm_alignr_epi8(c23_load_rslt14_m128i, c23_load_rslt13_m128i, 15);
//     __m128i c23_load_rslt15_m128i = _mm_loadu_si128(in + 14);
//     __m128i c23_alignr_rslt10_2_m128i = _mm_alignr_epi8(c23_load_rslt15_m128i, c23_load_rslt14_m128i, 10);
//     __m256i c23_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt10_1_m128i), c23_alignr_rslt10_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt10_m256i); // Unpack 10th 8 values.
//
//     __m128i c23_load_rslt16_m128i = _mm_loadu_si128(in + 15);
//     __m128i c23_alignr_rslt11_1_m128i = _mm_alignr_epi8(c23_load_rslt16_m128i, c23_load_rslt15_m128i, 6);
//     __m128i c23_alignr_rslt11_2_m128i = _mm_alignr_epi8(c23_load_rslt16_m128i, c23_load_rslt15_m128i, 17);
//     __m256i c23_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt11_1_m128i), c23_alignr_rslt11_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt11_m256i); // Unpack 11th 8 values.
//
//     __m128i c23_load_rslt17_m128i = _mm_loadu_si128(in + 16);
//     __m128i c23_alignr_rslt12_1_m128i = _mm_alignr_epi8(c23_load_rslt17_m128i, c23_load_rslt16_m128i, 13);
//     __m128i c23_load_rslt18_m128i = _mm_loadu_si128(in + 17);
//     __m128i c23_alignr_rslt12_2_m128i = _mm_alignr_epi8(c23_load_rslt18_m128i, c23_load_rslt17_m128i, 8);
//     __m256i c23_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt12_1_m128i), c23_alignr_rslt12_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt12_m256i); // Unpack 12th 8 values.
//
//     __m128i c23_load_rslt19_m128i = _mm_loadu_si128(in + 18);
//     __m128i c23_alignr_rslt13_1_m128i = _mm_alignr_epi8(c23_load_rslt19_m128i, c23_load_rslt18_m128i, 4);
//     __m128i c23_alignr_rslt13_2_m128i = _mm_alignr_epi8(c23_load_rslt19_m128i, c23_load_rslt18_m128i, 15);
//     __m256i c23_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt13_1_m128i), c23_alignr_rslt13_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt13_m256i); // Unpack 13th 8 values.
//
//     __m128i c23_load_rslt20_m128i = _mm_loadu_si128(in + 19);
//     __m128i c23_alignr_rslt14_1_m128i = _mm_alignr_epi8(c23_load_rslt20_m128i, c23_load_rslt19_m128i, 11);
//     __m128i c23_load_rslt21_m128i = _mm_loadu_si128(in + 20);
//     __m128i c23_alignr_rslt14_2_m128i = _mm_alignr_epi8(c23_load_rslt21_m128i, c23_load_rslt20_m128i, 6);
//     __m256i c23_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt14_1_m128i), c23_alignr_rslt14_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt14_m256i); // Unpack 14th 8 values.
//
//     __m128i c23_load_rslt22_m128i = _mm_loadu_si128(in + 21);
//     __m128i c23_alignr_rslt15_1_m128i = _mm_alignr_epi8(c23_load_rslt22_m128i, c23_load_rslt21_m128i, 2);
//     __m128i c23_alignr_rslt15_2_m128i = _mm_alignr_epi8(c23_load_rslt22_m128i, c23_load_rslt21_m128i, 13);
//     __m256i c23_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt15_1_m128i), c23_alignr_rslt15_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt15_m256i); // Unpack 15th 8 values.
//
//     __m128i c23_load_rslt23_m128i = _mm_loadu_si128(in + 22);
//     __m128i c23_alignr_rslt16_1_m128i = _mm_alignr_epi8(c23_load_rslt23_m128i, c23_load_rslt22_m128i, 9);
//     __m128i c23_alignr_rslt16_2_m128i = _mm_alignr_epi8(c23_load_rslt23_m128i, c23_load_rslt22_m128i, 20);
//     __m256i c23_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c23_alignr_rslt16_1_m128i), c23_alignr_rslt16_2_m128i, 1);
//     hor_avx2_unpack8_c23(out, c23_insert_rslt16_m256i); // Unpack 16th 8 values.
//
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c23(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c23_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[23]);
//		__m256i c23_srlv_rslt_m256i = _mm256_srlv_epi32(c23_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[23]);
//		__m256i c23_rslt_m256i = _mm256_and_si256(c23_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[23]);
//		_mm256_storeu_si256(out++, c23_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c23_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[23]);
//		__m256i c23_srlv_rslt_m256i = _mm256_srlv_epi32(c23_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[23]);
//		__m256i c23_and_rslt_m256i = _mm256_and_si256(c23_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[23]);
//		__m256i c23_rslt_m256i = _mm256_or_si256(c23_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 23));
//		_mm256_storeu_si256(out++, c23_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 24-bit values.
 * Load 24 SSE vectors, each containing 5 24-bit values. (6th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c24(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 16) {
		__m128i c24_load_rslt1_m128i = _mm_loadu_si128(in++);
		__m128i c24_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c24_alignr_rslt1_m128i = _mm_alignr_epi8(c24_load_rslt2_m128i, c24_load_rslt1_m128i, 12);
		__m256i c24_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c24_load_rslt1_m128i), c24_alignr_rslt1_m128i, 1);
		hor_avx2_unpack8_c24<0>(out, c24_insert_rslt1_m256i); // Unpack 1st 8 values.

		__m128i c24_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c24_alignr_rslt2_m128i = _mm_alignr_epi8(c24_load_rslt3_m128i, c24_load_rslt2_m128i, 8);
		__m256i c24_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c24_alignr_rslt2_m128i), c24_load_rslt3_m128i, 1);
		hor_avx2_unpack8_c24<4>(out, c24_insert_rslt2_m256i); // Unpack 2nd 8 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c24(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c24_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte + 11, byte + 10, byte + 9,
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, byte + 5, byte + 4, byte + 3,
				0xFF, byte + 2, byte + 1, byte + 0,
				0xFF, 11, 10, 9,
				0xFF, 8, 7, 6,
				0xFF, 5, 4, 3,
				0xFF, 2, 1, 0);
		__m256i c24_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c24_shfl_msk_m256i);
		_mm256_storeu_si256(out++, c24_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c24_shfl_msk_m256i = _mm256_set_epi8(
				0xFF, byte + 11, byte + 10, byte + 9,
				0xFF, byte + 8, byte + 7, byte + 6,
				0xFF, byte + 5, byte + 4, byte + 3,
				0xFF, byte + 2, byte + 1, byte + 0,
				0xFF, 11, 10, 9,
				0xFF, 8, 7, 6,
				0xFF, 5, 4, 3,
				0xFF, 2, 1, 0);
		__m256i c24_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c24_shfl_msk_m256i);
		__m256i c24_rslt_m256i = _mm256_or_si256(c24_shfl_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 24));
		_mm256_storeu_si256(out++, c24_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 24-bit values.
// * Load 24 SSE vectors, each containing 5 24-bit values. (6th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c24(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 16) {
//		__m128i c24_load_rslt1_m128i = _mm_loadu_si128(in++);
//		__m128i c24_load_rslt2_m128i = _mm_loadu_si128(in++);
//		__m128i c24_alignr_rslt1_m128i = _mm_alignr_epi8(c24_load_rslt2_m128i, c24_load_rslt1_m128i, 12);
//		__m256i c24_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c24_load_rslt1_m128i), c24_alignr_rslt1_m128i, 1);
//		hor_avx2_unpack8_c24(out, c24_insert_rslt1_m256i); // Unpack 1st 8 values.
//
//		__m128i c24_load_rslt3_m128i = _mm_loadu_si128(in++);
//		__m128i c24_alignr_rslt2_1_m128i = _mm_alignr_epi8(c24_load_rslt3_m128i, c24_load_rslt2_m128i, 8);
//		__m128i c24_alignr_rslt2_2_m128i = _mm_alignr_epi8(c24_load_rslt3_m128i, c24_load_rslt2_m128i, 20);
//		__m256i c24_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c24_alignr_rslt2_1_m128i), c24_alignr_rslt2_2_m128i, 1);
//		hor_avx2_unpack8_c24(out, c24_insert_rslt2_m256i); // Unpack 2nd 8 values.
//	}
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c24(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c24_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[24]);
//		_mm256_storeu_si256(out++, c24_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c24_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[24]);
//		__m256i c24_rslt_m256i = _mm256_or_si256(c24_shfl_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 24));
//		_mm256_storeu_si256(out++, c24_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 25-bit values.
 * Load 25 SSE vectors, each containing 5 25-bit values. (6th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c25(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
    __m128i c25_load_rslt1_m128i = _mm_loadu_si128(in + 0);
    __m128i c25_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c25_alignr_rslt1_m128i = _mm_alignr_epi8(c25_load_rslt2_m128i, c25_load_rslt1_m128i, 12);
    __m256i c25_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_load_rslt1_m128i), c25_alignr_rslt1_m128i, 1);
    hor_avx2_unpack8_c25<0, 0>(out, c25_insert_rslt1_m256i); // Unpack 1st 8 values.

    __m128i c25_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c25_alignr_rslt2_1_m128i = _mm_alignr_epi8(c25_load_rslt3_m128i, c25_load_rslt2_m128i, 9);
    __m128i c25_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c25_alignr_rslt2_2_m128i = _mm_alignr_epi8(c25_load_rslt4_m128i, c25_load_rslt3_m128i, 5);
    __m256i c25_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt2_1_m128i), c25_alignr_rslt2_2_m128i, 1);
    hor_avx2_unpack8_c25<0, 0>(out, c25_insert_rslt2_m256i); // Unpack 2nd 8 values.

    __m128i c25_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c25_alignr_rslt3_m128i = _mm_alignr_epi8(c25_load_rslt5_m128i, c25_load_rslt4_m128i, 14);
    __m256i c25_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_load_rslt4_m128i), c25_alignr_rslt3_m128i, 1);
    hor_avx2_unpack8_c25<2, 0>(out, c25_insert_rslt3_m256i); // Unpack 3rd 8 values.

    __m128i c25_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c25_alignr_rslt4_1_m128i = _mm_alignr_epi8(c25_load_rslt6_m128i, c25_load_rslt5_m128i, 11);
    __m128i c25_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c25_alignr_rslt4_2_m128i = _mm_alignr_epi8(c25_load_rslt7_m128i, c25_load_rslt6_m128i, 7);
    __m256i c25_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt4_1_m128i), c25_alignr_rslt4_2_m128i, 1);
    hor_avx2_unpack8_c25<0, 0>(out, c25_insert_rslt4_m256i); // Unpack 4th 8 values.

    __m128i c25_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c25_alignr_rslt5_m128i = _mm_alignr_epi8(c25_load_rslt8_m128i, c25_load_rslt7_m128i, 4);
    __m256i c25_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt5_m128i), c25_load_rslt8_m128i, 1);
    hor_avx2_unpack8_c25<0, 0>(out, c25_insert_rslt5_m256i); // Unpack 5th 8 values.

    __m128i c25_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c25_alignr_rslt6_1_m128i = _mm_alignr_epi8(c25_load_rslt9_m128i, c25_load_rslt8_m128i, 13);
    __m128i c25_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c25_alignr_rslt6_2_m128i = _mm_alignr_epi8(c25_load_rslt10_m128i, c25_load_rslt9_m128i, 9);
    __m256i c25_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt6_1_m128i), c25_alignr_rslt6_2_m128i, 1);
    hor_avx2_unpack8_c25<0, 0>(out, c25_insert_rslt6_m256i); // Unpack 6th 8 values.

    __m128i c25_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c25_alignr_rslt7_m128i = _mm_alignr_epi8(c25_load_rslt11_m128i, c25_load_rslt10_m128i, 6);
    __m256i c25_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt7_m128i), c25_load_rslt11_m128i, 1);
    hor_avx2_unpack8_c25<0, 2>(out, c25_insert_rslt7_m256i); // Unpack 7th 8 values.

    __m128i c25_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c25_alignr_rslt8_1_m128i = _mm_alignr_epi8(c25_load_rslt12_m128i, c25_load_rslt11_m128i, 15);
    __m128i c25_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c25_alignr_rslt8_2_m128i = _mm_alignr_epi8(c25_load_rslt13_m128i, c25_load_rslt12_m128i, 11);
    __m256i c25_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt8_1_m128i), c25_alignr_rslt8_2_m128i, 1);
    hor_avx2_unpack8_c25<0, 0>(out, c25_insert_rslt8_m256i); // Unpack 8th 8 values.

    __m128i c25_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c25_alignr_rslt9_1_m128i = _mm_alignr_epi8(c25_load_rslt14_m128i, c25_load_rslt13_m128i, 8);
    __m128i c25_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c25_alignr_rslt9_2_m128i = _mm_alignr_epi8(c25_load_rslt15_m128i, c25_load_rslt14_m128i, 4);
    __m256i c25_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt9_1_m128i), c25_alignr_rslt9_2_m128i, 1);
    hor_avx2_unpack8_c25<0, 0>(out, c25_insert_rslt9_m256i); // Unpack 9th 8 values.

    __m128i c25_load_rslt16_m128i = _mm_loadu_si128(in + 15);
    __m128i c25_alignr_rslt10_m128i = _mm_alignr_epi8(c25_load_rslt16_m128i, c25_load_rslt15_m128i, 13);
    __m256i c25_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_load_rslt15_m128i), c25_alignr_rslt10_m128i, 1);
    hor_avx2_unpack8_c25<1, 0>(out, c25_insert_rslt10_m256i); // Unpack 10th 8 values.

    __m128i c25_load_rslt17_m128i = _mm_loadu_si128(in + 16);
    __m128i c25_alignr_rslt11_1_m128i = _mm_alignr_epi8(c25_load_rslt17_m128i, c25_load_rslt16_m128i, 10);
    __m128i c25_load_rslt18_m128i = _mm_loadu_si128(in + 17);
    __m128i c25_alignr_rslt11_2_m128i = _mm_alignr_epi8(c25_load_rslt18_m128i, c25_load_rslt17_m128i, 6);
    __m256i c25_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt11_1_m128i), c25_alignr_rslt11_2_m128i, 1);
    hor_avx2_unpack8_c25<0, 0>(out, c25_insert_rslt11_m256i); // Unpack 11th 8 values.

    __m128i c25_load_rslt19_m128i = _mm_loadu_si128(in + 18);
    __m128i c25_alignr_rslt12_m128i = _mm_alignr_epi8(c25_load_rslt19_m128i, c25_load_rslt18_m128i, 15);
    __m256i c25_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_load_rslt18_m128i), c25_alignr_rslt12_m128i, 1);
    hor_avx2_unpack8_c25<3, 0>(out, c25_insert_rslt12_m256i); // Unpack 12th 8 values.

    __m128i c25_load_rslt20_m128i = _mm_loadu_si128(in + 19);
    __m128i c25_alignr_rslt13_1_m128i = _mm_alignr_epi8(c25_load_rslt20_m128i, c25_load_rslt19_m128i, 12);
    __m128i c25_load_rslt21_m128i = _mm_loadu_si128(in + 20);
    __m128i c25_alignr_rslt13_2_m128i = _mm_alignr_epi8(c25_load_rslt21_m128i, c25_load_rslt20_m128i, 8);
    __m256i c25_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt13_1_m128i), c25_alignr_rslt13_2_m128i, 1);
    hor_avx2_unpack8_c25<0, 0>(out, c25_insert_rslt13_m256i); // Unpack 13th 8 values.

    __m128i c25_load_rslt22_m128i = _mm_loadu_si128(in + 21);
    __m128i c25_alignr_rslt14_m128i = _mm_alignr_epi8(c25_load_rslt22_m128i, c25_load_rslt21_m128i, 5);
    __m256i c25_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt14_m128i), c25_load_rslt22_m128i, 1);
    hor_avx2_unpack8_c25<0, 1>(out, c25_insert_rslt14_m256i); // Unpack 14th 8 values.

    __m128i c25_load_rslt23_m128i = _mm_loadu_si128(in + 22);
    __m128i c25_alignr_rslt15_1_m128i = _mm_alignr_epi8(c25_load_rslt23_m128i, c25_load_rslt22_m128i, 14);
    __m128i c25_load_rslt24_m128i = _mm_loadu_si128(in + 23);
    __m128i c25_alignr_rslt15_2_m128i = _mm_alignr_epi8(c25_load_rslt24_m128i, c25_load_rslt23_m128i, 10);
    __m256i c25_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt15_1_m128i), c25_alignr_rslt15_2_m128i, 1);
    hor_avx2_unpack8_c25<0, 0>(out, c25_insert_rslt15_m256i); // Unpack 15th 8 values.

    __m128i c25_load_rslt25_m128i = _mm_loadu_si128(in + 24);
    __m128i c25_alignr_rslt16_m128i = _mm_alignr_epi8(c25_load_rslt25_m128i, c25_load_rslt24_m128i, 7);
    __m256i c25_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt16_m128i), c25_load_rslt25_m128i, 1);
    hor_avx2_unpack8_c25<0, 3>(out, c25_insert_rslt16_m256i); // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c25(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c25_shfl_msk_m256i = _mm256_set_epi8(
				byte2 + 12, byte2 + 11, byte2 + 10, byte2 + 9,
				byte2 + 9, byte2 + 8, byte2 + 7, byte2 + 6,
				byte2 + 6, byte2 + 5, byte2 + 4, byte2 + 3,
				byte2 + 3, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 12, byte1 + 11, byte1 + 10, byte1 + 9,
				byte1 + 9, byte1 + 8, byte1 + 7, byte1 + 6,
				byte1 + 6, byte1 + 5, byte1 + 4, byte1 + 3,
				byte1 + 3, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c25_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c25_shfl_msk_m256i);
		__m256i c25_srlv_rslt_m256i = _mm256_srlv_epi32(c25_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[25]);
		__m256i c25_rslt_m256i = _mm256_and_si256(c25_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[25]);
		_mm256_storeu_si256(out++, c25_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c25_shfl_msk_m256i = _mm256_set_epi8(
				byte2 + 12, byte2 + 11, byte2 + 10, byte2 + 9,
				byte2 + 9, byte2 + 8, byte2 + 7, byte2 + 6,
				byte2 + 6, byte2 + 5, byte2 + 4, byte2 + 3,
				byte2 + 3, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 12, byte1 + 11, byte1 + 10, byte1 + 9,
				byte1 + 9, byte1 + 8, byte1 + 7, byte1 + 6,
				byte1 + 6, byte1 + 5, byte1 + 4, byte1 + 3,
				byte1 + 3, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c25_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c25_shfl_msk_m256i);
		__m256i c25_srlv_rslt_m256i = _mm256_srlv_epi32(c25_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[25]);
		__m256i c25_and_rslt_m256i = _mm256_and_si256(c25_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[25]);
		__m256i c25_rslt_m256i = _mm256_or_si256(c25_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 25));
		_mm256_storeu_si256(out++, c25_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 25-bit values.
// * Load 25 SSE vectors, each containing 5 25-bit values. (6th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c25(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//    __m128i c25_load_rslt1_m128i = _mm_loadu_si128(in + 0);
//    __m128i c25_load_rslt2_m128i = _mm_loadu_si128(in + 1);
//    __m128i c25_alignr_rslt1_m128i = _mm_alignr_epi8(c25_load_rslt2_m128i, c25_load_rslt1_m128i, 12);
//    __m256i c25_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_load_rslt1_m128i), c25_alignr_rslt1_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt1_m256i); // Unpack 1st 8 values.
//
//    __m128i c25_load_rslt3_m128i = _mm_loadu_si128(in + 2);
//    __m128i c25_alignr_rslt2_1_m128i = _mm_alignr_epi8(c25_load_rslt3_m128i, c25_load_rslt2_m128i, 9);
//    __m128i c25_load_rslt4_m128i = _mm_loadu_si128(in + 3);
//    __m128i c25_alignr_rslt2_2_m128i = _mm_alignr_epi8(c25_load_rslt4_m128i, c25_load_rslt3_m128i, 5);
//    __m256i c25_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt2_1_m128i), c25_alignr_rslt2_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt2_m256i); // Unpack 2nd 8 values.
//
//    __m128i c25_load_rslt5_m128i = _mm_loadu_si128(in + 4);
//    __m128i c25_alignr_rslt3_1_m128i = _mm_alignr_epi8(c25_load_rslt5_m128i, c25_load_rslt4_m128i, 2);
//    __m128i c25_alignr_rslt3_2_m128i = _mm_alignr_epi8(c25_load_rslt5_m128i, c25_load_rslt4_m128i, 14);
//    __m256i c25_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt3_1_m128i), c25_alignr_rslt3_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt3_m256i); // Unpack 3rd 8 values.
//
//    __m128i c25_load_rslt6_m128i = _mm_loadu_si128(in + 5);
//    __m128i c25_alignr_rslt4_1_m128i = _mm_alignr_epi8(c25_load_rslt6_m128i, c25_load_rslt5_m128i, 11);
//    __m128i c25_load_rslt7_m128i = _mm_loadu_si128(in + 6);
//    __m128i c25_alignr_rslt4_2_m128i = _mm_alignr_epi8(c25_load_rslt7_m128i, c25_load_rslt6_m128i, 7);
//    __m256i c25_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt4_1_m128i), c25_alignr_rslt4_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt4_m256i); // Unpack 4th 8 values.
//
//    __m128i c25_load_rslt8_m128i = _mm_loadu_si128(in + 7);
//    __m128i c25_alignr_rslt5_1_m128i = _mm_alignr_epi8(c25_load_rslt8_m128i, c25_load_rslt7_m128i, 4);
//    __m128i c25_alignr_rslt5_2_m128i = _mm_alignr_epi8(c25_load_rslt8_m128i, c25_load_rslt7_m128i, 16);
//    __m256i c25_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt5_1_m128i), c25_alignr_rslt5_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt5_m256i); // Unpack 5th 8 values.
//
//    __m128i c25_load_rslt9_m128i = _mm_loadu_si128(in + 8);
//    __m128i c25_alignr_rslt6_1_m128i = _mm_alignr_epi8(c25_load_rslt9_m128i, c25_load_rslt8_m128i, 13);
//    __m128i c25_load_rslt10_m128i = _mm_loadu_si128(in + 9);
//    __m128i c25_alignr_rslt6_2_m128i = _mm_alignr_epi8(c25_load_rslt10_m128i, c25_load_rslt9_m128i, 9);
//    __m256i c25_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt6_1_m128i), c25_alignr_rslt6_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt6_m256i); // Unpack 6th 8 values.
//
//    __m128i c25_load_rslt11_m128i = _mm_loadu_si128(in + 10);
//    __m128i c25_alignr_rslt7_1_m128i = _mm_alignr_epi8(c25_load_rslt11_m128i, c25_load_rslt10_m128i, 6);
//    __m128i c25_alignr_rslt7_2_m128i = _mm_alignr_epi8(c25_load_rslt11_m128i, c25_load_rslt10_m128i, 18);
//    __m256i c25_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt7_1_m128i), c25_alignr_rslt7_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt7_m256i); // Unpack 7th 8 values.
//
//    __m128i c25_load_rslt12_m128i = _mm_loadu_si128(in + 11);
//    __m128i c25_alignr_rslt8_1_m128i = _mm_alignr_epi8(c25_load_rslt12_m128i, c25_load_rslt11_m128i, 15);
//    __m128i c25_load_rslt13_m128i = _mm_loadu_si128(in + 12);
//    __m128i c25_alignr_rslt8_2_m128i = _mm_alignr_epi8(c25_load_rslt13_m128i, c25_load_rslt12_m128i, 11);
//    __m256i c25_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt8_1_m128i), c25_alignr_rslt8_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt8_m256i); // Unpack 8th 8 values.
//
//    __m128i c25_load_rslt14_m128i = _mm_loadu_si128(in + 13);
//    __m128i c25_alignr_rslt9_1_m128i = _mm_alignr_epi8(c25_load_rslt14_m128i, c25_load_rslt13_m128i, 8);
//    __m128i c25_load_rslt15_m128i = _mm_loadu_si128(in + 14);
//    __m128i c25_alignr_rslt9_2_m128i = _mm_alignr_epi8(c25_load_rslt15_m128i, c25_load_rslt14_m128i, 4);
//    __m256i c25_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt9_1_m128i), c25_alignr_rslt9_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt9_m256i); // Unpack 9th 8 values.
//
//    __m128i c25_load_rslt16_m128i = _mm_loadu_si128(in + 15);
//    __m128i c25_alignr_rslt10_1_m128i = _mm_alignr_epi8(c25_load_rslt16_m128i, c25_load_rslt15_m128i, 1);
//    __m128i c25_alignr_rslt10_2_m128i = _mm_alignr_epi8(c25_load_rslt16_m128i, c25_load_rslt15_m128i, 13);
//    __m256i c25_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt10_1_m128i), c25_alignr_rslt10_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt10_m256i); // Unpack 10th 8 values.
//
//    __m128i c25_load_rslt17_m128i = _mm_loadu_si128(in + 16);
//    __m128i c25_alignr_rslt11_1_m128i = _mm_alignr_epi8(c25_load_rslt17_m128i, c25_load_rslt16_m128i, 10);
//    __m128i c25_load_rslt18_m128i = _mm_loadu_si128(in + 17);
//    __m128i c25_alignr_rslt11_2_m128i = _mm_alignr_epi8(c25_load_rslt18_m128i, c25_load_rslt17_m128i, 6);
//    __m256i c25_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt11_1_m128i), c25_alignr_rslt11_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt11_m256i); // Unpack 11th 8 values.
//
//    __m128i c25_load_rslt19_m128i = _mm_loadu_si128(in + 18);
//    __m128i c25_alignr_rslt12_1_m128i = _mm_alignr_epi8(c25_load_rslt19_m128i, c25_load_rslt18_m128i, 3);
//    __m128i c25_alignr_rslt12_2_m128i = _mm_alignr_epi8(c25_load_rslt19_m128i, c25_load_rslt18_m128i, 15);
//    __m256i c25_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt12_1_m128i), c25_alignr_rslt12_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt12_m256i); // Unpack 12th 8 values.
//
//    __m128i c25_load_rslt20_m128i = _mm_loadu_si128(in + 19);
//    __m128i c25_alignr_rslt13_1_m128i = _mm_alignr_epi8(c25_load_rslt20_m128i, c25_load_rslt19_m128i, 12);
//    __m128i c25_load_rslt21_m128i = _mm_loadu_si128(in + 20);
//    __m128i c25_alignr_rslt13_2_m128i = _mm_alignr_epi8(c25_load_rslt21_m128i, c25_load_rslt20_m128i, 8);
//    __m256i c25_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt13_1_m128i), c25_alignr_rslt13_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt13_m256i); // Unpack 13th 8 values.
//
//    __m128i c25_load_rslt22_m128i = _mm_loadu_si128(in + 21);
//    __m128i c25_alignr_rslt14_1_m128i = _mm_alignr_epi8(c25_load_rslt22_m128i, c25_load_rslt21_m128i, 5);
//    __m128i c25_alignr_rslt14_2_m128i = _mm_alignr_epi8(c25_load_rslt22_m128i, c25_load_rslt21_m128i, 17);
//    __m256i c25_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt14_1_m128i), c25_alignr_rslt14_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt14_m256i); // Unpack 14th 8 values.
//
//    __m128i c25_load_rslt23_m128i = _mm_loadu_si128(in + 22);
//    __m128i c25_alignr_rslt15_1_m128i = _mm_alignr_epi8(c25_load_rslt23_m128i, c25_load_rslt22_m128i, 14);
//    __m128i c25_load_rslt24_m128i = _mm_loadu_si128(in + 23);
//    __m128i c25_alignr_rslt15_2_m128i = _mm_alignr_epi8(c25_load_rslt24_m128i, c25_load_rslt23_m128i, 10);
//    __m256i c25_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt15_1_m128i), c25_alignr_rslt15_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt15_m256i); // Unpack 15th 8 values.
//
//    __m128i c25_load_rslt25_m128i = _mm_loadu_si128(in + 24);
//    __m128i c25_alignr_rslt16_1_m128i = _mm_alignr_epi8(c25_load_rslt25_m128i, c25_load_rslt24_m128i, 7);
//    __m128i c25_alignr_rslt16_2_m128i = _mm_alignr_epi8(c25_load_rslt25_m128i, c25_load_rslt24_m128i, 19);
//    __m256i c25_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c25_alignr_rslt16_1_m128i), c25_alignr_rslt16_2_m128i, 1);
//    hor_avx2_unpack8_c25(out, c25_insert_rslt16_m256i); // Unpack 16th 8 values.
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c25(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c25_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[25]);
//		__m256i c25_srlv_rslt_m256i = _mm256_srlv_epi32(c25_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[25]);
//		__m256i c25_rslt_m256i = _mm256_and_si256(c25_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[25]);
//		_mm256_storeu_si256(out++, c25_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c25_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[25]);
//		__m256i c25_srlv_rslt_m256i = _mm256_srlv_epi32(c25_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[25]);
//		__m256i c25_and_rslt_m256i = _mm256_and_si256(c25_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[25]);
//		__m256i c25_rslt_m256i = _mm256_or_si256(c25_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 25));
//		_mm256_storeu_si256(out++, c25_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 26-bit values.
 * Load 26 SSE vectors, each containing 4 26-bit values. (5th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c26(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
		__m128i c26_load_rslt1_m128i = _mm_loadu_si128(in++);
		__m128i c26_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt1_m128i = _mm_alignr_epi8(c26_load_rslt2_m128i, c26_load_rslt1_m128i, 13);
		__m256i c26_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_load_rslt1_m128i), c26_alignr_rslt1_m128i, 1);
		hor_avx2_unpack8_c26<0, 0>(out, c26_insert_rslt1_m256i); // Unpack 1st 8 values.

		__m128i c26_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt2_1_m128i = _mm_alignr_epi8(c26_load_rslt3_m128i, c26_load_rslt2_m128i, 10);
		__m128i c26_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt2_2_m128i = _mm_alignr_epi8(c26_load_rslt4_m128i, c26_load_rslt3_m128i, 7);
		__m256i c26_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt2_1_m128i), c26_alignr_rslt2_2_m128i, 1);
		hor_avx2_unpack8_c26<0, 0>(out, c26_insert_rslt2_m256i); // Unpack 2nd 8 values.

		__m128i c26_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt3_m128i = _mm_alignr_epi8(c26_load_rslt5_m128i, c26_load_rslt4_m128i, 4);
		__m256i c26_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt3_m128i), c26_load_rslt5_m128i, 1);
		hor_avx2_unpack8_c26<0, 1>(out, c26_insert_rslt3_m256i); // Unpack 3rd 8 values.

		__m128i c26_load_rslt6_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt4_1_m128i = _mm_alignr_epi8(c26_load_rslt6_m128i, c26_load_rslt5_m128i, 14);
		__m128i c26_load_rslt7_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt4_2_m128i = _mm_alignr_epi8(c26_load_rslt7_m128i, c26_load_rslt6_m128i, 11);
		__m256i c26_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt4_1_m128i), c26_alignr_rslt4_2_m128i, 1);
		hor_avx2_unpack8_c26<0, 0>(out, c26_insert_rslt4_m256i); // Unpack 4th 8 values.

		__m128i c26_load_rslt8_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt5_1_m128i = _mm_alignr_epi8(c26_load_rslt8_m128i, c26_load_rslt7_m128i, 8);
		__m128i c26_load_rslt9_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt5_2_m128i = _mm_alignr_epi8(c26_load_rslt9_m128i, c26_load_rslt8_m128i, 5);
		__m256i c26_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt5_1_m128i), c26_alignr_rslt5_2_m128i, 1);
		hor_avx2_unpack8_c26<0, 0>(out, c26_insert_rslt5_m256i); // Unpack 5th 8 values.

		__m128i c26_load_rslt10_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt6_m128i = _mm_alignr_epi8(c26_load_rslt10_m128i, c26_load_rslt9_m128i, 15);
		__m256i c26_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_load_rslt9_m128i), c26_alignr_rslt6_m128i, 1);
		hor_avx2_unpack8_c26<2, 0>(out, c26_insert_rslt6_m256i); // Unpack 6th 8 values.

		__m128i c26_load_rslt11_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt7_1_m128i = _mm_alignr_epi8(c26_load_rslt11_m128i, c26_load_rslt10_m128i, 12);
		__m128i c26_load_rslt12_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt7_2_m128i = _mm_alignr_epi8(c26_load_rslt12_m128i, c26_load_rslt11_m128i, 9);
		__m256i c26_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt7_1_m128i), c26_alignr_rslt7_2_m128i, 1);
		hor_avx2_unpack8_c26<0, 0>(out, c26_insert_rslt7_m256i); // Unpack 7th 8 values.

		__m128i c26_load_rslt13_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt8_m128i = _mm_alignr_epi8(c26_load_rslt13_m128i, c26_load_rslt12_m128i, 6);
		__m256i c26_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt8_m128i), c26_load_rslt13_m128i, 1);
		hor_avx2_unpack8_c26<0, 3>(out, c26_insert_rslt8_m256i); // Unpack 8th 8 values.
	}
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c26(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c26_shfl_msk_m256i = _mm256_set_epi8(
				byte2 + 12, byte2 + 11, byte2 + 10, byte2 + 9,
				byte2 + 9, byte2 + 8, byte2 + 7, byte2 + 6,
				byte2 + 6, byte2 + 5, byte2 + 4, byte2 + 3,
				byte2 + 3, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 12, byte1 + 11, byte1 + 10, byte1 + 9,
				byte1 + 9, byte1 + 8, byte1 + 7, byte1 + 6,
				byte1 + 6, byte1 + 5, byte1 + 4, byte1 + 3,
				byte1 + 3, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c26_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c26_shfl_msk_m256i);
		__m256i c26_srlv_rslt_m256i = _mm256_srlv_epi32(c26_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[26]);
		__m256i c26_rslt_m256i = _mm256_and_si256(c26_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[26]);
		_mm256_storeu_si256(out++, c26_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c26_shfl_msk_m256i = _mm256_set_epi8(
				byte2 + 12, byte2 + 11, byte2 + 10, byte2 + 9,
				byte2 + 9, byte2 + 8, byte2 + 7, byte2 + 6,
				byte2 + 6, byte2 + 5, byte2 + 4, byte2 + 3,
				byte2 + 3, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 12, byte1 + 11, byte1 + 10, byte1 + 9,
				byte1 + 9, byte1 + 8, byte1 + 7, byte1 + 6,
				byte1 + 6, byte1 + 5, byte1 + 4, byte1 + 3,
				byte1 + 3, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c26_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c26_shfl_msk_m256i);
		__m256i c26_srlv_rslt_m256i = _mm256_srlv_epi32(c26_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[26]);
		__m256i c26_and_rslt_m256i = _mm256_and_si256(c26_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[26]);
		__m256i c26_rslt_m256i = _mm256_or_si256(c26_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 26));
		_mm256_storeu_si256(out++, c26_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 26-bit values.
// * Load 26 SSE vectors, each containing 4 26-bit values. (5th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c26(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
//		__m128i c26_load_rslt1_m128i = _mm_loadu_si128(in++);
//		__m128i c26_load_rslt2_m128i = _mm_loadu_si128(in++);
//		__m128i c26_alignr_rslt1_m128i = _mm_alignr_epi8(c26_load_rslt2_m128i, c26_load_rslt1_m128i, 13);
//		__m256i c26_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_load_rslt1_m128i), c26_alignr_rslt1_m128i, 1);
//		hor_avx2_unpack8_c26(out, c26_insert_rslt1_m256i); // Unpack 1st 8 values.
//
//		__m128i c26_load_rslt3_m128i = _mm_loadu_si128(in++);
//		__m128i c26_alignr_rslt2_1_m128i = _mm_alignr_epi8(c26_load_rslt3_m128i, c26_load_rslt2_m128i, 10);
//		__m128i c26_load_rslt4_m128i = _mm_loadu_si128(in++);
//		__m128i c26_alignr_rslt2_2_m128i = _mm_alignr_epi8(c26_load_rslt4_m128i, c26_load_rslt3_m128i, 7);
//		__m256i c26_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt2_1_m128i), c26_alignr_rslt2_2_m128i, 1);
//		hor_avx2_unpack8_c26(out, c26_insert_rslt2_m256i); // Unpack 2nd 8 values.
//
//		__m128i c26_load_rslt5_m128i = _mm_loadu_si128(in++);
//		__m128i c26_alignr_rslt3_1_m128i = _mm_alignr_epi8(c26_load_rslt5_m128i, c26_load_rslt4_m128i, 4);
//		__m128i c26_alignr_rslt3_2_m128i = _mm_alignr_epi8(c26_load_rslt5_m128i, c26_load_rslt4_m128i, 17);
//		__m256i c26_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt3_1_m128i), c26_alignr_rslt3_2_m128i, 1);
//		hor_avx2_unpack8_c26(out, c26_insert_rslt3_m256i); // Unpack 3rd 8 values.
//
//		__m128i c26_load_rslt6_m128i = _mm_loadu_si128(in++);
//		__m128i c26_alignr_rslt4_1_m128i = _mm_alignr_epi8(c26_load_rslt6_m128i, c26_load_rslt5_m128i, 14);
//		__m128i c26_load_rslt7_m128i = _mm_loadu_si128(in++);
//		__m128i c26_alignr_rslt4_2_m128i = _mm_alignr_epi8(c26_load_rslt7_m128i, c26_load_rslt6_m128i, 11);
//		__m256i c26_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt4_1_m128i), c26_alignr_rslt4_2_m128i, 1);
//		hor_avx2_unpack8_c26(out, c26_insert_rslt4_m256i); // Unpack 4th 8 values.
//
//		__m128i c26_load_rslt8_m128i = _mm_loadu_si128(in++);
//		__m128i c26_alignr_rslt5_1_m128i = _mm_alignr_epi8(c26_load_rslt8_m128i, c26_load_rslt7_m128i, 8);
//		__m128i c26_load_rslt9_m128i = _mm_loadu_si128(in++);
//		__m128i c26_alignr_rslt5_2_m128i = _mm_alignr_epi8(c26_load_rslt9_m128i, c26_load_rslt8_m128i, 5);
//		__m256i c26_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt5_1_m128i), c26_alignr_rslt5_2_m128i, 1);
//		hor_avx2_unpack8_c26(out, c26_insert_rslt5_m256i); // Unpack 5th 8 values.
//
//		__m128i c26_load_rslt10_m128i = _mm_loadu_si128(in++);
//		__m128i c26_alignr_rslt6_1_m128i = _mm_alignr_epi8(c26_load_rslt10_m128i, c26_load_rslt9_m128i, 2);
//		__m128i c26_alignr_rslt6_2_m128i = _mm_alignr_epi8(c26_load_rslt10_m128i, c26_load_rslt9_m128i, 15);
//		__m256i c26_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt6_1_m128i), c26_alignr_rslt6_2_m128i, 1);
//		hor_avx2_unpack8_c26(out, c26_insert_rslt6_m256i); // Unpack 6th 8 values.
//
//		__m128i c26_load_rslt11_m128i = _mm_loadu_si128(in++);
//		__m128i c26_alignr_rslt7_1_m128i = _mm_alignr_epi8(c26_load_rslt11_m128i, c26_load_rslt10_m128i, 12);
//		__m128i c26_load_rslt12_m128i = _mm_loadu_si128(in++);
//		__m128i c26_alignr_rslt7_2_m128i = _mm_alignr_epi8(c26_load_rslt12_m128i, c26_load_rslt11_m128i, 9);
//		__m256i c26_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt7_1_m128i), c26_alignr_rslt7_2_m128i, 1);
//		hor_avx2_unpack8_c26(out, c26_insert_rslt7_m256i); // Unpack 7th 8 values.
//
//		__m128i c26_load_rslt13_m128i = _mm_loadu_si128(in++);
//		__m128i c26_alignr_rslt8_1_m128i = _mm_alignr_epi8(c26_load_rslt13_m128i, c26_load_rslt12_m128i, 6);
//		__m128i c26_alignr_rslt8_2_m128i = _mm_alignr_epi8(c26_load_rslt13_m128i, c26_load_rslt12_m128i, 19);
//		__m256i c26_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c26_alignr_rslt8_1_m128i), c26_alignr_rslt8_2_m128i, 1);
//		hor_avx2_unpack8_c26(out, c26_insert_rslt8_m256i); // Unpack 8th 8 values.
//	}
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c26(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c26_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[26]);
//		__m256i c26_srlv_rslt_m256i = _mm256_srlv_epi32(c26_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[26]);
//		__m256i c26_rslt_m256i = _mm256_and_si256(c26_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[26]);
//		_mm256_storeu_si256(out++, c26_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c26_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[26]);
//		__m256i c26_srlv_rslt_m256i = _mm256_srlv_epi32(c26_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[26]);
//		__m256i c26_and_rslt_m256i = _mm256_and_si256(c26_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[26]);
//		__m256i c26_rslt_m256i = _mm256_or_si256(c26_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 26));
//		_mm256_storeu_si256(out++, c26_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 27-bit values.
 * Load 27 SSE vectors, each containing 4 27-bit values. (5th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c27(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c27_load_rslt1_m128i = _mm_loadu_si128(in + 0);
    __m128i c27_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c27_alignr_rslt1_m128i = _mm_alignr_epi8(c27_load_rslt2_m128i, c27_load_rslt1_m128i, 13);
    __m256i c27_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_load_rslt1_m128i), c27_alignr_rslt1_m128i, 1);
    hor_avx2_unpack8_c27<0, 0>(out, c27_insert_rslt1_m256i); // Unpack 1st 8 values.

    __m128i c27_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c27_alignr_rslt2_1_m128i = _mm_alignr_epi8(c27_load_rslt3_m128i, c27_load_rslt2_m128i, 11);
    __m128i c27_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c27_alignr_rslt2_2_m128i = _mm_alignr_epi8(c27_load_rslt4_m128i, c27_load_rslt3_m128i, 8);
    __m256i c27_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt2_1_m128i), c27_alignr_rslt2_2_m128i, 1);
    hor_avx2_unpack8_c27<0, 0>(out, c27_insert_rslt2_m256i); // Unpack 2nd 8 values.

    __m128i c27_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c27_alignr_rslt3_1_m128i = _mm_alignr_epi8(c27_load_rslt5_m128i, c27_load_rslt4_m128i, 6);
    __m128i c27_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c27_alignr_rslt3_2_m128i = _mm_alignr_epi8(c27_load_rslt6_m128i, c27_load_rslt5_m128i, 3);
    __m256i c27_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt3_1_m128i), c27_alignr_rslt3_2_m128i, 1);
    hor_avx2_unpack8_c27<0, 0>(out, c27_insert_rslt3_m256i); // Unpack 3rd 8 values.

    __m128i c27_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c27_alignr_rslt4_m128i = _mm_alignr_epi8(c27_load_rslt7_m128i, c27_load_rslt6_m128i, 14);
    __m256i c27_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_load_rslt6_m128i), c27_alignr_rslt4_m128i, 1);
    hor_avx2_unpack8_c27<1, 0>(out, c27_insert_rslt4_m256i); // Unpack 4th 8 values.

    __m128i c27_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c27_alignr_rslt5_1_m128i = _mm_alignr_epi8(c27_load_rslt8_m128i, c27_load_rslt7_m128i, 12);
    __m128i c27_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c27_alignr_rslt5_2_m128i = _mm_alignr_epi8(c27_load_rslt9_m128i, c27_load_rslt8_m128i, 9);
    __m256i c27_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt5_1_m128i), c27_alignr_rslt5_2_m128i, 1);
    hor_avx2_unpack8_c27<0, 0>(out, c27_insert_rslt5_m256i); // Unpack 5th 8 values.

    __m128i c27_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c27_alignr_rslt6_1_m128i = _mm_alignr_epi8(c27_load_rslt10_m128i, c27_load_rslt9_m128i, 7);
    __m128i c27_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c27_alignr_rslt6_2_m128i = _mm_alignr_epi8(c27_load_rslt11_m128i, c27_load_rslt10_m128i, 4);
    __m256i c27_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt6_1_m128i), c27_alignr_rslt6_2_m128i, 1);
    hor_avx2_unpack8_c27<0, 0>(out, c27_insert_rslt6_m256i); // Unpack 6th 8 values.

    __m128i c27_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c27_alignr_rslt7_m128i = _mm_alignr_epi8(c27_load_rslt12_m128i, c27_load_rslt11_m128i, 15);
    __m256i c27_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_load_rslt11_m128i), c27_alignr_rslt7_m128i, 1);
    hor_avx2_unpack8_c27<2, 0>(out, c27_insert_rslt7_m256i); // Unpack 7th 8 values.

    __m128i c27_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c27_alignr_rslt8_1_m128i = _mm_alignr_epi8(c27_load_rslt13_m128i, c27_load_rslt12_m128i, 13);
    __m128i c27_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c27_alignr_rslt8_2_m128i = _mm_alignr_epi8(c27_load_rslt14_m128i, c27_load_rslt13_m128i, 10);
    __m256i c27_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt8_1_m128i), c27_alignr_rslt8_2_m128i, 1);
    hor_avx2_unpack8_c27<0, 0>(out, c27_insert_rslt8_m256i); // Unpack 8th 8 values.

    __m128i c27_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c27_alignr_rslt9_1_m128i = _mm_alignr_epi8(c27_load_rslt15_m128i, c27_load_rslt14_m128i, 8);
    __m128i c27_load_rslt16_m128i = _mm_loadu_si128(in + 15);
    __m128i c27_alignr_rslt9_2_m128i = _mm_alignr_epi8(c27_load_rslt16_m128i, c27_load_rslt15_m128i, 5);
    __m256i c27_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt9_1_m128i), c27_alignr_rslt9_2_m128i, 1);
    hor_avx2_unpack8_c27<0, 0>(out, c27_insert_rslt9_m256i); // Unpack 9th 8 values.

    __m128i c27_load_rslt17_m128i = _mm_loadu_si128(in + 16);
    __m128i c27_alignr_rslt10_m128i = _mm_alignr_epi8(c27_load_rslt17_m128i, c27_load_rslt16_m128i, 3);
    __m256i c27_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt10_m128i), c27_load_rslt17_m128i, 1);
    hor_avx2_unpack8_c27<0, 0>(out, c27_insert_rslt10_m256i); // Unpack 10th 8 values.

    __m128i c27_load_rslt18_m128i = _mm_loadu_si128(in + 17);
    __m128i c27_alignr_rslt11_1_m128i = _mm_alignr_epi8(c27_load_rslt18_m128i, c27_load_rslt17_m128i, 14);
    __m128i c27_load_rslt19_m128i = _mm_loadu_si128(in + 18);
    __m128i c27_alignr_rslt11_2_m128i = _mm_alignr_epi8(c27_load_rslt19_m128i, c27_load_rslt18_m128i, 11);
    __m256i c27_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt11_1_m128i), c27_alignr_rslt11_2_m128i, 1);
    hor_avx2_unpack8_c27<0, 0>(out, c27_insert_rslt11_m256i); // Unpack 11th 8 values.

    __m128i c27_load_rslt20_m128i = _mm_loadu_si128(in + 19);
    __m128i c27_alignr_rslt12_1_m128i = _mm_alignr_epi8(c27_load_rslt20_m128i, c27_load_rslt19_m128i, 9);
    __m128i c27_load_rslt21_m128i = _mm_loadu_si128(in + 20);
    __m128i c27_alignr_rslt12_2_m128i = _mm_alignr_epi8(c27_load_rslt21_m128i, c27_load_rslt20_m128i, 6);
    __m256i c27_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt12_1_m128i), c27_alignr_rslt12_2_m128i, 1);
    hor_avx2_unpack8_c27<0, 0>(out, c27_insert_rslt12_m256i); // Unpack 12th 8 values.

    __m128i c27_load_rslt22_m128i = _mm_loadu_si128(in + 21);
    __m128i c27_alignr_rslt13_m128i = _mm_alignr_epi8(c27_load_rslt22_m128i, c27_load_rslt21_m128i, 4);
    __m256i c27_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt13_m128i), c27_load_rslt22_m128i, 1);
    hor_avx2_unpack8_c27<0, 1>(out, c27_insert_rslt13_m256i); // Unpack 13th 8 values.

    __m128i c27_load_rslt23_m128i = _mm_loadu_si128(in + 22);
    __m128i c27_alignr_rslt14_1_m128i = _mm_alignr_epi8(c27_load_rslt23_m128i, c27_load_rslt22_m128i, 15);
    __m128i c27_load_rslt24_m128i = _mm_loadu_si128(in + 23);
    __m128i c27_alignr_rslt14_2_m128i = _mm_alignr_epi8(c27_load_rslt24_m128i, c27_load_rslt23_m128i, 12);
    __m256i c27_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt14_1_m128i), c27_alignr_rslt14_2_m128i, 1);
    hor_avx2_unpack8_c27<0, 0>(out, c27_insert_rslt14_m256i); // Unpack 14th 8 values.

    __m128i c27_load_rslt25_m128i = _mm_loadu_si128(in + 24);
    __m128i c27_alignr_rslt15_1_m128i = _mm_alignr_epi8(c27_load_rslt25_m128i, c27_load_rslt24_m128i, 10);
    __m128i c27_load_rslt26_m128i = _mm_loadu_si128(in + 25);
    __m128i c27_alignr_rslt15_2_m128i = _mm_alignr_epi8(c27_load_rslt26_m128i, c27_load_rslt25_m128i, 7);
    __m256i c27_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt15_1_m128i), c27_alignr_rslt15_2_m128i, 1);
    hor_avx2_unpack8_c27<0, 0>(out, c27_insert_rslt15_m256i); // Unpack 15th 8 values.

    __m128i c27_load_rslt27_m128i = _mm_loadu_si128(in + 26);
    __m128i c27_alignr_rslt16_m128i = _mm_alignr_epi8(c27_load_rslt27_m128i, c27_load_rslt26_m128i, 5);
    __m256i c27_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt16_m128i), c27_load_rslt27_m128i, 1);
    hor_avx2_unpack8_c27<0, 2>(out, c27_insert_rslt16_m256i); // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c27(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_shfl_msk_m256i = _mm256_set_epi8(
				byte2 + 13, byte2 + 12, byte2 + 11, byte2 + 10,
				byte2 + 10, byte2 + 9, byte2 + 8, byte2 + 7,
				byte2 + 7, byte2 + 6, byte2 + 5, byte2 + 4,
				byte2 + 3, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 13, byte1 + 12, byte1 + 11, byte1 + 10,
				byte1 + 9, byte1 + 8, byte1 + 7, byte1 + 6,
				byte1 + 6, byte1 + 5, byte1 + 4, byte1 + 3,
				byte1 + 3, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c27_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_shfl_msk_m256i);
		__m256i c27_sllv_rslt_m256i = _mm256_sllv_epi64(c27_shfl_rslt_m256i, _mm256_set_epi64x(0, 1, 0, 0));
		__m256i c27_srlv_rslt1_m256i = _mm256_srlv_epi64(c27_sllv_rslt_m256i, _mm256_set_epi64x(0, 0, 1, 0));
		__m256i c27_srlv_rslt2_m256i = _mm256_srlv_epi32(c27_srlv_rslt1_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[27]);
		__m256i c27_rslt_m256i = _mm256_and_si256(c27_srlv_rslt2_m256i, SIMDMasks::AVX2_and_msk_m256i[27]);
		_mm256_storeu_si256(out++, c27_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_shfl_msk_m256i = _mm256_set_epi8(
				byte2 + 13, byte2 + 12, byte2 + 11, byte2 + 10,
				byte2 + 10, byte2 + 9, byte2 + 8, byte2 + 7,
				byte2 + 7, byte2 + 6, byte2 + 5, byte2 + 4,
				byte2 + 3, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 13, byte1 + 12, byte1 + 11, byte1 + 10,
				byte1 + 9, byte1 + 8, byte1 + 7, byte1 + 6,
				byte1 + 6, byte1 + 5, byte1 + 4, byte1 + 3,
				byte1 + 3, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c27_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_shfl_msk_m256i);
		__m256i c27_sllv_rslt_m256i = _mm256_sllv_epi64(c27_shfl_rslt_m256i, _mm256_set_epi64x(0, 1, 0, 0));
		__m256i c27_srlv_rslt1_m256i = _mm256_srlv_epi64(c27_sllv_rslt_m256i, _mm256_set_epi64x(0, 0, 1, 0));
		__m256i c27_srlv_rslt2_m256i = _mm256_srlv_epi32(c27_srlv_rslt1_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[27]);
		__m256i c27_and_rslt_m256i = _mm256_and_si256(c27_srlv_rslt2_m256i, SIMDMasks::AVX2_and_msk_m256i[27]);
		__m256i c27_rslt_m256i = _mm256_or_si256(c27_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 27));
		_mm256_storeu_si256(out++, c27_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 27-bit values.
// * Load 27 SSE vectors, each containing 4 27-bit values. (5th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c27(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	__m128i c27_load_rslt1_m128i = _mm_loadu_si128(in + 0);
//    __m128i c27_load_rslt2_m128i = _mm_loadu_si128(in + 1);
//    __m128i c27_alignr_rslt1_m128i = _mm_alignr_epi8(c27_load_rslt2_m128i, c27_load_rslt1_m128i, 13);
//    __m256i c27_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_load_rslt1_m128i), c27_alignr_rslt1_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt1_m256i); // Unpack 1st 8 values.
//
//    __m128i c27_load_rslt3_m128i = _mm_loadu_si128(in + 2);
//    __m128i c27_alignr_rslt2_1_m128i = _mm_alignr_epi8(c27_load_rslt3_m128i, c27_load_rslt2_m128i, 11);
//    __m128i c27_load_rslt4_m128i = _mm_loadu_si128(in + 3);
//    __m128i c27_alignr_rslt2_2_m128i = _mm_alignr_epi8(c27_load_rslt4_m128i, c27_load_rslt3_m128i, 8);
//    __m256i c27_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt2_1_m128i), c27_alignr_rslt2_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt2_m256i); // Unpack 2nd 8 values.
//
//    __m128i c27_load_rslt5_m128i = _mm_loadu_si128(in + 4);
//    __m128i c27_alignr_rslt3_1_m128i = _mm_alignr_epi8(c27_load_rslt5_m128i, c27_load_rslt4_m128i, 6);
//    __m128i c27_load_rslt6_m128i = _mm_loadu_si128(in + 5);
//    __m128i c27_alignr_rslt3_2_m128i = _mm_alignr_epi8(c27_load_rslt6_m128i, c27_load_rslt5_m128i, 3);
//    __m256i c27_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt3_1_m128i), c27_alignr_rslt3_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt3_m256i); // Unpack 3rd 8 values.
//
//    __m128i c27_load_rslt7_m128i = _mm_loadu_si128(in + 6);
//    __m128i c27_alignr_rslt4_1_m128i = _mm_alignr_epi8(c27_load_rslt7_m128i, c27_load_rslt6_m128i, 1);
//    __m128i c27_alignr_rslt4_2_m128i = _mm_alignr_epi8(c27_load_rslt7_m128i, c27_load_rslt6_m128i, 14);
//    __m256i c27_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt4_1_m128i), c27_alignr_rslt4_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt4_m256i); // Unpack 4th 8 values.
//
//    __m128i c27_load_rslt8_m128i = _mm_loadu_si128(in + 7);
//    __m128i c27_alignr_rslt5_1_m128i = _mm_alignr_epi8(c27_load_rslt8_m128i, c27_load_rslt7_m128i, 12);
//    __m128i c27_load_rslt9_m128i = _mm_loadu_si128(in + 8);
//    __m128i c27_alignr_rslt5_2_m128i = _mm_alignr_epi8(c27_load_rslt9_m128i, c27_load_rslt8_m128i, 9);
//    __m256i c27_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt5_1_m128i), c27_alignr_rslt5_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt5_m256i); // Unpack 5th 8 values.
//
//    __m128i c27_load_rslt10_m128i = _mm_loadu_si128(in + 9);
//    __m128i c27_alignr_rslt6_1_m128i = _mm_alignr_epi8(c27_load_rslt10_m128i, c27_load_rslt9_m128i, 7);
//    __m128i c27_load_rslt11_m128i = _mm_loadu_si128(in + 10);
//    __m128i c27_alignr_rslt6_2_m128i = _mm_alignr_epi8(c27_load_rslt11_m128i, c27_load_rslt10_m128i, 4);
//    __m256i c27_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt6_1_m128i), c27_alignr_rslt6_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt6_m256i); // Unpack 6th 8 values.
//
//    __m128i c27_load_rslt12_m128i = _mm_loadu_si128(in + 11);
//    __m128i c27_alignr_rslt7_1_m128i = _mm_alignr_epi8(c27_load_rslt12_m128i, c27_load_rslt11_m128i, 2);
//    __m128i c27_alignr_rslt7_2_m128i = _mm_alignr_epi8(c27_load_rslt12_m128i, c27_load_rslt11_m128i, 15);
//    __m256i c27_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt7_1_m128i), c27_alignr_rslt7_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt7_m256i); // Unpack 7th 8 values.
//
//    __m128i c27_load_rslt13_m128i = _mm_loadu_si128(in + 12);
//    __m128i c27_alignr_rslt8_1_m128i = _mm_alignr_epi8(c27_load_rslt13_m128i, c27_load_rslt12_m128i, 13);
//    __m128i c27_load_rslt14_m128i = _mm_loadu_si128(in + 13);
//    __m128i c27_alignr_rslt8_2_m128i = _mm_alignr_epi8(c27_load_rslt14_m128i, c27_load_rslt13_m128i, 10);
//    __m256i c27_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt8_1_m128i), c27_alignr_rslt8_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt8_m256i); // Unpack 8th 8 values.
//
//    __m128i c27_load_rslt15_m128i = _mm_loadu_si128(in + 14);
//    __m128i c27_alignr_rslt9_1_m128i = _mm_alignr_epi8(c27_load_rslt15_m128i, c27_load_rslt14_m128i, 8);
//    __m128i c27_load_rslt16_m128i = _mm_loadu_si128(in + 15);
//    __m128i c27_alignr_rslt9_2_m128i = _mm_alignr_epi8(c27_load_rslt16_m128i, c27_load_rslt15_m128i, 5);
//    __m256i c27_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt9_1_m128i), c27_alignr_rslt9_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt9_m256i); // Unpack 9th 8 values.
//
//    __m128i c27_load_rslt17_m128i = _mm_loadu_si128(in + 16);
//    __m128i c27_alignr_rslt10_1_m128i = _mm_alignr_epi8(c27_load_rslt17_m128i, c27_load_rslt16_m128i, 3);
//    __m128i c27_alignr_rslt10_2_m128i = _mm_alignr_epi8(c27_load_rslt17_m128i, c27_load_rslt16_m128i, 16);
//    __m256i c27_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt10_1_m128i), c27_alignr_rslt10_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt10_m256i); // Unpack 10th 8 values.
//
//    __m128i c27_load_rslt18_m128i = _mm_loadu_si128(in + 17);
//    __m128i c27_alignr_rslt11_1_m128i = _mm_alignr_epi8(c27_load_rslt18_m128i, c27_load_rslt17_m128i, 14);
//    __m128i c27_load_rslt19_m128i = _mm_loadu_si128(in + 18);
//    __m128i c27_alignr_rslt11_2_m128i = _mm_alignr_epi8(c27_load_rslt19_m128i, c27_load_rslt18_m128i, 11);
//    __m256i c27_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt11_1_m128i), c27_alignr_rslt11_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt11_m256i); // Unpack 11th 8 values.
//
//    __m128i c27_load_rslt20_m128i = _mm_loadu_si128(in + 19);
//    __m128i c27_alignr_rslt12_1_m128i = _mm_alignr_epi8(c27_load_rslt20_m128i, c27_load_rslt19_m128i, 9);
//    __m128i c27_load_rslt21_m128i = _mm_loadu_si128(in + 20);
//    __m128i c27_alignr_rslt12_2_m128i = _mm_alignr_epi8(c27_load_rslt21_m128i, c27_load_rslt20_m128i, 6);
//    __m256i c27_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt12_1_m128i), c27_alignr_rslt12_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt12_m256i); // Unpack 12th 8 values.
//
//    __m128i c27_load_rslt22_m128i = _mm_loadu_si128(in + 21);
//    __m128i c27_alignr_rslt13_1_m128i = _mm_alignr_epi8(c27_load_rslt22_m128i, c27_load_rslt21_m128i, 4);
//    __m128i c27_alignr_rslt13_2_m128i = _mm_alignr_epi8(c27_load_rslt22_m128i, c27_load_rslt21_m128i, 17);
//    __m256i c27_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt13_1_m128i), c27_alignr_rslt13_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt13_m256i); // Unpack 13th 8 values.
//
//    __m128i c27_load_rslt23_m128i = _mm_loadu_si128(in + 22);
//    __m128i c27_alignr_rslt14_1_m128i = _mm_alignr_epi8(c27_load_rslt23_m128i, c27_load_rslt22_m128i, 15);
//    __m128i c27_load_rslt24_m128i = _mm_loadu_si128(in + 23);
//    __m128i c27_alignr_rslt14_2_m128i = _mm_alignr_epi8(c27_load_rslt24_m128i, c27_load_rslt23_m128i, 12);
//    __m256i c27_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt14_1_m128i), c27_alignr_rslt14_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt14_m256i); // Unpack 14th 8 values.
//
//    __m128i c27_load_rslt25_m128i = _mm_loadu_si128(in + 24);
//    __m128i c27_alignr_rslt15_1_m128i = _mm_alignr_epi8(c27_load_rslt25_m128i, c27_load_rslt24_m128i, 10);
//    __m128i c27_load_rslt26_m128i = _mm_loadu_si128(in + 25);
//    __m128i c27_alignr_rslt15_2_m128i = _mm_alignr_epi8(c27_load_rslt26_m128i, c27_load_rslt25_m128i, 7);
//    __m256i c27_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt15_1_m128i), c27_alignr_rslt15_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt15_m256i); // Unpack 15th 8 values.
//
//    __m128i c27_load_rslt27_m128i = _mm_loadu_si128(in + 26);
//    __m128i c27_alignr_rslt16_1_m128i = _mm_alignr_epi8(c27_load_rslt27_m128i, c27_load_rslt26_m128i, 5);
//    __m128i c27_alignr_rslt16_2_m128i = _mm_alignr_epi8(c27_load_rslt27_m128i, c27_load_rslt26_m128i, 18);
//    __m256i c27_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c27_alignr_rslt16_1_m128i), c27_alignr_rslt16_2_m128i, 1);
//    hor_avx2_unpack8_c27(out, c27_insert_rslt16_m256i); // Unpack 16th 8 values.
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c27(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c27_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[27]);
//		__m256i c27_sllv_rslt_m256i = _mm256_sllv_epi64(c27_shfl_rslt_m256i, _mm256_set_epi64x(0, 1, 0, 0));
//		__m256i c27_srlv_rslt1_m256i = _mm256_srlv_epi64(c27_sllv_rslt_m256i, _mm256_set_epi64x(0, 0, 1, 0));
//		__m256i c27_srlv_rslt2_m256i = _mm256_srlv_epi32(c27_srlv_rslt1_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[27]);
//		__m256i c27_rslt_m256i = _mm256_and_si256(c27_srlv_rslt2_m256i, SIMDMasks::AVX2_and_msk_m256i[27]);
//		_mm256_storeu_si256(out++, c27_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c27_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[27]);
//		__m256i c27_sllv_rslt_m256i = _mm256_sllv_epi64(c27_shfl_rslt_m256i, _mm256_set_epi64x(0, 1, 0, 0));
//		__m256i c27_srlv_rslt1_m256i = _mm256_srlv_epi64(c27_sllv_rslt_m256i, _mm256_set_epi64x(0, 0, 1, 0));
//		__m256i c27_srlv_rslt2_m256i = _mm256_srlv_epi32(c27_srlv_rslt1_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[27]);
//		__m256i c27_and_rslt_m256i = _mm256_and_si256(c27_srlv_rslt2_m256i, SIMDMasks::AVX2_and_msk_m256i[27]);
//		__m256i c27_rslt_m256i = _mm256_or_si256(c27_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 27));
//		_mm256_storeu_si256(out++, c27_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 28-bit values.
 * Load 28 SSE vectors, each containing 4 28-bit values. (5th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c28(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
		__m128i c28_load_rslt1_m128i = _mm_loadu_si128(in++);
		__m128i c28_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt1_m128i = _mm_alignr_epi8(c28_load_rslt2_m128i, c28_load_rslt1_m128i, 14);
		__m256i c28_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c28_load_rslt1_m128i), c28_alignr_rslt1_m128i, 1);
		hor_avx2_unpack8_c28<0>(out, c28_insert_rslt1_m256i); // Unpack 1st 8 values.

		__m128i c28_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt2_1_m128i = _mm_alignr_epi8(c28_load_rslt3_m128i, c28_load_rslt2_m128i, 12);
		__m128i c28_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt2_2_m128i = _mm_alignr_epi8(c28_load_rslt4_m128i, c28_load_rslt3_m128i, 10);
		__m256i c28_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c28_alignr_rslt2_1_m128i), c28_alignr_rslt2_2_m128i, 1);
		hor_avx2_unpack8_c28<0>(out, c28_insert_rslt2_m256i); // Unpack 2nd 8 values.

		__m128i c28_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt3_1_m128i = _mm_alignr_epi8(c28_load_rslt5_m128i, c28_load_rslt4_m128i, 8);
		__m128i c28_load_rslt6_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt3_2_m128i = _mm_alignr_epi8(c28_load_rslt6_m128i, c28_load_rslt5_m128i, 6);
		__m256i c28_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c28_alignr_rslt3_1_m128i), c28_alignr_rslt3_2_m128i, 1);
		hor_avx2_unpack8_c28<0>(out, c28_insert_rslt3_m256i); // Unpack 3rd 8 values.

		__m128i c28_load_rslt7_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt4_m128i = _mm_alignr_epi8(c28_load_rslt7_m128i, c28_load_rslt6_m128i, 4);
		__m256i c28_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c28_alignr_rslt4_m128i), c28_load_rslt7_m128i, 1);
		hor_avx2_unpack8_c28<2>(out, c28_insert_rslt4_m256i); // Unpack 4th 8 values.

	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c28(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_shfl_msk_m256i = _mm256_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0,
				13, 12, 11, 10,
				10, 9, 8, 7,
				6, 5, 4, 3,
				3, 2, 1, 0);
		__m256i c28_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_shfl_msk_m256i);
		__m256i c28_srlv_rslt_m256i = _mm256_srlv_epi32(c28_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[28]);
		__m256i c28_rslt_m256i = _mm256_and_si256(c28_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[28]);
		_mm256_storeu_si256(out++, c28_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_shfl_msk_m256i = _mm256_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0,
				13, 12, 11, 10,
				10, 9, 8, 7,
				6, 5, 4, 3,
				3, 2, 1, 0);
		__m256i c28_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_shfl_msk_m256i);
		__m256i c28_srlv_rslt_m256i = _mm256_srlv_epi32(c28_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[28]);
		__m256i c28_and_rslt_m256i = _mm256_and_si256(c28_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[28]);
		__m256i c28_rslt_m256i = _mm256_or_si256(c28_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 28));
		_mm256_storeu_si256(out++, c28_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 28-bit values.
// * Load 28 SSE vectors, each containing 4 28-bit values. (5th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c28(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
//		__m128i c28_load_rslt1_m128i = _mm_loadu_si128(in++);
//		__m128i c28_load_rslt2_m128i = _mm_loadu_si128(in++);
//		__m128i c28_alignr_rslt1_m128i = _mm_alignr_epi8(c28_load_rslt2_m128i, c28_load_rslt1_m128i, 14);
//		__m256i c28_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c28_load_rslt1_m128i), c28_alignr_rslt1_m128i, 1);
//		hor_avx2_unpack8_c28(out, c28_insert_rslt1_m256i); // Unpack 1st 8 values.
//
//		__m128i c28_load_rslt3_m128i = _mm_loadu_si128(in++);
//		__m128i c28_alignr_rslt2_1_m128i = _mm_alignr_epi8(c28_load_rslt3_m128i, c28_load_rslt2_m128i, 12);
//		__m128i c28_load_rslt4_m128i = _mm_loadu_si128(in++);
//		__m128i c28_alignr_rslt2_2_m128i = _mm_alignr_epi8(c28_load_rslt4_m128i, c28_load_rslt3_m128i, 10);
//		__m256i c28_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c28_alignr_rslt2_1_m128i), c28_alignr_rslt2_2_m128i, 1);
//		hor_avx2_unpack8_c28(out, c28_insert_rslt2_m256i); // Unpack 2nd 8 values.
//
//		__m128i c28_load_rslt5_m128i = _mm_loadu_si128(in++);
//		__m128i c28_alignr_rslt3_1_m128i = _mm_alignr_epi8(c28_load_rslt5_m128i, c28_load_rslt4_m128i, 8);
//		__m128i c28_load_rslt6_m128i = _mm_loadu_si128(in++);
//		__m128i c28_alignr_rslt3_2_m128i = _mm_alignr_epi8(c28_load_rslt6_m128i, c28_load_rslt5_m128i, 6);
//		__m256i c28_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c28_alignr_rslt3_1_m128i), c28_alignr_rslt3_2_m128i, 1);
//		hor_avx2_unpack8_c28(out, c28_insert_rslt3_m256i); // Unpack 3rd 8 values.
//
//		__m128i c28_load_rslt7_m128i = _mm_loadu_si128(in++);
//		__m128i c28_alignr_rslt4_1_m128i = _mm_alignr_epi8(c28_load_rslt7_m128i, c28_load_rslt6_m128i, 4);
//		__m128i c28_alignr_rslt4_2_m128i = _mm_alignr_epi8(c28_load_rslt7_m128i, c28_load_rslt6_m128i, 18);
//		__m256i c28_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c28_alignr_rslt4_1_m128i), c28_alignr_rslt4_2_m128i, 1);
//		hor_avx2_unpack8_c28(out, c28_insert_rslt4_m256i); // Unpack 4th 8 values.
//
//	}
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c28(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c28_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[28]);
//		__m256i c28_srlv_rslt_m256i = _mm256_srlv_epi32(c28_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[28]);
//		__m256i c28_rslt_m256i = _mm256_and_si256(c28_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[28]);
//		_mm256_storeu_si256(out++, c28_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c28_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[28]);
//		__m256i c28_srlv_rslt_m256i = _mm256_srlv_epi32(c28_shfl_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[28]);
//		__m256i c28_and_rslt_m256i = _mm256_and_si256(c28_srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[28]);
//		__m256i c28_rslt_m256i = _mm256_or_si256(c28_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 28));
//		_mm256_storeu_si256(out++, c28_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 29-bit values.
 * Load 29 SSE vectors, each containing 4 29-bit values. (5th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c29(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
     __m128i c29_load_rslt1_m128i = _mm_loadu_si128(in + 0);
     __m128i c29_load_rslt2_m128i = _mm_loadu_si128(in + 1);
     __m128i c29_alignr_rslt1_m128i = _mm_alignr_epi8(c29_load_rslt2_m128i, c29_load_rslt1_m128i, 14);
     __m256i c29_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_load_rslt1_m128i), c29_alignr_rslt1_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt1_m256i); // Unpack 1st 8 values.

     __m128i c29_load_rslt3_m128i = _mm_loadu_si128(in + 2);
     __m128i c29_alignr_rslt2_1_m128i = _mm_alignr_epi8(c29_load_rslt3_m128i, c29_load_rslt2_m128i, 13);
     __m128i c29_load_rslt4_m128i = _mm_loadu_si128(in + 3);
     __m128i c29_alignr_rslt2_2_m128i = _mm_alignr_epi8(c29_load_rslt4_m128i, c29_load_rslt3_m128i, 11);
     __m256i c29_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt2_1_m128i), c29_alignr_rslt2_2_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt2_m256i); // Unpack 2nd 8 values.

     __m128i c29_load_rslt5_m128i = _mm_loadu_si128(in + 4);
     __m128i c29_alignr_rslt3_1_m128i = _mm_alignr_epi8(c29_load_rslt5_m128i, c29_load_rslt4_m128i, 10);
     __m128i c29_load_rslt6_m128i = _mm_loadu_si128(in + 5);
     __m128i c29_alignr_rslt3_2_m128i = _mm_alignr_epi8(c29_load_rslt6_m128i, c29_load_rslt5_m128i, 8);
     __m256i c29_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt3_1_m128i), c29_alignr_rslt3_2_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt3_m256i); // Unpack 3rd 8 values.

     __m128i c29_load_rslt7_m128i = _mm_loadu_si128(in + 6);
     __m128i c29_alignr_rslt4_1_m128i = _mm_alignr_epi8(c29_load_rslt7_m128i, c29_load_rslt6_m128i, 7);
     __m128i c29_load_rslt8_m128i = _mm_loadu_si128(in + 7);
     __m128i c29_alignr_rslt4_2_m128i = _mm_alignr_epi8(c29_load_rslt8_m128i, c29_load_rslt7_m128i, 5);
     __m256i c29_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt4_1_m128i), c29_alignr_rslt4_2_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt4_m256i); // Unpack 4th 8 values.

     __m128i c29_load_rslt9_m128i = _mm_loadu_si128(in + 8);
     __m128i c29_alignr_rslt5_1_m128i = _mm_alignr_epi8(c29_load_rslt9_m128i, c29_load_rslt8_m128i, 4);
     __m128i c29_load_rslt10_m128i = _mm_loadu_si128(in + 9);
     __m128i c29_alignr_rslt5_2_m128i = _mm_alignr_epi8(c29_load_rslt10_m128i, c29_load_rslt9_m128i, 2);
     __m256i c29_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt5_1_m128i), c29_alignr_rslt5_2_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt5_m256i); // Unpack 5th 8 values.

     __m128i c29_load_rslt11_m128i = _mm_loadu_si128(in + 10);
     __m128i c29_alignr_rslt6_m128i = _mm_alignr_epi8(c29_load_rslt11_m128i, c29_load_rslt10_m128i, 15);
     __m256i c29_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_load_rslt10_m128i), c29_alignr_rslt6_m128i, 1);
     hor_avx2_unpack8_c29<1, 0>(out, c29_insert_rslt6_m256i); // Unpack 6th 8 values.

     __m128i c29_load_rslt12_m128i = _mm_loadu_si128(in + 11);
     __m128i c29_alignr_rslt7_1_m128i = _mm_alignr_epi8(c29_load_rslt12_m128i, c29_load_rslt11_m128i, 14);
     __m128i c29_load_rslt13_m128i = _mm_loadu_si128(in + 12);
     __m128i c29_alignr_rslt7_2_m128i = _mm_alignr_epi8(c29_load_rslt13_m128i, c29_load_rslt12_m128i, 12);
     __m256i c29_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt7_1_m128i), c29_alignr_rslt7_2_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt7_m256i); // Unpack 7th 8 values.

     __m128i c29_load_rslt14_m128i = _mm_loadu_si128(in + 13);
     __m128i c29_alignr_rslt8_1_m128i = _mm_alignr_epi8(c29_load_rslt14_m128i, c29_load_rslt13_m128i, 11);
     __m128i c29_load_rslt15_m128i = _mm_loadu_si128(in + 14);
     __m128i c29_alignr_rslt8_2_m128i = _mm_alignr_epi8(c29_load_rslt15_m128i, c29_load_rslt14_m128i, 9);
     __m256i c29_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt8_1_m128i), c29_alignr_rslt8_2_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt8_m256i); // Unpack 8th 8 values.

     __m128i c29_load_rslt16_m128i = _mm_loadu_si128(in + 15);
     __m128i c29_alignr_rslt9_1_m128i = _mm_alignr_epi8(c29_load_rslt16_m128i, c29_load_rslt15_m128i, 8);
     __m128i c29_load_rslt17_m128i = _mm_loadu_si128(in + 16);
     __m128i c29_alignr_rslt9_2_m128i = _mm_alignr_epi8(c29_load_rslt17_m128i, c29_load_rslt16_m128i, 6);
     __m256i c29_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt9_1_m128i), c29_alignr_rslt9_2_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt9_m256i); // Unpack 9th 8 values.

     __m128i c29_load_rslt18_m128i = _mm_loadu_si128(in + 17);
     __m128i c29_alignr_rslt10_1_m128i = _mm_alignr_epi8(c29_load_rslt18_m128i, c29_load_rslt17_m128i, 5);
     __m128i c29_load_rslt19_m128i = _mm_loadu_si128(in + 18);
     __m128i c29_alignr_rslt10_2_m128i = _mm_alignr_epi8(c29_load_rslt19_m128i, c29_load_rslt18_m128i, 3);
     __m256i c29_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt10_1_m128i), c29_alignr_rslt10_2_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt10_m256i); // Unpack 10th 8 values.

     __m128i c29_load_rslt20_m128i = _mm_loadu_si128(in + 19);
     __m128i c29_alignr_rslt11_m128i = _mm_alignr_epi8(c29_load_rslt20_m128i, c29_load_rslt19_m128i, 2);
     __m256i c29_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt11_m128i), c29_load_rslt20_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt11_m256i); // Unpack 11th 8 values.

     __m128i c29_load_rslt21_m128i = _mm_loadu_si128(in + 20);
     __m128i c29_alignr_rslt12_1_m128i = _mm_alignr_epi8(c29_load_rslt21_m128i, c29_load_rslt20_m128i, 15);
     __m128i c29_load_rslt22_m128i = _mm_loadu_si128(in + 21);
     __m128i c29_alignr_rslt12_2_m128i = _mm_alignr_epi8(c29_load_rslt22_m128i, c29_load_rslt21_m128i, 13);
     __m256i c29_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt12_1_m128i), c29_alignr_rslt12_2_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt12_m256i); // Unpack 12th 8 values.

     __m128i c29_load_rslt23_m128i = _mm_loadu_si128(in + 22);
     __m128i c29_alignr_rslt13_1_m128i = _mm_alignr_epi8(c29_load_rslt23_m128i, c29_load_rslt22_m128i, 12);
     __m128i c29_load_rslt24_m128i = _mm_loadu_si128(in + 23);
     __m128i c29_alignr_rslt13_2_m128i = _mm_alignr_epi8(c29_load_rslt24_m128i, c29_load_rslt23_m128i, 10);
     __m256i c29_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt13_1_m128i), c29_alignr_rslt13_2_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt13_m256i); // Unpack 13th 8 values.

     __m128i c29_load_rslt25_m128i = _mm_loadu_si128(in + 24);
     __m128i c29_alignr_rslt14_1_m128i = _mm_alignr_epi8(c29_load_rslt25_m128i, c29_load_rslt24_m128i, 9);
     __m128i c29_load_rslt26_m128i = _mm_loadu_si128(in + 25);
     __m128i c29_alignr_rslt14_2_m128i = _mm_alignr_epi8(c29_load_rslt26_m128i, c29_load_rslt25_m128i, 7);
     __m256i c29_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt14_1_m128i), c29_alignr_rslt14_2_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt14_m256i); // Unpack 14th 8 values.

     __m128i c29_load_rslt27_m128i = _mm_loadu_si128(in + 26);
     __m128i c29_alignr_rslt15_1_m128i = _mm_alignr_epi8(c29_load_rslt27_m128i, c29_load_rslt26_m128i, 6);
     __m128i c29_load_rslt28_m128i = _mm_loadu_si128(in + 27);
     __m128i c29_alignr_rslt15_2_m128i = _mm_alignr_epi8(c29_load_rslt28_m128i, c29_load_rslt27_m128i, 4);
     __m256i c29_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt15_1_m128i), c29_alignr_rslt15_2_m128i, 1);
     hor_avx2_unpack8_c29<0, 0>(out, c29_insert_rslt15_m256i); // Unpack 15th 8 values.

     __m128i c29_load_rslt29_m128i = _mm_loadu_si128(in + 28);
     __m128i c29_alignr_rslt16_m128i = _mm_alignr_epi8(c29_load_rslt29_m128i, c29_load_rslt28_m128i, 3);
     __m256i c29_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt16_m128i), c29_load_rslt29_m128i, 1);
     hor_avx2_unpack8_c29<0, 1>(out, c29_insert_rslt16_m256i); // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
template <int byte1, int byte2>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c29(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c29_shfl_msk_m256i = _mm256_set_epi8(
				byte2 + 14, byte2 + 13, byte2 + 12, byte2 + 11,
				byte2 + 10, byte2 + 9, byte2 + 8, byte2 + 7,
				byte2 + 7, byte2 + 6, byte2 + 5, byte2 + 4,
				byte2 + 3, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 14, byte1 + 13, byte1 + 12, byte1 + 11,
				byte1 + 10, byte1 + 9, byte1 + 8, byte1 + 7,
				byte1 + 7, byte1 + 6, byte1 + 5, byte1 + 4,
				byte1 + 3, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c29_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c29_shfl_msk_m256i);
		__m256i c29_sllv_rslt_m256i = _mm256_sllv_epi64(c29_shfl_rslt_m256i, _mm256_set_epi64x(0, 0, 1, 3));
		__m256i c29_srlv_rslt1_m256i = _mm256_srlv_epi64(c29_sllv_rslt_m256i, _mm256_set_epi64x(3, 1, 0, 0));
		__m256i c29_srlv_rslt2_m256i = _mm256_srlv_epi32(c29_srlv_rslt1_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[29]);
		__m256i c29_rslt_m256i = _mm256_and_si256(c29_srlv_rslt2_m256i, SIMDMasks::AVX2_and_msk_m256i[29]);
		_mm256_storeu_si256(out++, c29_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c29_shfl_msk_m256i = _mm256_set_epi8(
				byte2 + 14, byte2 + 13, byte2 + 12, byte2 + 11,
				byte2 + 10, byte2 + 9, byte2 + 8, byte2 + 7,
				byte2 + 7, byte2 + 6, byte2 + 5, byte2 + 4,
				byte2 + 3, byte2 + 2, byte2 + 1, byte2 + 0,
				byte1 + 14, byte1 + 13, byte1 + 12, byte1 + 11,
				byte1 + 10, byte1 + 9, byte1 + 8, byte1 + 7,
				byte1 + 7, byte1 + 6, byte1 + 5, byte1 + 4,
				byte1 + 3, byte1 + 2, byte1 + 1, byte1 + 0);
		__m256i c29_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c29_shfl_msk_m256i);
		__m256i c29_sllv_rslt_m256i = _mm256_sllv_epi64(c29_shfl_rslt_m256i, _mm256_set_epi64x(0, 0, 1, 3));
		__m256i c29_srlv_rslt1_m256i = _mm256_srlv_epi64(c29_shfl_rslt_m256i, _mm256_set_epi64x(3, 1, 0, 0));
		__m256i c29_srlv_rslt2_m256i = _mm256_srlv_epi32(c29_srlv_rslt1_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[29]);
		__m256i c29_and_rslt_m256i = _mm256_and_si256(c29_srlv_rslt2_m256i, SIMDMasks::AVX2_and_msk_m256i[29]);
		__m256i c29_rslt_m256i = _mm256_or_si256(c29_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 29));
		_mm256_storeu_si256(out++, c29_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 29-bit values.
// * Load 29 SSE vectors, each containing 4 29-bit values. (5th is incomplete)
// * Slower alternative.
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c29(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//     __m128i c29_load_rslt1_m128i = _mm_loadu_si128(in + 0);
//     __m128i c29_load_rslt2_m128i = _mm_loadu_si128(in + 1);
//     __m128i c29_alignr_rslt1_m128i = _mm_alignr_epi8(c29_load_rslt2_m128i, c29_load_rslt1_m128i, 14);
//     __m256i c29_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_load_rslt1_m128i), c29_alignr_rslt1_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt1_m256i); // Unpack 1st 8 values.
//
//     __m128i c29_load_rslt3_m128i = _mm_loadu_si128(in + 2);
//     __m128i c29_alignr_rslt2_1_m128i = _mm_alignr_epi8(c29_load_rslt3_m128i, c29_load_rslt2_m128i, 13);
//     __m128i c29_load_rslt4_m128i = _mm_loadu_si128(in + 3);
//     __m128i c29_alignr_rslt2_2_m128i = _mm_alignr_epi8(c29_load_rslt4_m128i, c29_load_rslt3_m128i, 11);
//     __m256i c29_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt2_1_m128i), c29_alignr_rslt2_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt2_m256i); // Unpack 2nd 8 values.
//
//     __m128i c29_load_rslt5_m128i = _mm_loadu_si128(in + 4);
//     __m128i c29_alignr_rslt3_1_m128i = _mm_alignr_epi8(c29_load_rslt5_m128i, c29_load_rslt4_m128i, 10);
//     __m128i c29_load_rslt6_m128i = _mm_loadu_si128(in + 5);
//     __m128i c29_alignr_rslt3_2_m128i = _mm_alignr_epi8(c29_load_rslt6_m128i, c29_load_rslt5_m128i, 8);
//     __m256i c29_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt3_1_m128i), c29_alignr_rslt3_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt3_m256i); // Unpack 3rd 8 values.
//
//     __m128i c29_load_rslt7_m128i = _mm_loadu_si128(in + 6);
//     __m128i c29_alignr_rslt4_1_m128i = _mm_alignr_epi8(c29_load_rslt7_m128i, c29_load_rslt6_m128i, 7);
//     __m128i c29_load_rslt8_m128i = _mm_loadu_si128(in + 7);
//     __m128i c29_alignr_rslt4_2_m128i = _mm_alignr_epi8(c29_load_rslt8_m128i, c29_load_rslt7_m128i, 5);
//     __m256i c29_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt4_1_m128i), c29_alignr_rslt4_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt4_m256i); // Unpack 4th 8 values.
//
//     __m128i c29_load_rslt9_m128i = _mm_loadu_si128(in + 8);
//     __m128i c29_alignr_rslt5_1_m128i = _mm_alignr_epi8(c29_load_rslt9_m128i, c29_load_rslt8_m128i, 4);
//     __m128i c29_load_rslt10_m128i = _mm_loadu_si128(in + 9);
//     __m128i c29_alignr_rslt5_2_m128i = _mm_alignr_epi8(c29_load_rslt10_m128i, c29_load_rslt9_m128i, 2);
//     __m256i c29_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt5_1_m128i), c29_alignr_rslt5_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt5_m256i); // Unpack 5th 8 values.
//
//     __m128i c29_load_rslt11_m128i = _mm_loadu_si128(in + 10);
//     __m128i c29_alignr_rslt6_1_m128i = _mm_alignr_epi8(c29_load_rslt11_m128i, c29_load_rslt10_m128i, 1);
//     __m128i c29_alignr_rslt6_2_m128i = _mm_alignr_epi8(c29_load_rslt11_m128i, c29_load_rslt10_m128i, 15);
//     __m256i c29_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt6_1_m128i), c29_alignr_rslt6_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt6_m256i); // Unpack 6th 8 values.
//
//     __m128i c29_load_rslt12_m128i = _mm_loadu_si128(in + 11);
//     __m128i c29_alignr_rslt7_1_m128i = _mm_alignr_epi8(c29_load_rslt12_m128i, c29_load_rslt11_m128i, 14);
//     __m128i c29_load_rslt13_m128i = _mm_loadu_si128(in + 12);
//     __m128i c29_alignr_rslt7_2_m128i = _mm_alignr_epi8(c29_load_rslt13_m128i, c29_load_rslt12_m128i, 12);
//     __m256i c29_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt7_1_m128i), c29_alignr_rslt7_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt7_m256i); // Unpack 7th 8 values.
//
//     __m128i c29_load_rslt14_m128i = _mm_loadu_si128(in + 13);
//     __m128i c29_alignr_rslt8_1_m128i = _mm_alignr_epi8(c29_load_rslt14_m128i, c29_load_rslt13_m128i, 11);
//     __m128i c29_load_rslt15_m128i = _mm_loadu_si128(in + 14);
//     __m128i c29_alignr_rslt8_2_m128i = _mm_alignr_epi8(c29_load_rslt15_m128i, c29_load_rslt14_m128i, 9);
//     __m256i c29_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt8_1_m128i), c29_alignr_rslt8_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt8_m256i); // Unpack 8th 8 values.
//
//     __m128i c29_load_rslt16_m128i = _mm_loadu_si128(in + 15);
//     __m128i c29_alignr_rslt9_1_m128i = _mm_alignr_epi8(c29_load_rslt16_m128i, c29_load_rslt15_m128i, 8);
//     __m128i c29_load_rslt17_m128i = _mm_loadu_si128(in + 16);
//     __m128i c29_alignr_rslt9_2_m128i = _mm_alignr_epi8(c29_load_rslt17_m128i, c29_load_rslt16_m128i, 6);
//     __m256i c29_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt9_1_m128i), c29_alignr_rslt9_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt9_m256i); // Unpack 9th 8 values.
//
//     __m128i c29_load_rslt18_m128i = _mm_loadu_si128(in + 17);
//     __m128i c29_alignr_rslt10_1_m128i = _mm_alignr_epi8(c29_load_rslt18_m128i, c29_load_rslt17_m128i, 5);
//     __m128i c29_load_rslt19_m128i = _mm_loadu_si128(in + 18);
//     __m128i c29_alignr_rslt10_2_m128i = _mm_alignr_epi8(c29_load_rslt19_m128i, c29_load_rslt18_m128i, 3);
//     __m256i c29_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt10_1_m128i), c29_alignr_rslt10_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt10_m256i); // Unpack 10th 8 values.
//
//     __m128i c29_load_rslt20_m128i = _mm_loadu_si128(in + 19);
//     __m128i c29_alignr_rslt11_1_m128i = _mm_alignr_epi8(c29_load_rslt20_m128i, c29_load_rslt19_m128i, 2);
//     __m128i c29_alignr_rslt11_2_m128i = _mm_alignr_epi8(c29_load_rslt20_m128i, c29_load_rslt19_m128i, 16);
//     __m256i c29_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt11_1_m128i), c29_alignr_rslt11_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt11_m256i); // Unpack 11th 8 values.
//
//     __m128i c29_load_rslt21_m128i = _mm_loadu_si128(in + 20);
//     __m128i c29_alignr_rslt12_1_m128i = _mm_alignr_epi8(c29_load_rslt21_m128i, c29_load_rslt20_m128i, 15);
//     __m128i c29_load_rslt22_m128i = _mm_loadu_si128(in + 21);
//     __m128i c29_alignr_rslt12_2_m128i = _mm_alignr_epi8(c29_load_rslt22_m128i, c29_load_rslt21_m128i, 13);
//     __m256i c29_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt12_1_m128i), c29_alignr_rslt12_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt12_m256i); // Unpack 12th 8 values.
//
//     __m128i c29_load_rslt23_m128i = _mm_loadu_si128(in + 22);
//     __m128i c29_alignr_rslt13_1_m128i = _mm_alignr_epi8(c29_load_rslt23_m128i, c29_load_rslt22_m128i, 12);
//     __m128i c29_load_rslt24_m128i = _mm_loadu_si128(in + 23);
//     __m128i c29_alignr_rslt13_2_m128i = _mm_alignr_epi8(c29_load_rslt24_m128i, c29_load_rslt23_m128i, 10);
//     __m256i c29_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt13_1_m128i), c29_alignr_rslt13_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt13_m256i); // Unpack 13th 8 values.
//
//     __m128i c29_load_rslt25_m128i = _mm_loadu_si128(in + 24);
//     __m128i c29_alignr_rslt14_1_m128i = _mm_alignr_epi8(c29_load_rslt25_m128i, c29_load_rslt24_m128i, 9);
//     __m128i c29_load_rslt26_m128i = _mm_loadu_si128(in + 25);
//     __m128i c29_alignr_rslt14_2_m128i = _mm_alignr_epi8(c29_load_rslt26_m128i, c29_load_rslt25_m128i, 7);
//     __m256i c29_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt14_1_m128i), c29_alignr_rslt14_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt14_m256i); // Unpack 14th 8 values.
//
//     __m128i c29_load_rslt27_m128i = _mm_loadu_si128(in + 26);
//     __m128i c29_alignr_rslt15_1_m128i = _mm_alignr_epi8(c29_load_rslt27_m128i, c29_load_rslt26_m128i, 6);
//     __m128i c29_load_rslt28_m128i = _mm_loadu_si128(in + 27);
//     __m128i c29_alignr_rslt15_2_m128i = _mm_alignr_epi8(c29_load_rslt28_m128i, c29_load_rslt27_m128i, 4);
//     __m256i c29_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt15_1_m128i), c29_alignr_rslt15_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt15_m256i); // Unpack 15th 8 values.
//
//     __m128i c29_load_rslt29_m128i = _mm_loadu_si128(in + 28);
//     __m128i c29_alignr_rslt16_1_m128i = _mm_alignr_epi8(c29_load_rslt29_m128i, c29_load_rslt28_m128i, 3);
//     __m128i c29_alignr_rslt16_2_m128i = _mm_alignr_epi8(c29_load_rslt29_m128i, c29_load_rslt28_m128i, 17);
//     __m256i c29_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c29_alignr_rslt16_1_m128i), c29_alignr_rslt16_2_m128i, 1);
//     hor_avx2_unpack8_c29(out, c29_insert_rslt16_m256i); // Unpack 16th 8 values.
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c29(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c29_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[29]);
//		__m256i c29_sllv_rslt_m256i = _mm256_sllv_epi64(c29_shfl_rslt_m256i, _mm256_set_epi64x(0, 0, 1, 3));
//		__m256i c29_srlv_rslt1_m256i = _mm256_srlv_epi64(c29_sllv_rslt_m256i, _mm256_set_epi64x(3, 1, 0, 0));
//		__m256i c29_srlv_rslt2_m256i = _mm256_srlv_epi32(c29_srlv_rslt1_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[29]);
//		__m256i c29_rslt_m256i = _mm256_and_si256(c29_srlv_rslt2_m256i, SIMDMasks::AVX2_and_msk_m256i[29]);
//		_mm256_storeu_si256(out++, c29_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c29_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[29]);
//		__m256i c29_sllv_rslt_m256i = _mm256_sllv_epi64(c29_shfl_rslt_m256i, _mm256_set_epi64x(0, 0, 1, 3));
//		__m256i c29_srlv_rslt1_m256i = _mm256_srlv_epi64(c29_shfl_rslt_m256i, _mm256_set_epi64x(3, 1, 0, 0));
//		__m256i c29_srlv_rslt2_m256i = _mm256_srlv_epi32(c29_srlv_rslt1_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[29]);
//		__m256i c29_and_rslt_m256i = _mm256_and_si256(c29_srlv_rslt2_m256i, SIMDMasks::AVX2_and_msk_m256i[29]);
//		__m256i c29_rslt_m256i = _mm256_or_si256(c29_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 29));
//		_mm256_storeu_si256(out++, c29_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 30-bit values.
 * Load 30 SSE vectors, each containing 4 30-bit values. (5th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c30(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
	     __m128i c30_load_rslt1_m128i = _mm_loadu_si128(in++);
	     __m128i c30_load_rslt2_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt1_m128i = _mm_alignr_epi8(c30_load_rslt2_m128i, c30_load_rslt1_m128i, 15);
	     __m256i c30_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_load_rslt1_m128i), c30_alignr_rslt1_m128i, 1);
	     hor_avx2_unpack8_c30<0>(out, c30_insert_rslt1_m256i); // Unpack 1st 8 values.

	     __m128i c30_load_rslt3_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt2_1_m128i = _mm_alignr_epi8(c30_load_rslt3_m128i, c30_load_rslt2_m128i, 14);
	     __m128i c30_load_rslt4_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt2_2_m128i = _mm_alignr_epi8(c30_load_rslt4_m128i, c30_load_rslt3_m128i, 13);
	     __m256i c30_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt2_1_m128i), c30_alignr_rslt2_2_m128i, 1);
	     hor_avx2_unpack8_c30<0>(out, c30_insert_rslt2_m256i); // Unpack 2nd 8 values.

	     __m128i c30_load_rslt5_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt3_1_m128i = _mm_alignr_epi8(c30_load_rslt5_m128i, c30_load_rslt4_m128i, 12);
	     __m128i c30_load_rslt6_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt3_2_m128i = _mm_alignr_epi8(c30_load_rslt6_m128i, c30_load_rslt5_m128i, 11);
	     __m256i c30_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt3_1_m128i), c30_alignr_rslt3_2_m128i, 1);
	     hor_avx2_unpack8_c30<0>(out, c30_insert_rslt3_m256i); // Unpack 3rd 8 values.

	     __m128i c30_load_rslt7_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt4_1_m128i = _mm_alignr_epi8(c30_load_rslt7_m128i, c30_load_rslt6_m128i, 10);
	     __m128i c30_load_rslt8_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt4_2_m128i = _mm_alignr_epi8(c30_load_rslt8_m128i, c30_load_rslt7_m128i, 9);
	     __m256i c30_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt4_1_m128i), c30_alignr_rslt4_2_m128i, 1);
	     hor_avx2_unpack8_c30<0>(out, c30_insert_rslt4_m256i); // Unpack 4th 8 values.

	     __m128i c30_load_rslt9_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt5_1_m128i = _mm_alignr_epi8(c30_load_rslt9_m128i, c30_load_rslt8_m128i, 8);
	     __m128i c30_load_rslt10_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt5_2_m128i = _mm_alignr_epi8(c30_load_rslt10_m128i, c30_load_rslt9_m128i, 7);
	     __m256i c30_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt5_1_m128i), c30_alignr_rslt5_2_m128i, 1);
	     hor_avx2_unpack8_c30<0>(out, c30_insert_rslt5_m256i); // Unpack 5th 8 values.

	     __m128i c30_load_rslt11_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt6_1_m128i = _mm_alignr_epi8(c30_load_rslt11_m128i, c30_load_rslt10_m128i, 6);
	     __m128i c30_load_rslt12_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt6_2_m128i = _mm_alignr_epi8(c30_load_rslt12_m128i, c30_load_rslt11_m128i, 5);
	     __m256i c30_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt6_1_m128i), c30_alignr_rslt6_2_m128i, 1);
	     hor_avx2_unpack8_c30<0>(out, c30_insert_rslt6_m256i); // Unpack 6th 8 values.

	     __m128i c30_load_rslt13_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt7_1_m128i = _mm_alignr_epi8(c30_load_rslt13_m128i, c30_load_rslt12_m128i, 4);
	     __m128i c30_load_rslt14_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt7_2_m128i = _mm_alignr_epi8(c30_load_rslt14_m128i, c30_load_rslt13_m128i, 3);
	     __m256i c30_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt7_1_m128i), c30_alignr_rslt7_2_m128i, 1);
	     hor_avx2_unpack8_c30<0>(out, c30_insert_rslt7_m256i); // Unpack 7th 8 values.

	     __m128i c30_load_rslt15_m128i = _mm_loadu_si128(in++);
	     __m128i c30_alignr_rslt8_m128i = _mm_alignr_epi8(c30_load_rslt15_m128i, c30_load_rslt14_m128i, 2);
	     __m256i c30_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt8_m128i), c30_load_rslt15_m128i, 1);
	     hor_avx2_unpack8_c30<1>(out, c30_insert_rslt8_m256i); // Unpack 8th 8 values.
	}
}

template <bool IsRiceCoding>
template <int byte>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c30(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		const __m256i Hor_AVX2_c30_shfl_msk_m256i = _mm256_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0,
				14, 13, 12, 11,
				10, 9, 8, 7,
				7, 6, 5, 4,
				3, 2, 1, 0);
		__m256i c30_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c30_shfl_msk_m256i);
		__m256i c30_sllv_rslt_m256i = _mm256_sllv_epi64(c30_shfl_rslt_m256i, _mm256_set_epi64x(0, 2, 0, 2));
		__m256i c30_srlv_rslt1_m256i = _mm256_srlv_epi64(c30_sllv_rslt_m256i, _mm256_set_epi64x(2, 0, 2, 0));
		__m256i c30_srlv_rslt2_m256i = _mm256_srlv_epi32(c30_srlv_rslt1_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[30]);
		__m256i c30_rslt_m256i = _mm256_and_si256(c30_srlv_rslt2_m256i, SIMDMasks::AVX2_and_msk_m256i[30]);
		_mm256_storeu_si256(out++, c30_rslt_m256i);
	}
	else { // For Rice and OptRice.
		const __m256i Hor_AVX2_c30_shfl_msk_m256i = _mm256_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0,
				14, 13, 12, 11,
				10, 9, 8, 7,
				7, 6, 5, 4,
				3, 2, 1, 0);
		__m256i c30_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, Hor_AVX2_c30_shfl_msk_m256i);
		__m256i c30_sllv_rslt_m256i = _mm256_sllv_epi64(c30_shfl_rslt_m256i, _mm256_set_epi64x(0, 2, 0, 2));
		__m256i c30_srlv_rslt1_m256i = _mm256_srlv_epi64(c30_sllv_rslt_m256i, _mm256_set_epi64x(2, 0, 2, 0));
		__m256i c30_srlv_rslt2_m256i = _mm256_srlv_epi32(c30_srlv_rslt1_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[30]);
		__m256i c30_and_rslt_m256i = _mm256_and_si256(c30_srlv_rslt2_m256i, SIMDMasks::AVX2_and_msk_m256i[30]);
		__m256i c30_rslt_m256i = _mm256_or_si256(c30_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 30));
		_mm256_storeu_si256(out++, c30_rslt_m256i);
	}
}

///**
// * AVX2-based unpacking 128 30-bit values.
// * Load 30 SSE vectors, each containing 4 30-bit values. (5th is incomplete)
// */
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c30(__m256i *  __restrict__  out,
//		const __m128i *  __restrict__  in) {
//	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 64) {
//	     __m128i c30_load_rslt1_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_load_rslt2_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt1_m128i = _mm_alignr_epi8(c30_load_rslt2_m128i, c30_load_rslt1_m128i, 15);
//	     __m256i c30_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_load_rslt1_m128i), c30_alignr_rslt1_m128i, 1);
//	     hor_avx2_unpack8_c30(out, c30_insert_rslt1_m256i); // Unpack 1st 8 values.
//
//	     __m128i c30_load_rslt3_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt2_1_m128i = _mm_alignr_epi8(c30_load_rslt3_m128i, c30_load_rslt2_m128i, 14);
//	     __m128i c30_load_rslt4_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt2_2_m128i = _mm_alignr_epi8(c30_load_rslt4_m128i, c30_load_rslt3_m128i, 13);
//	     __m256i c30_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt2_1_m128i), c30_alignr_rslt2_2_m128i, 1);
//	     hor_avx2_unpack8_c30(out, c30_insert_rslt2_m256i); // Unpack 2nd 8 values.
//
//	     __m128i c30_load_rslt5_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt3_1_m128i = _mm_alignr_epi8(c30_load_rslt5_m128i, c30_load_rslt4_m128i, 12);
//	     __m128i c30_load_rslt6_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt3_2_m128i = _mm_alignr_epi8(c30_load_rslt6_m128i, c30_load_rslt5_m128i, 11);
//	     __m256i c30_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt3_1_m128i), c30_alignr_rslt3_2_m128i, 1);
//	     hor_avx2_unpack8_c30(out, c30_insert_rslt3_m256i); // Unpack 3rd 8 values.
//
//	     __m128i c30_load_rslt7_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt4_1_m128i = _mm_alignr_epi8(c30_load_rslt7_m128i, c30_load_rslt6_m128i, 10);
//	     __m128i c30_load_rslt8_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt4_2_m128i = _mm_alignr_epi8(c30_load_rslt8_m128i, c30_load_rslt7_m128i, 9);
//	     __m256i c30_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt4_1_m128i), c30_alignr_rslt4_2_m128i, 1);
//	     hor_avx2_unpack8_c30(out, c30_insert_rslt4_m256i); // Unpack 4th 8 values.
//
//	     __m128i c30_load_rslt9_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt5_1_m128i = _mm_alignr_epi8(c30_load_rslt9_m128i, c30_load_rslt8_m128i, 8);
//	     __m128i c30_load_rslt10_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt5_2_m128i = _mm_alignr_epi8(c30_load_rslt10_m128i, c30_load_rslt9_m128i, 7);
//	     __m256i c30_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt5_1_m128i), c30_alignr_rslt5_2_m128i, 1);
//	     hor_avx2_unpack8_c30(out, c30_insert_rslt5_m256i); // Unpack 5th 8 values.
//
//	     __m128i c30_load_rslt11_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt6_1_m128i = _mm_alignr_epi8(c30_load_rslt11_m128i, c30_load_rslt10_m128i, 6);
//	     __m128i c30_load_rslt12_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt6_2_m128i = _mm_alignr_epi8(c30_load_rslt12_m128i, c30_load_rslt11_m128i, 5);
//	     __m256i c30_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt6_1_m128i), c30_alignr_rslt6_2_m128i, 1);
//	     hor_avx2_unpack8_c30(out, c30_insert_rslt6_m256i); // Unpack 6th 8 values.
//
//	     __m128i c30_load_rslt13_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt7_1_m128i = _mm_alignr_epi8(c30_load_rslt13_m128i, c30_load_rslt12_m128i, 4);
//	     __m128i c30_load_rslt14_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt7_2_m128i = _mm_alignr_epi8(c30_load_rslt14_m128i, c30_load_rslt13_m128i, 3);
//	     __m256i c30_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt7_1_m128i), c30_alignr_rslt7_2_m128i, 1);
//	     hor_avx2_unpack8_c30(out, c30_insert_rslt7_m256i); // Unpack 7th 8 values.
//
//	     __m128i c30_load_rslt15_m128i = _mm_loadu_si128(in++);
//	     __m128i c30_alignr_rslt8_1_m128i = _mm_alignr_epi8(c30_load_rslt15_m128i, c30_load_rslt14_m128i, 2);
//	     __m128i c30_alignr_rslt8_2_m128i = _mm_alignr_epi8(c30_load_rslt15_m128i, c30_load_rslt14_m128i, 17);
//	     __m256i c30_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c30_alignr_rslt8_1_m128i), c30_alignr_rslt8_2_m128i, 1);
//	     hor_avx2_unpack8_c30(out, c30_insert_rslt8_m256i); // Unpack 8th 8 values.
//	}
//}
//
//template <bool IsRiceCoding>
//void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c30(__m256i *  __restrict__  &out,
//		const __m256i &InReg) {
//	if (!IsRiceCoding) { // For NewPFor and OptPFor.
//		__m256i c30_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[30]);
//		__m256i c30_sllv_rslt_m256i = _mm256_sllv_epi64(c30_shfl_rslt_m256i, _mm256_set_epi64x(0, 2, 0, 2));
//		__m256i c30_srlv_rslt1_m256i = _mm256_srlv_epi64(c30_sllv_rslt_m256i, _mm256_set_epi64x(2, 0, 2, 0));
//		__m256i c30_srlv_rslt2_m256i = _mm256_srlv_epi32(c30_srlv_rslt1_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[30]);
//		__m256i c30_rslt_m256i = _mm256_and_si256(c30_srlv_rslt2_m256i, SIMDMasks::AVX2_and_msk_m256i[30]);
//		_mm256_storeu_si256(out++, c30_rslt_m256i);
//	}
//	else { // For Rice and OptRice.
//		__m256i c30_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[30]);
//		__m256i c30_sllv_rslt_m256i = _mm256_sllv_epi64(c30_shfl_rslt_m256i, _mm256_set_epi64x(0, 2, 0, 2));
//		__m256i c30_srlv_rslt1_m256i = _mm256_srlv_epi64(c30_sllv_rslt_m256i, _mm256_set_epi64x(2, 0, 2, 0));
//		__m256i c30_srlv_rslt2_m256i = _mm256_srlv_epi32(c30_srlv_rslt1_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[30]);
//		__m256i c30_and_rslt_m256i = _mm256_and_si256(c30_srlv_rslt2_m256i, SIMDMasks::AVX2_and_msk_m256i[30]);
//		__m256i c30_rslt_m256i = _mm256_or_si256(c30_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 30));
//		_mm256_storeu_si256(out++, c30_rslt_m256i);
//	}
//}


/**
 * AVX2-based unpacking 128 31-bit values.
 * Load 31 SSE vectors, each containing 4 31-bit values. (5th is incomplete)
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c31(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	__m128i c31_load_rslt1_m128i = _mm_loadu_si128(in + 0);
	__m128i c31_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c31_alignr_rslt1_m128i = _mm_alignr_epi8(c31_load_rslt2_m128i, c31_load_rslt1_m128i, 15);
	__m256i c31_insert_rslt1_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_load_rslt1_m128i), c31_alignr_rslt1_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt1_m256i); // Unpack 1st 8 values.

	__m128i c31_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c31_alignr_rslt2_1_m128i = _mm_alignr_epi8(c31_load_rslt3_m128i, c31_load_rslt2_m128i, 15);
	__m128i c31_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c31_alignr_rslt2_2_m128i = _mm_alignr_epi8(c31_load_rslt4_m128i, c31_load_rslt3_m128i, 14);
	__m256i c31_insert_rslt2_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt2_1_m128i), c31_alignr_rslt2_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt2_m256i); // Unpack 2nd 8 values.

	__m128i c31_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c31_alignr_rslt3_1_m128i = _mm_alignr_epi8(c31_load_rslt5_m128i, c31_load_rslt4_m128i, 14);
	__m128i c31_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c31_alignr_rslt3_2_m128i = _mm_alignr_epi8(c31_load_rslt6_m128i, c31_load_rslt5_m128i, 13);
	__m256i c31_insert_rslt3_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt3_1_m128i), c31_alignr_rslt3_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt3_m256i); // Unpack 3rd 8 values.

	__m128i c31_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c31_alignr_rslt4_1_m128i = _mm_alignr_epi8(c31_load_rslt7_m128i, c31_load_rslt6_m128i, 13);
	__m128i c31_load_rslt8_m128i = _mm_loadu_si128(in + 7);
	__m128i c31_alignr_rslt4_2_m128i = _mm_alignr_epi8(c31_load_rslt8_m128i, c31_load_rslt7_m128i, 12);
	__m256i c31_insert_rslt4_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt4_1_m128i), c31_alignr_rslt4_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt4_m256i); // Unpack 4th 8 values.

	__m128i c31_load_rslt9_m128i = _mm_loadu_si128(in + 8);
	__m128i c31_alignr_rslt5_1_m128i = _mm_alignr_epi8(c31_load_rslt9_m128i, c31_load_rslt8_m128i, 12);
	__m128i c31_load_rslt10_m128i = _mm_loadu_si128(in + 9);
	__m128i c31_alignr_rslt5_2_m128i = _mm_alignr_epi8(c31_load_rslt10_m128i, c31_load_rslt9_m128i, 11);
	__m256i c31_insert_rslt5_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt5_1_m128i), c31_alignr_rslt5_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt5_m256i); // Unpack 5th 8 values.

	__m128i c31_load_rslt11_m128i = _mm_loadu_si128(in + 10);
	__m128i c31_alignr_rslt6_1_m128i = _mm_alignr_epi8(c31_load_rslt11_m128i, c31_load_rslt10_m128i, 11);
	__m128i c31_load_rslt12_m128i = _mm_loadu_si128(in + 11);
	__m128i c31_alignr_rslt6_2_m128i = _mm_alignr_epi8(c31_load_rslt12_m128i, c31_load_rslt11_m128i, 10);
	__m256i c31_insert_rslt6_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt6_1_m128i), c31_alignr_rslt6_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt6_m256i); // Unpack 6th 8 values.

	__m128i c31_load_rslt13_m128i = _mm_loadu_si128(in + 12);
	__m128i c31_alignr_rslt7_1_m128i = _mm_alignr_epi8(c31_load_rslt13_m128i, c31_load_rslt12_m128i, 10);
	__m128i c31_load_rslt14_m128i = _mm_loadu_si128(in + 13);
	__m128i c31_alignr_rslt7_2_m128i = _mm_alignr_epi8(c31_load_rslt14_m128i, c31_load_rslt13_m128i, 9);
	__m256i c31_insert_rslt7_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt7_1_m128i), c31_alignr_rslt7_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt7_m256i); // Unpack 7th 8 values.

	__m128i c31_load_rslt15_m128i = _mm_loadu_si128(in + 14);
	__m128i c31_alignr_rslt8_1_m128i = _mm_alignr_epi8(c31_load_rslt15_m128i, c31_load_rslt14_m128i, 9);
	__m128i c31_load_rslt16_m128i = _mm_loadu_si128(in + 15);
	__m128i c31_alignr_rslt8_2_m128i = _mm_alignr_epi8(c31_load_rslt16_m128i, c31_load_rslt15_m128i, 8);
	__m256i c31_insert_rslt8_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt8_1_m128i), c31_alignr_rslt8_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt8_m256i); // Unpack 7th 8 values.

	__m128i c31_load_rslt17_m128i = _mm_loadu_si128(in + 16);
	__m128i c31_alignr_rslt9_1_m128i = _mm_alignr_epi8(c31_load_rslt17_m128i, c31_load_rslt16_m128i, 8);
	__m128i c31_load_rslt18_m128i = _mm_loadu_si128(in + 17);
	__m128i c31_alignr_rslt9_2_m128i = _mm_alignr_epi8(c31_load_rslt18_m128i, c31_load_rslt17_m128i, 7);
	__m256i c31_insert_rslt9_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt9_1_m128i), c31_alignr_rslt9_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt9_m256i); // Unpack 8th 8 values.

	__m128i c31_load_rslt19_m128i = _mm_loadu_si128(in + 18);
	__m128i c31_alignr_rslt10_1_m128i = _mm_alignr_epi8(c31_load_rslt19_m128i, c31_load_rslt18_m128i, 7);
	__m128i c31_load_rslt20_m128i = _mm_loadu_si128(in + 19);
	__m128i c31_alignr_rslt10_2_m128i = _mm_alignr_epi8(c31_load_rslt20_m128i, c31_load_rslt19_m128i, 6);
	__m256i c31_insert_rslt10_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt10_1_m128i), c31_alignr_rslt10_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt10_m256i); // Unpack 9th 8 values.

	__m128i c31_load_rslt21_m128i = _mm_loadu_si128(in + 20);
	__m128i c31_alignr_rslt11_1_m128i = _mm_alignr_epi8(c31_load_rslt21_m128i, c31_load_rslt20_m128i, 6);
	__m128i c31_load_rslt22_m128i = _mm_loadu_si128(in + 21);
	__m128i c31_alignr_rslt11_2_m128i = _mm_alignr_epi8(c31_load_rslt22_m128i, c31_load_rslt21_m128i, 5);
	__m256i c31_insert_rslt11_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt11_1_m128i), c31_alignr_rslt11_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt11_m256i); // Unpack 10th 8 values.

	__m128i c31_load_rslt23_m128i = _mm_loadu_si128(in + 22);
	__m128i c31_alignr_rslt12_1_m128i = _mm_alignr_epi8(c31_load_rslt23_m128i, c31_load_rslt22_m128i, 5);
	__m128i c31_load_rslt24_m128i = _mm_loadu_si128(in + 23);
	__m128i c31_alignr_rslt12_2_m128i = _mm_alignr_epi8(c31_load_rslt24_m128i, c31_load_rslt23_m128i, 4);
	__m256i c31_insert_rslt12_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt12_1_m128i), c31_alignr_rslt12_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt12_m256i); // Unpack 11th 8 values.

	__m128i c31_load_rslt25_m128i = _mm_loadu_si128(in + 24);
	__m128i c31_alignr_rslt13_1_m128i = _mm_alignr_epi8(c31_load_rslt25_m128i, c31_load_rslt24_m128i, 4);
	__m128i c31_load_rslt26_m128i = _mm_loadu_si128(in + 25);
	__m128i c31_alignr_rslt13_2_m128i = _mm_alignr_epi8(c31_load_rslt26_m128i, c31_load_rslt25_m128i, 3);
	__m256i c31_insert_rslt13_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt13_1_m128i), c31_alignr_rslt13_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt13_m256i); // Unpack 12th 8 values.

	__m128i c31_load_rslt27_m128i = _mm_loadu_si128(in + 26);
	__m128i c31_alignr_rslt14_1_m128i = _mm_alignr_epi8(c31_load_rslt27_m128i, c31_load_rslt26_m128i, 3);
	__m128i c31_load_rslt28_m128i = _mm_loadu_si128(in + 27);
	__m128i c31_alignr_rslt14_2_m128i = _mm_alignr_epi8(c31_load_rslt28_m128i, c31_load_rslt27_m128i, 2);
	__m256i c31_insert_rslt14_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt14_1_m128i), c31_alignr_rslt14_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt14_m256i); // Unpack 13th 8 values.

	__m128i c31_load_rslt29_m128i = _mm_loadu_si128(in + 28);
	__m128i c31_alignr_rslt15_1_m128i = _mm_alignr_epi8(c31_load_rslt29_m128i, c31_load_rslt28_m128i, 2);
	__m128i c31_load_rslt30_m128i = _mm_loadu_si128(in + 29);
	__m128i c31_alignr_rslt15_2_m128i = _mm_alignr_epi8(c31_load_rslt30_m128i, c31_load_rslt29_m128i, 1);
	__m256i c31_insert_rslt15_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt15_1_m128i), c31_alignr_rslt15_2_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt15_m256i); // Unpack 15th 8 values.

	__m128i c31_load_rslt31_m128i = _mm_loadu_si128(in + 30);
	__m128i c31_alignr_rslt16_m128i = _mm_alignr_epi8(c31_load_rslt31_m128i, c31_load_rslt30_m128i, 1);
	__m256i c31_insert_rslt16_m256i = _mm256_insertf128_si256(_mm256_castsi128_si256(c31_alignr_rslt16_m128i), c31_load_rslt31_m128i, 1);
	hor_avx2_unpack8_c31(out, c31_insert_rslt16_m256i); // Unpack 16th 8 values.
}

template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::hor_avx2_unpack8_c31(__m256i *  __restrict__  &out,
		const __m256i &InReg) {
	if (!IsRiceCoding) { // For NewPFor and OptPFor.
		__m256i c31_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[31]);
		__m256i c31_sllv_rstl1_m256i = _mm256_sllv_epi64(c31_shfl_rslt_m256i, _mm256_set_epi64x(0, 5, 0, 0));
		__m256i c31_srlv_rslt1_m256i = _mm256_srlv_epi64(c31_sllv_rstl1_m256i, _mm256_set_epi64x(0, 0, 5, 0));
		__m256i c31_sllv_rslt2_m256i = _mm256_sllv_epi64(InReg, _mm256_set_epi64x(0, 0, 3, 1));
		__m256i c31_srlv_rslt2_m256i = _mm256_srlv_epi64(c31_sllv_rslt2_m256i, _mm256_set_epi64x(1, 3, 0, 0));
		const __m256i mask = _mm256_set_epi8(
				0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00,
				0xFF, 0xFF, 0xFF, 0xFF,
				0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00,
				0xFF, 0xFF, 0xFF, 0xFF,
				0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00);
		__m256i c31_blend_rslt_m256i = _mm256_blendv_epi8(c31_srlv_rslt2_m256i, c31_srlv_rslt1_m256i, mask);

		__m256i c31_srlv_rslt3_m256i = _mm256_srlv_epi32(c31_blend_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[31]);
		__m256i c31_rslt_m256i = _mm256_and_si256(c31_srlv_rslt3_m256i, SIMDMasks::AVX2_and_msk_m256i[31]);
		_mm256_storeu_si256(out++, c31_rslt_m256i);
	}
	else { // For Rice and OptRice.
		__m256i c31_shfl_rslt_m256i = _mm256_shuffle_epi8(InReg, SIMDMasks::Hor_AVX2_shfl_msk_m256i[31]);
		__m256i c31_sllv_rstl1_m256i = _mm256_sllv_epi64(c31_shfl_rslt_m256i, _mm256_set_epi64x(0, 5, 0, 0));
		__m256i c31_srlv_rslt1_m256i = _mm256_srlv_epi64(c31_sllv_rstl1_m256i, _mm256_set_epi64x(0, 0, 5, 0));
		__m256i c31_sllv_rslt2_m256i = _mm256_sllv_epi64(InReg, _mm256_set_epi64x(0, 0, 3, 1));
		__m256i c31_srlv_rslt2_m256i = _mm256_srlv_epi64(c31_sllv_rslt2_m256i, _mm256_set_epi64x(1, 3, 0, 0));
		const __m256i mask = _mm256_set_epi8(
				0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00,
				0xFF, 0xFF, 0xFF, 0xFF,
				0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00,
				0xFF, 0xFF, 0xFF, 0xFF,
				0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00);
		__m256i c31_blend_rslt_m256i = _mm256_blendv_epi8(c31_srlv_rslt2_m256i, c31_srlv_rslt1_m256i, mask);

		__m256i c31_srlv_rslt3_m256i = _mm256_srlv_epi32(c31_blend_rslt_m256i, SIMDMasks::Hor_AVX2_srlv_msk_m256i[31]);
		__m256i c31_and_rslt_m256i = _mm256_and_si256(c31_srlv_rslt3_m256i, SIMDMasks::AVX2_and_msk_m256i[31]);
		__m256i c31_rslt_m256i = _mm256_or_si256(c31_and_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient++), 31));
		_mm256_storeu_si256(out++, c31_rslt_m256i);
	}
}


/**
 * AVX2-based unpacking 128 32-bit values.
 */
template <bool IsRiceCoding>
void HorUnpacker<AVX, IsRiceCoding>::horizontalunpack_c32(__m256i *  __restrict__  out,
		const __m128i *  __restrict__  in) {
	uint32_t *outPtr = reinterpret_cast<uint32_t *>(out);
	const uint32_t *inPtr = reinterpret_cast<const uint32_t *>(in);
	for (uint32_t valuesUnpacked = 0; valuesUnpacked < 128; valuesUnpacked += 32) {
		memcpy32(outPtr, inPtr);
		outPtr += 32;
		inPtr += 32;
	}
}


#endif // CODECS_HORAVXUNPACKER_H_
