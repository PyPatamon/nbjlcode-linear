/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *	   Created on: Mar 10, 2016
 */
/**
 * Based on code by
 *     Daniel Lemire, https://github.com/lemire/FastPFor
 * which was available under the Apache License, Version 2.0.
 */

#ifndef CODECS_SIMPLE9_H_
#define CODECS_SIMPLE9_H_

#if !defined(__GNUC__) && !defined(_MSC_VER)
#error Simple9.h requires GCC or MSVC
#endif

#include "IntegerCodec.h"
#include "Portability.h"

#if CODECS_SSE_PREREQ(4, 1)
namespace Codecs {
namespace SIMDMasks {
extern const __m128i Simple_SSE4_mul_msk_m128i[9];
extern const __m128i SSE2_and_msk_m128i[33];
} // namespace SIMDMasks
} // namespace Codecs
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
namespace Codecs {
namespace SIMDMasks {
extern const __m256i Simple_AVX2_srlv_msk_m256i[9][4];
extern const __m256i AVX2_and_msk_m256i[33];
} // namespace SIMDMasks
} // namespace Codecs
#endif /* __AVX2__ */

namespace Codecs {

template <typename InstructionSet, bool AllowOverShooting>
class Simple9;

using Simple9_Scalar = Simple9<Scalar, false>;

#if CODECS_SSE_PREREQ(4, 1)
using Simple9_SSE = Simple9<SSE, false>;
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
using Simple9_AVX = Simple9<AVX, false>;
#endif /* __AVX2__ */


/**
 * Simple-9 (S9) encoding for 32-bit integers.
 *
 * Simple-9 divides a 32-bit word into two parts, a 4-bit selector and 28 data bits.
 * There are 9 possible ways in which the 28 data bits can be partitioned into equal
 * length binary codes, together with possibly unused bits. For example, if the next
 * 3 integers are less than 512, then we can store them as three 9-bit binary codes,
 * leaving 1 data bit unused. The selector value describes which of the 9 partitions
 * is being used in the word. Note that integers >= 2^28 cannot be represented in S9.
 *
 * Follows
 *
 * V. N. Anh and A. Moffat. Improved word-aligned binary compression for text indexing.
 * IEEE Trans. Knowl. Data Eng., 18(6):857–861, 2006.
 */
template <typename InstructionSet, bool AllowOverShooting>
class Simple9Base : public IntegerCodec {
public:
    enum {
        SIMPLE9_LOGDESC = 4, // Number of bits for the selector.
        SIMPLE9_LEN = 9      // Number of possible partitions for the 28 data bits.
    };

    /**
     * We determine the size of buf based on:
     * 1) There're at most kMaxDecodingCount-1 values remaining to be decoded;
     * 2) Our codecs can write out at most kMaxDecodingCount-1 garbage values
     *    when decoding the last 32-bit word.
     * See decodeArray() for more details.
     */
    Simple9Base() : buf((Derived::kMaxDecodingCount-1) * 2) { }

    /**
     * Encode values into a 32-bit word.
     * This is called when there are >= 28 values to be coded. 
     */
    static void encodeWord(const uint32_t * &in, uint32_t &dword) {
        if (trymefull<28, 1>(in)) {
            writeSelector(0, dword);
            writeData<28, 1>(in, 28, dword);

            in += 28;
        }
        else if (trymefull<14, 2>(in)) {
            writeSelector(1, dword);
            writeData<14, 2>(in, 14, dword);

            in += 14;
        }
        else if (trymefull<9, 3>(in)) {
            writeSelector(2, dword);
            writeData<9, 3>(in, 9, dword);

            in += 9;
        }
        else if (trymefull<7, 4>(in)) {
            writeSelector(3, dword);
            writeData<7, 4>(in, 7, dword);

            in += 7;
        }
        else if (trymefull<5, 5>(in)) {
            writeSelector(4, dword);
            writeData<5, 5>(in, 5, dword);

            in += 5;
        }
        else if (trymefull<4, 7>(in)) {
            writeSelector(5, dword);
            writeData<4, 7>(in, 4, dword);

            in += 4;
        }
        else if (trymefull<3, 9>(in)) {
            writeSelector(6, dword);
            writeData<3, 9>(in, 3, dword);

            in += 3;
        }
        else if (trymefull<2, 14>(in)) {
            writeSelector(7, dword);
            writeData<2, 14>(in, 2, dword);

            in += 2;
        }
        else if (trymefull<1, 28>(in)) {
            writeSelector(8, dword);
            writeData<1, 28>(in, 1, dword);

            ++in;
        }
        else {
            std::cerr << "Input's out of range: " << *in << std::endl;
            throw std::runtime_error("You tried to apply " + Derived::codecname() + 
                                     " to an incompatible set of integers.");
        }
    }

    /**
     * Encode values into a 32-bit word.
     * This is called when there are < 28 values to be coded. 
     */
    static void encodeWord(const uint32_t * &in, uint32_t &valuesRemaining, uint32_t &dword) {
        uint32_t valuesEncoded = 0;
        if (tryme<28, 1>(in, valuesRemaining)) {
            valuesEncoded = (valuesRemaining < 28) ? valuesRemaining : 28;

            writeSelector(0, dword);
            writeData<28, 1>(in, valuesEncoded, dword);
        }
        else if (tryme<14, 2>(in, valuesRemaining)) {
            valuesEncoded = (valuesRemaining < 14) ? valuesRemaining : 14;

            writeSelector(1, dword);
            writeData<14, 2>(in, valuesEncoded, dword);
        }
        else if (tryme<9, 3>(in, valuesRemaining)) {
            valuesEncoded = (valuesRemaining < 9) ? valuesRemaining : 9;

            writeSelector(2, dword);
            writeData<9, 3>(in, valuesEncoded, dword);
        }
        else if (tryme<7, 4>(in, valuesRemaining)) {
            valuesEncoded = (valuesRemaining < 7) ? valuesRemaining : 7;

            writeSelector(3, dword);
            writeData<7, 4>(in, valuesEncoded, dword);
        }
        else if (tryme<5, 5>(in, valuesRemaining)) {
            valuesEncoded = (valuesRemaining < 5) ? valuesRemaining : 5;

            writeSelector(4, dword);
            writeData<5, 5>(in, valuesEncoded, dword);
        }
        else if (tryme<4, 7>(in, valuesRemaining)) {
            valuesEncoded = (valuesRemaining < 4) ? valuesRemaining : 4;

            writeSelector(5, dword);
            writeData<4, 7>(in, valuesEncoded, dword);
        }
        else if (tryme<3, 9>(in, valuesRemaining)) {
            valuesEncoded = (valuesRemaining < 3) ? valuesRemaining : 3;

            writeSelector(6, dword);
            writeData<3, 9>(in, valuesEncoded, dword);
        }
        else if (tryme<2, 14>(in, valuesRemaining)) {
            valuesEncoded = (valuesRemaining < 2) ? valuesRemaining : 2;

            writeSelector(7, dword);
            writeData<2, 14>(in, valuesEncoded, dword);
        }
        else if (tryme<1, 28>(in, valuesRemaining)) {
            valuesEncoded = 1;

            writeSelector(8, dword);
            writeData<1, 28>(in, valuesEncoded, dword);
        }
        else {
            std::cerr << "Input's out of range: " << *in << std::endl;
            throw std::runtime_error("You tried to apply " + Derived::codecname() + 
                                     " to an incompatible set of integers.");
        }

        in += valuesEncoded;
        valuesRemaining -= valuesEncoded;
    }

	virtual void encodeArray(const uint32_t *in, uint64_t nvalue,
            uint32_t *out, uint64_t &csize) {
        const uint32_t *const endin = in + nvalue;
        const uint32_t *const initout = out;

        // When there're >= 28 values to be coded,
        // encode values into complete 32-bit words.
        // Here complete means all encoded values are from in.
        while (endin - in >= 28) {
            uint32_t &dword = out[0];
            ++out;
            encodeWord(in, dword);
        }

        // Encode remaining (< 28) values. The
        // last 32-bit word might be incomplete.
        // Here incomplete means there might be 
        // garbage 0s padded in it.
        uint32_t valuesRemaining = endin - in;
        while (endin > in) {
            uint32_t &dword = out[0];
            ++out;
            encodeWord(in, valuesRemaining, dword);
        }

        csize = out - initout; // Number of 32-bit words consumed.
    }

    virtual const uint32_t *decodeArray(const uint32_t *in, uint64_t csize,
            uint32_t *out, uint64_t nvalue) {
        const uint32_t *const endout = out + nvalue;

        if (AllowOverShooting) { 
            // This implementation is more efficient but
            // can overshoot, which might result in disaster, 
            // e.g., if out points to an array of size nvalue.
            // For use with NewPFor and OptPFor.
            while (endout > out) {
                const uint32_t dword = in[0];
                ++in;
                Derived::decodeWord(dword, out);
            }
        }
        else { 
            // kMaxDecodingCount denotes the maximum number
            // of values decodeWord can write out at each call.
            // Therefore when there're >= kMaxDecodingCount values
            // to be decoded, it's safe to write the decoded
            // values directly to out.
            while (endout - out >= Derived::kMaxDecodingCount) {
                const uint32_t dword = in[0];
                ++in;
                Derived::decodeWord(dword, out);
            }

            // For the remaining (< kMaxDecodingCount) values,
            // to avoid overshooting, we write the decoded 
            // values to the array buf first, and then copy
            // only those we need to out.
            uint32_t valuesRemaining = endout - out;
            uint32_t *ptr = buf.data();
            const uint32_t *const endptr = ptr + valuesRemaining;
            while (endptr > ptr) {
                const uint32_t dword = in[0];
                ++in;
                Derived::decodeWord(dword, ptr);
            }
            // No need to check whether valuesRemaining is zero.
            memcpy(out, buf.data(), sizeof(uint32_t) * valuesRemaining);
        }

        return in;
    }

	virtual std::string name() const {
		return Derived::codecname();
	}

private:
    using Derived = Simple9<InstructionSet, AllowOverShooting>;
                          
    template <uint32_t num1, uint32_t log1>
    __attribute__ ((pure))
    static bool trymefull(const uint32_t *in) {
        for (uint32_t i = 0; i < num1; ++i) {
            if ((in[i]) >= (1U << log1))
                return false;
        }
        return true;
    }

    template <uint32_t num1, uint32_t log1>
    __attribute__ ((pure))
    static bool tryme(const uint32_t *in, uint32_t nvalue) {
        const uint32_t min = (nvalue < num1) ? nvalue : num1;
        for (uint32_t i = 0; i < min; ++i) {
            if ((in[i]) >= (1U << log1))
                return false;
        }
        return true;
    }

    static void writeSelector(uint32_t selector, uint32_t &dword) {
        assert(selector < SIMPLE9_LEN);
    	dword = selector << (32 - SIMPLE9_LOGDESC);
    }

    template <uint32_t num1, uint32_t log1>
    static void writeData(const uint32_t *in, uint32_t nvalue, uint32_t &dword) {
        assert(nvalue <= num1);
    	uint32_t shift = 32 - SIMPLE9_LOGDESC;
    	for (uint32_t i = 0; i < nvalue; ++i) {
    		shift -= log1;
    		dword |= (in[i] << shift);
    	}
    }

    std::vector<uint32_t> buf;
};


template <bool AllowOverShooting>
class Simple9<Scalar, AllowOverShooting> : public Simple9Base<Scalar, AllowOverShooting> {
public:
    using Simple9Base<Scalar, AllowOverShooting>::SIMPLE9_LOGDESC;
    using Simple9Base<Scalar, AllowOverShooting>::SIMPLE9_LEN;

    enum { kMaxDecodingCount = 28 }; // Maximum number of values decodeWord can
                                     // write out at each call. (case 0)

    static const std::string codecname() {
        return "Simple9_Scalar";
    }

    static void decodeWord(uint32_t dword, uint32_t * &out) {
        const uint32_t selector = dword >> (32 - SIMPLE9_LOGDESC);
        assert(selector < SIMPLE9_LEN);
		switch (selector) {
		case 0: // 28 * 1-bit
			out[0] = (dword >> 27) & 0x01;
			out[1] = (dword >> 26) & 0x01;
			out[2] = (dword >> 25) & 0x01;
			out[3] = (dword >> 24) & 0x01;
			out[4] = (dword >> 23) & 0x01;
			out[5] = (dword >> 22) & 0x01;
			out[6] = (dword >> 21) & 0x01;
			out[7] = (dword >> 20) & 0x01;
			out[8] = (dword >> 19) & 0x01;
			out[9] = (dword >> 18) & 0x01;
			out[10] = (dword >> 17) & 0x01;
			out[11] = (dword >> 16) & 0x01;
			out[12] = (dword >> 15) & 0x01;
			out[13] = (dword >> 14) & 0x01;
			out[14] = (dword >> 13) & 0x01;
			out[15] = (dword >> 12) & 0x01;
			out[16] = (dword >> 11) & 0x01;
			out[17] = (dword >> 10) & 0x01;
			out[18] = (dword >> 9) & 0x01;
			out[19] = (dword >> 8) & 0x01;
			out[20] = (dword >> 7) & 0x01;
			out[21] = (dword >> 6) & 0x01;
			out[22] = (dword >> 5) & 0x01;
			out[23] = (dword >> 4) & 0x01;
			out[24] = (dword >> 3) & 0x01;
			out[25] = (dword >> 2) & 0x01;
			out[26] = (dword >> 1) & 0x01;
			out[27] = dword & 0x01;

			out += 28;

			return;
		case 1: // 14 * 2-bit
			out[0] = (dword >> 26) & 0x03;
			out[1] = (dword >> 24) & 0x03;
			out[2] = (dword >> 22) & 0x03;
			out[3] = (dword >> 20) & 0x03;
			out[4] = (dword >> 18) & 0x03;
			out[5] = (dword >> 16) & 0x03;
			out[6] = (dword >> 14) & 0x03;
			out[7] = (dword >> 12) & 0x03;
			out[8] = (dword >> 10) & 0x03;
			out[9] = (dword >> 8) & 0x03;
			out[10] = (dword >> 6) & 0x03;
			out[11] = (dword >> 4) & 0x03;
			out[12] = (dword >> 2) & 0x03;
			out[13] = dword & 0x03;

			out += 14;

			return;
		case 2: // 9 * 3-bit
			out[0] = (dword >> 25) & 0x07;
			out[1] = (dword >> 22) & 0x07;
			out[2] = (dword >> 19) & 0x07;
			out[3] = (dword >> 16) & 0x07;
			out[4] = (dword >> 13) & 0x07;
			out[5] = (dword >> 10) & 0x07;
			out[6] = (dword >> 7) & 0x07;
			out[7] = (dword >> 4) & 0x07;
			out[8] = (dword >> 1) & 0x07;

			out += 9;

			return;
		case 3: // 7 * 4-bit
			out[0] = (dword >> 24) & 0x0f;
			out[1] = (dword >> 20) & 0x0f;
			out[2] = (dword >> 16) & 0x0f;
			out[3] = (dword >> 12) & 0x0f;
			out[4] = (dword >> 8) & 0x0f;
			out[5] = (dword >> 4) & 0x0f;
			out[6] = dword & 0x0f;

			out += 7;

			return;
		case 4: // 5 * 5-bit
			out[0] = (dword >> 23) & 0x1f;
			out[1] = (dword >> 18) & 0x1f;
			out[2] = (dword >> 13) & 0x1f;
			out[3] = (dword >> 8) & 0x1f;
			out[4] = (dword >> 3) & 0x1f;

			out += 5;

			return;
		case 5: // 4 * 7-bit
			out[0] = (dword >> 21) & 0x7f;
			out[1] = (dword >> 14) & 0x7f;
			out[2] = (dword >> 7) & 0x7f;
			out[3] = dword & 0x7f;

			out += 4;

			return;
		case 6: // 3 * 9-bit
			out[0] = (dword >> 19) & 0x01ff;
			out[1] = (dword >> 10) & 0x01ff;
			out[2] = (dword >> 1) & 0x01ff;

			out += 3;

			return;
		case 7: // 2 * 14-bit
			out[0] = (dword >> 14) & 0x3fff;
			out[1] = dword & 0x3fff;

			out += 2;

			return;
		case 8: // 1 * 28-bit
			out[0] = dword & 0x0fffffff;

			++out;

			return;
		default: // Invalid selector values.
			std::cerr << "Invalid selector: " << selector << std::endl;
			throw std::runtime_error("Invalid selector for " + codecname() + ".");
		}
    }
};


#if CODECS_SSE_PREREQ(4, 1)
template <bool AllowOverShooting>
class Simple9<SSE, AllowOverShooting> : public Simple9Base<SSE, AllowOverShooting> {
public:
    using Simple9Base<SSE, AllowOverShooting>::SIMPLE9_LOGDESC;
    using Simple9Base<SSE, AllowOverShooting>::SIMPLE9_LEN;

    enum { kMaxDecodingCount = 28 }; // Maximum number of values decodeWord can
                                     // write out at each call. (case 0)

    static const std::string codecname() {
        return "Simple9_SSE";
    }

    static void decodeWord(uint32_t dword, uint32_t * &out) {
        __m128i set1_rslt_m128i;
        __m128i mul_rslt_m128i, mul_rslt2_m128i;
        __m128i srli_rslt_m128i;
        __m128i rslt_m128i;

        set1_rslt_m128i = _mm_set1_epi32(dword);
		const uint32_t selector = dword >> (32 - SIMPLE9_LOGDESC);
        assert(selector < SIMPLE9_LEN);
		switch (selector) {
		case 0: // 28 * 1-bit
            mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, SIMDMasks::Simple_SSE4_mul_msk_m128i[0]);

            // 4 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 27);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[1]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

            // 4 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 23);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[1]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

            // 4 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 19);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[1]);
			_mm_storeu_si128((__m128i *)(out + 4 * 2), rslt_m128i);

            // 4 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 15);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[1]);
			_mm_storeu_si128((__m128i *)(out + 4 * 3), rslt_m128i);

            // 4 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 11);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[1]);
			_mm_storeu_si128((__m128i *)(out + 4 * 4), rslt_m128i);

            // 4 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 7);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[1]);
			_mm_storeu_si128((__m128i *)(out + 4 * 5), rslt_m128i);

            // 4 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 3);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[1]);
			_mm_storeu_si128((__m128i *)(out + 4 * 6), rslt_m128i);

			out += 28;

			return;
		case 1: // 14 * 2-bit + 2 garbage values
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, SIMDMasks::Simple_SSE4_mul_msk_m128i[1]);

            // 4 * 2-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 26);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[2]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

            // 4 * 2-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 18);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[2]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

            // 4 * 2-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 10);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[2]);
			_mm_storeu_si128((__m128i *)(out + 4 * 2), rslt_m128i);

            // 2 * 2-bit + 2 garbage values
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 2);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[2]);
			_mm_storeu_si128((__m128i *)(out + 4 * 3), rslt_m128i);

			out += 14;

			return;
		case 2: // 9 * 3-bit + 3 garbage values
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, SIMDMasks::Simple_SSE4_mul_msk_m128i[2]);

            // 4 * 3-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 25);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[3]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

            // 4 * 3-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 13);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[3]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

            // 1 * 3-bit + 3 garbage values
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 1);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[3]);
			_mm_storeu_si128((__m128i *)(out + 4 * 2), rslt_m128i);

			out += 9;

			return;
		case 3: // 7 * 4-bit + 1 garbage value
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, SIMDMasks::Simple_SSE4_mul_msk_m128i[3]);

            // 4 * 4-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 24);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[4]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

            // 3 * 4-bit + 1 garbage value
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 8);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[4]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

			out += 7;

			return;
		case 4: // 5 * 5-bit + 3 garbage value
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, SIMDMasks::Simple_SSE4_mul_msk_m128i[4]);

            // 4 * 5-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 23);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[5]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

            // 1 * 5-bit + 3 garbage value
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 3);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[5]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

			out += 5;

			return;
		case 5: // 4 * 7-bit
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, SIMDMasks::Simple_SSE4_mul_msk_m128i[5]);

            // 4 * 7-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 21);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[7]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			out += 4;

			return;
		case 6: // 3 * 9-bit + 1 garbage value
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, SIMDMasks::Simple_SSE4_mul_msk_m128i[6]);

            // 3 * 9-bit + 1 garbage value
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 19);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[9]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			out += 3;

			return;
		case 7: // 2 * 14-bit + 2 garbage value.
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, SIMDMasks::Simple_SSE4_mul_msk_m128i[7]);

            // 2 * 14-bit + 2 garbage values
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 14);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[14]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			out += 2;

			return;
		case 8: // 1 * 28-bit + 3 garbage values
            // 1 * 28-bit + 3 garbage values
			rslt_m128i = _mm_and_si128(set1_rslt_m128i, SIMDMasks::SSE2_and_msk_m128i[28]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			++out;

			return;
		default: // invalid selector
			std::cerr << "Invalid selector: " << selector << std::endl;
			throw std::runtime_error("Invalid selector for " + codecname() + ".");
		}
    }
};
#endif /* __SSE4_1__ */


#if CODECS_AVX_PREREQ(2, 0)
template <bool AllowOverShooting>
class Simple9<AVX, AllowOverShooting> : public Simple9Base<AVX, AllowOverShooting> {
public:
    using Simple9Base<AVX, AllowOverShooting>::SIMPLE9_LOGDESC;
    using Simple9Base<AVX, AllowOverShooting>::SIMPLE9_LEN;

    enum { kMaxDecodingCount = 32 }; // Maximum number of values decodeWord can
                                     // write out at each call. (case 0)

    static const std::string codecname() {
        return "Simple9_AVX";
    }

    static void decodeWord(uint32_t dword, uint32_t * &out) {
        __m256i set1_rslt_m256i;
        __m256i srlv_rslt_m256i;
        __m256i rslt_m256i;

		set1_rslt_m256i = _mm256_set1_epi32(dword);
		const uint32_t selector = dword >> (32 - SIMPLE9_LOGDESC);
        assert(selector < SIMPLE9_LEN);
		switch (selector) {
		case 0: // 28 * 1-bit + 4 garbage values
            // 8 * 1-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[0][0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[1]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

            // 8 * 1-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[0][1]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[1]);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 1), rslt_m256i);

            // 8 * 1-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[0][2]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[1]);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 2), rslt_m256i);

            // 4 * 1-bit + 4 garbage values
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[0][3]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[1]);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 3), rslt_m256i);

			out += 28;

			return;
		case 1: // 14 * 2-bit + 2 garbage values
            // 8 * 2-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[1][0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[2]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

            // 6 * 2-bit + 2 garbage values
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[1][1]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[2]);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 1), rslt_m256i);

			out += 14;

			return;
		case 2: // 9 * 3-bit + 7 garbage values
            // 8 * 3-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[2][0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[3]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

            // 1 * 3-bit + 7 garbage values
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[2][1]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[3]);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 1), rslt_m256i);

			out += 9;

			return;
		case 3: // 7 * 4-bit + 1 garbage value
            // 7 * 4-bit + 1 garbage value
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[3][0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[4]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 7;

			return;
		case 4: // 5 * 5-bit + 3 garbage values
            // 5 * 5-bit + 3 garbage values
			srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[4][0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[5]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 5;

			return;
		case 5: // 4 * 7-bit + 4 garbage values
            // 4 * 7-bit + 4 garbage values
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[5][0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[7]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 4;

			return;
		case 6: // 3 * 9-bit + 5 garbage values
            // 3 * 9-bit + 5 garbage values
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[6][0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[9]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 3;

			return;
		case 7: // 2 * 14-bit + 6 garbage values
            // 2 * 14-bit + 6 garbage values
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, SIMDMasks::Simple_AVX2_srlv_msk_m256i[7][0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[14]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 2;

			return;
		case 8: // 1 * 28-bit + 7 garbage values
            // 1 * 28-bit + 7 garbage values
		    rslt_m256i = _mm256_and_si256(set1_rslt_m256i, SIMDMasks::AVX2_and_msk_m256i[28]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			++out;

			return;
		default: // invalid selector
			std::cerr << "Invalid selector: " << selector << std::endl;
			throw std::runtime_error("Invalid selector for " + codecname() + ".");
		}
    }
};
#endif /* __AVX2__ */

} // namespace Codecs

#endif // CODECS_SIMPLE9_H_
