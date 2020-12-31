/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Mar 10, 2015
 */

#ifndef CODECS_VARINTG8CU_H_
#define CODECS_VARINTG8CU_H_ 

#if !defined(__GNUC__) && !defined(_MSC_VER)
#error varintG8IU.h requires GCC or MSVC.
#endif

#include "Bits.h"
#include "IntegerCodec.h"
#include "Portability.h"

#if !CODECS_X64 && !defined(__i386__) && !CODECS_PPC64
#error varintG8IU.h requires x86[_64].
#endif


#if CODECS_SSE_PREREQ(3, 1)
namespace Codecs {
namespace varintTables {
extern const __m128i varintGU_SSSE3_shfl_msk_m128i[4][256][2];
} // namespace varintTables
} // namespace Codecs
#endif /* __SSSE3__ */

#if CODECS_AVX_PREREQ(2, 0)
namespace Codecs {
namespace varintTables {
extern const __m256i varintGU_AVX2_shfl_msk_m256i[4][256];
} // namespace varintTables
} // namespace Codecs
#endif /* __AVX2__ */

namespace Codecs {
namespace varintTables {
extern const uint8_t varintGULengths[256][8];
extern const uint8_t varintG8CUOutputOffsets[4][256];
extern const uint8_t varintG8CUStates[256];
extern const uint32_t kMask[4];
} // namespace varintTables
} // namespace Codecs


namespace Codecs {

template <typename InstructionSet, bool AllowOverShooting>
class varintG8CU;

using varintG8CU_Scalar = varintG8CU<Scalar, false>;

#if CODECS_SSE_PREREQ(3, 1)
using varintG8CU_SSE = varintG8CU<SSE, false>;
#endif /* __SSSE3__ */

#if CODECS_AVX_PREREQ(2, 0)
using varintG8CU_AVX = varintG8CU<AVX, false>;
#endif /* __AVX2__ */

/**
 * varintG8CU encoding for 32-bit integers.
 *
 * varintGU Groups 8 data bytes together along with 1 descriptor byte containing the unary
 * representations of the lengths of each encoded integer. (1=1 byte 10=2 bytes .. 1000=4 bytes)
 * The 8 data bytes may encode as few as 2 and as many as 8 integers, depending on their size.
 * In the complete block variation, which we call varintG8CU, we always fill all all eight bytes
 * in a data block.
 *
 * This implementation assumes little-endian and does unaligned 32-bit
 * accesses, so it's basically not portable outside of the x86[_64] world.
 *
 * Follows
 *
 * A. A. Stepanov, A. R. Gangolli, D. E. Rose, R. J. Ernst, and P. S. Oberoi. SIMD-based decoding
 * of posting lists. In Proc. Information and Knowledge Management CIKM, pages 317–326, 2011.
 */
template <typename InstructionSet, bool AllowOverShooting>
class varintG8CUBase : public IntegerCodec {
public:
    enum {
    	kHeaderSize = 1, // Number of bytes for the descriptor.
    	kGroupSize = 8,  // Number of bytes for the data, which is also
                         // the maximum possible number of encoded values.
		kFullGroupSize = kHeaderSize + kGroupSize // Number of bytes for the group.
    };

    static void encodeGroup(const uint32_t * &src, const uint32_t *const endsrc, uint8_t *dst, uint8_t &state) {
        // Reserve 1 byte for the descriptor.
    	uint8_t *const initdst = dst;
    	++dst; 

    	uint8_t desc = 0;
    	uint32_t totalLength = 0;
    	while (endsrc > src) {
			uint8_t length = bytes(src[0]) - state;
			totalLength += length;
			if (totalLength > kGroupSize) {
				break;
			}

			// Flip the corresponding termination bit in desc.
			desc |= static_cast<uint8_t>(1 << (totalLength - 1));

            // Right shift src[0] by state bytes, and write out the
            // least significant length bytes of the shift result to dst.
			*reinterpret_cast<uint32_t *>(dst) = src[0] >> (state * 8);
			dst += length;
			++src;

			state = 0;
    	}

        if (endsrc > src) { 
            // For all but the last group, write out the
            // least significant state bytes of src[0] to dst.
            state = varintTables::varintG8CUStates[desc];
            if (state > 0) {
                *reinterpret_cast<uint32_t *>(dst) = src[0];
                dst += state;
            }
        }
        else { 
            // For the last group, to make sure length <= 4 as required
            // by the function decodeGroup of class varintG8CU_Scalar,
            // flip all leading 0 bits of desc.
            state = varintTables::varintG8CUStates[desc];
            desc |= static_cast<uint8_t>(~((1 << (8-state)) - 1));
        }

    	initdst[0] = desc;
    }

    virtual void encodeArray(const uint32_t *in, uint64_t nvalue,
                uint32_t *out, uint64_t &csize) {
        const uint32_t *const endin = in + nvalue;
        uint8_t *dst = reinterpret_cast<uint8_t *>(out);
        const uint8_t *const initdst = dst;
        uint8_t state = 0;
        while (endin > in) {
            encodeGroup(in, endin, dst, state);
            dst += kFullGroupSize;
        }

        // Align to 4-byte boundary.
        csize = ((dst - initdst) + 3) / 4;
    }

    virtual const uint32_t * decodeArray(const uint32_t *in, uint64_t csize,
                uint32_t *out, uint64_t nvalue) {
        const uint8_t *src = reinterpret_cast<const uint8_t *>(in);
        const uint8_t *const initsrc = src;
        uint8_t *dst = reinterpret_cast<uint8_t *>(out);
        const uint8_t* const enddst = reinterpret_cast<uint8_t *>(out + nvalue);
        uint8_t state = 0;

        if (AllowOverShooting) {
            // This implementation is more efficient but
            // can overshoot, which might result in disaster, 
            // e.g., if out points to an array of size nvalue.
            while (enddst > dst) {
                Derived::decodeGroup(src, dst, state);
                src += kFullGroupSize;
            }
        }
        else {
            // When decoding a group, varintG8CU_SSE and varintG8CU_AVX
        	// always output sizeof(uint32_t)*kGroupSize bytes.
        	// Therefore when there're >= sizeof(uint32_t)*kGroupSize
        	// bytes to be decoded, it's safe to write the decoded bytes
        	// directly to out.
            while (enddst - dst >= sizeof(uint32_t) * kGroupSize) {
                Derived::decodeGroup(src, dst, state);
                src += kFullGroupSize;
            }

            // For the remaining (< sizeof(uint32_t)*kGroupSize)
            // bytes, to avoid overshooting, we write the decoded
            // bytes to the array buf first, and then copy only
            // those we need to out.
            uint32_t bytesRemaining = enddst - dst;
            uint8_t *ptr = buf;
            const uint8_t *const endptr = ptr + bytesRemaining;
            while (endptr > ptr) {
                Derived::decodeGroup(src, ptr, state);
                src += kFullGroupSize;
            }
            memcpy(dst, buf, sizeof(uint8_t) * bytesRemaining);
        }

        csize = ((src - initsrc) + 3) / 4;
        return in + csize;
    }

    virtual std::string name() const {
        return Derived::codecname();
    }

private:
    using Derived = varintG8CU<InstructionSet, AllowOverShooting>;

    // We determine the size of buf based on:
    // 1) There are at most sizeof(uint32_t)*kGroupSize-1 bytes
    //    remaining to be decoded;
    // 2) varintG8CU_SSE and varintG8CU_AVX can write out
    //    sizeof(uint32_t)*kGroupSize-1 garbage bytes when
    //    decoding the last group.
    uint8_t buf[(sizeof(uint32_t)*kGroupSize-1) * 2];
};


template <bool AllowOverShooting>
class varintG8CU<Scalar, AllowOverShooting> : public varintG8CUBase<Scalar, AllowOverShooting> {
public:
    using varintG8CUBase<Scalar, AllowOverShooting>::kHeaderSize;

    static const std::string codecname() {
        return "varintG8CU_Scalar";
    }

    static void decodeGroup(const uint8_t *src, uint8_t * &dst, uint8_t &state) {
        assert(state < 4);
    	uint8_t desc = src[0];
    	src += kHeaderSize;
    	const uint8_t outputOffset = varintTables::varintG8CUOutputOffsets[state][desc];
        uint8_t length = varintTables::varintGULengths[desc][0];
        assert(length <= 4);
        *reinterpret_cast<uint32_t *>(dst) = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[length - 1];
        src += length;

        uint8_t i = 0;
        for (uint8_t offset = 4 - state; offset < outputOffset; offset += 4) {
            length = varintTables::varintGULengths[desc][++i];
            assert(length <= 4);
            *reinterpret_cast<uint32_t *>(dst + offset) = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[length - 1];
            src += length;
        }

        dst += outputOffset;
        state = varintTables::varintG8CUStates[desc];
    }
};


#if CODECS_SSE_PREREQ(3, 1)
template <bool AllowOverShooting>
class varintG8CU<SSE, AllowOverShooting> : public varintG8CUBase<SSE, AllowOverShooting> {
public:
    using varintG8CUBase<SSE, AllowOverShooting>::kHeaderSize;

    static const std::string codecname() {
        return "varintG8CU_SSE";
    }

    template <bool branchless = true>
    static void decodeGroup(const uint8_t *src, uint8_t * &dst, uint8_t &state) {
        assert(state < 4);
    	uint8_t desc = src[0];
    	src += kHeaderSize;
    	const uint8_t outputOffset = varintTables::varintG8CUOutputOffsets[state][desc];
        const __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src));
        __m128i result = _mm_shuffle_epi8(val, varintTables::varintGU_SSSE3_shfl_msk_m128i[state][desc][0]);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), result);

        // If-conversion
        if (branchless) {
            result = _mm_shuffle_epi8(val, varintTables::varintGU_SSSE3_shfl_msk_m128i[state][desc][1]);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 16), result);
        }
        else {
            if (outputOffset + state > 16) { // branch misprediction is costly
                result = _mm_shuffle_epi8(val, varintTables::varintGU_SSSE3_shfl_msk_m128i[state][desc][1]);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 16), result);
            }
        }

        dst += outputOffset;
        state = varintTables::varintG8CUStates[desc];
	}
};
#endif /* __SSSE3__ */


#if CODECS_AVX_PREREQ(2, 0)
template <bool AllowOverShooting>
class varintG8CU<AVX, AllowOverShooting> : public varintG8CUBase<AVX, AllowOverShooting> {
public:
    using varintG8CUBase<AVX, AllowOverShooting>::kHeaderSize;

    static const std::string codecname() {
        return "varintG8CU_AVX";
    }

    static void decodeGroup(const uint8_t *src, uint8_t * &dst, uint8_t &state) {
        assert(state < 4);
    	uint8_t desc = src[0];
    	src += kHeaderSize;
    	const uint8_t outputOffset = varintTables::varintG8CUOutputOffsets[state][desc];

        const __m128i data = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src));
        const __m256i val = _mm256_broadcastsi128_si256(data);
//        const __m256i val = _mm256_inserti128_si256(_mm256_castsi128_si256(data), data, 1); // Slower alternative.
        __m256i result = _mm256_shuffle_epi8(val, varintTables::varintGU_AVX2_shfl_msk_m256i[state][desc]);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), result);

		dst += outputOffset;
		state = varintTables::varintG8CUStates[desc];
	}
};
#endif /* __AVX2__ */

} // namespace Codecs

#endif // CODECS_VARINTG8CU_H_ 
